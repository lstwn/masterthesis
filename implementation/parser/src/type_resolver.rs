//! This module provides a type resolver for our Datalog dialect.

use crate::{
    analysis::AggregatedRule,
    ast::{Atom, Body, Head, Predicate},
};
use compute::{
    error::SyntaxError,
    expr::Expr,
    relation::RelationType,
    type_resolver::{
        ExprType, ScalarType, TypeResolver as IncLogTypeResolver,
        TypeResolverContext as IncLogTypeResolverContext,
    },
};
use std::collections::HashMap;

#[derive(Default)]
pub struct TypeResolver {
    rules: HashMap<String, RelationType>,
    body_vars: HashMap<String, ScalarType>,
    current_rule: Option<String>,
}

impl TypeResolver {
    fn new_rule_context(&mut self, rule: &AggregatedRule) {
        self.body_vars.clear();
        self.current_rule = Some(rule.name().clone());
    }
    pub fn inquire_rule<T: AsRef<str>>(&self, name: T) -> Option<RelationType> {
        self.rules.get(name.as_ref()).cloned()
    }
    pub fn resolve_rule(&mut self, rule: &AggregatedRule) -> Result<&RelationType, SyntaxError> {
        self.new_rule_context(rule);

        // For extensional rules, there will be no bodies containing any atoms.
        rule.bodies()
            .try_for_each(|body| self.resolve_body(body))
            .map_err(|mut err| {
                // TODO: Use a proper error stack (like anyhow) to add context to errors.
                err.message.push_str(&format!(
                    " while type resolving predicate '{}'",
                    rule.name()
                ));
                err
            })?;
        // Having resolved all bodies' atoms, we have enough information to
        // resolve the head.
        let relation_type = self.resolve_head(rule.head())?;

        self.rules.insert(rule.name().clone(), relation_type);
        Ok(self.rules.get(rule.name()).unwrap())
    }
    fn resolve_head(&mut self, head: &Head) -> Result<RelationType, SyntaxError> {
        head.variables
            .iter()
            .map(|variable| {
                let identifier = &variable.identifier.inner;
                let scalar_type = variable
                    .initializer
                    .as_ref()
                    .map(|expr| {
                        let mut ctx = IncLogTypeResolverContext::new(&mut self.body_vars);
                        IncLogTypeResolver::default()
                            .resolve_expr(expr, &mut ctx)
                            .map(|expr_type| match expr_type {
                                ExprType::Scalar(scalar_type) => scalar_type,
                                _ => {
                                    unreachable!(
                                        "Only scalar types are allowed in expr of variables"
                                    )
                                }
                            })
                    })
                    // Until we support types in the grammar, we use a dummy type.
                    .unwrap_or(Ok(ScalarType::Null))?;
                Ok((identifier, scalar_type))
            })
            .collect::<Result<RelationType, SyntaxError>>()
    }
    fn resolve_body(&mut self, body: &Body) -> Result<(), SyntaxError> {
        body.atoms
            .iter()
            .try_for_each(|atom| self.resolve_atom(atom))
    }
    fn resolve_atom(&mut self, atom: &Atom) -> Result<(), SyntaxError> {
        match atom {
            // Only positive atoms introduce new variables.
            Atom::Positive(predicate) => self.resolve_predicate(predicate).map(|_| ()),
            // We can ignore comparisons, as they do not introduce new variables.
            // Also, we can ignore predicates of negative atoms, as they are only
            // valid if the variables they negate occur in another predicate
            // of a positive atom.
            Atom::Comparison(_) | Atom::Negative(_) => Ok(()),
        }
    }
    pub fn resolve_predicate(
        &mut self,
        predicate: &Predicate,
    ) -> Result<RelationType, SyntaxError> {
        #[allow(clippy::type_complexity)]
        let resolve_field_type: Box<
            dyn for<'a> Fn(&'a str) -> Result<ScalarType, SyntaxError>,
        > = {
            let predicate_type = self.rules.get(predicate.name());
            let is_self_recursive = self.current_rule.as_ref() == Some(predicate.name());

            match predicate_type {
                Some(predicate_type) => Box::new(move |field_name| {
                    predicate_type
                        .field_type(field_name)
                        .copied()
                        .ok_or_else(|| {
                            SyntaxError::new(format!(
                                "Unknown field '{}' of predicate '{}'",
                                field_name,
                                predicate.name()
                            ))
                        })
                }),
                None if is_self_recursive => {
                    // If the predicate is self-recursive, we can use the previously
                    // found body_vars to resolve the field's scalar type.
                    Box::new(|field_name| {
                        self.body_vars.get(field_name).copied().ok_or_else(|| {
                            SyntaxError::new(format!(
                                "Unknown field '{}' of self-recursive predicate '{}'",
                                field_name,
                                predicate.name()
                            ))
                        })
                    })
                }
                // As the rules are topologically sorted, we can assume that all
                // referenced predicates are already defined or self-recursive.
                None => unreachable!("Unknown predicate '{}'", predicate.name()),
            }
        };

        let projected_type: RelationType = predicate
            .variables
            .iter()
            .filter_map(move |var_stmt| {
                if var_stmt.is_unused() {
                    return None;
                }
                let target_name = &var_stmt.identifier.inner;
                let source_name = var_stmt
                    .initializer
                    .as_ref()
                    .map(|expr| match expr {
                        Expr::Var(var_expr) => &var_expr.name,
                        _ => {
                            unreachable!("Only var expressions are allowed in predicate variables")
                        }
                    })
                    .unwrap_or(target_name);
                Some(resolve_field_type(source_name).map(|scalar_type| (target_name, scalar_type)))
            })
            .collect::<Result<RelationType, SyntaxError>>()?;

        projected_type.into_iter().for_each(|(name, scalar_type)| {
            // Make the variables known to the rule context.
            self.body_vars.insert(name.clone(), *scalar_type);
        });

        Ok(projected_type)
    }
}

//! This module translates a valid execution order of
//! [`AggregatedRules`](crate::analysis::AggregatedRule) into an equivalent
//! list of [`Stmts`](compute::stmt::Stmt) that can be executed incrementally.

use crate::{
    analysis::{AggregatedRule, DistinctFlaggedBody},
    ast::{Atom, Body, Head, Predicate, VarStmt},
    type_resolver::TypeResolver,
};
use compute::{
    dbsp::{DbspInput, DbspInputs, RootCircuit},
    error::SyntaxError,
    expr::{
        AntiJoinExpr, BinaryExpr, DistinctExpr, EquiJoinExpr, Expr, ProjectionExpr, SelectionExpr,
        UnionExpr, VarExpr as IncLogVarExpr,
    },
    operator::Operator,
    relation::{RelationSchema, RelationType},
    stmt::{Code, ExprStmt, Stmt, VarStmt as IncLogVarStmt},
};
use compute::{expr::FixedPointIterExpr, stmt::BlockStmt};

pub struct Translator<'a> {
    aggregated_rules: Vec<AggregatedRule>,
    inputs: DbspInputs,
    root_circuit: &'a mut RootCircuit,
    type_resolver: TypeResolver,
}

impl<'a> Translator<'a> {
    /// The `from` argument is assumed to be in a valid execution order.
    pub fn new(root_circuit: &'a mut RootCircuit, from: Vec<AggregatedRule>) -> Self {
        Self {
            aggregated_rules: from,
            root_circuit,
            inputs: Default::default(),
            type_resolver: TypeResolver::default(),
        }
    }
    pub fn translate(mut self) -> Result<(DbspInputs, Code), SyntaxError> {
        let code: Code = std::mem::take(&mut self.aggregated_rules)
            .into_iter()
            .map(|rule| {
                // This ensures that all predicates and variables are already
                // known in case we later inquire about them.
                let _rule_type = self.type_resolver.resolve_rule(&rule)?;
                match rule {
                    rule if rule.is_extensional() => self.translate_extensional_rule(rule),
                    rule if rule.is_self_recursive() => self.translate_self_recursive_rule(rule),
                    rule if rule.is_intensional() => self.translate_intensional_rule(rule),
                    _ => unreachable!(),
                }
            })
            .collect::<Result<_, _>>()?;
        Ok((self.inputs, code))
    }
    fn translate_extensional_rule(&mut self, rule: AggregatedRule) -> Result<Stmt, SyntaxError> {
        debug_assert!(rule.is_extensional());

        let (head, bodies) = rule.into_head_and_bodies();

        let name = head.name.identifier.inner;
        let tuple_fields: Vec<String> = head
            .variables
            .into_iter()
            .map(|variable| {
                if variable.initializer.is_some() {
                    Err(SyntaxError::new(
                        "Extensional rules do not support initializers for their fields.",
                    ))
                } else {
                    Ok(variable.identifier.inner)
                }
            })
            .collect::<Result<_, SyntaxError>>()?;
        let key_fields = tuple_fields.clone();

        Ok(Stmt::from(IncLogVarStmt {
            name: name.clone(),
            initializer: Some(Expr::from(DbspInput::add(
                RelationSchema::new(name, tuple_fields, key_fields)?,
                self.root_circuit,
                &mut self.inputs,
            ))),
        }))
    }
    fn translate_intensional_rule(&mut self, rule: AggregatedRule) -> Result<Stmt, SyntaxError> {
        debug_assert!(rule.is_intensional() && !rule.is_self_recursive());

        let (head, bodies) = rule.into_head_and_bodies();
        let unionized_bodies = self.unionize_bodies(&head, bodies)?;

        Ok(Stmt::from(IncLogVarStmt {
            name: head.name.identifier.inner,
            initializer: Some(unionized_bodies),
        }))
    }
    fn translate_self_recursive_rule(&mut self, rule: AggregatedRule) -> Result<Stmt, SyntaxError> {
        debug_assert!(rule.is_intensional() && rule.is_self_recursive());

        let imports: Vec<(String, Expr)> = rule
            .recursive_atoms()
            .flat_map(|atom| match atom {
                Atom::Positive(predicate) | Atom::Negative(predicate) => {
                    if predicate.name() == rule.name() {
                        // We do not import the self-recursive predicate itself.
                        return None;
                    }
                    let name = predicate.name().clone();
                    let var_expr = Expr::from(IncLogVarExpr::new(predicate.name()));
                    Some((name, var_expr))
                }
                Atom::Comparison(_) => None,
            })
            .collect();

        let (head, non_rec_bodies, rec_bodies) = rule.into_head_and_non_rec_rec_bodies();

        let accumulator: (String, Expr) = (
            head.name().clone(),
            self.unionize_bodies(&head, non_rec_bodies.into_iter())?,
        );

        let step = BlockStmt {
            stmts: vec![Stmt::from(ExprStmt {
                expr: self.unionize_bodies(&head, rec_bodies.into_iter())?,
            })],
        };

        Ok(Stmt::from(IncLogVarStmt {
            name: head.name.identifier.inner,
            initializer: Some(Expr::from(FixedPointIterExpr {
                circuit: self.root_circuit.clone(),
                imports,
                accumulator,
                step,
            })),
        }))
    }
    fn translate_body(
        &mut self,
        body: DistinctFlaggedBody,
    ) -> Result<(RelationType, Expr), SyntaxError> {
        let (distinct, body) = body;
        let (condition, positive, negative) = Self::partition_atoms(body);

        let (positive_relation_type, folded_positive_atoms) = Self::try_fold_helper(
            positive,
            |predicate| Ok(self.translate_predicate(predicate)),
            |(left_type, left), (right_type, right)| {
                let attributes = None; // Potential issue here with name collisions.
                let join_keys = left_type
                    .intersect(&right_type)
                    .map(|(field, _scalar_type)| {
                        (
                            Expr::from(IncLogVarExpr::new(field)),
                            Expr::from(IncLogVarExpr::new(field)),
                        )
                    });
                let joined_expr = Expr::from(EquiJoinExpr {
                    left,
                    right,
                    on: join_keys.collect(),
                    attributes,
                });
                let joined_type = left_type.join(right_type);
                Ok((joined_type, joined_expr))
            },
        )?;

        let with_negative_atoms =
            negative
                .into_iter()
                .fold(folded_positive_atoms, |left, predicate| {
                    let (right_type, right) = self.translate_predicate(predicate);
                    Expr::from(AntiJoinExpr {
                        left,
                        right,
                        // We can just take the fields from the negative relation type
                        // and due to Datalog's safety condition, the variables referenced
                        // in the negative atom must be a subset of the positive ones.
                        on: right_type
                            .into_iter()
                            .map(|field| {
                                (
                                    Expr::from(IncLogVarExpr::new(field.0)),
                                    Expr::from(IncLogVarExpr::new(field.0)),
                                )
                            })
                            .collect(),
                    })
                });

        let with_free_floating_conditions = match condition {
            Some(condition) => Expr::from(SelectionExpr {
                relation: with_negative_atoms,
                condition,
            }),
            None => with_negative_atoms,
        };

        let with_distinct = if distinct {
            Expr::from(DistinctExpr {
                relation: with_free_floating_conditions,
            })
        } else {
            with_free_floating_conditions
        };

        Ok((positive_relation_type, with_distinct))
    }
    fn translate_predicate(&mut self, predicate: Predicate) -> (RelationType, Expr) {
        // The rule type is the type of the rule the predicate is referencing.
        let rule_type = self.get_rule_type(predicate.name());
        let predicate_type = self.get_predicate_type(&predicate);
        let relation = Expr::from(IncLogVarExpr::from(predicate.name));
        let translated = self.possibly_project((rule_type, relation), predicate.variables);
        (predicate_type, translated)
    }
    fn unionize_bodies(
        &mut self,
        head: &Head,
        bodies: impl DoubleEndedIterator<Item = DistinctFlaggedBody>
            + ExactSizeIterator<Item = DistinctFlaggedBody>,
    ) -> Result<Expr, SyntaxError> {
        let bodies_count = bodies.len();
        Self::try_fold_helper(
            bodies,
            |body| {
                let typed_relation = self.translate_body(body)?;
                // Each body needs to be projected to the rule's head's variables
                // to be union-compatible.
                Ok(self.possibly_project(typed_relation, head.variables.clone()))
            },
            |mut left, right| {
                if let Expr::Union(union) = &mut left {
                    union.relations.push(right);
                    return Ok(left);
                }
                let mut relations = Vec::with_capacity(bodies_count);
                relations.push(left);
                relations.push(right);
                Ok(Expr::from(UnionExpr { relations }))
            },
        )
    }
    fn possibly_project(
        &mut self,
        typed_input_relation: (RelationType, Expr),
        attributes: Vec<VarStmt>,
    ) -> Expr {
        let (input_relation_type, mut relation) = typed_input_relation;
        // The comparison below is vulnerable to duplicates in `attributes` but
        // these things should be addressed by a (future) type checker.
        if input_relation_type == attributes.iter().map(|var| &var.identifier.inner) {
            // If the relation's type is the same as the attributes,
            // we can optimize away the projection.
            return relation;
        }
        let attributes: Vec<(String, Expr)> = attributes
            .into_iter()
            .filter_map(|variable| variable.into_projection_attribute())
            .collect();
        match relation {
            Expr::EquiJoin(ref mut join_expr) => {
                // If the relation is an EquiJoin, we can do the projection
                // as part of the join and again optimize away the projection.
                join_expr.attributes = Some(attributes);
                relation
            }
            _ => Expr::from(ProjectionExpr {
                relation,
                attributes,
            }),
        }
    }
    fn get_rule_type<T: AsRef<str>>(&mut self, name: T) -> RelationType {
        self.type_resolver.inquire_rule(name).expect(
            "Rule has already been successfully type resolved and referenced rule must be known",
        )
    }
    fn get_predicate_type(&mut self, predicate: &Predicate) -> RelationType {
        self.type_resolver
            .resolve_predicate(predicate)
            .expect("Rule has already been successfully type resolved and predicate must be known")
    }
    fn partition_atoms(body: Body) -> (Option<Expr>, Vec<Predicate>, Vec<Predicate>) {
        let mut positive: Vec<Predicate> = Vec::with_capacity(body.atoms.len());
        let mut negative: Vec<Predicate> = Vec::with_capacity(body.atoms.len() / 2);
        let mut comparisons: Vec<Expr> = Vec::with_capacity(body.atoms.len() / 2);

        for atom in body.atoms {
            match atom {
                Atom::Positive(predicate) => positive.push(predicate),
                Atom::Negative(predicate) => negative.push(predicate),
                Atom::Comparison(expr) => comparisons.push(expr),
            }
        }

        let condition: Option<Expr> = comparisons
            .into_iter()
            // We AND all filters into one condition expression to rule them all.
            .reduce(|acc, expr| {
                Expr::from(BinaryExpr {
                    operator: Operator::And,
                    left: acc,
                    right: expr,
                })
            });

        (condition, positive, negative)
    }
    fn try_fold_helper<I, T, U, Map, Fold>(
        sequence: impl IntoIterator<IntoIter = I>,
        mut map: Map,
        mut fold: Fold,
    ) -> Result<U, SyntaxError>
    where
        I: DoubleEndedIterator<Item = T>,
        Map: FnMut(T) -> Result<U, SyntaxError>,
        Fold: FnMut(U, U) -> Result<U, SyntaxError>,
    {
        // We start with the rightmost predicate and fold the rest into it.
        // We go from right to left to avoid extra allocations and have the
        // rightmost element end up as the innermost relation expression.
        let mut iter = sequence.into_iter().rev();
        let first = iter
            .next()
            .ok_or_else(|| {
                SyntaxError::new("At least one element is required to fold the sequence.")
            })
            .and_then(&mut map)?;

        let folded = iter.try_fold(first, |acc, unmapped| {
            let mapped = map(unmapped)?;
            // We reverse the arguments to account for the right-to-left folding
            // and provide the illusion of left-to-right folding for callers.
            fold(mapped, acc)
        })?;

        Ok(folded)
    }
    // TODO: Check safety condition
    // fn check_all_vars_at_least_once_positive()
    // TODO: Check range restriction
    // fn check_range_restriction()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{crdts::mvr_crdt_store_datalog, Parser};

    fn parse_and_translate(input: &str) -> Result<(DbspInputs, Code), SyntaxError> {
        // A hacky way to obtain/leak a `RootCircuit` for testing purposes.
        let (_handle, mut root_circuit) =
            RootCircuit::build(|circuit| Ok(circuit.clone())).unwrap();
        let mut parser = Parser::new(&mut root_circuit);
        let (inputs, naive_program) = parser.parse(input)?;
        Ok((inputs, naive_program))
    }

    #[test]
    fn test_translation() -> Result<(), anyhow::Error> {
        // TODO:
        // 1. [ ] How to specify which rule should be the result of the query?
        //        Implicitly, the rule that is last in the topological sort order.
        //        In case of a tie, the last rule in the input via a _stable_
        //        topological sort. Add this to thesis!
        let (inputs, code) = parse_and_translate(mvr_crdt_store_datalog())?;
        println!("Inputs: {:#?}", inputs);
        println!("Code: {:#?}", code);
        Ok(())
    }
}

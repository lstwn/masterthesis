//! This module translates a valid execution order of
//! [`AggregatedRules`](crate::analysis::AggregatedRule) into an equivalent
//! list of [`Stmts`](compute::stmt::Stmt) that can be executed incrementally.

use crate::{
    analysis::AggregatedRule,
    ast::{Atom, Body, Predicate},
};
use compute::{
    dbsp::{DbspInput, DbspInputs, RootCircuit},
    error::SyntaxError,
    expr::{BinaryExpr, CartesianProductExpr, Expr, ProjectionExpr, SelectionExpr, UnionExpr},
    operator::Operator,
    relation::RelationSchema,
    stmt::{Code, Stmt, VarStmt},
};

pub struct Translator<'a> {
    from: Vec<AggregatedRule>,
    inputs: DbspInputs,
    root_circuit: &'a mut RootCircuit,
}

impl AggregatedRule {
    pub fn is_extensional(&self) -> bool {
        self.bodies.is_empty()
    }
    pub fn is_intensional(&self) -> bool {
        !self.is_extensional()
    }
    pub fn is_self_recursive(&self) -> bool {
        self.atoms().any(|atom| match atom {
            Atom::Positive(predicate) => predicate.name() == self.head.name(),
            Atom::Negative(predicate) => predicate.name() == self.head.name(),
            Atom::Comparison(_) => false,
        })
    }
}

impl<'a> Translator<'a> {
    pub fn new(root_circuit: &'a mut RootCircuit, from: Vec<AggregatedRule>) -> Self {
        Self {
            from,
            root_circuit,
            inputs: Default::default(),
        }
    }
    pub fn translate(mut self) -> Result<(DbspInputs, Code), SyntaxError> {
        let code: Code = std::mem::take(&mut self.from)
            .into_iter()
            .map(|rule| match rule {
                rule if rule.is_extensional() => self.translate_extensional_rule(rule),
                rule if rule.is_self_recursive() => self.translate_self_recursive_rule(rule),
                rule if rule.is_intensional() => self.translate_intensional_rule(rule),
                _ => unreachable!(),
            })
            .collect::<Result<_, _>>()?;
        Ok((self.inputs, code))
    }
    fn translate_extensional_rule(&mut self, rule: AggregatedRule) -> Result<Stmt, SyntaxError> {
        debug_assert!(rule.is_extensional());

        let name = rule.head.name.identifier.inner;
        let tuple_fields: Vec<String> = rule
            .head
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

        Ok(Stmt::from(VarStmt {
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

        let mapper = |body: Body| -> Result<Expr, SyntaxError> {
            let relation = Translator::translate_body(body)?;
            // Each body needs to be projected to the head variables.
            Ok(Expr::from(ProjectionExpr {
                relation,
                attributes: rule
                    .head
                    .variables
                    .clone()
                    .into_iter()
                    .map(|variable| variable.into_projection_attribute())
                    .collect(),
            }))
        };

        let bodies_count = rule.bodies.len();
        let unionized_bodies = Self::fold_utility(rule.bodies, mapper, |mut left, body| {
            let right = mapper(body)?;
            if let Expr::Union(union) = &mut left {
                union.relations.push(right);
                return Ok(left);
            }
            let mut relations = Vec::with_capacity(bodies_count);
            relations.push(left);
            relations.push(right);
            Ok(Expr::from(UnionExpr { relations }))
        })?;

        Ok(Stmt::from(VarStmt {
            name: rule.head.name.identifier.inner,
            initializer: Some(unionized_bodies),
        }))
    }
    fn translate_self_recursive_rule(&mut self, rule: AggregatedRule) -> Result<Stmt, SyntaxError> {
        debug_assert!(rule.is_intensional() && rule.is_self_recursive());

        todo!("TODO: How to handle self-recursive rules?");
    }
    fn translate_body(body: Body) -> Result<Expr, SyntaxError> {
        let (condition, positive, negative) = Self::partition_atoms(body);

        if !negative.is_empty() {
            // TODO: How to translate negative predicates?
            return Err(SyntaxError::new(
                "Negative predicates are not supported yet.",
            ));
        }

        // TODO: Support at most one negative predicate.

        let folded_positive_atoms = Self::fold_utility(
            positive,
            |predicate| Ok(Self::translate_predicate(predicate)),
            |left, predicate| {
                let attributes = None; // Potential issue here with name collisions.
                let right = Self::translate_predicate(predicate);
                Ok(Expr::from(CartesianProductExpr::new(
                    left, right, attributes,
                )))
            },
        )?;

        // TODO: How to handle the variable name reuse and turn them into filters?

        let relation = match condition {
            Some(condition) => Expr::from(SelectionExpr {
                relation: folded_positive_atoms,
                condition,
            }),
            None => folded_positive_atoms,
        };

        Ok(relation)
    }
    fn translate_predicate(predicate: Predicate) -> Expr {
        Expr::from(ProjectionExpr {
            relation: Expr::from(predicate.name),
            attributes: predicate
                .variables
                .into_iter()
                .map(|variable| variable.into_projection_attribute())
                .collect(),
        })
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
    fn fold_utility<T, Map, Fold>(
        mut sequence: Vec<T>,
        map: Map,
        fold: Fold,
    ) -> Result<Expr, SyntaxError>
    where
        Map: FnMut(T) -> Result<Expr, SyntaxError>,
        Fold: FnMut(Expr, T) -> Result<Expr, SyntaxError>,
    {
        // We take the last predicate and fold the rest into it.
        // We go from right to left to avoid extra allocations and have it end
        // up as the innermost relation expression.
        let first = sequence
            .pop()
            .ok_or_else(|| {
                SyntaxError::new("At least one element is required to fold the sequence.")
            })
            .and_then(map)?;

        let folded = sequence.into_iter().try_rfold(first, fold)?;

        Ok(folded)
    }
    // fn check_all_vars_at_least_once_positive()
    // fn check_range_restriction()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Parser;

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
        let input = r#"
            // These are extensional database predicates (EDBPs).
            pred(FromNodeId, FromCounter, ToNodeId, ToCounter)  :- .
            set(NodeId, Counter, Key, Value)                    :- .

            // These are intensional database predicates (IDBPs).
            overwritten(NodeId, Counter)     :- pred(NodeId = FromNodeId, Counter = FromCounter, _ToNodeId, _ToCounter).
            overwrites(NodeId, Counter)      :- pred(_FromNodeId, _FromCounter, NodeId = ToNodeId, Counter = ToCounter).
        "#;
        let (inputs, code) = parse_and_translate(input)?;
        println!("Inputs: {:#?}", inputs);
        // The outermost projection could be eliminated, prevent here or in optimizer?
        println!("Code: {:#?}", code);
        Ok(())
    }
}

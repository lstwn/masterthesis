//! This module parses the following grammar of a Datalog variant:
//! ```ebnf
//! program     = rule* EOF ;
//! rule        = head ":-" body "." ;
//! head        = "distinct"? IDENTIFIER "(" field ( "," field )* ")" ;
//! field       = IDENTIFIER ( "=" expression )? ;
//! body        = ( atom ( "," atom )* )? ;
//! atom        = ( "not"? predicate ) | "(" expression ")" | comparison ;
//! predicate   = IDENTIFIER "(" variable ( "," variable )* ")" ;
//! variable    = IDENTIFIER ( "=" IDENTIFIER )? ;
//! ```
//! An empty body is allowed to define extensional database predicates (EDBPs).
//!
//! All parser functions assume that there is no leading whitespace in their inputs.

use crate::{
    ast::{Atom, Body, Head, Predicate, Program, Rule, VarExpr, VarStmt},
    expr::{comparison, expression},
    literal::identifier,
    parser_helper::{lead_ws, lead_ws_cmt},
};
use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::tag,
    character::complete::multispace1,
    combinator::{eof, map, opt, value},
    multi::{fold_many0, separated_list1},
    sequence::{delimited, pair, preceded, terminated},
};

const DIVIDER: &str = ":-";
const COMMA: &str = ",";
const DOT: &str = ".";
const NOT: &str = "not";
const ASSIGN: &str = "=";
const DISTINCT: &str = "distinct";

const LEFT_PAREN: &str = "(";
const RIGHT_PAREN: &str = ")";

pub fn program(input: &str) -> IResult<&str, Program> {
    fold_many0(lead_ws_cmt(rule), Program::default, |mut program, rule| {
        program.rules.push(rule);
        program
    })
    .parse(input)
    // We eat trailing whitespace as well as comments and finally match EOL.
    .and_then(|(input, program)| value(program, lead_ws_cmt(eof)).parse(input))
}

fn rule(input: &str) -> IResult<&str, Rule> {
    head.parse(input).and_then(|(input, head)| {
        delimited(lead_ws(tag(DIVIDER)), lead_ws(body), lead_ws(tag(DOT)))
            .parse(input)
            .map(|(input, body)| (input, Rule { head, body }))
    })
}

fn head(input: &str) -> IResult<&str, Head> {
    let distinct = terminated(tag(DISTINCT), multispace1);
    pair(opt(distinct), lead_ws(identifier))
        .parse(input)
        .and_then(|(input, (distinct, name))| {
            delimited(
                lead_ws(tag(LEFT_PAREN)),
                separated_list1(lead_ws(tag(COMMA)), lead_ws(field)),
                lead_ws(tag(RIGHT_PAREN)),
            )
            .parse(input)
            .map(|(input, variables)| {
                (
                    input,
                    if distinct.is_some() {
                        Head::with_distinct(name, variables)
                    } else {
                        Head::new(name, variables)
                    },
                )
            })
        })
}

fn field(input: &str) -> IResult<&str, VarStmt> {
    let (input, name) = identifier.parse(input)?;
    let (input, expr) = opt(preceded(lead_ws(tag(ASSIGN)), lead_ws(expression))).parse(input)?;
    if let Some(source_name) = expr {
        Ok((input, VarStmt::with_expr(name, source_name)))
    } else {
        Ok((input, VarStmt::new(name)))
    }
}

fn body(input: &str) -> IResult<&str, Body> {
    map(
        opt(separated_list1(lead_ws(tag(COMMA)), lead_ws_cmt(atom))),
        |atoms| Body {
            atoms: atoms.unwrap_or_default(),
        },
    )
    .parse(input)
}

fn atom(input: &str) -> IResult<&str, Atom> {
    let not = terminated(tag(NOT), multispace1);
    let positive_or_negative = map(pair(opt(not), lead_ws(predicate)), |(not, predicate)| {
        if not.is_none() {
            Atom::Positive(predicate)
        } else {
            Atom::Negative(predicate)
        }
    });
    // Only parenthesized expressions are allowed to contain logical operators,
    // such as `and` and `or`, on the top level. Otherwise, we cannot escape
    // to the next atom containing another comparison, but are forced to extend
    // the current expression with an `and`.
    let parenthesized_expression = delimited(
        tag(LEFT_PAREN),
        map(lead_ws(expression), Atom::Comparison),
        lead_ws(tag(RIGHT_PAREN)),
    );
    // But we allow comparisons to use logical operators on a _nested_ level.
    // See test [`test_program_with_complex_filters`] below.
    let comparison = map(comparison, Atom::Comparison);

    alt((positive_or_negative, parenthesized_expression, comparison)).parse(input)
}

fn predicate(input: &str) -> IResult<&str, Predicate> {
    identifier.parse(input).and_then(|(input, name)| {
        delimited(
            lead_ws(tag(LEFT_PAREN)),
            separated_list1(lead_ws(tag(COMMA)), lead_ws(variable)),
            lead_ws(tag(RIGHT_PAREN)),
        )
        .parse(input)
        .map(|(input, variables)| {
            (
                input,
                Predicate {
                    name: VarExpr::from(name),
                    variables,
                },
            )
        })
    })
}

fn variable(input: &str) -> IResult<&str, VarStmt> {
    let (input, target_name) = identifier.parse(input)?;
    let (input, source_name) =
        opt(preceded(lead_ws(tag(ASSIGN)), lead_ws(identifier))).parse(input)?;
    if let Some(source_name) = source_name {
        Ok((input, VarStmt::with_alias(target_name, source_name)))
    } else {
        Ok((input, VarStmt::new(target_name)))
    }
}

#[cfg(test)]
pub mod test {
    use crate::{key_value_store_crdts::MVR_KV_STORE_CRDT_DATALOG, list_crdt::LIST_CRDT_DATALOG};

    use super::*;
    use compute::{
        expr::{BinaryExpr, Expr, GroupingExpr, LiteralExpr, VarExpr as IncLogVarExpr},
        operator::Operator,
    };

    #[test]
    fn test_atom() {
        let input = "y(a, b)";
        let result = atom(input);
        let expected = Atom::Positive(Predicate {
            name: VarExpr::new("y"),
            variables: vec![VarStmt::new("a"), VarStmt::new("b")],
        });
        assert_eq!(result, Ok(("", expected)));

        let input = "not y(a, b)";
        let result = atom(input);
        let expected = Atom::Negative(Predicate {
            name: VarExpr::new("y"),
            variables: vec![VarStmt::new("a"), VarStmt::new("b")],
        });
        assert_eq!(result, Ok(("", expected)));
    }

    #[test]
    fn test_rule() {
        let input = "notName1(a) :- notName2(a).";
        let result = rule(input);
        let expected = Rule {
            head: Head::new("notName1", ["a"]),
            body: Body {
                // We ensure that this is still a positive atom and insist
                // that the name contains the prefix `not` keyword.
                atoms: vec![Atom::Positive(Predicate {
                    name: VarExpr::new("notName2"),
                    variables: vec![VarStmt::new("a")],
                })],
            },
        };
        assert_eq!(result, Ok(("", expected)));
    }

    #[test]
    fn test_program() {
        let input = r#"
            // Leading eol-comment.
            x(a, b = b + 1)  :- y(a, b, c), c > 2.
            // First line of eol-comment.
            // Second line of eol-comment.
            distinct z(a, b) :- y(a, b), not y(a, b).
            // Trailing eol-comment.
        "#;
        let result = program(input);
        let expected = Program {
            rules: vec![
                Rule {
                    head: Head::new(
                        "x",
                        [
                            VarStmt::new("a"),
                            VarStmt::with_expr(
                                "b",
                                Expr::from(BinaryExpr {
                                    operator: Operator::Addition,
                                    left: Expr::from(IncLogVarExpr::new("b")),
                                    right: Expr::from(LiteralExpr::from(1_u64)),
                                }),
                            ),
                        ],
                    ),
                    body: Body {
                        atoms: vec![
                            Atom::Positive(Predicate {
                                name: VarExpr::new("y"),
                                variables: vec![
                                    VarStmt::new("a"),
                                    VarStmt::new("b"),
                                    VarStmt::new("c"),
                                ],
                            }),
                            Atom::Comparison(Expr::from(BinaryExpr {
                                operator: Operator::Greater,
                                left: Expr::from(IncLogVarExpr::new("c")),
                                right: Expr::from(LiteralExpr::from(2_u64)),
                            })),
                        ],
                    },
                },
                Rule {
                    head: Head::with_distinct("z", ["a", "b"]),
                    body: Body {
                        atoms: vec![
                            Atom::Positive(Predicate {
                                name: VarExpr::new("y"),
                                variables: vec![VarStmt::new("a"), VarStmt::new("b")],
                            }),
                            Atom::Negative(Predicate {
                                name: VarExpr::new("y"),
                                variables: vec![VarStmt::new("a"), VarStmt::new("b")],
                            }),
                        ],
                    },
                },
            ],
        };
        assert_eq!(result, Ok(("", expected)));
    }

    #[test]
    fn test_program_with_complex_filters() {
        let input = r#"
            x(a, b, c) :- y(a, b, c),
                          // This is just a comparison.
                          a == 0,
                          // This is a logical expression which requires parenthesis.
                          (a > 2; b == 2, c == 3),
                          // This comparison does use logical operators but not
                          // on the top level, hence, no parenthesis are required.
                          true == (a == b; b == c).
        "#;
        let result = program(input);
        println!("{result:#?}");
        let expected = Program {
            rules: vec![Rule {
                head: Head::new(
                    "x",
                    [VarStmt::new("a"), VarStmt::new("b"), VarStmt::new("c")],
                ),
                body: Body {
                    atoms: vec![
                        Atom::Positive(Predicate {
                            name: VarExpr::new("y"),
                            variables: vec![
                                VarStmt::new("a"),
                                VarStmt::new("b"),
                                VarStmt::new("c"),
                            ],
                        }),
                        Atom::Comparison(Expr::from(BinaryExpr {
                            operator: Operator::Equal,
                            left: Expr::from(IncLogVarExpr::new("a")),
                            right: Expr::from(LiteralExpr::from(0_u64)),
                        })),
                        Atom::Comparison(Expr::from(BinaryExpr {
                            operator: Operator::Or,
                            left: Expr::from(BinaryExpr {
                                operator: Operator::Greater,
                                left: Expr::from(IncLogVarExpr::new("a")),
                                right: Expr::from(LiteralExpr::from(2_u64)),
                            }),
                            right: Expr::from(BinaryExpr {
                                operator: Operator::And,
                                left: Expr::from(BinaryExpr {
                                    operator: Operator::Equal,
                                    left: Expr::from(IncLogVarExpr::new("b")),
                                    right: Expr::from(LiteralExpr::from(2_u64)),
                                }),
                                right: Expr::from(BinaryExpr {
                                    operator: Operator::Equal,
                                    left: Expr::from(IncLogVarExpr::new("c")),
                                    right: Expr::from(LiteralExpr::from(3_u64)),
                                }),
                            }),
                        })),
                        Atom::Comparison(Expr::from(BinaryExpr {
                            operator: Operator::Equal,
                            left: Expr::from(LiteralExpr::from(true)),
                            right: Expr::from(GroupingExpr {
                                expr: Expr::from(BinaryExpr {
                                    operator: Operator::Or,
                                    left: Expr::from(BinaryExpr {
                                        operator: Operator::Equal,
                                        left: Expr::from(IncLogVarExpr::new("a")),
                                        right: Expr::from(IncLogVarExpr::new("b")),
                                    }),
                                    right: Expr::from(BinaryExpr {
                                        operator: Operator::Equal,
                                        left: Expr::from(IncLogVarExpr::new("b")),
                                        right: Expr::from(IncLogVarExpr::new("c")),
                                    }),
                                }),
                            }),
                        })),
                    ],
                },
            }],
        };
        assert_eq!(result, Ok(("", expected)));
    }

    #[test]
    fn test_mvr_store_crdt() {
        let result = program(MVR_KV_STORE_CRDT_DATALOG);
        // Here, we just check that the parser consumes the full input.
        assert_eq!(result.as_ref().map(|(input, program)| *input), Ok(""));
        println!("{result:#?}");
    }

    #[test]
    fn test_list_crdt() {
        let result = program(LIST_CRDT_DATALOG);
        // Here, we just check that the parser consumes the full input.
        assert_eq!(result.as_ref().map(|(input, program)| *input), Ok(""));
        println!("{result:#?}");
    }
}

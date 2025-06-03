//! This module parses the following grammar of a Datalog variant:
//! ```ebnf
//! program     = rule* EOF ;
//! rule        = head ":-" body "." ;
//! head        = IDENTIFIER "(" field ( "," field )* ")" ;
//! field       = IDENTIFIER ( "=" comparison )? ;
//! body        = ( atom ( "," atom )* )? ;
//! atom        = ( "not"? predicate ) | comparison ;
//! predicate   = IDENTIFIER "(" variable ( "," variable )* ")" ;
//! variable    = IDENTIFIER ( "=" IDENTIFIER )? ;
//! ```
//! An empty body is allowed to define extensional database predicates (EDBPs).
//!
//! All parser functions assume that there is no leading whitespace in their inputs.

use crate::{
    ast::{Atom, Body, Head, Predicate, Program, Rule, VarExpr, VarStmt},
    expr,
    literal::identifier,
    parser_helper::{lead_ws, lead_ws_cmt},
};
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::multispace1,
    combinator::{eof, map, opt, value},
    multi::{fold_many0, separated_list1},
    sequence::{delimited, pair, preceded, terminated},
    IResult, Parser,
};

const DIVIDER: &str = ":-";
const COMMA: &str = ",";
const DOT: &str = ".";
const NOT: &str = "not";
const ASSIGN: &str = "=";

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
    identifier.parse(input).and_then(|(input, name)| {
        delimited(
            lead_ws(tag(LEFT_PAREN)),
            separated_list1(lead_ws(tag(COMMA)), lead_ws(field)),
            lead_ws(tag(RIGHT_PAREN)),
        )
        .parse(input)
        .map(|(input, variables)| {
            (
                input,
                Head {
                    name: VarStmt::new(name),
                    variables,
                },
            )
        })
    })
}

fn field(input: &str) -> IResult<&str, VarStmt> {
    let (input, name) = identifier.parse(input)?;
    let (input, expr) =
        opt(preceded(lead_ws(tag(ASSIGN)), lead_ws(expr::comparison))).parse(input)?;
    if let Some(source_name) = expr {
        Ok((input, VarStmt::with_expr(name, source_name)))
    } else {
        Ok((input, VarStmt::new(name)))
    }
}

fn body(input: &str) -> IResult<&str, Body> {
    map(
        opt(separated_list1(lead_ws(tag(COMMA)), lead_ws(atom))),
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
    let comparison = map(expr::comparison, Atom::Comparison);

    alt((positive_or_negative, comparison)).parse(input)
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
    use super::*;
    use compute::{
        expr::{BinaryExpr, Expr, LiteralExpr, VarExpr as IncLogVarExpr},
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
            head: Head {
                name: VarStmt::new("notName1"),
                variables: vec![VarStmt::new("a")],
            },
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
            x(a, b = b + 1) :- y(a, b, c), c > 2.
            // First line of eol-comment.
            // Second line of eol-comment.
            z(a, b)     :- y(a, b), not y(a, b).
            // Trailing eol-comment.
        "#;
        let result = program(input);
        let expected = Program {
            rules: vec![
                Rule {
                    head: Head {
                        name: VarStmt::new("x"),
                        variables: vec![
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
                    },
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
                    head: Head {
                        name: VarStmt::new("z"),
                        variables: vec![VarStmt::new("a"), VarStmt::new("b")],
                    },
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

    fn mvr_store_crdt_program() -> &'static str {
        r#"
            // These are extensional database predicates (EDBPs).
            pred(FromNodeId, FromCounter, ToNodeId, ToCounter)  :- .
            set(NodeId, Counter, Key, Value)                    :- .

            // These are intensional database predicates (IDBPs).
            overwritten(NodeId, Counter)     :- pred(NodeId = FromNodeId, Counter = FromCounter, _ToNodeId, _ToCounter).
            overwrites(NodeId, Counter)      :- pred(_FromNodeId, _FromCounter, NodeId = ToNodeId, Counter = ToCounter).

            isRoot(NodeId, Counter)          :- set(NodeId, Counter, _Key, _Value),
                                                not overwrites(NodeId, Counter).

            isLeaf(NodeId, Counter)          :- set(NodeId, Counter, _Key, _Value),
                                                not overwritten(NodeId, Counter).

            isCausallyReady(NodeId, Counter) :- isRoot(NodeId, Counter).
            isCausallyReady(NodeId, Counter) :- isCausallyReady(FromNodeId = NodeId, FromCounter = Counter),
                                                pred(FromNodeId, FromCounter, NodeId = ToNodeId, Counter = ToCounter).

            mvrStore(Key, Value)             :- isLeaf(NodeId, Counter),
                                                isCausallyReady(NodeId, Counter),
                                                set(NodeId, Counter, Key, Value).
        "#
    }

    pub fn mvr_store_crdt_ast() -> Program {
        program(mvr_store_crdt_program()).unwrap().1
    }

    #[test]
    fn test_mvr_store_crdt() {
        let result = program(mvr_store_crdt_program());
        // TODO: Compare the full program but for now we just check that
        // the parser consumes the full input.
        assert_eq!(result.as_ref().map(|(input, program)| *input), Ok(""));
        println!("{:#?}", result);
    }
}

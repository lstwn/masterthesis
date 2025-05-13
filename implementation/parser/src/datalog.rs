//! This module parses the following grammar of a Datalog variant:
//! ```ebnf
//! program     = rule* EOF ;
//! rule        = head ":-" body "." ;
//! head        = IDENTIFIER "(" comparison ( "," comparison )* ")" ;
//! body        = ( atom ( "," atom )* )? ;
//! atom        = ( "not"? predicate ) | comparison ;
//! predicate   = IDENTIFIER "(" IDENTIFIER ( "," IDENTIFIER )* ")" ;
//! ```
//! An empty body is allowed to define extensional database predicates (EDBPs).

use crate::{
    ast::{Atom, Body, Head, Predicate, Program, Rule},
    expr,
    helper::ws_cmt,
    literal::identifier,
};
use compute::expr::VarExpr;
use nom::{
    branch::alt,
    bytes::complete::tag,
    combinator::{eof, map, opt, value},
    multi::{fold_many0, separated_list1},
    sequence::{delimited, pair},
    IResult, Parser,
};

const DIVIDER: &'static str = ":-";
const COMMA: &'static str = ",";
const DOT: &'static str = ".";
const NOT: &'static str = "not";

const LEFT_PAREN: &'static str = "(";
const RIGHT_PAREN: &'static str = ")";

pub fn program(input: &str) -> IResult<&str, Program> {
    fold_many0(ws_cmt(rule), Program::default, |mut program, rule| {
        program.rules.push(rule);
        program
    })
    .parse(input)
    // We eat trailing whitespace and comments and finally match EOL.
    .and_then(|(input, program)| value(program, ws_cmt(eof)).parse(input))
}

fn rule(input: &str) -> IResult<&str, Rule> {
    head.parse(input).and_then(|(input, head)| {
        delimited(ws_cmt(tag(DIVIDER)), ws_cmt(body), ws_cmt(tag(DOT)))
            .parse(input)
            .map(|(input, body)| (input, Rule { head, body }))
    })
}

fn head(input: &str) -> IResult<&str, Head> {
    identifier.parse(input).and_then(|(input, name)| {
        delimited(
            ws_cmt(tag(LEFT_PAREN)),
            separated_list1(ws_cmt(tag(COMMA)), ws_cmt(expr::comparison)),
            ws_cmt(tag(RIGHT_PAREN)),
        )
        .parse(input)
        .map(|(input, variables)| {
            (
                input,
                Head {
                    name: VarExpr::from(name),
                    variables,
                },
            )
        })
    })
}

fn body(input: &str) -> IResult<&str, Body> {
    map(
        opt(separated_list1(ws_cmt(tag(COMMA)), ws_cmt(atom))),
        |atoms| Body {
            atoms: atoms.unwrap_or_default(),
        },
    )
    .parse(input)
}

fn atom(input: &str) -> IResult<&str, Atom> {
    let positive_or_negative = map(
        pair(opt(tag(NOT)), ws_cmt(predicate)),
        |(not, predicate)| {
            if not.is_none() {
                Atom::Positive(predicate)
            } else {
                Atom::Negative(predicate)
            }
        },
    );
    let comparison = map(expr::comparison, |expr| Atom::Comparison(expr));

    alt((positive_or_negative, comparison)).parse(input)
}

fn predicate(input: &str) -> IResult<&str, Predicate> {
    identifier.parse(input).and_then(|(input, name)| {
        delimited(
            ws_cmt(tag(LEFT_PAREN)),
            separated_list1(ws_cmt(tag(COMMA)), map(ws_cmt(identifier), VarExpr::from)),
            ws_cmt(tag(RIGHT_PAREN)),
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

#[cfg(test)]
mod test {
    use compute::{
        expr::{BinaryExpr, Expr, LiteralExpr},
        operator::Operator,
    };

    use super::*;

    #[test]
    fn test_program() {
        let input = r#"
            // Leading eol-comment.
            x(a, b + 1) :- y(a, b, c), c > 2. // Inline eol-comment.
            z(a, b)     :- y(a, b), not y(a, b).
            // Trailing eol-comment.
        "#;
        let result = program(input);
        let expected = Program {
            rules: vec![
                Rule {
                    head: Head {
                        name: VarExpr::new("x"),
                        variables: vec![
                            Expr::from(VarExpr::new("a")),
                            Expr::from(BinaryExpr {
                                operator: Operator::Addition,
                                left: Expr::from(VarExpr::new("b")),
                                right: Expr::from(LiteralExpr::from(1_u64)),
                            }),
                        ],
                    },
                    body: Body {
                        atoms: vec![
                            Atom::Positive(Predicate {
                                name: VarExpr::new("y"),
                                variables: vec![
                                    VarExpr::new("a"),
                                    VarExpr::new("b"),
                                    VarExpr::new("c"),
                                ],
                            }),
                            Atom::Comparison(Expr::from(BinaryExpr {
                                operator: Operator::Greater,
                                left: Expr::from(VarExpr::new("c")),
                                right: Expr::from(LiteralExpr::from(2_u64)),
                            })),
                        ],
                    },
                },
                Rule {
                    head: Head {
                        name: VarExpr::new("z"),
                        variables: vec![
                            Expr::from(VarExpr::new("a")),
                            Expr::from(VarExpr::new("b")),
                        ],
                    },
                    body: Body {
                        atoms: vec![
                            Atom::Positive(Predicate {
                                name: VarExpr::new("y"),
                                variables: vec![VarExpr::new("a"), VarExpr::new("b")],
                            }),
                            Atom::Negative(Predicate {
                                name: VarExpr::new("y"),
                                variables: vec![VarExpr::new("a"), VarExpr::new("b")],
                            }),
                        ],
                    },
                },
            ],
        };
        assert_eq!(result, Ok(("", expected)));
    }

    #[test]
    fn test_mvr_store_crdt() {
        let input = r#"
            // These are extensional database predicates (EDBPs).
            pred(FromNodeId, FromCounter, ToNodeId, ToCounter)  :- .
            set(NodeId, Counter, Key, Value)                    :- .

            // These are intensional database predicates (IDBPs).
            overwritten(NodeId, Counter)     :- pred(NodeId, Counter, _ToNodeId, _ToCounter).
            overwrites(NodeId, Counter)      :- pred(_FromNodeId, _FromCounter, NodeId, Counter).

            isRoot(NodeId, Counter)          :- set(NodeId, Counter, _Key, _Value),
                                                not overwrites(NodeId, Counter).

            isLeaf(NodeId, Counter)          :- set(NodeId, Counter, _Key, _Value),
                                                not overwritten(NodeId, Counter).

            isCausallyReady(NodeId, Counter) :- isRoot(NodeId, Counter).
            isCausallyReady(NodeId, Counter) :- isCausallyReady(FromNodeId, FromCounter),
                                                pred(FromNodeId, FromCounter, NodeId, Counter).

            mvrStore(Key, Value)             :- isLeaf(NodeId, Counter),
                                                isCausallyReady(NodeId, Counter),
                                                set(NodeId, Counter, Key, Value).
        "#;
        let result = program(input);
        // TODO: Compare the full program but for now we just check that
        // the parser consumes the full input.
        assert_eq!(result.as_ref().map(|(input, program)| *input), Ok(""));
        println!("{:#?}", result);
    }
}

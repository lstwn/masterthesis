//! This module parses the following grammar of a Datalog variant:
//! ```ebnf
//! program     = rule* EOF ;
//! rule        = head ":-" body "." ;
//! head        = IDENTIFIER "(" comparison ( "," comparison )* ")" ;
//! body        = atom ( "," atom )* ;
//! atom        = ( "not"? predicate ) | comparison ;
//! predicate   = IDENTIFIER "(" IDENTIFIER ( "," IDENTIFIER )* ")" ;
//! ```

use crate::{
    ast::{Atom, Body, Head, Predicate, Program, Rule},
    expr,
    helper::{lead_trail_ws, lead_ws},
    literal::identifier,
};
use compute::expr::VarExpr;
use nom::{
    branch::alt,
    bytes::complete::tag,
    combinator::{map, opt},
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
    fold_many0(
        lead_trail_ws(rule),
        Program::default,
        |mut program, rule| {
            program.rules.push(rule);
            program
        },
    )
    .parse(input)
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
            separated_list1(lead_ws(tag(COMMA)), lead_ws(expr::comparison)),
            lead_ws(tag(RIGHT_PAREN)),
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
        separated_list1(lead_ws(tag(COMMA)), lead_ws(atom)),
        |atoms| Body { atoms },
    )
    .parse(input)
}

fn atom(input: &str) -> IResult<&str, Atom> {
    let positive_or_negative = map(
        pair(opt(tag(NOT)), lead_ws(predicate)),
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
            lead_ws(tag(LEFT_PAREN)),
            separated_list1(lead_ws(tag(COMMA)), map(lead_ws(identifier), VarExpr::from)),
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_program() {
        let input = r#"
            x(a, b + 1) :- y(a, b, c), c > 2.
            z(a, b)     :- y(a, b), not y(a, b).
        "#;
        let result = program(input);
        println!("{:#?}", result);
    }

    #[test]
    fn test_mvr_store_crdt() {
        let input = r#"
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
        println!("{:#?}", result);
    }
}

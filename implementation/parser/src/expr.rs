//! This module parses the following grammar of an expression language:
//! ```ebnf
//! comparison  = term ( ( "==" | "!=" | ">" | ">=" | "<" | "<=" ) term )? ;
//! term        = factor ( ( "+" | "-" ) factor )* ;
//! factor      = unary ( ( "*" | "/" ) unary )* ;
//! unary       = ( "-" | "!" ) unary | primary ;
//! primary     = literal | IDENTIFIER | "(" comparison ")" ;
//! literal     = BOOL | UINT | IINT | STRING | NULL ;
//! ```

use crate::{
    helper::ws_cmt,
    literal::{identifier, literal},
};
use compute::{
    expr::{BinaryExpr, Expr, GroupingExpr, LiteralExpr, UnaryExpr, VarExpr},
    operator::Operator,
};
use nom::{
    branch::alt,
    bytes::complete::tag,
    combinator::{map, opt},
    multi::fold_many0,
    sequence::{delimited, pair},
    IResult, Parser,
};

const EQUAL: &'static str = "==";
const NOT_EQUAL: &'static str = "!=";
const GREATER: &'static str = ">";
const GREATER_EQUAL: &'static str = ">=";
const LESS: &'static str = "<";
const LESS_EQUAL: &'static str = "<=";

const PLUS: &'static str = "+";
const MINUS: &'static str = "-";
const MULTIPLY: &'static str = "*";
const DIVIDE: &'static str = "/";
const BANG: &'static str = "!";

const LEFT_PAREN: &'static str = "(";
const RIGHT_PAREN: &'static str = ")";

pub fn comparison(input: &str) -> IResult<&str, Expr> {
    let equals = map(tag(EQUAL), |_: &str| Operator::Equal);
    let not_equals = map(tag(NOT_EQUAL), |_: &str| Operator::NotEqual);
    let greater = map(tag(GREATER), |_: &str| Operator::Greater);
    let greater_equals = map(tag(GREATER_EQUAL), |_: &str| Operator::GreaterEqual);
    let less = map(tag(LESS), |_: &str| Operator::Less);
    let less_equals = map(tag(LESS_EQUAL), |_: &str| Operator::LessEqual);
    let comparison_operator = ws_cmt(alt((
        equals,
        not_equals,
        greater,
        greater_equals,
        less,
        less_equals,
    )));

    term.parse(input).and_then(|(input, left)| {
        opt(pair(comparison_operator, ws_cmt(term)))
            .parse(input)
            .map(|(input, right)| {
                let expr = if let Some((operator, right)) = right {
                    Expr::from(BinaryExpr {
                        operator,
                        left,
                        right,
                    })
                } else {
                    left
                };
                (input, expr)
            })
    })
}

fn term(input: &str) -> IResult<&str, Expr> {
    let plus = map(tag(PLUS), |_: &str| Operator::Addition);
    let minus = map(tag(MINUS), |_: &str| Operator::Subtraction);
    let term_operator = ws_cmt(alt((plus, minus)));

    factor.parse(input).and_then(|(input, left)| {
        fold_many0(
            pair(term_operator, ws_cmt(factor)),
            // Why is this a FnMut() and not a FnOnce() to avoid the clone?
            move || left.clone(),
            |left, (operator, right)| {
                Expr::from(BinaryExpr {
                    operator,
                    left,
                    right,
                })
            },
        )
        .parse(input)
    })
}

fn factor(input: &str) -> IResult<&str, Expr> {
    let multiply = map(tag(MULTIPLY), |_: &str| Operator::Multiplication);
    let divide = map(tag(DIVIDE), |_: &str| Operator::Division);
    let factor_operator = ws_cmt(alt((multiply, divide)));

    unary.parse(input).and_then(|(input, left)| {
        fold_many0(
            pair(factor_operator, ws_cmt(unary)),
            // Why is this a FnMut() and not a FnOnce() to avoid the clone?
            move || left.clone(),
            |left, (operator, right)| {
                Expr::from(BinaryExpr {
                    operator,
                    left,
                    right,
                })
            },
        )
        .parse(input)
    })
}

fn unary(input: &str) -> IResult<&str, Expr> {
    let minus = map(tag(MINUS), |_: &str| Operator::Subtraction);
    let bang = map(tag(BANG), |_: &str| Operator::Not);
    let unary_operator = alt((minus, bang));

    alt((
        map(
            pair(unary_operator, ws_cmt(unary)),
            |(operator, operand)| Expr::from(UnaryExpr { operator, operand }),
        ),
        primary,
    ))
    .parse(input)
}

fn primary(input: &str) -> IResult<&str, Expr> {
    let literal = map(literal, |literal| Expr::from(LiteralExpr::from(literal)));
    let identifier = map(identifier, |ident| Expr::from(VarExpr::from(ident)));
    let grouping = map(
        delimited(
            tag(LEFT_PAREN),
            ws_cmt(comparison),
            ws_cmt(tag(RIGHT_PAREN)),
        ),
        |expr| Expr::from(GroupingExpr { expr }),
    );

    alt((literal, identifier, grouping)).parse(input)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_grouping() {
        let input = "(x + 1) * 2";
        let result = comparison(input);
        let expected = Expr::from(BinaryExpr {
            operator: Operator::Multiplication,
            left: Expr::from(GroupingExpr {
                expr: Expr::from(BinaryExpr {
                    operator: Operator::Addition,
                    left: Expr::from(VarExpr::new("x")),
                    right: Expr::from(LiteralExpr::from(1_u64)),
                }),
            }),
            right: Expr::from(LiteralExpr::from(2_u64)),
        });
        assert_eq!(result, Ok(("", expected)));

        let input = "x + 1 * 2";
        let result = comparison(input);
        let expected = Expr::from(BinaryExpr {
            operator: Operator::Addition,
            left: Expr::from(VarExpr::new("x")),
            right: Expr::from(BinaryExpr {
                operator: Operator::Multiplication,
                left: Expr::from(LiteralExpr::from(1_u64)),
                right: Expr::from(LiteralExpr::from(2_u64)),
            }),
        });
        assert_eq!(result, Ok(("", expected)));
    }

    #[test]
    fn test_comparison() {
        let input = "true";
        let result = factor(input);
        let expected = Expr::from(LiteralExpr::from(true));

        assert_eq!(result, Ok(("", expected)));
        let input = "x == 30 - 10";
        let result = comparison(input);
        let expected = Expr::from(BinaryExpr {
            operator: Operator::Equal,
            left: Expr::from(VarExpr::new("x")),
            right: Expr::from(BinaryExpr {
                operator: Operator::Subtraction,
                left: Expr::from(LiteralExpr::from(30_u64)),
                right: Expr::from(LiteralExpr::from(10_u64)),
            }),
        });
        assert_eq!(result, Ok(("", expected)));
    }

    #[test]
    fn test_term() {
        let input = "4 + 8 - 3 * 2";
        let result = term(input);
        let expected = Expr::from(BinaryExpr {
            operator: Operator::Subtraction,
            left: Expr::from(BinaryExpr {
                operator: Operator::Addition,
                left: Expr::from(LiteralExpr::from(4_u64)),
                right: Expr::from(LiteralExpr::from(8_u64)),
            }),
            right: Expr::from(BinaryExpr {
                operator: Operator::Multiplication,
                left: Expr::from(LiteralExpr::from(3_u64)),
                right: Expr::from(LiteralExpr::from(2_u64)),
            }),
        });
        assert_eq!(result, Ok(("", expected)));
    }

    #[test]
    fn test_factor() {
        let input = "true";
        let result = factor(input);
        let expected = Expr::from(LiteralExpr::from(true));
        assert_eq!(result, Ok(("", expected)));

        let input = "5 * 8 / 4 * 1";
        let result = factor(input);
        let expected = Expr::from(BinaryExpr {
            operator: Operator::Multiplication,
            left: Expr::from(BinaryExpr {
                operator: Operator::Division,
                left: Expr::from(BinaryExpr {
                    operator: Operator::Multiplication,
                    left: Expr::from(LiteralExpr::from(5_u64)),
                    right: Expr::from(LiteralExpr::from(8_u64)),
                }),
                right: Expr::from(LiteralExpr::from(4_u64)),
            }),
            right: Expr::from(LiteralExpr::from(1_u64)),
        });
        assert_eq!(result, Ok(("", expected)));
    }

    #[test]
    fn test_unary() {
        // TODO: Should this be a -Uint(5) (illegal) or a Int(-5) (legal)?
        // Maybe just offer an i64 type?
        let input = "- 5";
        let result = unary(input);
        assert_eq!(
            result,
            Ok((
                "",
                Expr::from(UnaryExpr {
                    operator: Operator::Subtraction,
                    operand: Expr::from(LiteralExpr::from(5_u64))
                })
            ))
        );

        let input = "! ! true";
        let result = unary(input);
        assert_eq!(
            result,
            Ok((
                "",
                Expr::from(UnaryExpr {
                    operator: Operator::Not,
                    operand: Expr::from(UnaryExpr {
                        operator: Operator::Not,
                        operand: Expr::from(LiteralExpr::from(true))
                    })
                })
            ))
        )
    }
}

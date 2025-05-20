//! This module parses the following grammar of literals:
//! ```ebnf
//! BOOL        = "true" | "false" ;
//! UINT        = DIGIT+ ;
//! IINT        = ( "-" | "+" )? DIGIT+ ;
//! STRING      = "\""<any char except "\"">*"\"" ;
//! IDENTIFIER  = ALPHA ( ALPHA | DIGIT )* ;
//! ALPHA       = "a".."z" | "A".."Z" | "_" ;
//! DIGIT       = "0".."9" ;
//! NULL        = "null" ;
//! ```

use crate::ast::Identifier;
use compute::expr::Literal;
use nom::{
    branch::alt,
    bytes::{complete::tag, take_until},
    character::complete::{alpha1, alphanumeric1, digit1},
    combinator::{map, map_res, opt, recognize, value},
    multi::{many0, many0_count, many1},
    sequence::{delimited, pair, preceded, terminated},
    IResult, Parser,
};

const UNDERSCORE: &str = "_";
const HYPHEN: &str = "-";
const PLUS: &str = "+";
const TRUE: &str = "true";
const FALSE: &str = "false";
const NULL: &str = "null";
const DOUBLE_QUOTE: &str = "\"";

pub fn identifier(input: &str) -> IResult<&str, Identifier> {
    let parse_identifier = recognize(pair(
        alt((alpha1, tag(UNDERSCORE))),
        many0_count(alt((alphanumeric1, tag(UNDERSCORE)))),
    ));
    map(parse_identifier, Identifier::from).parse(input)
}

pub fn literal(input: &str) -> IResult<&str, Literal> {
    alt((
        map(string, Literal::from),
        map(uint, Literal::from),
        map(iint, Literal::from),
        map(bool, Literal::from),
        map(null, Literal::from),
    ))
    .parse(input)
}

fn string(input: &str) -> IResult<&str, String> {
    // Currently, no string escaping is supported.
    let parse_string = delimited(
        tag(DOUBLE_QUOTE),
        take_until(DOUBLE_QUOTE),
        tag(DOUBLE_QUOTE),
    );
    map(parse_string, |str: &str| str.to_string()).parse(input)
}

fn uint(input: &str) -> IResult<&str, u64> {
    let parse_uint = recognize(many1(terminated(digit1, many0(tag(UNDERSCORE)))));
    map_res(parse_uint, |uint: &str| {
        uint.replace(UNDERSCORE, "").parse::<u64>()
    })
    .parse(input)
}

fn iint(input: &str) -> IResult<&str, i64> {
    let parse_iint = recognize(preceded(
        opt(alt((tag(HYPHEN), tag(PLUS)))),
        many1(terminated(digit1, many0(tag(UNDERSCORE)))),
    ));
    map_res(parse_iint, |iint: &str| {
        iint.replace(UNDERSCORE, "").parse::<i64>()
    })
    .parse(input)
}

fn bool(input: &str) -> IResult<&str, bool> {
    alt((value(true, tag(TRUE)), value(false, tag(FALSE)))).parse(input)
}

fn null(input: &str) -> IResult<&str, ()> {
    value((), tag(NULL)).parse(input)
}

#[cfg(test)]
mod test {
    use super::*;
    use nom::Err;
    use nom::Needed;

    #[test]
    fn test_identifier() {
        let alphanumeric = "hello_world1";
        assert_eq!(
            identifier(alphanumeric),
            Ok(("", Identifier::from("hello_world1")))
        );

        let leading_underscore = "_hello_world";
        assert_eq!(
            identifier(leading_underscore),
            Ok(("", Identifier::from("_hello_world")))
        );

        let no_leading_digit = "1hello_world";
        assert!(identifier(no_leading_digit).is_err());

        let empty = "";
        assert!(identifier(empty).is_err());
    }

    #[test]
    fn test_literal() {
        let input = "\"hello world\"";
        assert_eq!(literal(input), Ok(("", Literal::from("hello world"))));

        let input = "12_345";
        assert_eq!(literal(input), Ok(("", Literal::Uint(12_345))));

        let input = "+12_345";
        assert_eq!(literal(input), Ok(("", Literal::Iint(12_345))));

        let input = "-12_345";
        assert_eq!(literal(input), Ok(("", Literal::Iint(-12_345))));

        let input = "true";
        assert_eq!(literal(input), Ok(("", Literal::Bool(true))));

        let input = "false";
        assert_eq!(literal(input), Ok(("", Literal::Bool(false))));

        let input = "null";
        assert_eq!(literal(input), Ok(("", Literal::Null(()))));
    }

    #[test]
    fn test_string() {
        let input = "\"hello world\"";
        assert_eq!(string(input), Ok(("", "hello world".to_string())));

        let not_terminated = "\"hello world";
        assert_eq!(
            string(not_terminated),
            Err(Err::Incomplete(Needed::Unknown))
        );
    }

    #[test]
    fn test_uint() {
        let input = "12_345";
        assert_eq!(uint(input), Ok(("", 12_345)));
    }

    #[test]
    fn test_iint() {
        let input = "12_345";
        assert_eq!(iint(input), Ok(("", 12_345)));

        let input = "+12_345";
        assert_eq!(iint(input), Ok(("", 12_345)));

        let input = "-12_345";
        assert_eq!(iint(input), Ok(("", -12_345)));
    }

    #[test]
    fn test_bool() {
        let input = "true";
        assert_eq!(bool(input), Ok(("", true)));

        let input = "false";
        assert_eq!(bool(input), Ok(("", false)));

        let input = "notbool";
        assert!(bool(input).is_err());
    }

    #[test]
    fn test_null() {
        let input = "null";
        assert_eq!(null(input), Ok(("", ())));

        let input = "notnull";
        assert!(null(input).is_err());
    }
}

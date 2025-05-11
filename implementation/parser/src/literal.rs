use crate::ast::Identifier;
use compute::expr::Literal;
use nom::{
    branch::alt,
    bytes::{complete::tag, take_until},
    character::complete::{alpha1, alphanumeric1, digit1},
    combinator,
    multi::{many0, many0_count, many1},
    sequence::{delimited, pair, preceded, terminated},
    IResult, Parser,
};

const UNDERSCORE: &'static str = "_";
const HYPHEN: &'static str = "-";
const PLUS: &'static str = "+";
const TRUE: &'static str = "true";
const FALSE: &'static str = "false";
const NULL: &'static str = "null";
const DOUBLE_QUOTE: &'static str = "\"";

pub fn identifier(input: &str) -> IResult<&str, Identifier> {
    let parse_identifier = combinator::recognize(pair(
        alt((alpha1, tag(UNDERSCORE))),
        many0_count(alt((alphanumeric1, tag(UNDERSCORE)))),
    ));
    combinator::map(parse_identifier, Identifier::from).parse(input)
}

pub fn literal(input: &str) -> IResult<&str, Literal> {
    alt((
        combinator::map(string, Literal::from),
        combinator::map(uint, Literal::from),
        combinator::map(iint, Literal::from),
        combinator::map(bool, Literal::from),
        combinator::map(null, Literal::from),
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
    combinator::map(parse_string, |str: &str| str.to_string()).parse(input)
}

fn uint(input: &str) -> IResult<&str, u64> {
    let parse_uint = combinator::recognize(many1(terminated(digit1, many0(tag(UNDERSCORE)))));
    combinator::map_res(parse_uint, |uint: &str| {
        uint.replace(UNDERSCORE, "").parse::<u64>()
    })
    .parse(input)
}

fn iint(input: &str) -> IResult<&str, i64> {
    let parse_iint = combinator::recognize(preceded(
        combinator::opt(alt((tag(HYPHEN), tag(PLUS)))),
        many1(terminated(digit1, many0(tag(UNDERSCORE)))),
    ));
    combinator::map_res(parse_iint, |iint: &str| {
        iint.replace(UNDERSCORE, "").parse::<i64>()
    })
    .parse(input)
}

fn bool(input: &str) -> IResult<&str, bool> {
    alt((
        combinator::value(true, tag(TRUE)),
        combinator::value(false, tag(FALSE)),
    ))
    .parse(input)
}

fn null(input: &str) -> IResult<&str, ()> {
    combinator::value((), tag(NULL)).parse(input)
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

//! This module provides some helper combinators for parsing.

use nom::{
    bytes::{complete::tag, take_until},
    character::complete::multispace0,
    combinator::{map, opt, value},
    error::ParseError,
    multi::many0_count,
    sequence::{delimited, preceded},
    IResult, Parser,
};

/// A combinator that takes a parser `inner` and produces a parser that
/// consumes (and discards) leading withespace (but no trailing whitespace) as
/// well as comments. It returns the output of `inner`.
pub fn lead_ws_cmt<'a, O, E: ParseError<&'a str>, F>(
    inner: F,
) -> impl Parser<&'a str, Output = O, Error = E>
where
    F: Parser<&'a str, Output = O, Error = E>,
{
    lead_ws(preceded(opt(eol_comments), lead_ws(inner)))
}

pub fn lead_ws<'a, O, E: ParseError<&'a str>, F>(
    inner: F,
) -> impl Parser<&'a str, Output = O, Error = E>
where
    F: Parser<&'a str, Output = O, Error = E>,
{
    preceded(multispace0, inner)
}

/// A parser that consumes multiple eol comments and reports back how many it
/// encountered.
fn eol_comments<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, usize, E> {
    let maybe_more_eol_comments = map(
        many0_count(preceded(multispace0, eol_comment)),
        |count: usize| count + 1,
    );
    preceded(eol_comment, maybe_more_eol_comments).parse(input)
}

/// A parser that consumes a comment which starts with `//` and continues until
/// the end of the line. It throws away the output.
fn eol_comment<'a, E: ParseError<&'a str>>(input: &'a str) -> IResult<&'a str, (), E> {
    let left_delimiter = tag("//");
    let comment = take_until("\n");
    let right_delimiter = tag("\n");
    value(
        (), // We throw away the output.
        delimited(left_delimiter, comment, right_delimiter),
    )
    .parse(input)
}

/// A combinator that takes a parser `inner` and produces a parser that also
/// consumes leading and trailing whitespace, returning the output of `inner`.
#[deprecated]
pub fn lead_trail_ws<'a, O, E: ParseError<&'a str>, F>(
    inner: F,
) -> impl Parser<&'a str, Output = O, Error = E>
where
    F: Parser<&'a str, Output = O, Error = E>,
{
    delimited(multispace0, inner, multispace0)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_eol_comments() {
        let input = "// this is a comment\nbar";
        let result = eol_comments::<()>(input);
        assert_eq!(result, Ok(("bar", 1)));

        let input = "// first line\n   // second line\nbar";
        let result = eol_comments::<()>(input);
        assert_eq!(result, Ok(("bar", 2)));
    }
}

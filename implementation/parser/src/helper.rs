//! This module provides some helper combinators for parsing.

use nom::{
    bytes::complete::{is_not, tag},
    character::complete::multispace0,
    combinator::{opt, value},
    error::ParseError,
    sequence::{delimited, pair, preceded},
    IResult, Parser,
};

/// A combinator that takes a parser `inner` and produces a parser that
/// consumes (and discards) leading withespace (but no trailing whitespace) as
/// well as comments. It returns the output of `inner`.
pub fn ws_cmt<'a, O, E: ParseError<&'a str>, F>(
    inner: F,
) -> impl Parser<&'a str, Output = O, Error = E>
where
    F: Parser<&'a str, Output = O, Error = E>,
{
    let whitespace_or_comment = delimited(opt(multispace0), opt(eol_comment), opt(multispace0));
    preceded(whitespace_or_comment, inner)
}

/// A parser that consumes a comment which starts with `//` and continues until
/// the end of the line. It throws away the output.
pub fn eol_comment<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, (), E> {
    value(
        (), // Output is thrown away.
        pair(tag("//"), is_not("\n\r")),
    )
    .parse(i)
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

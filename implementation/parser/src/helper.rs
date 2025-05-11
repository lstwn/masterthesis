use nom::{
    character::complete::multispace0,
    error::ParseError,
    sequence::{delimited, preceded},
    Parser,
};

/// A combinator that takes a parser `inner` and produces a parser that also
/// consumes leading but no trailing whitespace, returning the output of `inner`.
pub fn ws<'a, O, E: ParseError<&'a str>, F>(inner: F) -> impl Parser<&'a str, Output = O, Error = E>
where
    F: Parser<&'a str, Output = O, Error = E>,
{
    preceded(multispace0, inner)
}

/// A combinator that takes a parser `inner` and produces a parser that also
/// consumes leading and trailing whitespace, returning the output of `inner`.
pub fn lead_trail_ws<'a, O, E: ParseError<&'a str>, F>(
    inner: F,
) -> impl Parser<&'a str, Output = O, Error = E>
where
    F: Parser<&'a str, Output = O, Error = E>,
{
    delimited(multispace0, inner, multispace0)
}

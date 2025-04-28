1. How does a grammar for Datalog look like?
   Program = [Rule\n]+
   Rule = Head :- Body.
   Atom = Identifier(Variable[, Variable]+)
   Head = Atom
   Body = Atom(, Atom)\*
   Identifier = [a-zA-Z0-9_]+
   Variable = [a-zA-Z0-9_]+
   NOT?
2. How to figure out the evaluation order? Dependency graph.
3. How to map the rules to the IR of relational algebra?
   Datalog works with binding values to variables but the IR works in terms of relations.
4. Soundness: How to deal with NOT? Stratified negation.
   Here, the dependency graph could be useful, too.
5. Query Optimizer

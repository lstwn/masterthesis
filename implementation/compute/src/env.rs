// IDEAS:
// - [x] add Lox class
// - [ ] string wrapper struct for identifier
// - [ ] develop Environment (context)
//   - [x] use mutable environment
// - [ ] try out idea from blog post
//   - [ ] Offer regular AST and flattened AST
//   - [ ] From regular AST to flattened AST: Post-order traversal
//   - [ ] From flattened AST to regular AST: ???
// - [ ] integrate with dbsp
// - [ ] POC:
//   - [ ] implement relation struct/TupleValue
//   - [ ] selection
//   - [ ] projection
//   - [ ] join
//   - [ ] iteration until fix point
// - [ ] add parser for string representation

use crate::{
    error::SyntaxError,
    expr::{BinaryExpr, Expr, ExprVisitor, LitExpr, TernaryExpr, UnaryExpr, VarExpr},
    scalar::ScalarTypedValue,
    stmt::{Stmt, StmtVisitor, VarStmt},
    util::MemAddr,
};
use std::collections::HashMap;

/// An AST node identifier.
/// Can be its address in memory if using a pointer-based AST
/// or its index if using a flattened AST.
#[derive(Eq, PartialEq, Hash, Clone, Copy)]
pub struct NodeRef(usize);

impl From<usize> for NodeRef {
    fn from(index: usize) -> Self {
        Self(index)
    }
}

impl<T: MemAddr> From<&T> for NodeRef {
    fn from(addr: &T) -> Self {
        Self(addr.mem_addr())
    }
}

/// Just an alias to a value of a variable. May be a closure in the future to
/// avoid eagerly filling all attributes of a relation _regardless_ if they are
/// used or not.
type Val = ScalarTypedValue;

/// First entry is the scope, second entry is the variable within that scope.
type VarIdent = (usize, usize);

pub struct Environment {
    // Outer vector is the stack of scopes.
    // Inner vector stores the variable slots in a scope.
    scopes: Vec<Vec<Val>>,
    // Side table to store the VarIdent for each variable.
    side_table: HashMap<NodeRef, VarIdent>,
}

impl Environment {
    fn new(side_table: HashMap<NodeRef, VarIdent>) -> Self {
        let mut env = Self {
            scopes: Vec::with_capacity(8),
            side_table,
        };
        // Create global environment.
        env.begin_scope();
        env
    }
    pub fn begin_scope(&mut self) -> () {
        self.scopes.push(Vec::new());
    }
    pub fn end_scope(&mut self) -> () {
        self.scopes.pop();
    }
    pub fn define_var(&mut self, val: Val) -> () {
        self.scopes.last_mut().expect("no root env").push(val);
    }
    pub fn assign_var(&mut self, node_ref: NodeRef, val: Val) -> () {
        let (scope, slot) = self.get_slots(&node_ref);
        self.scopes[scope][slot] = val;
    }
    pub fn lookup_var<T: Into<NodeRef>>(&self, node_ref: T) -> &ScalarTypedValue {
        let (scope, slot) = self.get_slots(&node_ref.into());
        &self.scopes[scope][slot]
    }
    fn get_slots(&self, node_ref: &NodeRef) -> VarIdent {
        *self
            .side_table
            .get(node_ref)
            .expect("No variable associated with NodeRef")
    }
}

struct Variable {
    initialized: bool,
    slot: usize,
}

impl Variable {
    fn new(slot: usize) -> Self {
        Self {
            initialized: false,
            slot,
        }
    }
}

pub struct Resolver {
    scopes: Vec<HashMap<String, Variable>>,
    side_table: HashMap<NodeRef, VarIdent>,
}

impl Resolver {
    pub fn new() -> Self {
        let mut resolver = Self {
            scopes: Vec::with_capacity(8),
            side_table: HashMap::new(),
        };
        // Create global environment.
        resolver.scopes.push(HashMap::new());
        resolver
    }
    pub fn to_environment(mut self, expr: &Expr) -> Result<Environment, SyntaxError> {
        self.resolve_expr(expr)
            .map(|()| Environment::new(self.side_table))
    }
    fn resolve_expr(&mut self, expr: &Expr) -> VisitorResult {
        self.visit_expr(expr, ())
    }
    fn begin_scope(&mut self) -> () {
        self.scopes.push(HashMap::new());
    }
    fn end_scope(&mut self) -> () {
        self.scopes.pop();
    }
    // declare in Lox
    fn declare_var(&mut self, name: &String) -> Result<(), SyntaxError> {
        match self.scopes.last_mut() {
            Some(scope) => scope.insert(name, Variable::new(scope.len())),
            None => Err(SyntaxError::new("No scope to declare variable in")),
        }
    }
    // define in Lox
    fn assign_var(&mut self, name: &String) -> Result<(), SyntaxError> {
        match self.scopes.last_mut() {
            Some(scope) => match scope.get_mut(name) {
                Some(var) => {
                    var.initialized = true;
                    Ok(())
                }
                None => Err(SyntaxError::new("Variable not declared in top-most scope")),
            },
            None => Err(SyntaxError::new("No scope to find variable to assign to")),
        }
    }
    // resolveLocal in Lox
    fn resolve_var(&mut self, expr: &Expr, name: &String) -> Result<(), SyntaxError> {
        for (i, scope) in self.scopes.iter().enumerate().rev() {
            if let Some(var) = scope.get(name) {
                let scope_idx = self.scopes.len() - 1 - i;
                self.side_table
                    .insert(NodeRef::from(expr), (scope_idx, var.slot));
                return Ok(());
            }
        }
        Err(SyntaxError::new("Variable not declared"))
    }
}

type VisitorResult = Result<(), SyntaxError>;
type VisitorCtx = ();

impl ExprVisitor<VisitorResult, VisitorCtx> for Resolver {
    fn visit_expr(&mut self, expr: &Expr, ctx: VisitorCtx) -> VisitorResult {
        match expr {
            Expr::Ternary(expr) => self.visit_ternary_expr(expr, ctx),
            Expr::Binary(expr) => self.visit_binary_expr(expr, ctx),
            Expr::Unary(expr) => self.visit_unary_expr(expr, ctx),
            Expr::Var(expr) => self.visit_var_expr(expr, ctx),
            Expr::Lit(expr) => self.visit_lit_expr(expr, ctx),
        }
    }

    fn visit_ternary_expr(&mut self, expr: &TernaryExpr, ctx: VisitorCtx) -> VisitorResult {
        self.resolve_expr(&expr.left)
            .and_then(|()| self.resolve_expr(&expr.mid))
            .and_then(|()| self.resolve_expr(&expr.right))
    }

    fn visit_binary_expr(&mut self, expr: &BinaryExpr, ctx: VisitorCtx) -> VisitorResult {
        self.resolve_expr(&expr.left)
            .and_then(|()| self.resolve_expr(&expr.right))
    }

    fn visit_unary_expr(&mut self, expr: &UnaryExpr, ctx: VisitorCtx) -> VisitorResult {
        self.resolve_expr(&expr.operand)
    }

    fn visit_var_expr(&mut self, expr: &VarExpr, ctx: VisitorCtx) -> VisitorResult {
        if let Some(var) = self.scopes.last().and_then(|scope| scope.get(&expr.name)) {
            if !var.initialized {
                return Err(SyntaxError::new("Variable used before initialization"));
            }
        } else {
            // TODO: I think this is valid because of mutual recursion!
            Err(SyntaxError::new("Variable not declared"))
        }
        self.resolve_var(expr, &expr.name)
    }

    fn visit_lit_expr(&mut self, expr: &LitExpr, ctx: VisitorCtx) -> VisitorResult {
        Ok(())
    }
}

impl StmtVisitor<VisitorResult, VisitorCtx> for Resolver {
    fn visit_stmt(&mut self, stmt: &Stmt, ctx: VisitorCtx) -> VisitorResult {
        match stmt {
            Stmt::Var(stmt) => self.visit_var_stmt(stmt, ctx),
        }
    }

    fn visit_var_stmt(&mut self, stmt: &VarStmt, ctx: VisitorCtx) -> VisitorResult {
        self.declare_var(&stmt.name)
            .and_then(|()| {
                if let Some(expr) = &stmt.initializer {
                    self.resolve_expr(expr)
                } else {
                    Ok(())
                }
            })
            .and_then(|()| self.assign_var(&stmt.name))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // #[test]
    // fn test() -> Result<(), RuntimeError> {
    //     let mut global = Environment::new_root();
    //     global.defineVar("a".to_owned(), ScalarTypedValue::Uint(0));

    //     {
    //         let mut local = global.new_child();
    //         local.assignVar("a".to_owned(), ScalarTypedValue::Uint(1))?;
    //         assert_eq!(local.lookupVar("a"), Some(&ScalarTypedValue::Uint(1)));
    //         local.defineVar("a".to_owned(), ScalarTypedValue::Uint(2));
    //         assert_eq!(local.lookupVar("a"), Some(&ScalarTypedValue::Uint(2)));
    //         assert_eq!(local.lookupVar("b"), None);
    //         local.defineVar("b".to_owned(), ScalarTypedValue::Uint(0));
    //         assert_eq!(local.lookupVar("b"), Some(&ScalarTypedValue::Uint(2)));
    //     }

    //     assert_eq!(global.lookupVar("a"), Some(&ScalarTypedValue::Uint(1)));
    //     assert_eq!(global.lookupVar("b"), None);

    //     Ok(())
    // }
}

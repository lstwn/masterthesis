// IDEAS:
// - [x] add Lox class
// - [ ] string wrapper struct for identifier
// - [ ] implement functions
// - [x] develop Environment (context)
//   - [x] use mutable environment
//   - [x] fix multiple instances of resolver in REPL
// - [x] allow printing last expr (statment) without using print
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
    expr::{BinaryExpr, CallExpr, Expr, ExprVisitor, LitExpr, TernaryExpr, UnaryExpr, VarExpr},
    function::Function,
    interpreter::Interpreter,
    scalar::ScalarTypedValue,
    stmt::{BlockStmt, ExprStmt, FunctionStmt, Program, Stmt, StmtVisitor, VarStmt},
    util::MemAddr,
};
use std::{
    cell::{Ref, RefCell},
    collections::HashMap,
    fmt,
    rc::Rc,
};

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

/// The value of a variable.
/// Idea: Turn into a closure in the future to avoid eagerly filling all
/// attributes of a relation _regardless_ if they are used or not. Or solve via
/// resolver?
#[derive(Clone, Debug)]
pub enum Val {
    /// String.
    String(String),
    /// Unsigned integer value of 64 bits.
    Uint(u64),
    /// Signed integer value of 64 bits.
    Iint(i64),
    /// Boolean.
    Bool(bool),
    /// Null.
    Null(()),
    /// Function.
    Function(Function),
}

impl Default for Val {
    fn default() -> Self {
        Val::Null(())
    }
}

impl From<ScalarTypedValue> for Val {
    fn from(value: ScalarTypedValue) -> Self {
        match value {
            ScalarTypedValue::String(value) => Val::String(value),
            ScalarTypedValue::Uint(value) => Val::Uint(value),
            ScalarTypedValue::Iint(value) => Val::Iint(value),
            ScalarTypedValue::Bool(value) => Val::Bool(value),
            ScalarTypedValue::Null(()) => Val::Null(()),
        }
    }
}

impl fmt::Display for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Val::String(value) => write!(f, "{}", value),
            Val::Uint(value) => write!(f, "{}", value),
            Val::Iint(value) => write!(f, "{}", value),
            Val::Bool(value) => write!(f, "{}", value),
            Val::Null(()) => write!(f, "null"),
            Val::Function(function) => write!(f, "{}", function),
        }
    }
}

/// First entry is the scope, second entry is the variable within that scope.
pub type VarIdent = (usize, usize);

#[derive(Clone)]
struct Scope {
    /// Variable slots of an environment.
    inner: Rc<RefCell<Vec<Val>>>,
}

impl Scope {
    fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(Vec::new())),
        }
    }
    fn define_var(&mut self, val: Val) -> () {
        self.inner.borrow_mut().push(val);
    }
    fn assign_var(&mut self, slot: usize, val: Val) -> () {
        self.inner.borrow_mut()[slot] = val;
    }
    fn lookup_var(&self, slot: usize) -> Ref<Val> {
        let vec = self.inner.borrow();
        Ref::map(vec, |vec| &vec[slot])
    }
}

#[derive(Clone)]
pub struct Environment {
    /// The vector models a stack of scopes with the root environment at
    /// the bottom and the innermost scope at the top.
    scopes: Vec<Scope>,
}

impl Environment {
    pub fn new() -> Self {
        Self {
            scopes: Vec::with_capacity(8),
        }
    }
    pub fn begin_scope(&mut self) -> () {
        self.scopes.push(Scope::new());
    }
    pub fn end_scope(&mut self) -> () {
        self.scopes.pop();
    }
    pub fn define_var<T: Into<Val>>(&mut self, val: T) -> () {
        self.scopes
            .last_mut()
            .expect("no root env")
            .define_var(val.into());
    }
    pub fn assign_var(&mut self, at: &VarIdent, val: Val) -> () {
        let (scope, slot) = *at;
        self.scopes[scope].assign_var(slot, val);
    }
    pub fn lookup_var(&self, at: &VarIdent) -> Ref<Val> {
        let (scope, slot) = *at;
        self.scopes[scope].lookup_var(slot)
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

pub struct Resolver<'a> {
    scopes: Vec<HashMap<String, Variable>>,
    interpreter: &'a mut Interpreter,
}

impl<'a> Resolver<'a> {
    pub fn new(interpreter: &'a mut Interpreter) -> Self {
        Self {
            scopes: Vec::with_capacity(8),
            interpreter,
        }
    }
    pub fn resolve(&mut self, program: &Program) -> Result<(), SyntaxError> {
        self.visit_block(&program.stmts, ())
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
            Some(scope) => {
                scope.insert(name.clone(), Variable::new(scope.len()));
                Ok(())
            }
            None => Err(SyntaxError::new("No scope to declare variable in")),
        }
    }
    // define in Lox
    fn define_var(&mut self, name: &String) -> Result<(), SyntaxError> {
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
    fn resolve_var(&mut self, expr: &VarExpr, name: &String) -> Result<(), SyntaxError> {
        for (i, scope) in self.scopes.iter().enumerate().rev() {
            if let Some(var) = scope.get(name) {
                let scope_idx = self.scopes.len() - 1 - i;
                let slot_idx = var.slot;
                self.interpreter.resolve(expr, (scope_idx, slot_idx));
                return Ok(());
            }
        }
        Err(SyntaxError::new("Variable not declared"))
    }
    fn visit_stmts(&mut self, stmts: &Vec<Stmt>, ctx: VisitorCtx) -> VisitorResult {
        for stmt in stmts {
            self.visit_stmt(stmt, ctx)?;
        }
        Ok(())
    }
    fn visit_block(&mut self, stmts: &Vec<Stmt>, ctx: VisitorCtx) -> Result<(), SyntaxError> {
        self.begin_scope();
        self.visit_stmts(stmts, ctx)?;
        self.end_scope();
        Ok(())
    }
}

type VisitorResult = Result<(), SyntaxError>;
type VisitorCtx = ();

impl ExprVisitor<VisitorResult, VisitorCtx> for Resolver<'_> {
    fn visit_expr(&mut self, expr: &Expr, ctx: VisitorCtx) -> VisitorResult {
        match expr {
            Expr::Ternary(expr) => self.visit_ternary_expr(expr, ctx),
            Expr::Binary(expr) => self.visit_binary_expr(expr, ctx),
            Expr::Unary(expr) => self.visit_unary_expr(expr, ctx),
            Expr::Var(expr) => self.visit_var_expr(expr, ctx),
            Expr::Lit(expr) => self.visit_lit_expr(expr, ctx),
            Expr::Call(expr) => self.visit_call_expr(expr, ctx),
        }
    }

    fn visit_ternary_expr(&mut self, expr: &TernaryExpr, ctx: VisitorCtx) -> VisitorResult {
        self.visit_expr(&expr.left, ctx)
            .and_then(|()| self.visit_expr(&expr.mid, ctx))
            .and_then(|()| self.visit_expr(&expr.right, ctx))
    }

    fn visit_binary_expr(&mut self, expr: &BinaryExpr, ctx: VisitorCtx) -> VisitorResult {
        self.visit_expr(&expr.left, ctx)
            .and_then(|()| self.visit_expr(&expr.right, ctx))
    }

    fn visit_unary_expr(&mut self, expr: &UnaryExpr, ctx: VisitorCtx) -> VisitorResult {
        self.visit_expr(&expr.operand, ctx)
    }

    fn visit_var_expr(&mut self, expr: &VarExpr, ctx: VisitorCtx) -> VisitorResult {
        if let Some(var) = self.scopes.last().and_then(|scope| scope.get(&expr.name)) {
            if !var.initialized {
                return Err(SyntaxError::new(
                    "Variable referenced in its own initializer",
                ));
            }
        }
        // `resolve_var` returns an error if the variable is not declared.
        self.resolve_var(expr, &expr.name)
    }

    fn visit_lit_expr(&mut self, expr: &LitExpr, ctx: VisitorCtx) -> VisitorResult {
        Ok(())
    }

    fn visit_call_expr(&mut self, expr: &CallExpr, ctx: VisitorCtx) -> VisitorResult {
        // IDEA: check for arity here!
        todo!()
    }
}

impl StmtVisitor<VisitorResult, VisitorCtx> for Resolver<'_> {
    fn visit_stmt(&mut self, stmt: &Stmt, ctx: VisitorCtx) -> VisitorResult {
        match stmt {
            Stmt::Var(stmt) => self.visit_var_stmt(stmt, ctx),
            Stmt::Expr(stmt) => self.visit_expr_stmt(stmt, ctx),
            Stmt::Block(stmt) => self.visit_block_stmt(stmt, ctx),
            Stmt::Function(stmt) => self.visit_function_stmt(stmt, ctx),
        }
    }

    fn visit_var_stmt(&mut self, stmt: &VarStmt, ctx: VisitorCtx) -> VisitorResult {
        self.declare_var(&stmt.name)
            .and_then(|()| {
                if let Some(expr) = &stmt.initializer {
                    self.visit_expr(expr, ctx)
                } else {
                    Ok(())
                }
            })
            .and_then(|()| self.define_var(&stmt.name))
    }

    fn visit_expr_stmt(&mut self, stmt: &ExprStmt, ctx: VisitorCtx) -> VisitorResult {
        self.visit_expr(&stmt.expr, ctx)
    }

    fn visit_block_stmt(&mut self, stmt: &BlockStmt, ctx: VisitorCtx) -> VisitorResult {
        self.visit_block(&stmt.stmts, ctx)
    }

    fn visit_function_stmt(&mut self, stmt: &FunctionStmt, ctx: VisitorCtx) -> VisitorResult {
        self.declare_var(&stmt.name)?;
        self.define_var(&stmt.name)?;

        self.begin_scope();
        for parameter in &stmt.parameters {
            self.declare_var(parameter)?;
            self.define_var(parameter)?;
        }
        self.visit_stmts(&stmt.body.stmts, ctx)?;
        self.end_scope();
        Ok(())
    }
}

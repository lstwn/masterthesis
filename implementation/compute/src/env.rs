// IDEAS:
// - [x] add Lox class
// - [x] Make functions first class citizens:
//   - [x] Add environment context to visitor functions.
//   - [x] Add function expressions.
//   - [x] Remove function statements.
// - [ ] string wrapper struct for identifier
// - [x] develop Environment (context)
//   - [x] use mutable environment
//   - [x] fix multiple instances of resolver in REPL
// - [x] allow printing last expr (statment) without using print
// - [ ] try out idea from blog post:
//       https://www.cs.cornell.edu/~asampson/blog/flattening.html
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
    context::ResolverContext,
    dbsp_playground::RelationRef,
    error::SyntaxError,
    expr::{
        AssignExpr, BinaryExpr, CallExpr, ExprVisitorMut, FunctionExpr, GroupingExpr, LitExpr,
        SelectionExpr, TernaryExpr, UnaryExpr, VarExpr,
    },
    function::FunctionRef,
    scalar::ScalarTypedValue,
    stmt::{BlockStmt, ExprStmt, Stmt, StmtVisitorMut, VarStmt},
    util::{MemAddr, Named, Resolvable},
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
#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
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
    // TODO: maybe make strings reference counted to avoid cloning costs?
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
    Function(FunctionRef),
    /// Relation.
    Relation(RelationRef),
}

impl Eq for Val {}

impl PartialEq for Val {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Val::String(a), Val::String(b)) => a == b,
            (Val::Uint(a), Val::Uint(b)) => a == b,
            (Val::Iint(a), Val::Iint(b)) => a == b,
            (Val::Bool(a), Val::Bool(b)) => a == b,
            (Val::Null(()), Val::Null(())) => true,
            (Val::Function(a), Val::Function(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }
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
            Val::Function(function) => write!(f, "{}", function.borrow()),
            Val::Relation(relation) => write!(f, "relation"),
        }
    }
}

/// First entry is the scope, second entry is the variable within that scope.
pub type VarIdent = (usize, usize);

#[derive(Clone, Debug)]
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
    fn assign_var(&mut self, slot_idx: usize, val: Val) -> () {
        self.inner.borrow_mut()[slot_idx] = val;
    }
    fn lookup_var(&self, slot_idx: usize) -> Ref<Val> {
        let vec = self.inner.borrow();
        Ref::map(vec, |vec| &vec[slot_idx])
    }
}

const SCOPES_CAPACITY: usize = 8;

#[derive(Clone, Debug)]
pub struct Environment {
    /// The vector models a stack of scopes with the root environment at
    /// the bottom and the innermost scope at the top.
    scopes: Vec<Scope>,
}

impl Environment {
    pub fn new() -> Self {
        let mut environment = Self {
            scopes: Vec::with_capacity(SCOPES_CAPACITY),
        };
        // Create the global scope.
        environment.begin_scope();
        environment
    }
    pub fn just_global(&self) -> bool {
        self.scopes.len() == 1
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
        let (scope_idx, slot_idx) = *at;
        self.scopes[scope_idx].assign_var(slot_idx, val);
    }
    pub fn lookup_var(&self, at: &VarIdent) -> Ref<Val> {
        let (scope_idx, slot_idx) = *at;
        self.scopes[scope_idx].lookup_var(slot_idx)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Variable {
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

pub struct ScopeStack {
    inner: Vec<HashMap<String, Variable>>,
}

impl ScopeStack {
    pub fn new() -> Self {
        let mut scope_stack = Self {
            inner: Vec::with_capacity(SCOPES_CAPACITY),
        };
        // Create the global scope.
        scope_stack.begin_scope();
        scope_stack
    }
    fn just_global(&self) -> bool {
        self.inner.len() == 1
    }
    fn begin_scope(&mut self) -> () {
        self.inner.push(HashMap::new());
    }
    fn end_scope(&mut self) -> () {
        self.inner.pop();
    }
    fn innermost(&self) -> Option<&HashMap<String, Variable>> {
        self.inner.last()
    }
    fn innermost_mut(&mut self) -> Option<&mut HashMap<String, Variable>> {
        self.inner.last_mut()
    }
}

pub struct Resolver {}

impl Resolver {
    pub fn new() -> Self {
        Self {}
    }
    pub fn resolve<'a>(
        &mut self,
        stmts: impl IntoIterator<Item = &'a mut Stmt>,
        ctx: VisitorCtx,
    ) -> Result<(), SyntaxError> {
        // Ensure we have a global scope before resolving.
        debug_assert!(ctx.scopes.just_global());
        // We do not call `visit_block` here because the root scope is created
        // in the `ScopeStack` constructor and should remain intact across
        // multiple calls to `resolve`.
        let ret = self.visit_stmts(stmts, ctx);
        // Ensure we have a global scope after resolving.
        debug_assert!(ctx.scopes.just_global());
        ret
    }
    // declare in Lox
    fn declare_var(&mut self, name: &String, ctx: VisitorCtx) -> Result<(), SyntaxError> {
        match ctx.scopes.innermost_mut() {
            Some(scope) => {
                scope.insert(name.clone(), Variable::new(scope.len()));
                Ok(())
            }
            None => Err(SyntaxError::new("No scope to declare variable in")),
        }
    }
    // define in Lox
    fn define_var(&mut self, name: &String, ctx: VisitorCtx) -> Result<(), SyntaxError> {
        match ctx.scopes.innermost_mut() {
            Some(scope) => match scope.get_mut(name) {
                Some(var) => {
                    var.initialized = true;
                    Ok(())
                }
                None => Err(SyntaxError::new("Variable not declared in innermost scope")),
            },
            None => Err(SyntaxError::new("No scope to find variable to assign to")),
        }
    }
    // resolveLocal in Lox
    fn resolve_var<T: Resolvable + Named>(
        &mut self,
        expr: &mut T,
        ctx: VisitorCtx,
    ) -> Result<(), SyntaxError> {
        for (scope_idx, scope) in ctx.scopes.inner.iter().enumerate().rev() {
            if let Some(var) = scope.get(expr.name()) {
                let slot_idx = var.slot;
                expr.set_resolved((scope_idx, slot_idx));
                return Ok(());
            }
        }
        // We have to tolerate unresolved variables because we are in a tuple context.
        // Later on, we can also do static analysis to determine if it is a valid
        // reference to a tuple variable by tracking a relation's schema.
        if ctx.is_tuple_context {
            Ok(())
        } else {
            Err(SyntaxError::new(format!(
                "Variable '{}' not declared",
                expr.name()
            )))
        }
    }
    fn visit_stmts<'a>(
        &mut self,
        stmts: impl IntoIterator<Item = &'a mut Stmt>,
        ctx: VisitorCtx,
    ) -> VisitorResult {
        for stmt in stmts {
            self.visit_stmt(stmt, ctx)?;
        }
        Ok(())
    }
    fn visit_block<'a, F>(
        &mut self,
        stmts: impl IntoIterator<Item = &'a mut Stmt>,
        ctx: VisitorCtx,
        after_new_scope_actions: F,
    ) -> Result<(), SyntaxError>
    where
        F: FnOnce(&mut Self, VisitorCtx) -> Result<(), SyntaxError>,
    {
        ctx.scopes.begin_scope();
        after_new_scope_actions(self, ctx)?;
        self.visit_stmts(stmts, ctx)?;
        ctx.scopes.end_scope();
        Ok(())
    }
}

type VisitorResult = Result<(), SyntaxError>;
type VisitorCtx<'a, 'b> = &'a mut ResolverContext<'b>;

impl<'a, 'b> ExprVisitorMut<VisitorResult, VisitorCtx<'a, 'b>> for Resolver {
    fn visit_ternary_expr(&mut self, expr: &mut TernaryExpr, ctx: VisitorCtx) -> VisitorResult {
        self.visit_expr(&mut expr.left, ctx)
            .and_then(|()| self.visit_expr(&mut expr.mid, ctx))
            .and_then(|()| self.visit_expr(&mut expr.right, ctx))
    }

    fn visit_binary_expr(&mut self, expr: &mut BinaryExpr, ctx: VisitorCtx) -> VisitorResult {
        self.visit_expr(&mut expr.left, ctx)
            .and_then(|()| self.visit_expr(&mut expr.right, ctx))
    }

    fn visit_unary_expr(&mut self, expr: &mut UnaryExpr, ctx: VisitorCtx) -> VisitorResult {
        self.visit_expr(&mut expr.operand, ctx)
    }

    fn visit_grouping_expr(&mut self, expr: &mut GroupingExpr, ctx: VisitorCtx) -> VisitorResult {
        self.visit_expr(&mut expr.expr, ctx)
    }

    fn visit_var_expr(&mut self, expr: &mut VarExpr, ctx: VisitorCtx) -> VisitorResult {
        if let Some(var) = ctx
            .scopes
            .innermost()
            .and_then(|scope| scope.get(&expr.name))
        {
            if !var.initialized {
                return Err(SyntaxError::new(
                    "Variable referenced in its own initializer",
                ));
            }
        }
        // `resolve_var` returns an error if the variable is not declared.
        self.resolve_var(expr, ctx)
    }

    fn visit_assign_expr(&mut self, expr: &mut AssignExpr, ctx: VisitorCtx) -> VisitorResult {
        self.visit_expr(&mut expr.value, ctx)?;
        // `resolve_var` returns an error if the variable is not declared.
        self.resolve_var(expr, ctx)
    }

    fn visit_lit_expr(&mut self, expr: &mut LitExpr, ctx: VisitorCtx) -> VisitorResult {
        Ok(())
    }

    fn visit_function_expr(&mut self, expr: &mut FunctionExpr, ctx: VisitorCtx) -> VisitorResult {
        self.visit_block(&mut expr.body.stmts, ctx, |resolver, ctx| {
            for parameter in &expr.parameters {
                resolver.declare_var(parameter, ctx)?;
                resolver.define_var(parameter, ctx)?;
            }
            Ok(())
        })
    }

    fn visit_call_expr(&mut self, expr: &mut CallExpr, ctx: VisitorCtx) -> VisitorResult {
        // TODO: check for arity here just once statically.
        self.visit_expr(&mut expr.callee, ctx)?;
        for arg in &mut expr.arguments {
            self.visit_expr(arg, ctx)?;
        }
        Ok(())
    }
    fn visit_selection_expr(&mut self, expr: &mut SelectionExpr, ctx: VisitorCtx) -> VisitorResult {
        self.visit_expr(&mut expr.relation, ctx).and_then(|()| {
            ctx.begin_tuple_context();
            let ret = self.visit_expr(&mut expr.condition, ctx);
            ctx.end_tuple_context();
            ret
        })
    }
}

impl<'a, 'b> StmtVisitorMut<VisitorResult, VisitorCtx<'a, 'b>> for Resolver {
    fn visit_var_stmt(&mut self, stmt: &mut VarStmt, ctx: VisitorCtx) -> VisitorResult {
        self.declare_var(&stmt.name, ctx)
            .and_then(|()| {
                if let Some(expr) = &mut stmt.initializer {
                    self.visit_expr(expr, ctx)
                } else {
                    Ok(())
                }
            })
            .and_then(|()| self.define_var(&stmt.name, ctx))
    }

    fn visit_expr_stmt(&mut self, stmt: &mut ExprStmt, ctx: VisitorCtx) -> VisitorResult {
        self.visit_expr(&mut stmt.expr, ctx)
    }

    fn visit_block_stmt(&mut self, stmt: &mut BlockStmt, ctx: VisitorCtx) -> VisitorResult {
        self.visit_block(&mut stmt.stmts, ctx, |_resolver, _ctx| Ok(()))
    }
}

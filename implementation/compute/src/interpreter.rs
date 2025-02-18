use crate::{
    env::{Environment, NodeRef, Val, VarIdent},
    error::RuntimeError,
    expr::{
        AssignExpr, BinaryExpr, CallExpr, ExprVisitor, FunctionExpr, GroupingExpr, LitExpr,
        TernaryExpr, UnaryExpr, VarExpr,
    },
    function::new_function,
    operator::Operator,
    stmt::{BlockStmt, ExprStmt, Program, Stmt, StmtVisitor, VarStmt},
};
use std::collections::HashMap;

type ScalarTypedValue = Val;

pub struct Interpreter {
    /// Side table to store the VarIdent for each variable.
    /// TODO: move to VisitorCtx? And what about environment? Put into `Context`?
    side_table: HashMap<NodeRef, VarIdent>,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            side_table: HashMap::new(),
        }
    }
    pub fn interpret(
        &mut self,
        program: &Program,
    ) -> Result<Option<ScalarTypedValue>, RuntimeError> {
        // Because we call `execute_block` here, we implicitly create a global
        // scope for the program.
        self.visit_block(&program.stmts, &mut Environment::new(), |_env| ())
    }
    pub fn resolve<T: Into<NodeRef>>(&mut self, expr: T, ident: VarIdent) -> () {
        self.side_table.insert(expr.into(), ident);
    }
    fn visit_stmts(
        &mut self,
        stmts: &Vec<Stmt>,
        ctx: VisitorCtx,
    ) -> Result<Option<ScalarTypedValue>, RuntimeError> {
        // Functional programming can be so beautiful. Return the last value
        // if any.
        stmts
            .iter()
            .try_fold(None, |_prev, stmt| self.visit_stmt(stmt, ctx))
    }
    fn visit_block<F: FnOnce(&mut Environment) -> ()>(
        &mut self,
        stmts: &Vec<Stmt>,
        mut ctx: VisitorCtx,
        environment: F,
    ) -> Result<Option<ScalarTypedValue>, RuntimeError> {
        ctx.begin_scope();
        environment(&mut ctx);
        let ret = self.visit_stmts(stmts, ctx);
        ctx.end_scope();
        ret
    }
}

macro_rules! comparison_helper {
    ($left:expr, $right:expr, $op:tt, $($variant:path),*) => {{
        match (&$left, &$right) {
            $(
                ($variant(left), $variant(right)) => Ok(ScalarTypedValue::Bool(left $op right)),
            )*
            _ => Err(RuntimeError::new(
                format!("expected comparable type, got: {:?} and {:?}", $left, $right),
            )),
        }
    }}
}

macro_rules! arithmetic_helper {
    ($left:expr, $right:expr, $op:tt, $($variant:path),*) => {{
        match (&$left, &$right) {
            $(
                ($variant(left), $variant(right)) => Ok($variant(left $op right)),
            )*
            _ => Err(RuntimeError::new(
                format!("expected number type, got: {:?} and {:?}", $left, $right),
            )),
        }
    }}
}

impl Interpreter {
    fn visit_lazy_binary_expr(&mut self, expr: &BinaryExpr, ctx: VisitorCtx) -> ExprVisitorResult {
        let left = is_truthy(&self.visit_expr(&expr.left, ctx)?);

        if let Operator::And | Operator::Or = expr.operator {
            if expr.operator == Operator::And && left || expr.operator == Operator::Or && !left {
                let right = is_truthy(&self.visit_expr(&expr.right, ctx)?);
                Ok(ScalarTypedValue::Bool(right))
            } else {
                Ok(ScalarTypedValue::Bool(left))
            }
        } else {
            Err(RuntimeError::new(format!(
                "unsupported (lazy) binary operator: {:?}",
                expr.operator
            )))
        }
    }
    fn visit_eager_binary_expr(&mut self, expr: &BinaryExpr, ctx: VisitorCtx) -> ExprVisitorResult {
        let left = self.visit_expr(&expr.left, ctx)?;
        let right = self.visit_expr(&expr.right, ctx)?;

        match expr.operator {
            Operator::Equal => {
                comparison_helper!(left, right, ==, ScalarTypedValue::Iint, ScalarTypedValue::Uint, ScalarTypedValue::Bool, ScalarTypedValue::String, ScalarTypedValue::Null)
            }
            Operator::NotEqual => {
                comparison_helper!(left, right, !=, ScalarTypedValue::Iint, ScalarTypedValue::Uint, ScalarTypedValue::Bool, ScalarTypedValue::String, ScalarTypedValue::Null)
            }
            Operator::Less => {
                comparison_helper!(left, right, <, ScalarTypedValue::Iint, ScalarTypedValue::Uint, ScalarTypedValue::Bool, ScalarTypedValue::String)
            }
            Operator::LessThan => {
                comparison_helper!(left, right, <=, ScalarTypedValue::Iint, ScalarTypedValue::Uint, ScalarTypedValue::Bool, ScalarTypedValue::String)
            }
            Operator::Greater => {
                comparison_helper!(left, right, >, ScalarTypedValue::Iint, ScalarTypedValue::Uint, ScalarTypedValue::Bool, ScalarTypedValue::String)
            }
            Operator::GreaterThan => {
                comparison_helper!(left, right, >=, ScalarTypedValue::Iint, ScalarTypedValue::Uint, ScalarTypedValue::Bool, ScalarTypedValue::String)
            }
            Operator::Addition => {
                if let (ScalarTypedValue::String(left), ScalarTypedValue::String(right)) =
                    (&left, &right)
                {
                    return Ok(ScalarTypedValue::String(format!("{}{}", left, right)));
                }
                arithmetic_helper!(left, right, +, ScalarTypedValue::Iint, ScalarTypedValue::Uint)
            }
            Operator::Subtraction => {
                arithmetic_helper!(left, right, -, ScalarTypedValue::Iint, ScalarTypedValue::Uint)
            }
            Operator::Multiplication => {
                arithmetic_helper!(left, right, *, ScalarTypedValue::Iint, ScalarTypedValue::Uint)
            }
            Operator::Division => {
                arithmetic_helper!(left, right, /, ScalarTypedValue::Iint, ScalarTypedValue::Uint)
            }
            _ => Err(RuntimeError::new(format!(
                "unsupported (eager) binary operator: {:?}",
                expr.operator
            ))),
        }
    }
}

type VisitorCtx<'a> = &'a mut Environment;

type ExprVisitorResult = Result<ScalarTypedValue, RuntimeError>;

impl ExprVisitor<ExprVisitorResult, VisitorCtx<'_>> for Interpreter {
    // TODO: Remove ternary expressions.
    fn visit_ternary_expr(&mut self, expr: &TernaryExpr, ctx: VisitorCtx) -> ExprVisitorResult {
        todo!()
    }

    fn visit_binary_expr(&mut self, expr: &BinaryExpr, ctx: VisitorCtx) -> ExprVisitorResult {
        if let Operator::And | Operator::Or = expr.operator {
            self.visit_lazy_binary_expr(expr, ctx)
        } else {
            self.visit_eager_binary_expr(expr, ctx)
        }
    }

    fn visit_unary_expr(&mut self, expr: &UnaryExpr, ctx: VisitorCtx) -> ExprVisitorResult {
        let operand = self.visit_expr(&expr.operand, ctx)?;

        match expr.operator {
            Operator::Subtraction => match operand {
                ScalarTypedValue::Iint(value) => Ok(ScalarTypedValue::Iint(-value)),
                _ => Err(RuntimeError::new(format!(
                    "expected signed int, got: {:?}",
                    operand
                ))),
            },
            Operator::Not => Ok(ScalarTypedValue::Bool(!is_truthy(&operand))),
            _ => Err(RuntimeError::new(format!(
                "unsupported unary operator: {:?}",
                expr.operator
            ))),
        }
    }

    fn visit_grouping_expr(&mut self, expr: &GroupingExpr, ctx: VisitorCtx) -> ExprVisitorResult {
        self.visit_expr(&expr.expr, ctx)
    }

    fn visit_var_expr(&mut self, expr: &VarExpr, ctx: VisitorCtx) -> ExprVisitorResult {
        let ident = self.side_table.get(&NodeRef::from(expr)).unwrap();
        // Maybe make values reference counted instead of cloning here?
        Ok(ctx.lookup_var(ident).clone())
    }

    fn visit_assign_expr(&mut self, expr: &AssignExpr, ctx: VisitorCtx) -> ExprVisitorResult {
        self.visit_expr(&expr.value, ctx).and_then(|value| {
            self.side_table
                .get(&NodeRef::from(expr))
                .ok_or_else(|| RuntimeError::new(format!("Undefined variable '{}'.", expr.name)))
                .map(|ident| {
                    ctx.assign_var(ident, value.clone());
                    value
                })
        })
    }

    fn visit_lit_expr(&mut self, expr: &LitExpr, ctx: VisitorCtx) -> ExprVisitorResult {
        // Maybe make values reference counted instead of cloning here?
        Ok(Val::from(expr.value.clone()))
    }

    fn visit_function_expr(&mut self, expr: &FunctionExpr, ctx: VisitorCtx) -> ExprVisitorResult {
        Ok(Val::Function(new_function(
            // For now, we assume that the function is anonymous, that is, nameless.
            None,
            // We also assume that the code (in form of the AST) lives at least as long
            // as this function struct.
            &expr,
            ctx.clone(), // Clone is cheap and necessary here.
        )))
    }

    fn visit_call_expr(&mut self, expr: &CallExpr, ctx: VisitorCtx) -> ExprVisitorResult {
        let callee = match self.visit_expr(&expr.callee, ctx)? {
            Val::Function(callee) => callee,
            _ => return Err(RuntimeError::new("Expected function".to_string())),
        };
        let mut callee = callee.borrow_mut();

        // TODO: check arity in resolver just _once_ statically.
        if expr.arguments.len() != callee.arity() {
            return Err(RuntimeError::new(format!(
                "Expected exactly {} arguments, but got {}",
                callee.arity(),
                expr.arguments.len()
            )));
        }

        let args = expr
            .arguments
            .iter()
            .map(|arg| self.visit_expr(arg, ctx))
            .collect::<Result<Vec<_>, _>>()?;

        let body: &Vec<Stmt> =
            // This is a sin, I know, please forgive me. I bet there is a nicer solution.
            // Yet, this is safe because the the `body` and `environment` are disjoint
            // borrows from the `callee` struct.
            unsafe { &*std::ptr::from_ref(&callee.declaration().body.stmts) as &Vec<Stmt> };
        let environment = &mut callee.environment;

        self.visit_block(body, environment, move |environment| {
            for arg in args.into_iter() {
                environment.define_var(arg);
            }
        })
        // We return the default value of `null` if the function does not return
        // anything.
        .map(|val| val.unwrap_or_default())
    }
}

type StmtVisitorResult = Result<Option<ScalarTypedValue>, RuntimeError>;

impl StmtVisitor<StmtVisitorResult, VisitorCtx<'_>> for Interpreter {
    fn visit_var_stmt(&mut self, stmt: &VarStmt, ctx: VisitorCtx) -> StmtVisitorResult {
        stmt.initializer
            .as_ref()
            .map_or_else(
                // We default to null if no initializer is provided.
                || Ok(ScalarTypedValue::default()),
                |expr| {
                    self.visit_expr(expr, ctx).map(|val| {
                        if let Val::Function(function) = &val {
                            // Here, a function turns from anonymous to named.
                            // If the function is later aliased, that is, reassigned to another
                            // variable, we stick to this original name (NodeJS does it, too).
                            function.borrow_mut().name = Some(stmt.name.clone());
                        }
                        val
                    })
                },
            )
            .map(|val| {
                ctx.define_var(val.clone());
                Some(val)
            })
    }

    fn visit_expr_stmt(&mut self, stmt: &ExprStmt, ctx: VisitorCtx) -> StmtVisitorResult {
        // Evaluate the expression and return the result.
        // This is the only statement that can return a value.
        self.visit_expr(&stmt.expr, ctx).map(|expr| Some(expr))
    }

    fn visit_block_stmt(&mut self, stmt: &BlockStmt, ctx: VisitorCtx) -> StmtVisitorResult {
        self.visit_block(&stmt.stmts, ctx, |_env| ())
    }
}

fn is_truthy(value: &ScalarTypedValue) -> bool {
    match value {
        ScalarTypedValue::Null(()) => false,
        ScalarTypedValue::Bool(value) => *value,
        _ => true,
    }
}

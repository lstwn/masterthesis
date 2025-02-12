use crate::{
    env::Environment,
    error::RuntimeError,
    expr::{BinaryExpr, CallExpr, Expr, ExprVisitor, LitExpr, TernaryExpr, UnaryExpr, VarExpr},
    operator::Operator,
    scalar::ScalarTypedValue,
    stmt::{BlockStmt, ExprStmt, FunctionStmt, Program, Stmt, StmtVisitor, VarStmt},
};

pub struct Interpreter {
    env: Environment,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            env: Environment::new(),
        }
    }
    pub fn interpret(
        &mut self,
        program: &Program,
    ) -> Result<Option<ScalarTypedValue>, RuntimeError> {
        // Because we call `execute_block` here, we implicitly create a global
        // scope for the program.
        self.execute_block(&program.stmts)
    }
    pub fn env_mut(&mut self) -> &mut Environment {
        &mut self.env
    }
    fn evaluate(&mut self, expr: &Expr) -> Result<ScalarTypedValue, RuntimeError> {
        self.visit_expr(expr, ())
    }
    fn execute(&mut self, stmt: &Stmt) -> Result<Option<ScalarTypedValue>, RuntimeError> {
        self.visit_stmt(stmt, ())
    }
    fn execute_block(
        &mut self,
        stmts: &Vec<Stmt>,
    ) -> Result<Option<ScalarTypedValue>, RuntimeError> {
        self.env.begin_scope();
        // Functional programming can be so beautiful.
        let ret = stmts
            .iter()
            .try_fold(None, |_prev, stmt| self.execute(stmt));
        self.env.end_scope();
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
    fn visit_lazy_binary_expr(
        &mut self,
        expr: &BinaryExpr,
        ctx: ExprVisitorCtx,
    ) -> ExprVisitorResult {
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
    fn visit_eager_binary_expr(
        &mut self,
        expr: &BinaryExpr,
        ctx: ExprVisitorCtx,
    ) -> ExprVisitorResult {
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

type ExprVisitorResult = Result<ScalarTypedValue, RuntimeError>;
type ExprVisitorCtx = ();

impl ExprVisitor<ExprVisitorResult, ExprVisitorCtx> for Interpreter {
    fn visit_expr(&mut self, expr: &Expr, ctx: ExprVisitorCtx) -> ExprVisitorResult {
        match expr {
            Expr::Ternary(expr) => self.visit_ternary_expr(expr, ctx),
            Expr::Binary(expr) => self.visit_binary_expr(expr, ctx),
            Expr::Unary(expr) => self.visit_unary_expr(expr, ctx),
            Expr::Var(expr) => self.visit_var_expr(expr, ctx),
            Expr::Lit(expr) => self.visit_lit_expr(expr, ctx),
            Expr::Call(expr) => self.visit_call_expr(expr, ctx),
        }
    }

    fn visit_ternary_expr(&mut self, expr: &TernaryExpr, ctx: ExprVisitorCtx) -> ExprVisitorResult {
        todo!()
    }

    fn visit_binary_expr(&mut self, expr: &BinaryExpr, ctx: ExprVisitorCtx) -> ExprVisitorResult {
        if let Operator::And | Operator::Or = expr.operator {
            self.visit_lazy_binary_expr(expr, ctx)
        } else {
            self.visit_eager_binary_expr(expr, ctx)
        }
    }

    fn visit_unary_expr(&mut self, expr: &UnaryExpr, ctx: ExprVisitorCtx) -> ExprVisitorResult {
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

    fn visit_var_expr(&mut self, expr: &VarExpr, ctx: ExprVisitorCtx) -> ExprVisitorResult {
        // Maybe make values reference counted instead of cloning here?
        Ok(self.env.lookup_var(expr).clone())
    }

    fn visit_lit_expr(&mut self, expr: &LitExpr, ctx: ExprVisitorCtx) -> ExprVisitorResult {
        // Maybe make values reference counted instead of cloning here?
        Ok(expr.value.clone())
    }

    fn visit_call_expr(&mut self, expr: &CallExpr, ctx: ExprVisitorCtx) -> ExprVisitorResult {
        todo!()
    }
}

type StmtVisitorResult = Result<Option<ScalarTypedValue>, RuntimeError>;
type StmtVisitorCtx = ();

impl StmtVisitor<StmtVisitorResult, StmtVisitorCtx> for Interpreter {
    fn visit_stmt(&mut self, stmt: &Stmt, ctx: StmtVisitorCtx) -> StmtVisitorResult {
        match stmt {
            Stmt::Var(stmt) => self.visit_var_stmt(stmt, ctx),
            Stmt::Expr(stmt) => self.visit_expr_stmt(stmt, ctx),
            Stmt::Block(stmt) => self.visit_block_stmt(stmt, ctx),
            Stmt::Function(stmt) => self.visit_function_stmt(stmt, ctx),
        }
    }

    fn visit_var_stmt(&mut self, stmt: &VarStmt, ctx: StmtVisitorCtx) -> StmtVisitorResult {
        stmt.initializer
            .as_ref()
            .map_or_else(
                // We default to null if no initializer is provided.
                || Ok(ScalarTypedValue::default()),
                |expr| self.evaluate(expr),
            )
            .map(|val| {
                self.env.define_var(val);
                None
            })
    }

    fn visit_expr_stmt(&mut self, stmt: &ExprStmt, ctx: StmtVisitorCtx) -> StmtVisitorResult {
        // Evaluate the expression and return the result.
        // This is the only statement that can return a value.
        self.evaluate(&stmt.expr).map(|expr| Some(expr))
    }

    fn visit_block_stmt(&mut self, stmt: &BlockStmt, ctx: StmtVisitorCtx) -> StmtVisitorResult {
        self.execute_block(&stmt.stmts)
    }

    fn visit_function_stmt(
        &mut self,
        stmt: &FunctionStmt,
        ctx: StmtVisitorCtx,
    ) -> StmtVisitorResult {
        todo!()
    }
}

fn is_truthy(value: &ScalarTypedValue) -> bool {
    match value {
        ScalarTypedValue::Null(()) => false,
        ScalarTypedValue::Bool(value) => *value,
        _ => true,
    }
}

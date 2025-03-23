use crate::{
    context::InterpreterContext,
    error::RuntimeError,
    expr::{
        AssignExpr, BinaryExpr, CallExpr, EquiJoinExpr, Expr, ExprVisitor, FunctionExpr,
        GroupingExpr, LiteralExpr, ProjectionExpr, SelectionExpr, TernaryExpr, ThetaJoinExpr,
        UnaryExpr, VarExpr,
    },
    function::new_function,
    operator::Operator,
    relation::{SchemaTuple, TupleKey, TupleValue, new_relation},
    scalar::ScalarTypedValue,
    stmt::{BlockStmt, ExprStmt, Stmt, StmtVisitor, VarStmt},
    variable::{Environment, Value},
};
use std::rc::Rc;

pub struct Interpreter {}

impl Interpreter {
    pub fn new() -> Self {
        Self {}
    }
    pub fn interpret<'a>(
        &mut self,
        stmts: impl IntoIterator<Item = &'a Stmt>,
        ctx: &mut InterpreterContext,
    ) -> Result<Option<Value>, RuntimeError> {
        // Ensure we have a global scope before interpreting.
        debug_assert!(ctx.environment.just_global());
        // We do not call `visit_block` here because the root scope is created
        // in the `Environment` constructor and should remain intact across
        // multiple calls to `interpret`.
        let ret = self.visit_stmts(stmts, ctx);
        // Ensure we have a global scope after interpreting.
        debug_assert!(ctx.environment.just_global());
        ret
    }
    fn evaluate(&mut self, expr: &Expr, ctx: VisitorCtx) -> Result<Value, RuntimeError> {
        self.visit_expr(expr, ctx)
    }
    fn visit_stmts<'a>(
        &mut self,
        stmts: impl IntoIterator<Item = &'a Stmt>,
        ctx: VisitorCtx,
    ) -> Result<Option<Value>, RuntimeError> {
        // Functional programming can be so beautiful. Return the last value
        // if any.
        stmts
            .into_iter()
            .try_fold(None, |_prev, stmt| self.visit_stmt(stmt, ctx))
    }
    fn visit_block<'a, F: FnOnce(&mut Environment) -> ()>(
        &mut self,
        stmts: impl IntoIterator<Item = &'a Stmt>,
        ctx: VisitorCtx,
        environment: F,
    ) -> Result<Option<Value>, RuntimeError> {
        ctx.environment.begin_scope();
        environment(&mut ctx.environment);
        let ret = self.visit_stmts(stmts, ctx);
        ctx.environment.end_scope();
        ret
    }
}

macro_rules! comparison_helper {
    ($left:expr, $right:expr, $op:tt, $($variant:path),*) => {{
        match (&$left, &$right) {
            $(
                ($variant(left), $variant(right)) => Ok(Value::Bool(left $op right)),
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

macro_rules! assert_type {
    ($value:expr, $variant:path, $kind:expr) => {
        match $value {
            $variant(relation) => relation,
            _ => {
                return Err(RuntimeError::new(format!(
                    "Expected {} type, got: {:?}",
                    $kind, $value
                )));
            }
        }
    };
}

impl Interpreter {
    fn visit_lazy_binary_expr(&mut self, expr: &BinaryExpr, ctx: VisitorCtx) -> ExprVisitorResult {
        let left = is_truthy(&self.visit_expr(&expr.left, ctx)?);

        if let Operator::And | Operator::Or = expr.operator {
            if expr.operator == Operator::And && left || expr.operator == Operator::Or && !left {
                let right = is_truthy(&self.visit_expr(&expr.right, ctx)?);
                Ok(Value::Bool(right))
            } else {
                Ok(Value::Bool(left))
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
                comparison_helper!(left, right, ==, Value::Iint, Value::Uint, Value::Bool, Value::String, Value::Null)
            }
            Operator::NotEqual => {
                comparison_helper!(left, right, !=, Value::Iint, Value::Uint, Value::Bool, Value::String, Value::Null)
            }
            Operator::Less => {
                comparison_helper!(left, right, <, Value::Iint, Value::Uint, Value::Bool, Value::String)
            }
            Operator::LessEqual => {
                comparison_helper!(left, right, <=, Value::Iint, Value::Uint, Value::Bool, Value::String)
            }
            Operator::Greater => {
                comparison_helper!(left, right, >, Value::Iint, Value::Uint, Value::Bool, Value::String)
            }
            Operator::GreaterEqual => {
                comparison_helper!(left, right, >=, Value::Iint, Value::Uint, Value::Bool, Value::String)
            }
            Operator::Addition => {
                if let (Value::String(left), Value::String(right)) = (&left, &right) {
                    return Ok(Value::String(format!("{}{}", left, right)));
                }
                arithmetic_helper!(left, right, +, Value::Iint, Value::Uint)
            }
            Operator::Subtraction => {
                arithmetic_helper!(left, right, -, Value::Iint, Value::Uint)
            }
            Operator::Multiplication => {
                arithmetic_helper!(left, right, *, Value::Iint, Value::Uint)
            }
            Operator::Division => {
                arithmetic_helper!(left, right, /, Value::Iint, Value::Uint)
            }
            _ => Err(RuntimeError::new(format!(
                "unsupported (eager) binary operator: {:?}",
                expr.operator
            ))),
        }
    }
}

type VisitorCtx<'a, 'b> = &'a mut InterpreterContext<'b>;

type ExprVisitorResult = Result<Value, RuntimeError>;

impl<'a, 'b> ExprVisitor<ExprVisitorResult, VisitorCtx<'a, 'b>> for Interpreter {
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
                Value::Iint(value) => Ok(Value::Iint(-value)),
                _ => Err(RuntimeError::new(format!(
                    "expected signed int, got: {:?}",
                    operand
                ))),
            },
            Operator::Not => Ok(Value::Bool(!is_truthy(&operand))),
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
        let name = &expr.name;
        ctx.tuple_vars.get(name).map_or_else(
            || {
                let resolved = expr
                    .resolved
                    .as_ref()
                    // This should never happen because the resolver should have resolved
                    // all non-tuple variables before the interpreter starts.
                    .expect(&format!("Unresolved variable '{name}'."));
                // Maybe make values reference counted instead of cloning here?
                Ok(ctx.environment.lookup_var(resolved).clone())
            },
            |value| Ok(Value::from(value.clone())),
        )
    }

    fn visit_assign_expr(&mut self, expr: &AssignExpr, ctx: VisitorCtx) -> ExprVisitorResult {
        let name = &expr.name;
        self.visit_expr(&expr.value, ctx).and_then(|value| {
            let resolved = expr
                .resolved
                .as_ref()
                // This should never happen because the resolver should have resolved
                // all non-tuple variables before the interpreter starts and assigning
                // to tuple variables is not allowed.
                .expect(&format!("Unresolved variable '{name}'."));
            ctx.environment.assign_var(resolved, value.clone());
            Ok(value)
        })
    }

    fn visit_literal_expr(&mut self, expr: &LiteralExpr, ctx: VisitorCtx) -> ExprVisitorResult {
        // Maybe make values reference counted instead of cloning here?
        Ok(Value::from(expr.value.clone()))
    }

    fn visit_function_expr(&mut self, expr: &FunctionExpr, ctx: VisitorCtx) -> ExprVisitorResult {
        Ok(Value::Function(new_function(
            // For now, we assume that the function is anonymous, that is, nameless.
            None,
            // We clone here to let functions own their code and thus,
            // we do not have to worry about the lifetime of their code.
            expr.clone(),
            ctx.environment.clone(), // Clone is cheap and necessary here.
        )))
    }

    fn visit_call_expr(&mut self, expr: &CallExpr, ctx: VisitorCtx) -> ExprVisitorResult {
        let callee = assert_type!(
            self.visit_expr(&expr.callee, ctx)?,
            Value::Function,
            "function"
        );
        let mut callee = callee.borrow_mut();

        // TODO: Optimize by checking arity in resolver just _once_ statically.
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
        let mut new_ctx = InterpreterContext::new(&mut callee.environment);

        self.visit_block(body, &mut new_ctx, move |environment| {
            for arg in args.into_iter() {
                environment.define_var(arg);
            }
        })
        // We return the default value of `null` if the function does not return
        // anything.
        .map(|value| value.unwrap_or_default())
    }
    fn visit_selection_expr(&mut self, expr: &SelectionExpr, ctx: VisitorCtx) -> ExprVisitorResult {
        let relation = assert_type!(
            self.visit_expr(&expr.relation, ctx)?,
            Value::Relation,
            "relation"
        );
        let relation_ref = relation.borrow();
        let relation_clone = Rc::clone(&relation);

        let condition = expr.condition.clone();
        let environment = ctx.environment.clone();
        let selected = relation_ref.inner.filter(move |(key, tuple)| {
            // No need to run resolver here, already resolved!
            let schema = &relation_clone.borrow().schema;
            let mut environment = &mut environment.clone();
            let mut new_ctx = InterpreterContext::new(&mut environment);
            new_ctx.begin_tuple_ctx(schema, tuple);
            let value = Interpreter::new()
                .evaluate(&condition, &mut new_ctx)
                .expect("Runtime error while interpreting selection condition");
            is_truthy(&value)
        });

        Ok(Value::Relation(new_relation(
            format!("{}-selected", relation_ref.name),
            relation_ref.schema.select(),
            selected,
        )))
    }
    fn visit_projection_expr(
        &mut self,
        expr: &ProjectionExpr,
        ctx: VisitorCtx,
    ) -> ExprVisitorResult {
        let relation = assert_type!(
            self.visit_expr(&expr.relation, ctx)?,
            Value::Relation,
            "relation"
        );
        let relation_ref = relation.borrow();

        let (attributes, requires_dbsp) = expr.attributes.iter().fold(
            (Vec::with_capacity(expr.attributes.len()), false),
            |(mut attributes, mut requires_dbsp), (name, map_fn)| {
                if map_fn.is_some() {
                    requires_dbsp = true;
                }
                attributes.push(name.clone());
                (attributes, requires_dbsp)
            },
        );

        let (schema, projected) = if requires_dbsp {
            let projected = relation_ref.inner.map_index({
                let relation_clone = Rc::clone(&relation);
                let environment = ctx.environment.clone();
                let map_fns: Vec<Expr> = expr
                    .attributes
                    .iter()
                    .map(|(name, map_fn)| {
                        let map_fn_or_identity = map_fn.clone().unwrap_or_else(|| {
                            Expr::Var(Box::new(VarExpr {
                                name: name.clone(),
                                resolved: None,
                            }))
                        });
                        map_fn_or_identity
                    })
                    .collect();
                move |(key, tuple)| {
                    let schema = &relation_clone.borrow().schema;
                    let mut environment = &mut environment.clone();
                    let mut new_ctx = InterpreterContext::new(&mut environment);
                    new_ctx.begin_tuple_ctx(schema, tuple);
                    let mapped_tuple: TupleValue = map_fns
                        .iter()
                        .map(|map_fn| {
                            ScalarTypedValue::try_from(
                                Interpreter::new()
                                    .evaluate(&map_fn, &mut new_ctx)
                                    .expect("Runtime error while interpreting projection function"),
                            )
                            .expect("Type error while interpreting projection function")
                        })
                        .collect();
                    // For now we leave the key as is.
                    (key.clone(), mapped_tuple)
                }
            });
            let schema = relation_ref.schema.project(attributes);
            (schema, projected)
        } else {
            let picked = relation_ref.inner.clone();
            let schema = relation_ref.schema.pick(&attributes);
            (schema, picked)
        };

        Ok(Value::Relation(new_relation(
            format!("{}-projected", relation_ref.name),
            schema,
            projected,
        )))
    }

    fn visit_equi_join_expr(&mut self, expr: &EquiJoinExpr, ctx: VisitorCtx) -> ExprVisitorResult {
        let left = assert_type!(
            self.visit_expr(&expr.left, ctx)?,
            Value::Relation,
            "relation"
        );
        let left_ref = left.borrow();

        let right = assert_type!(
            self.visit_expr(&expr.right, ctx)?,
            Value::Relation,
            "relation"
        );
        let right_ref = right.borrow();

        let left_indexed = left_ref.inner.map_index({
            let left = Rc::clone(&left);
            let key_fields = expr.attributes.clone();
            move |(key, tuple)| {
                let key: TupleKey = SchemaTuple::new(&left.borrow().schema.tuple, tuple)
                    .pick(&key_fields)
                    .collect();
                (key, tuple.clone())
            }
        });
        let right_indexed = right_ref.inner.map_index({
            let right = Rc::clone(&right);
            let key_fields = expr.attributes.clone();
            move |(key, tuple)| {
                let key: TupleKey = SchemaTuple::new(&right.borrow().schema.tuple, tuple)
                    .pick(&key_fields)
                    .collect();
                (key, tuple.clone())
            }
        });
        let joined = left_indexed.join_index(&right_indexed, {
            let left_rel = Rc::clone(&left);
            let right_rel = Rc::clone(&right);
            move |key, left, right| {
                let tuple = SchemaTuple::new(&left_rel.borrow().schema.tuple, left)
                    .join(&SchemaTuple::new(&right_rel.borrow().schema.tuple, right))
                    .collect();
                Some((key.clone(), tuple))
            }
        });

        let schema = left_ref
            .schema
            .join(&right_ref.schema, expr.attributes.clone());

        Ok(Value::Relation(new_relation(
            format!("{}-{}-joined", left_ref.name, right_ref.name),
            schema,
            joined,
        )))
    }

    fn visit_theta_join_expr(
        &mut self,
        expr: &ThetaJoinExpr,
        ctx: VisitorCtx,
    ) -> ExprVisitorResult {
        // Unfortunately, this will be a cartesian product followed by a selection,
        // as done by most DB systems out there.
        // Yet, there are smarter but more involved approaches:
        // https://hpi.de/oldsite/fileadmin/user_upload/fachgebiete/naumann/publications/PDFs/2021_weise_optimized.pdf
        todo!()
    }
}

type StmtVisitorResult = Result<Option<Value>, RuntimeError>;

impl<'a, 'b> StmtVisitor<StmtVisitorResult, VisitorCtx<'a, 'b>> for Interpreter {
    fn visit_var_stmt(&mut self, stmt: &VarStmt, ctx: VisitorCtx) -> StmtVisitorResult {
        stmt.initializer
            .as_ref()
            .map_or_else(
                // We default to null if no initializer is provided.
                || Ok(Value::default()),
                |expr| {
                    self.visit_expr(expr, ctx).map(|val| {
                        if let Value::Function(function) = &val {
                            // Here, a function turns from anonymous to named.
                            // If the function is later aliased, that is, reassigned to another
                            // variable, we stick to this original name (NodeJS does it, too).
                            function.borrow_mut().name = Some(stmt.name.clone());
                        }
                        val
                    })
                },
            )
            .map(|value| {
                ctx.environment.define_var(value.clone());
                Some(value)
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

fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null(()) => false,
        Value::Bool(value) => *value,
        _ => true,
    }
}

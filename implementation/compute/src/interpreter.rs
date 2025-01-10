use crate::{
    expr::{BinaryExpr, Expr, IdentExpr, NaryExpr, TernaryExpr, UnaryExpr, Visitor},
    operator::Operator,
    scalar::ScalarTypedValue,
};
use thiserror::Error;

struct Interpreter {}

impl Interpreter {
    fn interpret(&mut self, expr: &Expr) -> () {
        match self.visit_expr(expr, ()) {
            Ok(value) => println!("{:?}", value),
            Err(error) => eprintln!("{:?}", error),
        }
    }
}

type ExprResult = Result<ScalarTypedValue, RuntimeError>;
type Ctx = ();

macro_rules! comparison_helper {
    ($left:expr, $right:expr, $op:tt, $($variant:path),*) => {{
        match (&$left, &$right) {
            $(
                ($variant(left), $variant(right)) => Ok(ScalarTypedValue::Bool(left $op right)),
            )*
            _ => Err(RuntimeError {
                message: format!("expected comparable type, got: {:?} and {:?}", $left, $right),
            }),
        }
    }}
}

macro_rules! arithmetic_helper {
    ($left:expr, $right:expr, $op:tt, $($variant:path),*) => {{
        match (&$left, &$right) {
            $(
                ($variant(left), $variant(right)) => Ok($variant(left $op right)),
            )*
            _ => Err(RuntimeError {
                message: format!("expected number type, got: {:?} and {:?}", $left, $right),
            }),
        }
    }}
}

impl Interpreter {
    fn visit_lazy_binary_expr(&mut self, expr: &BinaryExpr, ctx: Ctx) -> ExprResult {
        let left = is_truthy(&self.visit_expr(&expr.left, ctx)?);

        if let Operator::And | Operator::Or = expr.operator {
            if expr.operator == Operator::And && left || expr.operator == Operator::Or && !left {
                let right = is_truthy(&self.visit_expr(&expr.right, ctx)?);
                Ok(ScalarTypedValue::Bool(right))
            } else {
                Ok(ScalarTypedValue::Bool(left))
            }
        } else {
            Err(RuntimeError {
                message: format!("unsupported (lazy) binary operator: {:?}", expr.operator),
            })
        }
    }
    fn visit_eager_binary_expr(&mut self, expr: &BinaryExpr, ctx: Ctx) -> ExprResult {
        let left = self.visit_expr(&expr.left, ctx)?;
        let right = self.visit_expr(&expr.right, ctx)?;

        match expr.operator {
            Operator::Equal => {
                comparison_helper!(left, right, ==, ScalarTypedValue::Iint, ScalarTypedValue::Uint, ScalarTypedValue::Bool, ScalarTypedValue::String, ScalarTypedValue::Null)
            },
            Operator::NotEqual => {
                comparison_helper!(left, right, !=, ScalarTypedValue::Iint, ScalarTypedValue::Uint, ScalarTypedValue::Bool, ScalarTypedValue::String, ScalarTypedValue::Null)
            },
            Operator::Less => {
                comparison_helper!(left, right, <, ScalarTypedValue::Iint, ScalarTypedValue::Uint, ScalarTypedValue::Bool, ScalarTypedValue::String)
            },
            Operator::LessThan => {
                comparison_helper!(left, right, <=, ScalarTypedValue::Iint, ScalarTypedValue::Uint, ScalarTypedValue::Bool, ScalarTypedValue::String)
            },
            Operator::Greater => {
                comparison_helper!(left, right, >, ScalarTypedValue::Iint, ScalarTypedValue::Uint, ScalarTypedValue::Bool, ScalarTypedValue::String)
            },
            Operator::GreaterThan => {
                comparison_helper!(left, right, >=, ScalarTypedValue::Iint, ScalarTypedValue::Uint, ScalarTypedValue::Bool, ScalarTypedValue::String)
            },
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
            _ => Err(RuntimeError {
                message: format!("unsupported (eager) binary operator: {:?}", expr.operator),
            }),
        }
    }
}

impl Visitor<ExprResult, Ctx> for Interpreter {
    fn visit_expr(&mut self, expr: &Expr, ctx: Ctx) -> ExprResult {
        match expr {
            Expr::Nary(expr) => self.visit_nary_expr(expr, ctx),
            Expr::Ternary(expr) => self.visit_ternary_expr(expr, ctx),
            Expr::Binary(expr) => self.visit_binary_expr(expr, ctx),
            Expr::Unary(expr) => self.visit_unary_expr(expr, ctx),
            Expr::Variable(expr) => self.visit_ident_expr(expr, ctx),
            Expr::Literal(expr) => self.visit_lit_expr(expr, ctx),
        }
    }

    fn visit_nary_expr(&mut self, expr: &NaryExpr, ctx: Ctx) -> ExprResult {
        todo!()
    }

    fn visit_ternary_expr(&mut self, expr: &TernaryExpr, ctx: Ctx) -> ExprResult {
        todo!()
    }

    fn visit_binary_expr(&mut self, expr: &BinaryExpr, ctx: Ctx) -> ExprResult {
        if let Operator::And | Operator::Or = expr.operator {
            self.visit_lazy_binary_expr(expr, ctx)
        } else {
            self.visit_eager_binary_expr(expr, ctx)
        }
    }

    fn visit_unary_expr(&mut self, expr: &UnaryExpr, ctx: Ctx) -> ExprResult {
        let operand = self.visit_expr(&expr.operand, ctx)?;

        match expr.operator {
            Operator::Subtraction => match operand {
                ScalarTypedValue::Iint(value) => Ok(ScalarTypedValue::Iint(-value)),
                _ => Err(RuntimeError {
                    message: format!("expected signed int, got: {:?}", operand),
                }),
            },
            Operator::Not => Ok(ScalarTypedValue::Bool(!is_truthy(&operand))),
            _ => Err(RuntimeError {
                message: format!("unsupported unary operator: {:?}", expr.operator),
            }),
        }
    }

    fn visit_ident_expr(&mut self, expr: &IdentExpr, ctx: Ctx) -> ExprResult {
        todo!()
    }

    fn visit_lit_expr(&mut self, expr: &crate::expr::LitExpr, ctx: Ctx) -> ExprResult {
        Ok(expr.value.clone())
    }
}

fn is_truthy(value: &ScalarTypedValue) -> bool {
    match value {
        ScalarTypedValue::Null(()) => false,
        ScalarTypedValue::Bool(value) => *value,
        _ => true,
    }
}

#[derive(Error, Debug)]
#[error("runtime error: {message}")]
struct RuntimeError {
    message: String,
    // TODO: token: Token,
}

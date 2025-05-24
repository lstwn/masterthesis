use compute::{
    dbsp::{DBSPHandle, DbspError, DbspInputs, DbspOutput, RootCircuit, Runtime},
    error::{RuntimeError, SyntaxError},
    stmt::Code,
    variable::Value,
    IncLog,
};
use optimizer::Optimizer;
use parser::Parser;

fn main() {
    println!("Hello, world!");
}

#[derive(Clone)]
pub struct IncDataLog {
    threads: usize, // Use NonZeroUsize?
    optimize: bool,
}

impl Default for IncDataLog {
    fn default() -> Self {
        Self {
            threads: 1,
            optimize: true,
        }
    }
}

impl IncDataLog {
    pub fn build_circuit_from_ir<F>(
        &self,
        intermediate_representation: F,
    ) -> Result<(DBSPHandle, DbspInputs, DbspOutput), anyhow::Error>
    where
        F: Fn(&mut RootCircuit, &mut DbspInputs) -> Result<Code, SyntaxError>
            + Clone
            + Send
            + Sync
            + 'static,
    {
        let optimizer = self.init_optimizer();
        let (circuit, (inputs, output)) = self.init_dbsp_runtime(move |root_circuit| {
            let mut inputs = DbspInputs::default();
            let naive_program = intermediate_representation(root_circuit, &mut inputs)?;
            let optimized_program = if let Some(optimizer) = optimizer {
                optimizer.optimize(naive_program)?
            } else {
                naive_program
            };
            Self::build_circuit(inputs, optimized_program).map_err(anyhow::Error::from)
        })?;

        Ok((circuit, inputs, output))
    }
    pub fn build_circuit_from_datalog(
        &self,
        code: String,
    ) -> Result<(DBSPHandle, DbspInputs, DbspOutput), anyhow::Error> {
        let optimizer = self.init_optimizer();
        let (circuit, (inputs, output)) = self.init_dbsp_runtime(move |root_circuit| {
            let (inputs, naive_program) = Parser::new(root_circuit).parse(code.as_str())?;
            let optimized_program = if let Some(optimizer) = optimizer {
                optimizer.optimize(naive_program)?
            } else {
                naive_program
            };
            Self::build_circuit(inputs, optimized_program).map_err(anyhow::Error::from)
        })?;

        Ok((circuit, inputs, output))
    }
    fn init_dbsp_runtime<F, T>(&self, constructor: F) -> Result<(DBSPHandle, T), DbspError>
    where
        F: FnOnce(&mut RootCircuit) -> Result<T, anyhow::Error> + Clone + Send + 'static,
        T: Send + 'static,
    {
        Runtime::init_circuit(self.threads, constructor)
    }
    fn init_optimizer(&self) -> Option<Optimizer> {
        if self.optimize {
            Some(Optimizer::default())
        } else {
            None
        }
    }
    fn build_circuit(
        inputs: DbspInputs,
        program: Code,
    ) -> Result<(DbspInputs, DbspOutput), RuntimeError> {
        let output = IncLog::default().execute(program);

        let output = match output {
            Ok(Some(Value::Relation(relation))) => {
                let relation = relation.borrow();
                let output_handle = relation.inner.output();
                let output_schema = relation.schema.clone();
                DbspOutput::new(output_schema, output_handle)
            }
            result => {
                return Err(RuntimeError::new(format!(
                    "Expected a relation as program's output, got {:?}",
                    result
                )));
            }
        };

        Ok((inputs, output))
    }
}

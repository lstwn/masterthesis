use parser::list_crdt::ListOperation;
use std::ops::Range;

pub trait TraceProvider {
    fn list_ops(&self) -> &Vec<ListOperation>;
    fn final_result(&self) -> &String;
    fn result_until(&self, until: usize) -> String {
        let mut result = Vec::new();
        for op in self.list_ops_range(0..until) {
            match op {
                ListOperation::InsertAt(idx, char) => {
                    result.insert(*idx, *char);
                }
                ListOperation::DeleteAt(idx) => {
                    result.remove(*idx);
                }
            }
        }
        result.into_iter().collect()
    }
    fn list_ops_range(&self, range: Range<usize>) -> &[ListOperation] {
        &self.list_ops()[range]
    }
}

#[allow(dead_code, unused_variables)]
pub mod automerge_paper {
    use super::*;
    use std::{fs::File, io::BufReader};

    /// This module contains struct definitions to parse Seph Gentle's
    /// editing traces from https://github.com/josephg/editing-traces.
    mod json_definition {
        use serde::Deserialize;

        #[derive(Clone, Debug, Deserialize)]
        pub struct Trace {
            #[serde(rename = "startContent")]
            pub start_content: String,
            #[serde(rename = "endContent")]
            pub end_content: String,
            #[serde(rename = "txns")]
            pub transactions: Vec<Transaction>,
        }

        #[derive(Clone, Debug, Deserialize)]
        pub struct Transaction {
            pub time: String,
            pub patches: Vec<Patch>,
        }

        /// In JS' `array.splice(start, deleteCount, content)` format.
        #[derive(Clone, Debug, Deserialize)]
        #[serde(from = "TriplePatch")]
        pub struct Patch {
            pub start: usize,
            pub delete_count: usize,
            pub content: String,
        }

        #[derive(Clone, Debug, Deserialize)]
        struct TriplePatch(usize, usize, String);

        impl From<TriplePatch> for Patch {
            fn from(value: TriplePatch) -> Self {
                Patch {
                    start: value.0,
                    delete_count: value.1,
                    content: value.2,
                }
            }
        }
    }

    pub struct AutomergePaperTrace {
        final_result: String,
        list_ops: Vec<ListOperation>,
    }

    impl AutomergePaperTrace {
        pub fn load() -> Self {
            let file =
                File::open("benches/automerge-paper.json").expect("Failed to open trace file");
            let reader = BufReader::new(file);
            let trace: json_definition::Trace =
                serde_json::from_reader(reader).expect("Failed to parse trace file");
            let final_result = trace.end_content;
            let list_ops = trace
                .transactions
                .into_iter()
                .flat_map(|tx| {
                    tx.patches.into_iter().map(|patch| {
                        if patch.delete_count > 0 {
                            assert!(patch.content.is_empty(), "no replace operations");
                            ListOperation::DeleteAt(patch.start)
                        } else {
                            assert!(patch.content.len() == 1, "only single character inserts");
                            assert!(patch.delete_count == 0, "no replace operations");
                            ListOperation::InsertAt(
                                patch.start,
                                patch.content.chars().next().unwrap(),
                            )
                        }
                    })
                })
                .collect::<Vec<_>>();
            Self {
                final_result,
                list_ops,
            }
        }
    }

    impl TraceProvider for AutomergePaperTrace {
        fn list_ops(&self) -> &Vec<ListOperation> {
            &self.list_ops
        }
        fn final_result(&self) -> &String {
            &self.final_result
        }
    }
}

pub mod from_string {
    use super::*;

    /// Generates [`ListOperation`]s from a string. Hence, the operations contain
    /// only [`ListOperation::InsertAt`]s which are consecutive and without any
    /// cursor jumps in between.
    pub struct StringTrace {
        final_result: String,
        list_ops: Vec<ListOperation>,
    }

    impl StringTrace {
        pub fn new<T: AsRef<str>>(string: T) -> Self {
            let list_ops = string
                .as_ref()
                .chars()
                .enumerate()
                .map(|(idx, char)| ListOperation::InsertAt(idx, char))
                .collect();
            Self {
                list_ops,
                final_result: string.as_ref().to_string(),
            }
        }
        pub fn with_len<T: AsRef<str>>(base_string: T, len: usize) -> Self {
            let string = base_string
                .as_ref()
                .chars()
                .cycle()
                .take(len)
                .collect::<String>();
            Self::new(string)
        }
    }

    impl TraceProvider for StringTrace {
        fn list_ops(&self) -> &Vec<ListOperation> {
            &self.list_ops
        }
        fn final_result(&self) -> &String {
            &self.final_result
        }
    }

    pub const TEST_STRING: &str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";
}

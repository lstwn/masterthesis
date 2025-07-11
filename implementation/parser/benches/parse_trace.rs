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

pub fn parse_trace<R: std::io::Read>(reader: R) -> Result<Trace, serde_json::Error> {
    serde_json::from_reader(reader)
}

use crate::env::VarIdent;

pub trait MemAddr {
    /// Make sure that the value is not moved in memory!
    fn mem_addr(&self) -> usize {
        self as *const Self as *const () as usize
    }
}

pub trait Resolvable {
    fn set_resolved(&mut self, resolved: VarIdent) -> ();
}

pub trait Named {
    fn name(&self) -> &str;
}

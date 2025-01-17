pub trait MemAddr {
    /// Make sure that the value is not moved in memory!
    fn mem_addr(&self) -> usize {
        self as *const Self as *const () as usize
    }
}

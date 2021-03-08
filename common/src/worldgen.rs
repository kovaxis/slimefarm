use crate::prelude::*;

pub struct ChunkFillArgs<'a> {
    pub center: ChunkPos,
    pub pos: ChunkPos,
    pub chunk: &'a mut Chunk,
}

pub type ChunkFillRet = Option<()>;

pub trait ChunkGen {
    type Shared: Send + Sync;
    type Local;

    fn split(self) -> (Self::Shared, Self::Local);
    fn fill(shared: &Self::Shared, local: &mut Self::Local, args: ChunkFillArgs) -> ChunkFillRet;
}

pub struct ChunkGenerator {
    shared: *const u8,
    local: *mut u8,
    fill: unsafe fn(*const u8, *mut u8, ChunkFillArgs) -> ChunkFillRet,
    clone_shared: unsafe fn(*const u8) -> *const u8,
    drop_shared: unsafe fn(*const u8),
    drop_local: unsafe fn(*mut u8),
}
unsafe impl Send for ChunkGenerator {}
impl ChunkGenerator {
    pub fn new<G: ChunkGen + 'static>(gen: G) -> Self {
        unsafe fn fill_raw<G: ChunkGen>(
            shared: *const u8,
            local: *mut u8,
            args: ChunkFillArgs,
        ) -> Option<()> {
            let shared =
                mem::ManuallyDrop::new(Arc::<G::Shared>::from_raw(shared as *const G::Shared));
            let mut local =
                mem::ManuallyDrop::new(Box::<G::Local>::from_raw(local as *mut G::Local));
            G::fill(&shared, &mut local, args)
        }
        unsafe fn clone_shared_raw<G: ChunkGen>(shared: *const u8) -> *const u8 {
            let shared =
                mem::ManuallyDrop::new(Arc::<G::Shared>::from_raw(shared as *const G::Shared));
            Arc::into_raw(Arc::clone(&shared)) as *const u8
        }
        unsafe fn drop_shared_raw<G: ChunkGen>(shared: *const u8) {
            drop(Arc::<G::Shared>::from_raw(shared as *const G::Shared));
        }
        unsafe fn drop_local_raw<G: ChunkGen>(local: *mut u8) {
            drop(Box::<G::Local>::from_raw(local as *mut G::Local));
        }
        let (shared, local) = gen.split();
        let shared = Arc::into_raw(Arc::new(shared));
        let local = Box::into_raw(Box::new(local));
        Self {
            shared: shared as *const u8,
            local: local as *mut u8,
            fill: fill_raw::<G>,
            clone_shared: clone_shared_raw::<G>,
            drop_shared: drop_shared_raw::<G>,
            drop_local: drop_local_raw::<G>,
        }
    }

    /// Drops the shared part of `self` and replaces it with a shared copy of the shared part of
    /// `other`.
    ///
    /// # Safety
    ///
    /// Chunk generators must be of the same type.
    pub unsafe fn share_with(&mut self, other: &ChunkGenerator) {
        (self.drop_shared)(self.shared);
        self.shared = (other.clone_shared)(other.shared);
    }

    pub fn fill(&mut self, args: ChunkFillArgs) -> ChunkFillRet {
        unsafe { (self.fill)(self.shared, self.local, args) }
    }
}
impl Drop for ChunkGenerator {
    fn drop(&mut self) {
        unsafe {
            (self.drop_shared)(self.shared);
            (self.drop_local)(self.local);
        }
    }
}

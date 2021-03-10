use crate::prelude::*;

pub struct ChunkFillArgs {
    pub center: ChunkPos,
    pub pos: ChunkPos,
}

pub type ChunkFillRet = Option<ChunkBox>;

pub const CHUNK_COLOR_DOWNSCALE: i32 = 8;
pub const CHUNK_COLOR_BUF_WIDTH: i32 = CHUNK_SIZE / CHUNK_COLOR_DOWNSCALE + 1;
pub const CHUNK_COLOR_BUF_LEN: usize = (CHUNK_SIZE / CHUNK_COLOR_DOWNSCALE + 1).pow(3) as usize;
pub type ChunkColorBuf = [[f32; 3]; CHUNK_COLOR_BUF_LEN];

pub struct BlockColorArgs {
    pub pos: BlockPos,
    pub id: u8,
    pub out: &'static mut ChunkColorBuf,
}

pub type BlockColorRet = ();

pub trait SplitCall {
    type Args;
    type Ret;

    type Shared: Send + Sync;
    type Local;

    fn split(self) -> (Self::Shared, Self::Local);
    fn fill(shared: &Self::Shared, local: &mut Self::Local, args: Self::Args) -> Self::Ret;
}

pub struct DynSplitCall<A, R> {
    shared: *const u8,
    local: *mut u8,
    fill: unsafe fn(*const u8, *mut u8, A) -> R,
    clone_shared: unsafe fn(*const u8) -> *const u8,
    drop_shared: unsafe fn(*const u8),
    drop_local: unsafe fn(*mut u8),
}
unsafe impl<A, R> Send for DynSplitCall<A, R> {}
impl<A, R> DynSplitCall<A, R> {
    pub fn new<G>(gen: G) -> Self
    where
        G: SplitCall<Args = A, Ret = R> + 'static,
    {
        unsafe fn fill_raw<G: SplitCall>(
            shared: *const u8,
            local: *mut u8,
            args: G::Args,
        ) -> G::Ret {
            let shared =
                mem::ManuallyDrop::new(Arc::<G::Shared>::from_raw(shared as *const G::Shared));
            let mut local =
                mem::ManuallyDrop::new(Box::<G::Local>::from_raw(local as *mut G::Local));
            G::fill(&shared, &mut local, args)
        }
        unsafe fn clone_shared_raw<G: SplitCall>(shared: *const u8) -> *const u8 {
            let shared =
                mem::ManuallyDrop::new(Arc::<G::Shared>::from_raw(shared as *const G::Shared));
            Arc::into_raw(Arc::clone(&shared)) as *const u8
        }
        unsafe fn drop_shared_raw<G: SplitCall>(shared: *const u8) {
            drop(Arc::<G::Shared>::from_raw(shared as *const G::Shared));
        }
        unsafe fn drop_local_raw<G: SplitCall>(local: *mut u8) {
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
    /// The type of the shared part of both must be the same.
    pub unsafe fn share_with<A2, R2>(&mut self, other: &DynSplitCall<A2, R2>) {
        (self.drop_shared)(self.shared);
        self.shared = (other.clone_shared)(other.shared);
    }

    pub fn call(&mut self, args: A) -> R {
        unsafe { (self.fill)(self.shared, self.local, args) }
    }
}
impl<A, R> Drop for DynSplitCall<A, R> {
    fn drop(&mut self) {
        unsafe {
            (self.drop_shared)(self.shared);
            (self.drop_local)(self.local);
        }
    }
}

pub type AnyChunkFill = DynSplitCall<ChunkFillArgs, ChunkFillRet>;
pub type AnyBlockColor = DynSplitCall<BlockColorArgs, BlockColorRet>;

/// A set of chunkfill and colorgen functions that share the same shared part.
pub struct AnyGen {
    fill: AnyChunkFill,
    color: AnyBlockColor,
}
impl AnyGen {
    pub fn new<F, C>(fill: F, color: C) -> Self
    where
        F: SplitCall<Args = ChunkFillArgs, Ret = ChunkFillRet> + 'static,
        C: SplitCall<Args = BlockColorArgs, Ret = BlockColorRet, Shared = F::Shared> + 'static,
    {
        let fill = DynSplitCall::new(fill);
        let mut color = DynSplitCall::new(color);
        unsafe {
            color.share_with(&fill);
        }
        Self { fill, color }
    }

    /// # Safety
    ///
    /// The type of the shared parts of both `AnyGen`s must be the same.
    pub unsafe fn share_with(&mut self, other: &AnyGen) {
        self.fill.share_with(&other.fill);
        self.color.share_with(&other.color);
    }

    pub fn split(self) -> (AnyChunkFill, AnyBlockColor) {
        (self.fill, self.color)
    }
}

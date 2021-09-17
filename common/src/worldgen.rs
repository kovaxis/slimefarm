use crate::prelude::*;

pub struct ChunkFillArgs {
    pub center: ChunkPos,
    pub pos: ChunkPos,
    pub layer: i32,
    pub substrate: ChunkView<'static>,
    pub output: ChunkViewOut<'static>,
}

pub type ChunkFillRet = Option<()>;

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
    fn call(shared: &Self::Shared, local: &mut Self::Local, args: Self::Args) -> Self::Ret;
}

#[repr(C)]
pub struct DynSplitCall<A, R> {
    shared: *const u8,
    local: *mut u8,
    call: unsafe extern "C" fn(*const u8, *mut u8, A) -> R,
    clone_shared: unsafe extern "C" fn(*const u8) -> *const u8,
    drop_shared: unsafe extern "C" fn(*const u8),
    drop_local: unsafe extern "C" fn(*mut u8),
}
unsafe impl<A, R> Send for DynSplitCall<A, R> {}
impl<A, R> DynSplitCall<A, R> {
    pub fn new<G>(gen: G) -> Self
    where
        G: SplitCall<Args = A, Ret = R> + 'static,
    {
        unsafe extern "C" fn call_raw<G: SplitCall>(
            shared: *const u8,
            local: *mut u8,
            args: G::Args,
        ) -> G::Ret {
            let shared =
                mem::ManuallyDrop::new(Arc::<G::Shared>::from_raw(shared as *const G::Shared));
            let mut local =
                mem::ManuallyDrop::new(Box::<G::Local>::from_raw(local as *mut G::Local));
            G::call(&shared, &mut local, args)
        }
        unsafe extern "C" fn clone_shared_raw<G: SplitCall>(shared: *const u8) -> *const u8 {
            let shared =
                mem::ManuallyDrop::new(Arc::<G::Shared>::from_raw(shared as *const G::Shared));
            Arc::into_raw(Arc::clone(&shared)) as *const u8
        }
        unsafe extern "C" fn drop_shared_raw<G: SplitCall>(shared: *const u8) {
            drop(Arc::<G::Shared>::from_raw(shared as *const G::Shared));
        }
        unsafe extern "C" fn drop_local_raw<G: SplitCall>(local: *mut u8) {
            drop(Box::<G::Local>::from_raw(local as *mut G::Local));
        }
        let (shared, local) = gen.split();
        let shared = Arc::into_raw(Arc::new(shared));
        let local = Box::into_raw(Box::new(local));
        Self {
            shared: shared as *const u8,
            local: local as *mut u8,
            call: call_raw::<G>,
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
        unsafe { (self.call)(self.shared, self.local, args) }
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
    pub layers: Vec<i32>,
    fill: AnyChunkFill,
    color: AnyBlockColor,
}
impl AnyGen {
    pub fn new<F, C>(layers: Vec<i32>, fill: F, color: C) -> Self
    where
        F: SplitCall<Args = ChunkFillArgs, Ret = ChunkFillRet> + 'static,
        C: SplitCall<Args = BlockColorArgs, Ret = BlockColorRet, Shared = F::Shared> + 'static,
    {
        let fill = DynSplitCall::new(fill);
        let mut color = DynSplitCall::new(color);
        unsafe {
            color.share_with(&fill);
        }
        Self {
            layers,
            fill,
            color,
        }
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

pub struct ChunkView<'a> {
    size: i32,
    chunks: *const ChunkRef<'a>,
    marker: PhantomData<&'a ChunkRef<'a>>,
}
impl<'a> ChunkView<'a> {
    pub fn new(size: i32, chunks: &'a [ChunkRef<'a>]) -> Self {
        assert_eq!(
            (size * size * size) as usize,
            chunks.len(),
            "size does not meet chunks provided"
        );
        Self {
            size,
            chunks: chunks.as_ptr(),
            marker: PhantomData,
        }
    }

    pub fn center_chunk(&self) -> ChunkRef<'a> {
        unsafe { self.chunk_from_idx((self.size * self.size * self.size) as usize / 2) }
    }

    pub fn chunks(&self) -> &'a [ChunkRef<'a>] {
        unsafe {
            std::slice::from_raw_parts(self.chunks, (self.size * self.size * self.size) as usize)
        }
    }

    unsafe fn chunk_from_idx(&self, idx: usize) -> ChunkRef<'a> {
        *self.chunks.offset(idx as isize)
    }

    pub fn try_get(&self, pos: [i32; 3]) -> Option<BlockData> {
        let chunk_pos = [
            (pos[0] as u32 / CHUNK_SIZE as u32) as usize,
            (pos[1] as u32 / CHUNK_SIZE as u32) as usize,
            (pos[2] as u32 / CHUNK_SIZE as u32) as usize,
        ];
        let sub_pos = [
            ((pos[0] + CHUNK_SIZE * (self.size / 2)) as u32 % CHUNK_SIZE as u32) as i32,
            ((pos[1] + CHUNK_SIZE * (self.size / 2)) as u32 % CHUNK_SIZE as u32) as i32,
            ((pos[2] + CHUNK_SIZE * (self.size / 2)) as u32 % CHUNK_SIZE as u32) as i32,
        ];
        if chunk_pos[0] < self.size as usize
            && chunk_pos[1] < self.size as usize
            && chunk_pos[2] < self.size as usize
        {
            let chunk = unsafe {
                self.chunk_from_idx(
                    chunk_pos[0]
                        + chunk_pos[1] * self.size as usize
                        + chunk_pos[2] * (self.size * self.size) as usize,
                )
            };
            Some(chunk.sub_get(sub_pos))
        } else {
            None
        }
    }

    pub fn get(&self, pos: [i32; 3]) -> BlockData {
        self.try_get(pos).unwrap_or(BlockData { data: 0 })
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct ChunkOut<'a> {
    chunk: &'a mut ChunkBox,
}
impl<'a> ChunkOut<'a> {
    pub fn new(chunk: &'a mut ChunkBox) -> Self {
        Self { chunk }
    }

    pub fn make_solid(&mut self) {
        self.chunk.make_solid();
    }

    pub fn make_empty(&mut self) {
        self.chunk.make_empty();
    }

    pub fn sub_set(&mut self, pos: [i32; 3], block: BlockData) {
        *self.chunk.blocks_mut().sub_get_mut(pos) = block;
    }

    pub fn set_idx(&mut self, idx: usize, block: BlockData) {
        self.chunk.blocks_mut().blocks[idx] = block;
    }

    pub fn try_drop_blocks(&mut self) {
        self.chunk.try_drop_blocks();
    }
}

pub struct ChunkViewOut<'a> {
    size: i32,
    chunks: *mut ChunkOut<'a>,
    marker: PhantomData<&'a mut [ChunkOut<'a>]>,
}
impl<'a> ChunkViewOut<'a> {
    pub fn new(size: i32, chunks: &'a mut [&'a mut ChunkBox]) -> Self {
        debug_assert!(size > 0, "size must be positive");
        debug_assert!(size < (1 << 10), "size too large");
        debug_assert!(size % 2 == 1, "size must be odd");
        assert_eq!(
            (size * size * size) as usize,
            chunks.len(),
            "size does not meet chunks provided"
        );
        Self {
            size,
            chunks: chunks.as_mut_ptr() as *mut ChunkOut,
            marker: PhantomData,
        }
    }

    pub fn chunks(&mut self) -> &'a mut [ChunkOut<'a>] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.chunks,
                (self.size * self.size * self.size) as usize,
            )
        }
    }

    pub fn center_chunk(&mut self) -> ChunkOut {
        unsafe { self.chunk_from_idx((self.size * self.size * self.size) as usize / 2) }
    }

    unsafe fn chunk_from_idx(&mut self, idx: usize) -> ChunkOut {
        ptr::read(self.chunks.offset(idx as isize))
    }

    pub fn set(&mut self, pos: [i32; 3], block: BlockData) -> Option<()> {
        let chunk_pos = [
            ((pos[0] + CHUNK_SIZE * (self.size / 2)) >> CHUNK_BITS) as usize,
            ((pos[1] + CHUNK_SIZE * (self.size / 2)) >> CHUNK_BITS) as usize,
            ((pos[2] + CHUNK_SIZE * (self.size / 2)) >> CHUNK_BITS) as usize,
        ];
        let sub_pos = [
            pos[0] & CHUNK_MASK,
            pos[1] & CHUNK_MASK,
            pos[2] & CHUNK_MASK,
        ];
        if chunk_pos[0] < self.size as usize
            && chunk_pos[1] < self.size as usize
            && chunk_pos[2] < self.size as usize
        {
            let mut chunk = unsafe {
                self.chunk_from_idx(
                    chunk_pos[0]
                        + chunk_pos[1] * self.size as usize
                        + chunk_pos[2] * (self.size * self.size) as usize,
                )
            };
            chunk.sub_set(sub_pos, block);
            Some(())
        } else {
            None
        }
    }
}

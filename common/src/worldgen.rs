use crate::prelude::*;

pub trait GenCapsule<'a> {
    fn cast_trait_obj(&self, name: &str) -> (usize, usize);
    fn init(&'a self, _store: GenStoreRef<'a>) {}
    fn cast_concrete(&self) -> *const u8 {
        self as *const Self as *const u8
    }
}

#[macro_export]
macro_rules! impl_cast_to {
    ($($name:expr => $into:path),*) => {
        fn cast_trait_obj(&self, name: &str) -> (usize, usize) {
            match name {
                $(
                    $name => {
                        let newref: &dyn $into = &*self;
                        unsafe { mem::transmute(newref) }
                    }
                )*
                _ => (0, 0),
            }
        }
    };
}

pub unsafe fn cast_capsule_concrete<'a, T: Sized>(cap: &'a dyn GenCapsule<'a>) -> &'a T {
    &*(cap.cast_concrete() as *const T)
}

pub unsafe fn cast_capsule<'a, T: ?Sized>(cap: &'a dyn GenCapsule<'a>, name: &str) -> &'a T {
    assert_eq!(
        mem::size_of::<&'a T>(),
        mem::size_of::<(usize, usize)>(),
        "cannot cast to thin pointer"
    );
    let ptr = cap.cast_trait_obj(name);
    if ptr.0 == 0 || ptr.1 == 0 {
        panic!("cannot cast capsule to trait object `{}`", name);
    }
    mem::transmute_copy(&ptr)
}

pub trait GenStore {
    fn register<'a>(&'a self, name: &str, capsule: Box<dyn 'a + GenCapsule<'a>>);
    fn lookup_capsule<'a>(&'a self, name: &str) -> Option<&'a (dyn 'a + GenCapsule<'a>)>;
}

impl<'a> dyn 'a + GenStore {
    pub unsafe fn try_lookup<T: ?Sized>(&'a self, name: &str) -> Option<&'a T> {
        let cap = self.lookup_capsule(name)?;
        Some(cast_capsule(cap, name))
    }

    pub unsafe fn lookup<T: ?Sized>(&'a self, name: &str) -> &'a T {
        match self.try_lookup(name) {
            Some(cap) => cap,
            None => panic!("failed to find capsule with name `{}`", name),
        }
    }

    pub unsafe fn try_lookup_concrete<T: Sized>(&'a self, name: &str) -> Option<&'a T> {
        let cap = self.lookup_capsule(name)?;
        Some(cast_capsule_concrete(cap))
    }

    pub unsafe fn lookup_concrete<T: Sized>(&'a self, name: &str) -> &'a T {
        match self.try_lookup_concrete(name) {
            Some(cap) => cap,
            None => panic!("failed to find capsule with name `{}`", name),
        }
    }
}

pub type GenStoreRef<'a> = &'a (dyn 'a + GenStore);

pub trait GenStage {
    fn require(&self, min: ChunkPos, max: ChunkPos, layer: i32) -> Option<()>;
    fn place(&self, min: ChunkPos, max: ChunkPos, landmark: LandmarkId) -> Option<()>;
    /// Unsafe because the landmark box kind must match the actual type of the landmark box.
    unsafe fn create_landmark(&self, landmark: LandmarkBox) -> LandmarkId;
    fn landmark_kind(&self, name: &str) -> LandmarkKind;
}

pub struct LandmarkCreator<'a, T> {
    stage: &'a dyn GenStage,
    kind: LandmarkKind,
    _marker: PhantomData<T>,
}
impl<'a, T> LandmarkCreator<'a, T> {
    /// Safety: Guarantee that the given name is associated **only** with the type `T`.
    pub unsafe fn new(stage: &'a dyn GenStage, name: &str) -> Self {
        let kind = stage.landmark_kind(name);
        Self {
            stage,
            kind,
            _marker: PhantomData,
        }
    }
    pub fn create(&self, landmark: T) -> LandmarkId {
        unsafe {
            self.stage
                .create_landmark(LandmarkBox::new(self.kind, landmark))
        }
    }
    pub fn downcast_box(&self, b: LandmarkBox) -> Option<T> {
        if b.kind() == self.kind {
            unsafe { Some(b.into_inner::<T>()) }
        } else {
            None
        }
    }
}

#[repr(C)]
struct LandmarkWrap<T> {
    head: LandmarkHead,
    inner: T,
}

pub struct LandmarkBox {
    ptr: NonNull<LandmarkHead>,
}
impl LandmarkBox {
    pub fn new<T>(kind: LandmarkKind, landmark: T) -> Self {
        unsafe fn drop_raw<T>(ptr: NonNull<LandmarkHead>, drop_all: bool) {
            if drop_all {
                drop(Box::from_raw(ptr.as_ptr() as *mut LandmarkWrap<T>));
            } else {
                drop(Box::from_raw(
                    ptr.as_ptr() as *mut LandmarkWrap<mem::ManuallyDrop<T>>
                ));
            }
        }
        let b = Box::<LandmarkWrap<T>>::new(LandmarkWrap {
            head: LandmarkHead {
                refs: 0,
                kind,
                drop: drop_raw::<T>,
            },
            inner: landmark,
        });
        LandmarkBox {
            ptr: unsafe { NonNull::new_unchecked(Box::into_raw(b) as *mut LandmarkHead) },
        }
    }
}
impl LandmarkBox {
    pub fn kind(&self) -> LandmarkKind {
        unsafe { self.ptr.as_ref().kind }
    }

    /// Safety: The caller guarantees that the contained landmark is of type `T`.
    pub unsafe fn into_inner<T>(self) -> T {
        let ptr = self.ptr;
        mem::forget(self);
        let inner = ptr::read(&(*(ptr.as_ptr() as *mut LandmarkWrap<T>)).inner);
        (ptr.as_ref().drop)(ptr, false);
        inner
    }
}
impl Drop for LandmarkBox {
    fn drop(&mut self) {
        unsafe {
            (self.ptr.as_ref().drop)(self.ptr, true);
        }
    }
}

#[derive(PartialEq, Eq, Copy, Clone)]
pub struct LandmarkKind(pub i32);

#[repr(C)]
pub struct LandmarkHead {
    pub refs: u32,
    pub kind: LandmarkKind,
    /// If true, drop everything.
    /// If false, drop only the memory, but keep the inner landmark body undropped.
    pub drop: unsafe fn(NonNull<LandmarkHead>, bool),
}

pub struct LandmarkId(pub u32);

/// Generator logic.
/// The top level generator should go into the `base.chunkfill` slot in the gen store.
pub trait ChunkFill {
    fn layer_count(&self) -> i32;
    fn fill(&self, args: ChunkFillArgs) -> ChunkFillRet;
    fn colorizer(&self) -> Box<dyn BlockColorizer>;
}

/// SAFETY: The values in this struct have an implicit lifetime.
/// Specifically, if this type is received as an argument, its lifetime must not be extended beyond
/// the argument lifetime.
pub struct ChunkFillArgs {
    pub pos: ChunkPos,
    pub layer: i32,
    pub blocks: Cell<*mut ChunkBox>,
}
impl ChunkFillArgs {
    pub fn take_blocks(&self) -> &mut ChunkBox {
        let val = self.blocks.replace(ptr::null_mut());
        unsafe {
            if val.is_null() {
                panic!("attempt to take blocks twice");
            }
            &mut *val
        }
    }
}

pub type ChunkFillRet = Option<()>;

pub const CHUNK_COLOR_DOWNSCALE: i32 = 8;
pub const CHUNK_COLOR_BUF_WIDTH: i32 = CHUNK_SIZE / CHUNK_COLOR_DOWNSCALE + 1;
pub const CHUNK_COLOR_BUF_LEN: usize = (CHUNK_SIZE / CHUNK_COLOR_DOWNSCALE + 1).pow(3) as usize;
pub type ChunkColorBuf = [[f32; 3]; CHUNK_COLOR_BUF_LEN];

pub trait BlockColorizer: Send {
    fn colorize(&mut self, args: BlockColorArgs) -> BlockColorRet;
}

pub struct BlockColorArgs {
    pub pos: BlockPos,
    pub id: u8,
    pub out: &'static mut ChunkColorBuf,
}

pub type BlockColorRet = ();

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

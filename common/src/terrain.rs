use crate::{
    arena::{Box as ArenaBox, BoxUninit as ArenaBoxUninit},
    prelude::*,
};

/// Base-2 logarithm of the chunk size.
pub const CHUNK_BITS: i32 = 5;

/// Guaranteed to be a power of 2.
pub const CHUNK_SIZE: i32 = 1 << CHUNK_BITS;

/// Masks a block coordinate into a chunk-relative block coordinate.
pub const CHUNK_MASK: i32 = CHUNK_SIZE - 1;

/// A 2D slice (or loaf) of an arbitrary type.
pub type LoafBox<T> = crate::arena::Box<[T; (CHUNK_SIZE * CHUNK_SIZE) as usize]>;

/// Saves chunk tags on the last 2 bits of the chunk pointer.
/// This is why `ChunkData` has an alignment of 4.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct ChunkRef<'a> {
    blocks: NonNull<ChunkData>,
    marker: PhantomData<&'a ChunkData>,
}
unsafe impl Send for ChunkRef<'_> {}
unsafe impl Sync for ChunkRef<'_> {}
impl fmt::Debug for ChunkRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_normal() {
            write!(f, "ChunkRef({}KB)", mem::size_of::<ChunkData>() / 1024)
        } else if self.is_solid() {
            write!(f, "ChunkRef::Solid")
        } else if self.is_empty() {
            write!(f, "ChunkRef::Empty")
        } else {
            write!(f, "ChunkRef::Unknown")
        }
    }
}
impl<'a> ChunkRef<'a> {
    const MASK: usize = 0b11;
    const TAG_NORMAL: usize = 0;
    const TAG_SOLID: usize = 1;
    const TAG_EMPTY: usize = 2;

    #[inline]
    fn tag(&self) -> usize {
        self.blocks.as_ptr() as usize & Self::MASK
    }

    #[inline]
    fn make_tag(tag: usize) -> NonNull<ChunkData> {
        unsafe { NonNull::new_unchecked((0x100 | tag) as *mut ChunkData) }
    }

    #[inline]
    pub fn is_normal(&self) -> bool {
        self.tag() == Self::TAG_NORMAL
    }

    #[inline]
    pub fn is_solid(&self) -> bool {
        self.tag() == Self::TAG_SOLID
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tag() == Self::TAG_EMPTY
    }

    #[inline]
    pub fn blocks(&self) -> Option<&ChunkData> {
        if self.is_normal() {
            unsafe { Some(self.blocks.as_ref()) }
        } else {
            None
        }
    }

    #[inline]
    pub fn sub_get(&self, pos: Int3) -> BlockData {
        if let Some(blocks) = self.blocks() {
            blocks.sub_get(pos)
        } else if self.is_empty() {
            BlockData { data: 0 }
        } else {
            BlockData { data: 1 }
        }
    }

    pub fn into_raw(self) -> *const ChunkData {
        self.blocks.as_ptr()
    }

    pub unsafe fn from_raw(raw: *const ChunkData) -> Self {
        Self {
            blocks: NonNull::new_unchecked(raw as *mut ChunkData),
            marker: PhantomData,
        }
    }

    pub fn clone_chunk(&self) -> ChunkBox {
        if self.is_normal() {
            let mut uninit = ArenaBoxUninit::<ChunkData>::new();
            unsafe {
                ptr::copy_nonoverlapping(self.blocks.as_ptr(), uninit.as_mut().as_mut_ptr(), 1);
                ChunkBox {
                    blocks: ArenaBox::into_raw(uninit.assume_init()),
                }
            }
        } else {
            ChunkBox {
                blocks: self.blocks,
            }
        }
    }
}

#[repr(transparent)]
pub struct ChunkBox {
    blocks: NonNull<ChunkData>,
}
unsafe impl Send for ChunkBox {}
unsafe impl Sync for ChunkBox {}
impl ops::Deref for ChunkBox {
    type Target = ChunkRef<'static>;
    fn deref(&self) -> &ChunkRef<'static> {
        unsafe { mem::transmute(self) }
    }
}
impl fmt::Debug for ChunkBox {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_normal() {
            write!(f, "ChunkBox({}KB)", mem::size_of::<ChunkData>() / 1024)
        } else if self.is_solid() {
            write!(f, "ChunkBox::Solid")
        } else if self.is_empty() {
            write!(f, "ChunkBox::Empty")
        } else {
            write!(f, "ChunkBox::Unknown")
        }
    }
}
impl Default for ChunkBox {
    fn default() -> Self {
        Self::new_empty()
    }
}
impl ChunkBox {
    /// Create a new chunk box with memory allocated to its blocks, but all of them set to 0.
    #[inline]
    pub fn new() -> Self {
        ChunkBox {
            blocks: unsafe { ArenaBox::into_raw(ArenaBoxUninit::<ChunkData>::new().init_zero()) },
        }
    }

    /// Create a new chunk box with memory allocated, but uninitialized.
    /// Inherently unsafe, but may work in most cases and speeds things up.
    #[inline]
    pub unsafe fn new_uninit() -> Self {
        ChunkBox {
            blocks: ArenaBox::into_raw(ArenaBoxUninit::<ChunkData>::new().assume_init()),
        }
    }

    /// Create a new empty chunk box without allocating any memory.
    /// This is basically a compile-time constant.
    #[inline]
    pub fn new_empty() -> Self {
        ChunkBox {
            blocks: ChunkRef::make_tag(ChunkRef::TAG_EMPTY),
        }
    }

    /// Create a new solid chunk box without allocating any memory.
    /// This is basically a compile-time constant.
    #[inline]
    pub fn new_solid() -> Self {
        ChunkBox {
            blocks: ChunkRef::make_tag(ChunkRef::TAG_SOLID),
        }
    }

    /// Create a new chunk box with unspecified but allocated contents.
    #[inline]
    pub fn new_quick() -> Self {
        //Change depending on how bold we feel
        //Self::new()
        unsafe { Self::new_uninit() }
    }

    /// Get an immutable reference to the chunk box.
    #[inline]
    pub fn as_ref(&self) -> ChunkRef {
        ChunkRef {
            blocks: self.blocks,
            marker: PhantomData,
        }
    }

    /// Get a mutable reference to the inner blocks.
    /// If there is no associated memory, this method will allocate and fill with empty or solid
    /// blocks, according to the current state.
    #[inline]
    pub fn blocks_mut(&mut self) -> &mut ChunkData {
        if !self.is_normal() {
            unsafe {
                let solid = self.is_solid();
                *self = ChunkBox::new_uninit();
                if solid {
                    ptr::write_bytes(self.blocks.as_ptr(), 1, 1);
                } else {
                    ptr::write_bytes(self.blocks.as_ptr(), 0, 1);
                }
            }
        }
        unsafe { self.blocks.as_mut() }
    }

    /// Gets the inner blocks if there is memory associated to this chunk box.
    #[inline]
    pub fn try_blocks_mut(&mut self) -> Option<&mut ChunkData> {
        if self.is_normal() {
            unsafe { Some(self.blocks.as_mut()) }
        } else {
            None
        }
    }

    /// The `new_tag` must be a valid, non-normal tag.
    #[inline]
    unsafe fn take_blocks(&mut self, new_tag: usize) -> Option<ArenaBox<ChunkData>> {
        let out = if self.is_normal() {
            let blocks = ArenaBox::from_raw(self.blocks);
            Some(blocks)
        } else {
            None
        };
        self.blocks = ChunkRef::make_tag(new_tag);
        out
    }

    /// Remove any memory associated with this box, and simply make all blocks solid.
    #[inline]
    pub fn make_solid(&mut self) {
        unsafe {
            self.take_blocks(ChunkRef::TAG_SOLID);
        }
    }

    /// Remove any memory associated with this box, and simply make all blocks empty.
    #[inline]
    pub fn make_empty(&mut self) {
        unsafe {
            self.take_blocks(ChunkRef::TAG_EMPTY);
        }
    }

    /// Check if all of the blocks are of the same type, and drop the data altogether and mark with
    /// a tag in that case.
    #[inline]
    pub fn try_drop_blocks(&mut self) {
        if let Some(chunk_data) = self.blocks() {
            let blocks = &chunk_data.blocks;
            if blocks.iter().skip(1).all(|&b| b.data == blocks[0].data) {
                //All the same
                if blocks[0].data == 0 {
                    self.make_empty();
                } else if blocks[0].data == 1 {
                    self.make_solid();
                }
            }
        }
    }
}
impl Drop for ChunkBox {
    fn drop(&mut self) {
        if self.is_normal() {
            unsafe {
                drop(ArenaBox::from_raw(self.blocks));
            }
        }
    }
}

pub type ChunkPos = Int3;
#[allow(non_snake_case)]
pub fn ChunkPos(x: [i32; 3]) -> ChunkPos {
    Int3::new(x)
}

pub type BlockPos = Int3;
#[allow(non_snake_case)]
pub fn BlockPos(x: [i32; 3]) -> BlockPos {
    Int3::new(x)
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockData {
    pub data: u8,
}
impl BlockData {
    #[inline]
    pub fn is_solid(&self) -> bool {
        self.data != 0
    }
}

/// Holds the block data for a chunk.
/// Has an alignment of 4 in order for `ChunkBox` to store tags in the last 2 bits of the pointer.
#[derive(Clone)]
#[repr(align(4))]
pub struct ChunkData {
    pub blocks: [BlockData; (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize],
}
impl ChunkData {
    #[track_caller]
    #[inline]
    pub fn sub_get(&self, sub_pos: Int3) -> BlockData {
        match self.blocks.get(sub_pos.to_index([CHUNK_BITS; 3].into())) {
            Some(&b) => b,
            None => panic!(
                "block index [{}, {}, {}] outside chunk boundaries",
                sub_pos.x, sub_pos.y, sub_pos.z
            ),
        }
    }

    #[track_caller]
    #[inline]
    pub fn sub_get_mut(&mut self, sub_pos: Int3) -> &mut BlockData {
        match self
            .blocks
            .get_mut(sub_pos.to_index([CHUNK_BITS; 3].into()))
        {
            Some(b) => b,
            None => panic!(
                "block index [{}, {}, {}] outside chunk boundaries",
                sub_pos[0], sub_pos[1], sub_pos[2]
            ),
        }
    }

    #[track_caller]
    #[inline]
    pub fn set_idx(&mut self, idx: usize, block: BlockData) {
        self.blocks[idx] = block;
    }
}

pub struct GridKeeper<T> {
    corner_pos: ChunkPos,
    size_log2: u32,
    origin_idx: i32,
    slots: Vec<T>,
}
impl<T: GridSlot> GridKeeper<T> {
    pub fn with_radius(radius: f32, center: ChunkPos) -> Self {
        let size = ((radius * 2.).max(2.) as u32).next_power_of_two().max(1) as i32;
        Self::new(size, center)
    }

    pub fn new(size: i32, center: ChunkPos) -> Self {
        //Make sure length is a power of two
        let size_log2 = (mem::size_of_val(&size) * 8) as u32 - (size - 1).leading_zeros();
        assert_eq!(
            1 << size_log2,
            size,
            "GridKeeper size must be a power of two"
        );
        //Allocate space for meshes
        let total = (1 << (size_log2 * 3)) as usize;
        let mut slots = Vec::with_capacity(total);
        slots.resize_with(total, T::new);
        //Group em up
        Self {
            corner_pos: center - ChunkPos::splat(1 << (size_log2 - 1)),
            size_log2,
            origin_idx: 0,
            slots,
        }
    }

    /// Will slide chunks and remove chunks that went over the border.
    pub fn set_center(&mut self, new_center: ChunkPos) {
        let new_corner = new_center + [-self.half_size(), -self.half_size(), -self.half_size()];
        let adj = new_corner - self.corner_pos;
        let clear_range =
            |this: &mut Self, x: ops::Range<i32>, y: ops::Range<i32>, z: ops::Range<i32>| {
                // OPTIMIZE: Store lists of active slots in each plane slice, so as to only visit
                // actual active slots.
                for z in z.clone() {
                    for y in y.clone() {
                        for x in x.clone() {
                            this.sub_get_mut([x, y, z].into()).reset();
                        }
                    }
                }
            };
        if adj.x > 0 {
            clear_range(self, 0..adj.x, 0..self.size(), 0..self.size());
        } else if adj.x < 0 {
            clear_range(
                self,
                self.size() + adj.x..self.size(),
                0..self.size(),
                0..self.size(),
            );
        }
        if adj.y > 0 {
            clear_range(self, 0..self.size(), 0..adj.y, 0..self.size());
        } else if adj.y < 0 {
            clear_range(
                self,
                0..self.size(),
                self.size() + adj.y..self.size(),
                0..self.size(),
            );
        }
        if adj.z > 0 {
            clear_range(self, 0..self.size(), 0..self.size(), 0..adj.z);
        } else if adj.z < 0 {
            clear_range(
                self,
                0..self.size(),
                0..self.size(),
                self.size() + adj.z..self.size(),
            );
        }
        self.origin_idx =
            (self.origin_idx + adj.x + adj.y * self.size() + adj.z * (self.size() * self.size()))
                .rem_euclid(self.total_len());
        self.corner_pos = new_corner;
    }
}
impl<T> GridKeeper<T> {
    #[inline]
    pub fn center(&self) -> ChunkPos {
        self.corner_pos + [self.half_size(), self.half_size(), self.half_size()]
    }

    #[inline]
    pub fn size_log2(&self) -> u32 {
        self.size_log2
    }

    #[inline]
    pub fn size(&self) -> i32 {
        1 << self.size_log2
    }

    #[inline]
    pub fn half_size(&self) -> i32 {
        1 << (self.size_log2 - 1)
    }

    #[inline]
    pub fn total_len(&self) -> i32 {
        1 << (self.size_log2 * 3)
    }

    #[inline]
    pub fn get(&self, pos: ChunkPos) -> Option<&T> {
        let pos = pos - self.corner_pos;
        if pos.is_within(Int3::splat(self.size())) {
            Some(self.sub_get(pos))
        } else {
            None
        }
    }
    #[inline]
    pub fn get_mut(&mut self, pos: ChunkPos) -> Option<&mut T> {
        let pos = pos - self.corner_pos;
        if pos.is_within(Int3::splat(self.size())) {
            Some(self.sub_get_mut(pos))
        } else {
            None
        }
    }

    #[inline]
    pub fn get_by_idx(&self, idx: i32) -> &T {
        &self.slots[(self.origin_idx + idx).rem_euclid(self.total_len()) as usize]
    }
    #[inline]
    pub fn get_by_idx_mut(&mut self, idx: i32) -> &mut T {
        let idx = (self.origin_idx + idx).rem_euclid(self.total_len()) as usize;
        &mut self.slots[idx]
    }
    #[inline]
    pub fn sub_get(&self, pos: Int3) -> &T {
        &self.slots[(self.origin_idx
            + pos.x
            + pos.y * self.size()
            + pos.z * (self.size() * self.size()))
        .rem_euclid(self.total_len()) as usize]
    }
    #[inline]
    pub fn sub_get_mut(&mut self, pos: Int3) -> &mut T {
        let size = self.size();
        let total_len = self.total_len();
        &mut self.slots[(self.origin_idx + pos.x + pos.y * size + pos.z * (size * size))
            .rem_euclid(total_len) as usize]
    }

    #[inline]
    pub fn sub_idx_to_pos(&self, idx: i32) -> ChunkPos {
        self.corner_pos + Int3::from_index([self.size_log2 as i32; 3].into(), idx as usize)
    }
}

pub struct GridKeeper2d<T> {
    corner_pos: Int2,
    size_log2: u32,
    origin_idx: i32,
    slots: Vec<T>,
}
impl<T: GridSlot> GridKeeper2d<T> {
    pub fn with_radius(radius: f32, center: Int2) -> Self {
        let size = ((radius * 2.).max(2.) as u32).next_power_of_two().max(1) as i32;
        Self::new(size, center)
    }

    pub fn new(size: i32, center: Int2) -> Self {
        //Make sure length is a power of two
        let size_log2 = (mem::size_of_val(&size) * 8) as u32 - (size - 1).leading_zeros();
        assert_eq!(
            1 << size_log2,
            size,
            "GridKeeper2d size must be a power of two"
        );
        //Allocate space for meshes
        let total = (1 << (size_log2 * 2)) as usize;
        let mut slots = Vec::with_capacity(total);
        slots.resize_with(total, T::new);
        //Group em up
        Self {
            corner_pos: center - Int2::splat(1 << (size_log2 - 1)),
            size_log2,
            origin_idx: 0,
            slots,
        }
    }

    /// Will slide chunks and remove chunks that went over the border.
    pub fn set_center(&mut self, new_center: Int2) {
        let new_corner = new_center - Int2::splat(self.half_size());
        let adj = new_corner - self.corner_pos;
        let clear_range = |this: &mut Self, x: ops::Range<i32>, y: ops::Range<i32>| {
            for y in y.clone() {
                for x in x.clone() {
                    this.sub_get_mut([x, y].into()).reset();
                }
            }
        };
        if adj.x > 0 {
            clear_range(self, 0..adj.x, 0..self.size());
        } else if adj.x < 0 {
            clear_range(self, self.size() + adj.x..self.size(), 0..self.size());
        }
        if adj.y > 0 {
            clear_range(self, 0..self.size(), 0..adj.y);
        } else if adj.y < 0 {
            clear_range(self, 0..self.size(), self.size() + adj.y..self.size());
        }
        self.origin_idx =
            (self.origin_idx + adj.x + adj.y * self.size()).rem_euclid(self.total_len());
        self.corner_pos = new_corner;
    }
}
impl<T> GridKeeper2d<T> {
    #[inline]
    pub fn center(&self) -> Int2 {
        self.corner_pos + Int2::splat(self.half_size())
    }

    #[inline]
    pub fn size_log2(&self) -> u32 {
        self.size_log2
    }

    #[inline]
    pub fn size(&self) -> i32 {
        1 << self.size_log2
    }
    #[inline]

    pub fn half_size(&self) -> i32 {
        1 << (self.size_log2 - 1)
    }

    #[inline]
    pub fn total_len(&self) -> i32 {
        1 << (self.size_log2 * 2)
    }

    #[inline]
    pub fn get(&self, pos: Int2) -> Option<&T> {
        let pos = pos - self.corner_pos;
        if pos.is_within([self.size(); 2].into()) {
            Some(self.sub_get(pos))
        } else {
            None
        }
    }
    #[inline]
    pub fn get_mut(&mut self, pos: Int2) -> Option<&mut T> {
        let pos = pos - self.corner_pos;
        if pos.is_within([self.size(); 2].into()) {
            Some(self.sub_get_mut(pos))
        } else {
            None
        }
    }

    #[inline]
    pub fn get_by_idx(&self, idx: i32) -> &T {
        &self.slots[(self.origin_idx + idx).rem_euclid(self.total_len()) as usize]
    }
    #[inline]
    pub fn get_by_idx_mut(&mut self, idx: i32) -> &mut T {
        let idx = (self.origin_idx + idx).rem_euclid(self.total_len()) as usize;
        &mut self.slots[idx]
    }
    #[inline]
    pub fn sub_get(&self, pos: Int2) -> &T {
        &self.slots
            [(self.origin_idx + pos.x + pos.y * self.size()).rem_euclid(self.total_len()) as usize]
    }
    #[inline]
    pub fn sub_get_mut(&mut self, pos: Int2) -> &mut T {
        let size = self.size();
        let total_len = self.total_len();
        &mut self.slots[(self.origin_idx + pos.x + pos.y * size).rem_euclid(total_len) as usize]
    }

    #[inline]
    pub fn sub_idx_to_pos(&self, idx: i32) -> Int2 {
        self.corner_pos + Int2::from_index([self.size(); 2].into(), idx as usize)
    }
}

pub trait GridSlot {
    fn new() -> Self;
    fn reset(&mut self);
}
impl<T> GridSlot for T
where
    T: Default,
{
    #[inline]
    fn new() -> T {
        T::default()
    }

    #[inline]
    fn reset(&mut self) {
        *self = T::default();
    }
}

/// a dynamic growable buffer, representing an infinite space filled with "filler block".
/// any block within this space can be set or get, but setting blocks far away from the origin
/// will overallocate memory.
/// this buffer can be "transferred" over to a chunk, copying the corresponding blocks over from
/// the virtual buffer to the actual chunk.
pub struct BlockBuf {
    /// the actual buffer data.
    blocks: Vec<BlockData>,
    /// "filler block", the block that by default covers the entire space.
    fill: BlockData,
    /// the lowest coordinate represented by the physical block buffer, in buffer-local coordinates.
    /// usually negative.
    corner: Int3,
    /// log2 of the size.
    /// forces sizes to be a power of two.
    size: Int3,
}
impl BlockBuf {
    pub fn with_filler(fill: BlockData, corner: Int3, size: Int3) -> Self {
        let size_log2 = [
            (mem::size_of_val(&size.x) * 8) as i32 - (size.x - 1).leading_zeros() as i32,
            (mem::size_of_val(&size.y) * 8) as i32 - (size.y - 1).leading_zeros() as i32,
            (mem::size_of_val(&size.z) * 8) as i32 - (size.z - 1).leading_zeros() as i32,
        ]
        .into();
        Self {
            fill,
            corner,
            size: size_log2,
            blocks: vec![fill; 1 << (size_log2[0] + size_log2[1] + size_log2[2])],
        }
    }

    pub fn with_capacity(corner: Int3, size: Int3) -> Self {
        Self::with_filler(BlockData { data: 255 }, corner, size)
    }

    /// creates a new block buffer with its origin at [0, 0, 0].
    pub fn new() -> Self {
        Self::with_capacity([-2, -2, -2].into(), [4, 4, 4].into())
    }

    /// get the block data at the given position relative to the block buffer origin.
    /// if the position is out of bounds, return the filler block.
    pub fn get(&self, pos: BlockPos) -> BlockData {
        let pos = pos - self.corner;
        if pos.is_within([1 << self.size[0], 1 << self.size[1], 1 << self.size[2]].into()) {
            self.blocks[pos.to_index(self.size)]
        } else {
            self.fill
        }
    }

    /// extend the inner box buffer so that the given block is covered by it.
    pub fn reserve(&mut self, pos: BlockPos) {
        let mut pos = pos - self.corner;
        if !pos.is_within([1 << self.size[0], 1 << self.size[1], 1 << self.size[2]].into()) {
            // determine a new fitting bounding box
            let mut new_corner = self.corner;
            let mut new_size = self.size;
            let mut shift = Int3::zero();
            for i in 0..3 {
                while pos[i] < 0 {
                    new_corner[i] -= 1 << new_size[i];
                    pos[i] += 1 << new_size[i];
                    shift[i] += 1 << new_size[i];
                    new_size[i] += 1;
                }
                while pos[i] >= 1 << new_size[i] {
                    new_size[i] += 1;
                }
            }

            // Allocate space for new blocks and move block data around to fit new layout
            // OPTIMIZE: If only resizing on the Z axis, bulk move blocks around.
            self.blocks
                .resize(1 << (new_size.x + new_size.y + new_size.z), self.fill);
            let mut src = 1 << (self.size.x + self.size.y + self.size.z);
            let mut dst = 1 << (new_size.x + new_size.y + new_size.z);
            dst += shift.x + ((shift.y + (shift.z << new_size.y)) << new_size.x);
            dst -= ((1 << new_size.z) - (1 << self.size.z)) << (new_size.x + new_size.y);
            for _z in 0..1 << self.size.z {
                dst -= ((1 << new_size.y) - (1 << self.size.y)) << new_size.x;
                for _y in 0..1 << self.size.y {
                    src -= 1 << self.size.x;
                    dst -= 1 << new_size.x;
                    let (start, end) = (src, src + (1 << self.size.x));
                    self.blocks.copy_within(start..end, dst as usize);
                    self.blocks[start..end.min(dst as usize)].fill(self.fill);
                }
            }
            self.corner = new_corner;
            self.size = new_size;
        }
    }

    /// Fill a sphere with a certain block.
    pub fn fill_sphere(&mut self, center: Vec3, radius: f32, block: BlockData) {
        let mn = Int3::from_f32((center - Vec3::broadcast(radius)).map(f32::floor));
        let mx = Int3::from_f32((center + Vec3::broadcast(radius)).map(f32::ceil));
        self.reserve(mn);
        self.reserve(mx);
        let r2 = radius * radius;
        for z in mn.z..=mx.z {
            for y in mn.y..=mx.z {
                // OPTIMIZE: Use sqrt and math to figure out exactly where does this row start and
                // end, and write all the row blocks in one go.
                for x in mn.x..=mx.x {
                    let d = Vec3::new(x as f32, y as f32, z as f32) - center;
                    if d.mag_sq() <= r2 {
                        self.set([x, y, z].into(), block);
                    }
                }
            }
        }
    }

    /// Fill a cylinder with a certain block.
    /// The cylinder need not be axis-oriented.
    /// The cylinder endpoints are capped with a sphere.
    /// Receives the two endpoints of the cylinder skeleton segment, as well as two radii, one for
    /// each endpoint.
    pub fn fill_cylinder(&mut self, u0: Vec3, u1: Vec3, r0: f32, r1: f32, block: BlockData) {
        let mn = Int3::from_f32(
            (u0 - Vec3::broadcast(r0))
                .min_by_component(u1 - Vec3::broadcast(r1))
                .map(f32::floor),
        );
        let mx = Int3::from_f32(
            (u0 + Vec3::broadcast(r0))
                .max_by_component(u1 + Vec3::broadcast(r1))
                .map(f32::ceil),
        );
        self.reserve(mn);
        self.reserve(mx);
        let n = u1 - u0;
        let inv = n.mag_sq().recip();
        for z in mn.z..=mx.z {
            for y in mn.y..=mx.y {
                for x in mn.x..=mx.x {
                    let p = Vec3::new(x as f32, y as f32, z as f32) - u0;
                    let s = (n.dot(p) * inv).min(1.).max(0.);
                    let r = r0 + (r1 - r0) * s;
                    let d2 = (n * s - p).mag_sq();
                    if d2 <= r * r {
                        self.set([x, y, z].into(), block);
                    }
                }
            }
        }
    }

    /// set the block data at the given position relative to the block buffer origin.
    /// if the position is out of bounds, allocates new blocks filled with the filler block and
    /// only then sets the given position.
    pub fn set(&mut self, pos: BlockPos, block: BlockData) {
        self.reserve(pos);
        let pos = pos - self.corner;
        self.blocks[pos.to_index(self.size)] = block;
    }

    /// copy the contents of the block buffer over to the given chunk.
    /// locates the block buffer origin at the given origin position, and locates the chunk at the
    /// given chunk coordinates.
    pub fn transfer(&self, origin: BlockPos, chunkpos: ChunkPos, chunk: &mut ChunkBox) {
        let chunkpos = chunkpos << CHUNK_BITS;
        // position of the buffer relative to the destination chunk
        let pos = origin + self.corner - chunkpos;

        let skipchunk = pos.max(Int3::splat(0));
        let uptochunk = (pos
            + Int3::new([1 << self.size[0], 1 << self.size[1], 1 << self.size[2]]))
        .min(Int3::splat(CHUNK_SIZE));
        let skipbuf = -pos.min(Int3::splat(0));
        let uptobuf = uptochunk - pos;

        if uptobuf.x <= skipbuf.x
            || uptobuf.y <= skipbuf.y
            || uptobuf.z <= skipbuf.z
            || uptochunk.x <= skipchunk.x
            || uptochunk.y <= skipchunk.y
            || uptochunk.z <= skipchunk.z
        {
            // chunk and buffer do not overlap
            return;
        }

        // copy data from buffer to chunk
        let chunk = chunk.blocks_mut();
        let mut src = 0;
        let mut dst = 0;
        src += skipbuf.z << (self.size.x + self.size.y);
        dst += skipchunk.z * CHUNK_SIZE * CHUNK_SIZE;
        for _z in skipbuf.z..uptobuf.z {
            src += skipbuf.y << self.size.x;
            dst += skipchunk.y * CHUNK_SIZE;
            for _y in skipbuf.y..uptobuf.y {
                src += skipbuf.x;
                dst += skipchunk.x;
                for _x in skipbuf.x..uptobuf.x {
                    if self.blocks[src as usize] != self.fill {
                        chunk.set_idx(dst as usize, self.blocks[src as usize]);
                    }
                    src += 1;
                    dst += 1;
                }
                src += (1 << self.size.x) - uptobuf.x;
                dst += CHUNK_SIZE - uptochunk.x;
            }
            src += ((1 << self.size.y) - uptobuf.y) << self.size.x;
            dst += (CHUNK_SIZE - uptochunk.y) * CHUNK_SIZE;
        }
    }
}

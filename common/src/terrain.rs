use crate::{
    arena::{Box as ArenaBox, BoxUninit as ArenaBoxUninit},
    prelude::*,
};

/// Base-2 logarithm of the chunk size.
pub const CHUNK_BITS: u32 = 5;

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
    pub fn sub_get(&self, pos: [i32; 3]) -> BlockData {
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

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ChunkPos(pub [i32; 3]);
impl ops::Deref for ChunkPos {
    type Target = [i32; 3];
    fn deref(&self) -> &[i32; 3] {
        &self.0
    }
}
impl ops::DerefMut for ChunkPos {
    fn deref_mut(&mut self) -> &mut [i32; 3] {
        &mut self.0
    }
}
impl ChunkPos {
    pub fn offset(&self, x: i32, y: i32, z: i32) -> ChunkPos {
        ChunkPos([self[0] + x, self[1] + y, self[2] + z])
    }

    pub fn to_block_floor(&self) -> BlockPos {
        BlockPos([
            self[0] * CHUNK_SIZE,
            self[1] * CHUNK_SIZE,
            self[2] * CHUNK_SIZE,
        ])
    }
    pub fn to_block_center(&self) -> BlockPos {
        BlockPos([
            self[0] * CHUNK_SIZE + CHUNK_SIZE / 2,
            self[1] * CHUNK_SIZE + CHUNK_SIZE / 2,
            self[2] * CHUNK_SIZE + CHUNK_SIZE / 2,
        ])
    }

    pub fn xy(&self) -> [i32; 2] {
        [self[0], self[1]]
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockPos(pub [i32; 3]);
impl ops::Deref for BlockPos {
    type Target = [i32; 3];
    fn deref(&self) -> &[i32; 3] {
        &self.0
    }
}
impl ops::DerefMut for BlockPos {
    fn deref_mut(&mut self) -> &mut [i32; 3] {
        &mut self.0
    }
}
impl BlockPos {
    pub fn offset(&self, x: i32, y: i32, z: i32) -> BlockPos {
        BlockPos([self[0] + x, self[1] + y, self[2] + z])
    }
    pub fn from_float(pos: [f64; 3]) -> BlockPos {
        BlockPos([
            pos[0].floor() as i32,
            pos[1].floor() as i32,
            pos[2].floor() as i32,
        ])
    }
    pub fn to_float_floor(&self) -> [f64; 3] {
        [self[0] as f64, self[1] as f64, self[2] as f64]
    }
    pub fn to_float_center(&self) -> [f64; 3] {
        [
            self[0] as f64 + 0.5,
            self[1] as f64 + 0.5,
            self[2] as f64 + 0.5,
        ]
    }
    pub fn to_chunk(&self) -> ChunkPos {
        ChunkPos([
            self[0].div_euclid(CHUNK_SIZE),
            self[1].div_euclid(CHUNK_SIZE),
            self[2].div_euclid(CHUNK_SIZE),
        ])
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BlockData {
    pub data: u8,
}
impl BlockData {
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
    pub fn sub_get(&self, sub_pos: [i32; 3]) -> BlockData {
        match self.blocks.get(
            (sub_pos[0] | sub_pos[1] * CHUNK_SIZE | sub_pos[2] * (CHUNK_SIZE * CHUNK_SIZE))
                as usize,
        ) {
            Some(&b) => b,
            None => panic!(
                "block index [{}, {}, {}] outside chunk boundaries",
                sub_pos[0], sub_pos[1], sub_pos[2]
            ),
        }
    }

    #[track_caller]
    #[inline]
    pub fn sub_get_mut(&mut self, sub_pos: [i32; 3]) -> &mut BlockData {
        match self.blocks.get_mut(
            (sub_pos[0] | sub_pos[1] * CHUNK_SIZE | sub_pos[2] * (CHUNK_SIZE * CHUNK_SIZE))
                as usize,
        ) {
            Some(b) => b,
            None => panic!(
                "block index [{}, {}, {}] outside chunk boundaries",
                sub_pos[0], sub_pos[1], sub_pos[2]
            ),
        }
    }

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
            corner_pos: center.offset(
                -(1 << (size_log2 - 1)),
                -(1 << (size_log2 - 1)),
                -(1 << (size_log2 - 1)),
            ),
            size_log2,
            origin_idx: 0,
            slots,
        }
    }

    /// Will slide chunks and remove chunks that went over the border.
    pub fn set_center(&mut self, new_center: ChunkPos) {
        let new_corner = new_center.offset(-self.half_size(), -self.half_size(), -self.half_size());
        let adj_x = new_corner[0] - self.corner_pos[0];
        let adj_y = new_corner[1] - self.corner_pos[1];
        let adj_z = new_corner[2] - self.corner_pos[2];
        let clear_range =
            |this: &mut Self, x: ops::Range<i32>, y: ops::Range<i32>, z: ops::Range<i32>| {
                // OPTIMIZE: Store lists of active slots in each plane slice, so as to only visit
                // actual active slots.
                for z in z.clone() {
                    for y in y.clone() {
                        for x in x.clone() {
                            this.sub_get_mut([x, y, z]).reset();
                        }
                    }
                }
            };
        if adj_x > 0 {
            clear_range(self, 0..adj_x, 0..self.size(), 0..self.size());
        } else if adj_x < 0 {
            clear_range(
                self,
                self.size() + adj_x..self.size(),
                0..self.size(),
                0..self.size(),
            );
        }
        if adj_y > 0 {
            clear_range(self, 0..self.size(), 0..adj_y, 0..self.size());
        } else if adj_y < 0 {
            clear_range(
                self,
                0..self.size(),
                self.size() + adj_y..self.size(),
                0..self.size(),
            );
        }
        if adj_z > 0 {
            clear_range(self, 0..self.size(), 0..self.size(), 0..adj_z);
        } else if adj_z < 0 {
            clear_range(
                self,
                0..self.size(),
                0..self.size(),
                self.size() + adj_z..self.size(),
            );
        }
        self.origin_idx =
            (self.origin_idx + adj_x + adj_y * self.size() + adj_z * (self.size() * self.size()))
                .rem_euclid(self.total_len());
        self.corner_pos = new_corner;
    }
}
impl<T> GridKeeper<T> {
    #[inline]
    pub fn center(&self) -> ChunkPos {
        self.corner_pos
            .offset(self.half_size(), self.half_size(), self.half_size())
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
        if pos[0] >= self.corner_pos[0]
            && pos[0] < self.corner_pos[0] + self.size()
            && pos[1] >= self.corner_pos[1]
            && pos[1] < self.corner_pos[1] + self.size()
            && pos[2] >= self.corner_pos[2]
            && pos[2] < self.corner_pos[2] + self.size()
        {
            Some(self.sub_get([
                pos[0] - self.corner_pos[0],
                pos[1] - self.corner_pos[1],
                pos[2] - self.corner_pos[2],
            ]))
        } else {
            None
        }
    }
    #[inline]
    pub fn get_mut(&mut self, pos: ChunkPos) -> Option<&mut T> {
        if pos[0] >= self.corner_pos[0]
            && pos[0] < self.corner_pos[0] + self.size()
            && pos[1] >= self.corner_pos[1]
            && pos[1] < self.corner_pos[1] + self.size()
            && pos[2] >= self.corner_pos[2]
            && pos[2] < self.corner_pos[2] + self.size()
        {
            Some(self.sub_get_mut([
                pos[0] - self.corner_pos[0],
                pos[1] - self.corner_pos[1],
                pos[2] - self.corner_pos[2],
            ]))
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
    pub fn sub_get(&self, pos: [i32; 3]) -> &T {
        &self.slots[(self.origin_idx
            + pos[0]
            + pos[1] * self.size()
            + pos[2] * (self.size() * self.size()))
        .rem_euclid(self.total_len()) as usize]
    }
    #[inline]
    pub fn sub_get_mut(&mut self, pos: [i32; 3]) -> &mut T {
        let size = self.size();
        let total_len = self.total_len();
        &mut self.slots[(self.origin_idx + pos[0] + pos[1] * size + pos[2] * (size * size))
            .rem_euclid(total_len) as usize]
    }

    #[inline]
    pub fn sub_idx_to_pos(&self, idx: i32) -> ChunkPos {
        let x = idx % self.size();
        let y = idx / self.size() % self.size();
        let z = idx / self.size() / self.size();
        self.corner_pos.offset(x, y, z)
    }
}

pub struct GridKeeper2d<T> {
    corner_pos: [i32; 2],
    size_log2: u32,
    origin_idx: i32,
    slots: Vec<T>,
}
impl<T: GridSlot> GridKeeper2d<T> {
    pub fn with_radius(radius: f32, center: [i32; 2]) -> Self {
        let size = ((radius * 2.).max(2.) as u32).next_power_of_two().max(1) as i32;
        Self::new(size, center)
    }

    pub fn new(size: i32, center: [i32; 2]) -> Self {
        //Make sure length is a power of two
        let size_log2 = (mem::size_of_val(&size) * 8) as u32 - (size - 1).leading_zeros();
        assert_eq!(
            1 << size_log2,
            size,
            "GridKeeper size must be a power of two"
        );
        //Allocate space for meshes
        let total = (1 << (size_log2 * 2)) as usize;
        let mut slots = Vec::with_capacity(total);
        slots.resize_with(total, T::new);
        //Group em up
        Self {
            corner_pos: [
                center[0] - (1 << (size_log2 - 1)),
                center[1] - (1 << (size_log2 - 1)),
            ],
            size_log2,
            origin_idx: 0,
            slots,
        }
    }

    /// Will slide chunks and remove chunks that went over the border.
    pub fn set_center(&mut self, new_center: [i32; 2]) {
        let new_corner = [
            new_center[0] - self.half_size(),
            new_center[1] - self.half_size(),
        ];
        let adj_x = new_corner[0] - self.corner_pos[0];
        let adj_y = new_corner[1] - self.corner_pos[1];
        let clear_range = |this: &mut Self, x: ops::Range<i32>, y: ops::Range<i32>| {
            for y in y.clone() {
                for x in x.clone() {
                    this.sub_get_mut([x, y]).reset();
                }
            }
        };
        if adj_x > 0 {
            clear_range(self, 0..adj_x, 0..self.size());
        } else if adj_x < 0 {
            clear_range(self, self.size() + adj_x..self.size(), 0..self.size());
        }
        if adj_y > 0 {
            clear_range(self, 0..self.size(), 0..adj_y);
        } else if adj_y < 0 {
            clear_range(self, 0..self.size(), self.size() + adj_y..self.size());
        }
        self.origin_idx =
            (self.origin_idx + adj_x + adj_y * self.size()).rem_euclid(self.total_len());
        self.corner_pos = new_corner;
    }
}
impl<T> GridKeeper2d<T> {
    #[inline]
    pub fn center(&self) -> [i32; 2] {
        [
            self.corner_pos[0] + self.half_size(),
            self.corner_pos[1] + self.half_size(),
        ]
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
    pub fn get(&self, pos: [i32; 2]) -> Option<&T> {
        if pos[0] >= self.corner_pos[0]
            && pos[0] < self.corner_pos[0] + self.size()
            && pos[1] >= self.corner_pos[1]
            && pos[1] < self.corner_pos[1] + self.size()
        {
            Some(self.sub_get([pos[0] - self.corner_pos[0], pos[1] - self.corner_pos[1]]))
        } else {
            None
        }
    }
    #[inline]
    pub fn get_mut(&mut self, pos: [i32; 2]) -> Option<&mut T> {
        if pos[0] >= self.corner_pos[0]
            && pos[0] < self.corner_pos[0] + self.size()
            && pos[1] >= self.corner_pos[1]
            && pos[1] < self.corner_pos[1] + self.size()
        {
            Some(self.sub_get_mut([pos[0] - self.corner_pos[0], pos[1] - self.corner_pos[1]]))
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
    pub fn sub_get(&self, pos: [i32; 2]) -> &T {
        &self.slots[(self.origin_idx + pos[0] + pos[1] * self.size()).rem_euclid(self.total_len())
            as usize]
    }
    #[inline]
    pub fn sub_get_mut(&mut self, pos: [i32; 2]) -> &mut T {
        let size = self.size();
        let total_len = self.total_len();
        &mut self.slots[(self.origin_idx + pos[0] + pos[1] * size).rem_euclid(total_len) as usize]
    }

    #[inline]
    pub fn sub_idx_to_pos(&self, idx: i32) -> [i32; 2] {
        let x = idx % self.size();
        let y = idx / self.size();
        [self.corner_pos[0] + x, self.corner_pos[1] + y]
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

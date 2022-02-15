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

/// Maximum amount of portals that can touch a chunk.
pub const MAX_CHUNK_PORTALS: usize = 256;

/// A 2D slice (or loaf) of an arbitrary type.
pub type LoafBox<T> = crate::arena::Box<[T; (CHUNK_SIZE * CHUNK_SIZE) as usize]>;

/// Saves chunk tags on the last 2 bits of the chunk pointer.
/// This is why `ChunkData` has an alignment of 4.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct ChunkRef<'a> {
    ptr: NonNull<ChunkData>,
    marker: PhantomData<&'a ChunkData>,
}
unsafe impl Send for ChunkRef<'_> {}
unsafe impl Sync for ChunkRef<'_> {}
impl fmt::Debug for ChunkRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ChunkRef<")?;
        if self.is_solid() {
            write!(f, "solid")?;
        }
        if self.is_clear() {
            write!(f, "clear")?;
        }
        if let Some(block) = self.homogeneous_block() {
            write!(f, ">::Homogeneous({:?})", block)
        } else {
            write!(f, ">({}KB)", mem::size_of::<ChunkData>() / 1024)
        }
    }
}
impl<'a> ChunkRef<'a> {
    const MASK: usize = 0b111;
    const FLAG_HOMOGENEOUS: usize = 0b001;
    const FLAG_CLEAR: usize = 0b010;
    const FLAG_SOLID: usize = 0b100;

    #[inline]
    fn raw(self) -> usize {
        self.ptr.as_ptr() as usize
    }

    #[inline]
    pub fn is_homogeneous(self) -> bool {
        self.raw() & Self::FLAG_HOMOGENEOUS != 0
    }

    #[inline]
    pub fn is_clear(self) -> bool {
        self.raw() & Self::FLAG_CLEAR != 0
    }

    #[inline]
    pub fn is_solid(self) -> bool {
        self.raw() & Self::FLAG_SOLID != 0
    }

    /// Will return garbage if the chunk is not homogeneous, but it will be defined behaviour
    /// anyway.
    #[inline]
    pub fn homogeneous_block_unchecked(self) -> BlockData {
        BlockData {
            data: (self.raw() >> 8) as u8,
        }
    }

    #[inline]
    pub fn homogeneous_block(self) -> Option<BlockData> {
        if self.is_homogeneous() {
            Some(self.homogeneous_block_unchecked())
        } else {
            None
        }
    }

    #[inline]
    pub unsafe fn blocks_unchecked(self) -> &'a ChunkData {
        &*((self.raw() & !Self::MASK) as *const ChunkData)
    }

    #[inline]
    pub fn blocks(self) -> Option<&'a ChunkData> {
        if self.is_homogeneous() {
            None
        } else {
            unsafe { Some(self.blocks_unchecked()) }
        }
    }

    #[inline]
    pub fn sub_get(self, pos: Int3) -> BlockData {
        if let Some(blocks) = self.blocks() {
            blocks.sub_get(pos)
        } else {
            self.homogeneous_block_unchecked()
        }
    }

    #[inline]
    pub fn sub_portal_at(self, pos: Int3, axis: usize) -> Option<&'a PortalData> {
        if let Some(blocks) = self.blocks() {
            blocks.sub_portal_at(pos, axis)
        } else {
            None
        }
    }

    pub fn into_raw(self) -> *const ChunkData {
        self.ptr.as_ptr()
    }

    pub unsafe fn from_raw(raw: *const ChunkData) -> Self {
        Self {
            ptr: NonNull::new_unchecked(raw as *mut ChunkData),
            marker: PhantomData,
        }
    }

    pub fn clone_chunk(self) -> ChunkBox {
        if self.is_homogeneous() {
            ChunkBox { ptr: self.ptr }
        } else {
            let mut uninit = ArenaBoxUninit::<ChunkData>::new();
            let blockmem = unsafe {
                ptr::copy_nonoverlapping(self.blocks_unchecked(), uninit.as_mut().as_mut_ptr(), 1);
                ArenaBox::into_raw(uninit.assume_init())
            };
            unsafe {
                ChunkBox {
                    ptr: NonNull::new_unchecked(
                        (blockmem.as_ptr() as usize | (self.raw() & Self::MASK)) as *mut ChunkData,
                    ),
                }
            }
        }
    }
}

/// # `ChunkBox` pointer format
///
/// Since `ChunkData` has an alignment of 8, the `blocks` pointer has 3 bits to store tags.
/// If bit 0 is set, the chunk has no associated memory, and the entire chunk is made up of a
/// single block type, which is stored in bits 8..16.
/// If bit 1 is set, the chunk is made entirely out of clear blocks (although they
/// might be different). This flag is completely independent from bit 0.
/// If bit 2 is set, the chunk is made entirely out of solid blocks (although they might
/// be different types of solid blocks). This flag is completely independent from bit 0, although it
/// is mutually exclusive with bit 1.
///
/// Note that flags 1 and 2 are not hints. If they are set, the chunk **must** be made out of
/// entirely solid or clear blocks, otherwise it is a logic error.
#[repr(transparent)]
pub struct ChunkBox {
    ptr: NonNull<ChunkData>,
}
unsafe impl Send for ChunkBox {}
unsafe impl Sync for ChunkBox {}
impl ops::Deref for ChunkBox {
    type Target = ChunkRef<'static>;
    fn deref(&self) -> &ChunkRef<'static> {
        // SUPER-UNSAFE!
        unsafe { mem::transmute(self) }
    }
}
impl fmt::Debug for ChunkBox {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ChunkBox<")?;
        if self.is_solid() {
            write!(f, "solid")?;
        }
        if self.is_clear() {
            write!(f, "clear")?;
        }
        if let Some(block) = self.homogeneous_block() {
            write!(f, ">::Homogeneous({:?})", block)
        } else {
            write!(f, ">({}KB)", mem::size_of::<ChunkData>() / 1024)
        }
    }
}
impl ChunkBox {
    unsafe fn new_raw_nonhomogeneous(blockmem: NonNull<ChunkData>, tags: usize) -> Self {
        Self {
            ptr: NonNull::new_unchecked((blockmem.as_ptr() as usize | tags) as *mut ChunkData),
        }
    }

    /// `raw` must be nonzero! (and valid, of course)
    unsafe fn new_raw(raw: usize) -> Self {
        Self {
            ptr: NonNull::new_unchecked(raw as *mut ChunkData),
        }
    }

    /// Create a new chunk box with memory allocated, but uninitialized.
    /// Inherently unsafe, but may work in most cases and speeds things up.
    #[inline]
    pub unsafe fn new_uninit() -> Self {
        let mut blockmem = ArenaBox::into_raw(ArenaBoxUninit::<ChunkData>::new().assume_init());
        blockmem.as_mut().portal_count = 0;
        ChunkBox::new_raw_nonhomogeneous(blockmem, 0)
    }

    /// Create a new chunk filled with the given block, without allocating any memory.
    #[inline]
    pub fn new_homogeneous(block: BlockData) -> Self {
        unsafe { Self::new_raw(((block.data as usize) << 8) | ChunkRef::FLAG_HOMOGENEOUS) }
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
            ptr: self.ptr,
            marker: PhantomData,
        }
    }

    pub unsafe fn blocks_mut_unchecked(&mut self) -> &mut ChunkData {
        &mut *((self.ptr.as_ptr() as usize & !ChunkRef::MASK) as *mut ChunkData)
    }

    /// Get a mutable reference to the inner blocks.
    /// If there is no associated memory, this method will allocate and fill with the homogeneous
    /// block, according to the current state.
    #[inline]
    pub fn blocks_mut(&mut self) -> &mut ChunkData {
        if let Some(block) = self.homogeneous_block() {
            unsafe {
                *self = ChunkBox::new_uninit();
                ptr::write_bytes(&mut self.blocks_mut_unchecked().blocks, block.data, 1);
            }
        }
        unsafe { self.blocks_mut_unchecked() }
    }

    /// Gets the inner blocks if there is memory associated to this chunk box.
    #[inline]
    pub fn try_blocks_mut(&mut self) -> Option<&mut ChunkData> {
        if self.is_homogeneous() {
            None
        } else {
            unsafe { Some(self.blocks_mut_unchecked()) }
        }
    }

    /// Take the current blocks, if available, and replace by a given chunkbox.
    #[inline]
    fn take_blocks(&mut self, new_box: ChunkBox) -> Option<ArenaBox<ChunkData>> {
        unsafe {
            let out = if let Some(blocks) = self.try_blocks_mut() {
                Some(ArenaBox::from_raw(NonNull::new_unchecked(
                    blocks as *mut ChunkData,
                )))
            } else {
                None
            };
            ptr::write(self, new_box);
            out
        }
    }

    /// Remove any memory associated with this box, and make all blocks homogeneous.
    pub fn make_homogeneous(&mut self, block: BlockData) {
        self.take_blocks(Self::new_homogeneous(block));
    }

    /// Mark the chunk as solid.
    /// It is a logic error if the chunk is not actually solid!
    #[inline]
    pub fn mark_solid(&mut self) {
        unsafe {
            ptr::write(self, Self::new_raw(self.raw() | ChunkRef::FLAG_SOLID));
        }
    }

    /// Mark the chunk as clear.
    /// It is a logic error if the chunk is not actually clear!
    #[inline]
    pub fn mark_clear(&mut self) {
        unsafe {
            ptr::write(self, Self::new_raw(self.raw() | ChunkRef::FLAG_CLEAR));
        }
    }

    /// If the chunk is homogeneous, mark the appropiate solidity (solid/nonsolid) tags.
    #[inline]
    pub fn mark_solidity(&mut self, style: &StyleTable) {
        if let Some(block) = self.homogeneous_block() {
            if block.is_solid(style) {
                self.mark_solid();
            } else if block.is_clear(style) {
                self.mark_clear();
            }
        }
    }

    /// After generating a chunk, apply final checkups to optimize everything.
    pub fn consolidate(&mut self, style: &StyleTable) {
        // Mark the solidity of the chunk
        self.mark_solidity(style);
        // TODO: Sort portals to optimize lookups
    }

    /// Check if all of the blocks are of the same type, and drop the data altogether and mark with
    /// a tag in that case.
    #[inline]
    pub fn try_drop_blocks(&mut self) {
        if let Some(chunk_data) = self.blocks() {
            let blocks = &chunk_data.blocks;
            let block0 = blocks[0];
            if blocks.iter().skip(1).all(|&b| b == block0) {
                //All the same
                self.make_homogeneous(block0);
            }
        }
    }
}
impl Drop for ChunkBox {
    fn drop(&mut self) {
        unsafe {
            if let Some(blocks) = self.try_blocks_mut() {
                drop(ArenaBox::from_raw(NonNull::from(blocks)));
            }
        }
    }
}

/// An atomically reference-counted `ChunkBox`, with less indirections than `Arc<ChunkBox>`.
/// Also saves the atomic operations if the `ChunkBox` is homogeneous.
pub struct ChunkArc {
    chunk: mem::ManuallyDrop<ChunkBox>,
}
impl ChunkArc {
    pub fn new(chunk: ChunkBox) -> Self {
        if let Some(data) = chunk.blocks() {
            data.ref_count.store(1);
        }
        Self {
            chunk: mem::ManuallyDrop::new(chunk),
        }
    }

    /// Get an immutable reference to the chunk arc.
    #[inline]
    pub fn as_ref(&self) -> ChunkRef {
        self.chunk.as_ref()
    }
}
impl ops::Deref for ChunkArc {
    type Target = ChunkRef<'static>;
    fn deref(&self) -> &ChunkRef<'static> {
        // SUPER-UNSAFE! (maybe? not sure)
        &*self.chunk
    }
}
impl ops::Drop for ChunkArc {
    fn drop(&mut self) {
        if let Some(data) = self.chunk.blocks() {
            if data.ref_count.fetch_sub(1) == 1 {
                // Last reference to the `ChunkBox`
                unsafe {
                    mem::ManuallyDrop::drop(&mut self.chunk);
                }
            }
        }
    }
}
impl Clone for ChunkArc {
    fn clone(&self) -> Self {
        if let Some(data) = self.chunk.blocks() {
            data.ref_count.fetch_add(1);
        }
        unsafe {
            Self {
                chunk: ptr::read(&self.chunk),
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

/// Represents a single block id.
///
/// Currently a single byte, and the same byte always represents the same block anywhere in the
/// world.
/// In the future, however, the byte might represent a different block depending on context (ie.
/// position or dimension).
/// Additionally, at some point block ids might become 2 bytes.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BlockData {
    pub data: u8,
}
impl BlockData {
    #[inline]
    pub fn is_clear(self, table: &StyleTable) -> bool {
        table.is_clear(self)
    }
    #[inline]
    pub fn is_solid(self, table: &StyleTable) -> bool {
        table.is_solid(self)
    }
    #[inline]
    pub fn is_portal(self, table: &StyleTable) -> bool {
        table.is_portal(self)
    }
}

/// Represents a single portal within a chunk.
///
/// If a portal spans multiple chunks, it must be present in every chunk it touches.
#[derive(Copy, Clone, Debug)]
pub struct PortalData {
    /// The position of the portal min-corner relative to the chunk.
    pub pos: [i16; 3],
    /// The dimensions of the portal relative to the chunk.
    /// One of these dimensions must be zero.
    pub size: [i16; 3],
    /// Where does the portal teleport to, in coordinates relative to the portal.
    /// There should be an equal-size and reverse-jump portal at the destination.
    pub jump: [i32; 3],
}
impl PortalData {
    pub fn get_axis(&self) -> usize {
        if self.size[0] == 0 {
            0
        } else if self.size[1] == 0 {
            1
        } else {
            2
        }
    }
}

/// Holds the block and entity data for a chunk.
/// Has an alignment of 8 in order for `ChunkBox` to store tags in the last 3 bits of the pointer.
#[repr(align(8))]
pub struct ChunkData {
    /// All of the blocks in this chunk.
    pub blocks: [BlockData; (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize],
    /// All of the portals in this chunk.
    pub portals: [PortalData; MAX_CHUNK_PORTALS],
    /// The amount of portals in the portal buffer.
    pub portal_count: u32,
    /// A hidden reference count field for use in chunk reference counting.
    ref_count: AtomicCell<u32>,
}
impl Clone for ChunkData {
    fn clone(&self) -> Self {
        Self {
            blocks: self.blocks,
            portals: self.portals,
            portal_count: self.portal_count,
            ref_count: AtomicCell::new(self.ref_count.load()),
        }
    }
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
    pub fn sub_set(&mut self, sub_pos: Int3, block: BlockData) {
        match self
            .blocks
            .get_mut(sub_pos.to_index([CHUNK_BITS; 3].into()))
        {
            Some(b) => *b = block,
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

    #[track_caller]
    #[inline]
    pub fn portals(&self) -> &[PortalData] {
        &self.portals[..self.portal_count as usize]
    }

    #[inline]
    pub fn portals_mut(&mut self) -> &mut [PortalData] {
        &mut self.portals[..self.portal_count as usize]
    }

    #[track_caller]
    #[inline]
    pub fn push_portal(&mut self, portal: PortalData) {
        assert!(
            (self.portal_count as usize) < self.portals.len(),
            "portal buffer overflow (attempt to add more than {} portals to a chunk)",
            self.portals.len(),
        );
        self.portals[self.portal_count as usize] = portal;
        self.portal_count += 1;
    }

    /// Look through the portals in this chunk and find one at the given coordinates and
    /// orientation.
    #[inline]
    pub fn sub_portal_at(&self, at: Int3, axis: usize) -> Option<&PortalData> {
        for portal in self.portals() {
            let mn = Int3::new([
                portal.pos[0] as i32,
                portal.pos[1] as i32,
                portal.pos[2] as i32,
            ]);
            let mut mx = mn
                + Int3::new([
                    portal.size[0] as i32,
                    portal.size[1] as i32,
                    portal.size[2] as i32,
                ]);
            mx[axis] += 1;
            if mn.x <= at.x
                && at.x < mx.x
                && mn.y <= at.y
                && at.y < mx.y
                && mn.z <= at.z
                && at.z < mx.z
            {
                return Some(portal);
            }
        }
        None
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

#[derive(Default, Clone, Deserialize)]
pub struct BlockTexture {
    /// What are the physical and visual properties of the block.
    #[serde(default)]
    pub style: BlockStyle,
    /// Whether to set color at the vertices or per-block.
    /// If smooth, color is per-vertex, and therefore blocks represent a color gradient.
    #[serde(default = "default_true")]
    pub smooth: bool,
    /// Constant base block color.
    /// Red, Green, Blue, Specularity.
    /// Other texture effects are added onto this base color.
    #[serde(default)]
    pub base: [f32; 4],
    /// How much value noise to add of each scale.
    /// 0 -> per-block value noise
    /// 1 -> 2-block interpolated value noise
    /// K -> 2^K-block interpolated value noise
    #[serde(default)]
    pub noise: [[f32; 4]; Self::NOISE_LEVELS],
}
impl BlockTexture {
    pub const NOISE_LEVELS: usize = 6;
}

#[derive(Copy, Clone, Deserialize, PartialEq, Eq)]
#[repr(u8)]
pub enum BlockStyle {
    Solid,
    Clear,
    Portal,
    Custom,
}
impl Default for BlockStyle {
    fn default() -> Self {
        Self::Solid
    }
}
impl BlockStyle {
    #[inline]
    pub fn from_raw(raw: u8) -> Self {
        unsafe { mem::transmute::<u8, Self>(raw & 0b11) }
    }

    #[inline]
    pub fn as_raw(self) -> u8 {
        self as u8
    }

    #[inline]
    pub fn is_clear(self) -> bool {
        self == Self::Clear
    }
    #[inline]
    pub fn is_solid(self) -> bool {
        self == Self::Solid
    }
    #[inline]
    pub fn is_portal(self) -> bool {
        self == Self::Portal
    }
}

pub struct BlockTextures {
    pub blocks: Box<[Cell<BlockTexture>; 256]>,
}
impl BlockTextures {
    pub fn set(&self, id: BlockData, tex: BlockTexture) {
        self.blocks[id.data as usize].set(tex);
    }
}
impl Default for BlockTextures {
    fn default() -> Self {
        let mut arr: Uninit<[Cell<BlockTexture>; 256]> = Uninit::uninit();
        for i in 0..256 {
            unsafe {
                (arr.as_mut_ptr() as *mut Cell<BlockTexture>)
                    .offset(i)
                    .write(default());
            }
        }
        let arr = unsafe { arr.assume_init() };
        Self {
            blocks: Box::new(arr),
        }
    }
}
impl Clone for BlockTextures {
    fn clone(&self) -> Self {
        let new = Self::default();
        let mut tmp: BlockTexture = default();
        for (old, new) in self.blocks.iter().zip(new.blocks.iter()) {
            tmp = old.replace(tmp);
            new.set(tmp.clone());
            tmp = old.replace(tmp);
        }
        new
    }
}

pub struct StyleTable {
    words: [usize; Self::TOTAL_WORDS],
}
impl StyleTable {
    const BITS: usize = mem::size_of::<usize>() * 8;
    const BITS_PER_BLOCK: usize = 2;
    const BLOCKS_PER_WORD: usize = Self::BITS / Self::BITS_PER_BLOCK;
    const TOTAL_WORDS: usize = 256 / Self::BLOCKS_PER_WORD;

    pub fn new(tex: &BlockTextures) -> Self {
        let mut words = [0; Self::TOTAL_WORDS];
        for i in 0..256 {
            let texcell = &tex.blocks[i as usize];
            let tx = texcell.take();
            words[i / Self::BLOCKS_PER_WORD] |=
                (tx.style as u8 as usize) << (i % Self::BLOCKS_PER_WORD * Self::BITS_PER_BLOCK);
            texcell.set(tx);
        }
        Self { words }
    }

    #[inline]
    pub fn lookup(&self, id: BlockData) -> BlockStyle {
        let raw = self.words[id.data as usize / Self::BLOCKS_PER_WORD]
            >> (id.data as usize % Self::BLOCKS_PER_WORD * Self::BITS_PER_BLOCK);
        BlockStyle::from_raw(raw as u8)
    }

    #[inline]
    pub fn is_clear(&self, id: BlockData) -> bool {
        self.lookup(id).is_clear()
    }
    #[inline]
    pub fn is_solid(&self, id: BlockData) -> bool {
        self.lookup(id).is_solid()
    }
    #[inline]
    pub fn is_portal(&self, id: BlockData) -> bool {
        self.lookup(id).is_portal()
    }
}

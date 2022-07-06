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

/// A primitive unsigned integer with the bit width of the chunk size.
pub type ChunkSizedInt = u32;

/// Maximum amount of portals that can touch a chunk.
pub const MAX_CHUNK_PORTALS: usize = 256;

/// Maximum bytes of entity data per chunk.
pub const MAX_ENTITY_BYTES: usize = 2048;

/// A 2D slice (or loaf) of an arbitrary type.
pub type LoafBox<T> = crate::arena::Box<[T; (CHUNK_SIZE * CHUNK_SIZE) as usize]>;

/// Saves chunk tags on the last 1 bits of the chunk pointer.
/// This is why `ChunkData` has an alignment of 16.
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
        if self.is_homogeneous() {
            write!(f, "ChunkRef::Homogeneous({:?})", self.homogeneous_block())
        } else {
            write!(f, "ChunkRef({}KB)", mem::size_of::<ChunkData>() / 1024)
        }
    }
}
impl<'a> ChunkRef<'a> {
    const MASK: usize = 0b1;
    const FLAG_HOMOGENEOUS: usize = 0b1;

    const PACKED_SHINETHROUGH: usize = 1;
    const PACKED_BLOCK: usize = 8;
    const PACKED_SKYLIGHT: usize = 16;
    const PACKED_LIGHTMODE: usize = 24;

    #[inline]
    pub fn new_homogeneous(
        shinethrough: bool,
        block: BlockData,
        skylight: u8,
        lightmode: u8,
    ) -> ChunkRef<'static> {
        ChunkRef {
            ptr: unsafe {
                NonNull::new_unchecked(
                    (((lightmode as usize) << Self::PACKED_LIGHTMODE)
                        | ((skylight as usize) << Self::PACKED_SKYLIGHT)
                        | ((block.data as usize) << Self::PACKED_BLOCK)
                        | ((shinethrough as usize) << Self::PACKED_SHINETHROUGH)
                        | ChunkRef::FLAG_HOMOGENEOUS) as *mut ChunkData,
                )
            },
            marker: PhantomData,
        }
    }

    #[inline]
    pub fn placeholder() -> ChunkRef<'static> {
        ChunkRef::new_homogeneous(false, BlockData { data: 0 }, 0, 0)
    }

    #[inline]
    fn raw(self) -> usize {
        self.ptr.as_ptr() as usize
    }

    #[inline]
    pub fn is_homogeneous(self) -> bool {
        self.raw() & Self::FLAG_HOMOGENEOUS != 0
    }

    /// Will return garbage if the chunk is not homogeneous, but it will be defined behaviour
    /// anyway.
    #[inline]
    pub fn homogeneous_shinethrough(self) -> bool {
        (self.raw() >> Self::PACKED_SHINETHROUGH) & 1 != 0
    }

    /// Will return garbage if the chunk is not homogeneous, but it will be defined behaviour
    /// anyway.
    #[inline]
    pub fn homogeneous_block(self) -> BlockData {
        BlockData {
            data: (self.raw() >> Self::PACKED_BLOCK) as u8,
        }
    }

    /// Will return garbage if the chunk is not homogeneous, but it will be defined behaviour
    /// anyway.
    #[inline]
    pub fn homogeneous_skylight(self) -> u8 {
        (self.raw() >> Self::PACKED_SKYLIGHT) as u8
    }

    /// Will return garbage if the chunk is not homogeneous, but it will be defined behaviour
    /// anyway.
    #[inline]
    pub fn homogeneous_lightmode(self) -> u8 {
        (self.raw() >> Self::PACKED_LIGHTMODE) as u8
    }

    #[inline]
    pub unsafe fn data_uncheckeed(self) -> &'a ChunkData {
        &*((self.raw() & !Self::MASK) as *const ChunkData)
    }

    #[inline]
    pub fn data(self) -> Option<&'a ChunkData> {
        if self.is_homogeneous() {
            None
        } else {
            unsafe { Some(self.data_uncheckeed()) }
        }
    }

    #[inline]
    pub fn sub_get(self, pos: Int3) -> BlockData {
        if let Some(blocks) = self.data() {
            blocks.sub_get(pos)
        } else {
            self.homogeneous_block()
        }
    }

    #[inline]
    pub fn sub_portal_at(self, pos: Int3, axis: usize) -> Option<&'a PortalData> {
        if let Some(blocks) = self.data() {
            blocks.sub_portal_at(pos, axis)
        } else {
            None
        }
    }

    #[inline]
    pub fn light_mode(self) -> u8 {
        if let Some(data) = self.data() {
            data.light_mode
        } else {
            self.homogeneous_lightmode()
        }
    }

    #[inline]
    pub fn sub_skylight_idx(self, idx: usize) -> u8 {
        if let Some(data) = self.data() {
            data.skylight[idx]
        } else {
            self.homogeneous_skylight()
        }
    }

    #[inline]
    pub fn sub_skylight(self, pos: Int3) -> u8 {
        self.sub_skylight_idx(pos.to_index(CHUNK_BITS))
    }

    #[inline]
    pub fn is_clear(self, style: &StyleTable) -> bool {
        if !self.is_homogeneous() {
            return false;
        }
        style.is_clear(self.homogeneous_block())
    }

    #[inline]
    pub fn is_solid(self, style: &StyleTable) -> bool {
        if !self.is_homogeneous() {
            return false;
        }
        style.is_solid(self.homogeneous_block())
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
            let datamem = unsafe {
                ptr::copy_nonoverlapping(self.data_uncheckeed(), uninit.as_mut().as_mut_ptr(), 1);
                ArenaBox::into_raw(uninit.assume_init())
            };
            unsafe {
                ChunkBox {
                    ptr: NonNull::new_unchecked(
                        (datamem.as_ptr() as usize | (self.raw() & Self::MASK)) as *mut ChunkData,
                    ),
                }
            }
        }
    }
}

/// # `ChunkBox` pointer format
///
/// Since `ChunkData` has an alignment of 16, the `blocks` pointer has 4 bits to store tags.
/// If bit 0 is set, the chunk has no associated memory, and the entire chunk is made up of a
/// single block type, which is stored in bits 8..16.
/// If bit 1 is set, the chunk is made entirely out of clear blocks (although they
/// might be different). This flag is completely independent from bit 0.
/// If bit 2 is set, the chunk is made entirely out of solid blocks (although they might
/// be different types of solid blocks). This flag is completely independent from bit 0, although it
/// is mutually exclusive with bit 1.
/// If bit 3 is set, the chunk emits skylight. Every dimension without a roof _must_ have shiny
/// chunks at the top of the world. And not only a single layer, but all chunks above a certain
/// height should be shiny. To compute shadows, the world generator will render chunks up to the
/// sky level or until a roof is reached.
///
/// Note that flags 1 and 2 are not hints. If they are set, the chunk **must** be made out of
/// entirely solid or clear blocks, otherwise it is a logic error.
///
/// Also note that the pointer is never null. A chunk is in one of two states:
/// - Homogeneous: The bit 0 is set, so the pointer is non-null.
/// - Nonhomogeneous: The pointer (at least the top bits of it) point to a valid address, which
///     cannot be null.
/// Therefore there is always at least one bit set.
/// This is why `NonNull` can be used.
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
        if self.is_homogeneous() {
            write!(f, "ChunkBox::Homogeneous({:?})", self.homogeneous_block())
        } else {
            write!(f, "ChunkBox({}KB)", mem::size_of::<ChunkData>() / 1024)
        }
    }
}
impl ChunkBox {
    unsafe fn new_raw_nonhomogeneous(datamem: NonNull<ChunkData>, tags: usize) -> Self {
        Self {
            ptr: NonNull::new_unchecked((datamem.as_ptr() as usize | tags) as *mut ChunkData),
        }
    }

    /// Create a new chunk box with memory allocated, but with the blocks uninitialized.
    /// Inherently unsafe, but may work in most cases and speeds things up.
    #[inline]
    pub unsafe fn new_uninit() -> Self {
        let mut blockmem = ArenaBox::into_raw(ArenaBoxUninit::<ChunkData>::new().assume_init());
        blockmem.as_mut().portal_count = 0;
        blockmem.as_mut().entity_len = 0;
        ChunkBox::new_raw_nonhomogeneous(blockmem, 0)
    }

    /// Create a new chunk filled with the given block, without allocating any memory.
    #[inline]
    pub fn new_homogeneous(
        shinethrough: bool,
        block: BlockData,
        skylight: u8,
        lightmode: u8,
    ) -> Self {
        Self {
            ptr: ChunkRef::new_homogeneous(shinethrough, block, skylight, lightmode).ptr,
        }
    }

    #[inline]
    pub fn placeholder() -> Self {
        Self::new_homogeneous(false, BlockData { data: 0 }, 0, 0)
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

    pub unsafe fn data_mut_unchecked(&mut self) -> &mut ChunkData {
        &mut *((self.ptr.as_ptr() as usize & !ChunkRef::MASK) as *mut ChunkData)
    }

    /// Get a mutable reference to the inner blocks.
    /// If there is no associated memory, this method will allocate and fill with the homogeneous
    /// block, according to the current state.
    #[inline]
    pub fn data_mut(&mut self) -> &mut ChunkData {
        if self.is_homogeneous() {
            let shinethrough = self.homogeneous_shinethrough();
            let block = self.homogeneous_block();
            let skylight = self.homogeneous_skylight();
            let lightmode = self.homogeneous_lightmode();
            unsafe {
                *self = ChunkBox::new_uninit();
                ptr::write_bytes(&mut self.data_mut_unchecked().blocks, block.data, 1);
                ptr::write_bytes(&mut self.data_mut_unchecked().skylight, skylight, 1);
                ptr::write_bytes(
                    &mut self.data_mut_unchecked().shinethrough,
                    if shinethrough { 0xff } else { 0x00 },
                    1,
                );
                self.data_mut_unchecked().light_mode = lightmode;
            }
        }
        unsafe { self.data_mut_unchecked() }
    }

    /// Gets the inner blocks if there is memory associated to this chunk box.
    #[inline]
    pub fn try_data_mut(&mut self) -> Option<&mut ChunkData> {
        if self.is_homogeneous() {
            None
        } else {
            unsafe { Some(self.data_mut_unchecked()) }
        }
    }

    /// Take the current blocks, if available, and replace by a given chunkbox.
    #[inline]
    fn take_blocks(&mut self, new_box: ChunkBox) -> Option<ArenaBox<ChunkData>> {
        unsafe {
            let out = if let Some(data) = self.try_data_mut() {
                Some(ArenaBox::from_raw(NonNull::new_unchecked(
                    data as *mut ChunkData,
                )))
            } else {
                None
            };
            ptr::write(self, new_box);
            out
        }
    }

    /// Remove any memory associated with this box, and make all blocks homogeneous.
    pub fn make_homogeneous(
        &mut self,
        shinethrough: bool,
        block: BlockData,
        skylight: u8,
        lightmode: u8,
    ) {
        self.take_blocks(Self::new_homogeneous(
            shinethrough,
            block,
            skylight,
            lightmode,
        ));
    }

    /// Check if all of the blocks are of the same type, and drop the data altogether and mark with
    /// a tag in that case.
    #[inline]
    pub fn try_drop_data(&mut self) {
        if let Some(data) = self.data() {
            if data.portal_count != 0 {
                // Portals make a chunk nonhomogeneous
                return;
            }
            let block0 = data.blocks[0];
            if data.blocks.iter().skip(1).any(|&b| b != block0) {
                // Blocks are not homogeneous
                return;
            }
            let shinethrough = match data.shinethrough[0] {
                0 => 0,
                ChunkSizedInt::MAX => ChunkSizedInt::MAX,
                // Shinethrough is not homogeneous
                _ => return,
            };
            let light0 = data.skylight[0];
            if data.skylight.iter().skip(1).any(|&s| s != light0) {
                // Skylight is not homogeneous
                return;
            }
            self.make_homogeneous(shinethrough != 0, block0, light0, data.light_mode);
        }
    }
}
impl Drop for ChunkBox {
    fn drop(&mut self) {
        unsafe {
            if let Some(blocks) = self.try_data_mut() {
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
        if let Some(data) = chunk.data() {
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
        if let Some(data) = self.chunk.data() {
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
        if let Some(data) = self.chunk.data() {
            data.ref_count.fetch_add(1);
        }
        unsafe {
            Self {
                chunk: ptr::read(&self.chunk),
            }
        }
    }
}

/// An integer coordinate within the universe.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Int4 {
    pub coords: Int3,
    pub dim: u32,
}
impl Hash for Int4 {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        h.write_u128(
            self.coords.x as u32 as u128
                | ((self.coords.y as u32 as u128) << 32)
                | ((self.coords.z as u32 as u128) << 64)
                | ((self.dim as u128) << 96),
        );
    }
}
impl Int4 {
    pub fn block_to_chunk(&self) -> Int4 {
        Int4 {
            coords: self.coords >> CHUNK_BITS,
            dim: self.dim,
        }
    }
    pub fn chunk_to_block(&self) -> Int4 {
        Int4 {
            coords: self.coords << CHUNK_BITS,
            dim: self.dim,
        }
    }

    pub fn world_pos(&self) -> WorldPos {
        WorldPos {
            coords: self.coords.to_f64(),
            dim: self.dim,
        }
    }
}

pub type ChunkPos = Int4;
pub type BlockPos = Int4;

/// A floating-point valued coordinate within the universe.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct WorldPos {
    pub coords: [f64; 3],
    pub dim: u32,
}
impl WorldPos {
    pub fn block_pos(&self) -> BlockPos {
        BlockPos {
            coords: Int3::from_f64(self.coords),
            dim: self.dim,
        }
    }
}

/// Represents a single block id.
///
/// Currently a single byte, and the same byte always represents the same block anywhere in the
/// world.
/// In the future, however, the byte might represent a different block depending on context (ie.
/// position or dimension).
/// Alternatively, at some point block ids might become 2 bytes.
#[derive(Copy, Clone, Debug, PartialEq, Deserialize)]
#[serde(transparent)]
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
    /// The target dimension of the portal.
    pub dim: u32,
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

    pub fn get_center(&self) -> Int3 {
        [
            (self.pos[0] + (self.size[0] >> 1)) as i32,
            (self.pos[1] + (self.size[1] >> 1)) as i32,
            (self.pos[2] + (self.size[2] >> 1)) as i32,
        ]
        .into()
    }
}

/// Holds the block and entity data for a chunk.
/// Has an alignment of 16 in order for `ChunkBox` to store tags in the last 4 bits of the pointer.
#[repr(align(16))]
pub struct ChunkData {
    /// All of the blocks in this chunk.
    pub blocks: [BlockData; (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize],
    /// Skylight data for each block on this chunk.
    pub skylight: [u8; (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize],
    /// A byte referencing a certain lighting mode.
    pub light_mode: u8,
    /// Which columns of this chunk does sunlight go through.
    pub shinethrough: [ChunkSizedInt; CHUNK_SIZE as usize],
    /// All of the portals in this chunk.
    pub portals: [PortalData; MAX_CHUNK_PORTALS],
    /// The amount of portals in the portal buffer.
    pub portal_count: u32,
    /// Landmarks in this chunk.
    /// These usually correspond to entities.
    /// Each entity consists of a 2-byte length field followed by entity data.
    pub entity_data: [u8; MAX_ENTITY_BYTES],
    /// Length of the entity buffer.
    pub entity_len: u16,
    /// A hidden reference count field for use in chunk reference counting.
    ref_count: AtomicCell<u32>,
}
impl Clone for ChunkData {
    fn clone(&self) -> Self {
        Self {
            blocks: self.blocks,
            skylight: self.skylight,
            light_mode: self.light_mode,
            shinethrough: self.shinethrough,
            portals: self.portals,
            portal_count: self.portal_count,
            entity_data: self.entity_data,
            entity_len: self.entity_len,
            ref_count: AtomicCell::new(0),
        }
    }
}
impl ChunkData {
    #[track_caller]
    #[inline]
    pub fn sub_get(&self, sub_pos: Int3) -> BlockData {
        match self.blocks.get(sub_pos.to_index(CHUNK_BITS)) {
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
        match self.blocks.get_mut(sub_pos.to_index(CHUNK_BITS)) {
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

    /// Add an entity prototype to the end of the entity buffer.
    ///
    /// `dpos` is the entity position relative to the chunk corner.
    /// It must hold that `0.0 <= dpos[k] < 32.0` for `k` in 0..3
    ///
    /// `raw` is a raw bytestring passed as-is to lua as an entity spawn event when the chunk is
    /// loaded.
    ///
    /// Returns `None` if there is no space in the entity buffer for the new entity.
    #[inline]
    pub fn push_entity(&mut self, dpos: Vec3, raw: &[u8]) -> Option<()> {
        let old_len = self.entity_len as usize;
        let new_len = old_len + raw.len() + 2 + 6;
        if new_len > self.entity_data.len() {
            return None;
        }

        let dpos = dpos * 256.;
        let dpos = [dpos.x as u16, dpos.y as u16, dpos.z as u16];

        let data = &mut self.entity_data;
        data[old_len..old_len + 2].copy_from_slice(&(raw.len() as u16).to_le_bytes());
        data[old_len + 2..old_len + 4].copy_from_slice(&dpos[0].to_le_bytes());
        data[old_len + 4..old_len + 6].copy_from_slice(&dpos[1].to_le_bytes());
        data[old_len + 6..old_len + 8].copy_from_slice(&dpos[2].to_le_bytes());
        data[old_len + 8..new_len].copy_from_slice(raw);
        self.entity_len = new_len as u16;
        Some(())
    }

    /// Get the next entity at index `idx`.
    /// The first call to `next_entity` should start with `idx = 0`, and afterwards `next_entity`
    /// will advance the index automatically.
    ///
    /// Returns a pair of entity position relative to the chunk corner and a raw entity prototype.
    ///
    /// Returns `None` if there is no next entity, leaving the index untouched.
    #[inline]
    pub fn next_entity(&self, idx: &mut usize) -> Option<(Vec3, &[u8])> {
        let i = *idx;
        if i + 2 + 6 > self.entity_len as usize {
            return None;
        }

        let data = &self.entity_data;
        let len = u16::from_le_bytes([data[i + 0], data[i + 1]]) as usize;
        let dx = u16::from_le_bytes([data[i + 2], data[i + 3]]) as f32;
        let dy = u16::from_le_bytes([data[i + 4], data[i + 5]]) as f32;
        let dz = u16::from_le_bytes([data[i + 6], data[i + 7]]) as f32;
        let dpos = Vec3::new(dx, dy, dz) * (1. / 256.);
        *idx = i + 8 + len;
        Some((dpos, &data[i + 8..i + 8 + len]))
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

fn now_u32(epoch: &Instant) -> u32 {
    epoch.elapsed().as_secs() as u32
}

pub struct GridKeeperSlot<T> {
    pub last_use: std::sync::atomic::AtomicU32,
    pub item: T,
}
impl<T> GridKeeperSlot<T> {
    pub fn new(epoch: &Instant, t: T) -> Self {
        Self {
            last_use: now_u32(epoch).into(),
            item: t,
        }
    }

    pub fn touch(&self, epoch: &Instant) {
        self.last_use
            .store(now_u32(epoch), std::sync::atomic::Ordering::Relaxed);
    }
}

const KEEPALIVE_DURATION: Duration = Duration::from_secs(10);

pub struct GridKeeperN<K, T> {
    pub map: HashMap<K, GridKeeperSlot<T>>,
    pub epoch: Instant,
    pub keepalive: u32,
}
impl<K, T> GridKeeperN<K, T>
where
    K: Hash + Eq,
{
    pub fn new() -> Self {
        Self {
            map: default(),
            epoch: Instant::now(),
            keepalive: KEEPALIVE_DURATION.as_secs() as u32,
        }
    }

    pub fn get(&self, pos: K) -> Option<&T> {
        self.map.get(&pos).map(|t| {
            t.touch(&self.epoch);
            &t.item
        })
    }

    pub fn get_mut(&mut self, pos: K) -> Option<&mut T> {
        let epoch = &self.epoch;
        self.map.get_mut(&pos).map(|t| {
            t.touch(epoch);
            &mut t.item
        })
    }

    pub fn insert(&mut self, pos: K, t: T) {
        self.map.insert(pos, GridKeeperSlot::new(&self.epoch, t));
    }

    pub fn or_insert(&mut self, pos: K, f: impl FnOnce() -> T) -> &mut T {
        use std::collections::hash_map::Entry;
        match self.map.entry(pos) {
            Entry::Occupied(entry) => {
                let t = entry.into_mut();
                t.touch(&self.epoch);
                &mut t.item
            }
            Entry::Vacant(entry) => &mut entry.insert(GridKeeperSlot::new(&self.epoch, f())).item,
        }
    }

    pub fn try_or_insert<E, F>(&mut self, pos: K, f: F) -> StdResult<&mut T, E>
    where
        F: FnOnce() -> StdResult<T, E>,
    {
        use std::collections::hash_map::Entry;
        match self.map.entry(pos) {
            Entry::Occupied(entry) => {
                let t = entry.into_mut();
                t.touch(&self.epoch);
                Ok(&mut t.item)
            }
            Entry::Vacant(entry) => {
                Ok(&mut entry.insert(GridKeeperSlot::new(&self.epoch, f()?)).item)
            }
        }
    }

    pub fn gc_with<F>(&mut self, mut finalize: F)
    where
        F: FnMut(&K, &mut T),
    {
        let cutoff = now_u32(&self.epoch).saturating_sub(self.keepalive);
        self.map.retain(|k, t| {
            let keep = t.last_use.load(std::sync::atomic::Ordering::Relaxed) >= cutoff;
            if !keep {
                finalize(k, &mut t.item);
            }
            keep
        })
    }

    pub fn gc(&mut self) {
        let cutoff = now_u32(&self.epoch).saturating_sub(self.keepalive);
        self.map
            .retain(|_k, t| t.last_use.load(std::sync::atomic::Ordering::Relaxed) >= cutoff)
    }
}
impl<K: Hash + Eq, T> Default for GridKeeperN<K, T> {
    fn default() -> Self {
        Self::new()
    }
}

pub type GridKeeper2<T> = GridKeeperN<Int2, T>;
pub type GridKeeper3<T> = GridKeeperN<Int3, T>;
pub type GridKeeper4<T> = GridKeeperN<Int4, T>;

#[derive(Default, Clone, Deserialize)]
pub struct BlendConf {
    pub base: u32,
    pub mul: u32,
    pub shr: u32,
}

#[derive(Default, Clone, Deserialize)]
pub struct LightingConf {
    /// When converting multiple light values to a single vertex light value, this constant
    /// specifies the base lighting, per-block multiplication coefficient and final right-shift
    /// amount.
    pub light: BlendConf,
    /// When generating the per-block decay dither, the base decay is added with an 8-bit random
    /// value which is multiplied by `mul` and then the final result is shifted right by `shr`.
    pub decay: BlendConf,
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

pub struct WorldInfo {
    pub blocks: [BlockTexture; 256],
    pub light_modes: [LightingConf; 256],
}
impl Default for WorldInfo {
    fn default() -> Self {
        Self {
            blocks: arr![default(); 256],
            light_modes: arr![default(); 256],
        }
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

    pub fn new(info: &WorldInfo) -> Self {
        let mut words = [0; Self::TOTAL_WORDS];
        for i in 0..256 {
            let tx = &info.blocks[i];
            words[i / Self::BLOCKS_PER_WORD] |=
                (tx.style as u8 as usize) << (i % Self::BLOCKS_PER_WORD * Self::BITS_PER_BLOCK);
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

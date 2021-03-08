use crate::prelude::*;

/// Guaranteed to be a power of 2.
pub const CHUNK_SIZE: i32 = 32;

pub type ChunkBox = crate::arena::Box<Chunk>;
pub type LoafBox<T> = crate::arena::Box<[T; (CHUNK_SIZE * CHUNK_SIZE) as usize]>;

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

    pub fn xz(&self) -> [i32; 2] {
        [self[0], self[2]]
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

#[derive(Clone)]
pub struct Chunk {
    pub blocks: [BlockData; (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize],
}
impl Chunk {
    #[track_caller]
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
    pub fn _sub_get_mut(&mut self, sub_pos: [i32; 3]) -> &mut BlockData {
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

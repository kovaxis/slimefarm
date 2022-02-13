use crate::prelude::*;

/// a dynamic growable buffer, representing an infinite space filled with "filler block".
/// any block within this space can be set or get, but setting blocks far away from the origin
/// will overallocate memory.
/// this buffer can be "transferred" over to a chunk, copying the corresponding blocks over from
/// the virtual buffer to the actual chunk.
pub struct BlockBuf {
    /// Where does the block buffer origin map to in the real world.
    origin: Int3,
    /// The actual buffer data.
    blocks: Vec<BlockData>,
    /// "Filler block", the block that by default covers the entire space.
    fill: BlockData,
    /// The lowest coordinate represented by the physical block buffer, in buffer-local coordinates.
    /// usually negative.
    corner: Int3,
    /// Log2 of the size.
    /// Forces sizes to be a power of two.
    size: Int3,
}
impl BlockBuf {
    pub fn with_capacity(origin: BlockPos, fill: BlockData, corner: Int3, size: Int3) -> Self {
        let size_log2 = [
            (mem::size_of_val(&size.x) * 8) as i32 - (size.x - 1).leading_zeros() as i32,
            (mem::size_of_val(&size.y) * 8) as i32 - (size.y - 1).leading_zeros() as i32,
            (mem::size_of_val(&size.z) * 8) as i32 - (size.z - 1).leading_zeros() as i32,
        ]
        .into();
        Self {
            origin,
            fill,
            corner,
            size: size_log2,
            blocks: vec![fill; 1 << (size_log2[0] + size_log2[1] + size_log2[2])],
        }
    }

    /// Creates a new block buffer with its origin at [0, 0, 0].
    pub fn new(origin: BlockPos, fill: BlockData) -> Self {
        Self::with_capacity(origin, fill, [-2, -2, -2].into(), [4, 4, 4].into())
    }

    /// Get the block data at the given position relative to the block buffer origin.
    /// If the position is out of bounds, return the filler block.
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
            for y in mn.y..=mx.y {
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

    /// Set the block data at the given position relative to the block buffer origin.
    /// If the position is out of bounds, allocates new blocks filled with the filler block and
    /// only then sets the given position.
    pub fn set(&mut self, pos: BlockPos, block: BlockData) {
        self.reserve(pos);
        let pos = pos - self.corner;
        self.blocks[pos.to_index(self.size)] = block;
    }

    /// Copy the contents of the block buffer over to the given chunk.
    /// Locates the block buffer origin at the buffer origin position, and locates the chunk at the
    /// given chunk coordinates.
    pub fn transfer(&self, chunkpos: ChunkPos, chunk: &mut ChunkBox) {
        let chunkpos = chunkpos << CHUNK_BITS;
        // position of the buffer relative to the destination chunk
        let pos = self.origin + self.corner - chunkpos;

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

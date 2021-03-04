use std::f64::INFINITY;

use crate::prelude::*;

/// Guaranteed to be a power of 2.
pub const CHUNK_SIZE: i32 = 32;

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
    fn offset(&self, x: i32, y: i32, z: i32) -> ChunkPos {
        ChunkPos([self[0] + x, self[1] + y, self[2] + z])
    }

    fn to_block_floor(&self) -> BlockPos {
        BlockPos([
            self[0] * CHUNK_SIZE,
            self[1] * CHUNK_SIZE,
            self[2] * CHUNK_SIZE,
        ])
    }
    fn _to_block_center(&self) -> BlockPos {
        BlockPos([
            self[0] * CHUNK_SIZE + CHUNK_SIZE / 2,
            self[1] * CHUNK_SIZE + CHUNK_SIZE / 2,
            self[2] * CHUNK_SIZE + CHUNK_SIZE / 2,
        ])
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
    fn offset(&self, x: i32, y: i32, z: i32) -> BlockPos {
        BlockPos([self[0] + x, self[1] + y, self[2] + z])
    }
    fn _from_float(pos: [f64; 3]) -> BlockPos {
        BlockPos([
            pos[0].floor() as i32,
            pos[1].floor() as i32,
            pos[2].floor() as i32,
        ])
    }
    fn _to_float_floor(&self) -> [f64; 3] {
        [self[0] as f64, self[1] as f64, self[2] as f64]
    }
    fn _to_float_center(&self) -> [f64; 3] {
        [
            self[0] as f64 + 0.5,
            self[1] as f64 + 0.5,
            self[2] as f64 + 0.5,
        ]
    }
    fn to_chunk(&self) -> ChunkPos {
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
    fn is_solid(&self) -> bool {
        self.data != 0
    }
}

const MIN_CHUNKS_MESHED_PER_SEC: f32 = 20.;

struct Terrain {
    mesher: Mesher,
    state: Rc<State>,
    chunks: HashMap<ChunkPos, Box<Chunk>>,
    meshes: MeshKeeper,
    meshing_time_estimate: f32,
    last_meshing: Instant,
    generator: GeneratorHandle,
}
impl Terrain {
    fn new(state: &Rc<State>, gen_cfg: crate::gen::GenConfig) -> Terrain {
        Terrain {
            mesher: Mesher::new(),
            state: state.clone(),
            chunks: default(),
            meshes: MeshKeeper::new(0., ChunkPos([0, 0, 0])),
            meshing_time_estimate: 0.,
            last_meshing: Instant::now(),
            generator: GeneratorHandle::new(gen_cfg),
        }
    }

    fn set_view_radius(&mut self, radius: f32) {
        self.meshes = MeshKeeper::new(radius / CHUNK_SIZE as f32, self.meshes.center());
    }

    fn book_keep(&mut self, center: BlockPos, limit: Duration) {
        //measure_time!(start book_keep);
        //Keep track of time
        let now = Instant::now();
        let deadline = Instant::now() + limit;
        //Adjust center
        let center = center
            .offset(CHUNK_SIZE / 2, CHUNK_SIZE / 2, CHUNK_SIZE / 2)
            .to_chunk();
        self.meshes.set_center(center);
        //Receive chunks from the generator
        {
            let mut req = self.generator.request.lock();
            //Mark which chunks are in-progress, so as to not request them again
            for chunk_pos in req.in_progress.drain() {
                if let Some(slot) = self.meshes.get_mut(chunk_pos) {
                    slot.generating = true;
                }
            }
            //Receive chunks
            for (pos, chunk) in req.ready.drain(..) {
                self.chunks.insert(pos, chunk);
            }
        }
        //Look for candidate chunks
        let meshing_time_estimate = Duration::from_secs_f32(self.meshing_time_estimate);
        let mut force_mesh =
            ((now - self.last_meshing).as_secs_f32() * MIN_CHUNKS_MESHED_PER_SEC).floor() as i32;
        let mut chunks_meshed = 0;
        let mut request_queue = self.generator.swap_request_vec(default());
        let request_count = self.generator.request_hint();
        request_queue.clear();
        request_queue.reserve(request_count);
        let mut order_idx = 0;
        while order_idx < self.meshes.render_order.len() {
            let idx = self.meshes.render_order[order_idx];
            order_idx += 1;
            let mesh_slot = self.meshes.get_by_idx(idx);
            let pos = self.meshes.sub_idx_to_pos(idx);
            let mut should_generate = !mesh_slot.generating;
            if mesh_slot.mesh.is_none()
                && (force_mesh > 0 || Instant::now() + meshing_time_estimate < deadline)
            {
                (|| {
                    let start_time = Instant::now();
                    let mesh_slot = self.meshes.get_by_idx_mut(idx);
                    let chunk = self.chunks.get(&pos)?;
                    //Don't generate, it already exists
                    should_generate = false;
                    //Get all of its adjacent neighbors
                    /*let neighbors = [
                        &**self.chunks.get(&pos.offset(-1, 0, 0))?,
                        &**self.chunks.get(&pos.offset(1, 0, 0))?,
                        &**self.chunks.get(&pos.offset(0, -1, 0))?,
                        &**self.chunks.get(&pos.offset(0, 1, 0))?,
                        &**self.chunks.get(&pos.offset(0, 0, -1))?,
                        &**self.chunks.get(&pos.offset(0, 0, 1))?,
                    ];*/
                    macro_rules! build_neighbors {
                        (@1, 1, 1) => {{
                            &**chunk
                        }};
                        (@$x:expr, $y:expr, $z:expr) => {{
                            &**self.chunks.get(&pos.offset($x-1, $y-1, $z-1))?
                        }};
                        ($($x:tt, $y:tt, $z:tt;)*) => {{
                            [$(
                                build_neighbors!(@$x, $y, $z),
                            )*]
                        }};
                    }
                    let neighbors = build_neighbors![
                        0, 0, 0;
                        1, 0, 0;
                        2, 0, 0;
                        0, 1, 0;
                        1, 1, 0;
                        2, 1, 0;
                        0, 2, 0;
                        1, 2, 0;
                        2, 2, 0;

                        0, 0, 1;
                        1, 0, 1;
                        2, 0, 1;
                        0, 1, 1;
                        1, 1, 1;
                        2, 1, 1;
                        0, 2, 1;
                        1, 2, 1;
                        2, 2, 1;

                        0, 0, 2;
                        1, 0, 2;
                        2, 0, 2;
                        0, 1, 2;
                        1, 1, 2;
                        2, 1, 2;
                        0, 2, 2;
                        1, 2, 2;
                        2, 2, 2;
                    ];
                    //Mesh the chunk
                    let mesh = self.mesher.make_mesh(&neighbors);
                    //Upload the chunk
                    let mesh = ChunkMesh {
                        buf: if mesh.indices.is_empty() {
                            None
                        } else {
                            Some(mesh.make_buffer(&self.state.display))
                        },
                    };
                    mesh_slot.mesh = Some(mesh);
                    force_mesh -= 1;
                    chunks_meshed += 1;
                    //Use the time measure to estimate the time meshing a chunk takes
                    let time_taken = start_time.elapsed().as_secs_f32();
                    const RAISE_WEIGHT: f32 = 0.75;
                    const LOWER_WEIGHT: f32 = 0.1;
                    let weight = if time_taken > self.meshing_time_estimate {
                        RAISE_WEIGHT
                    } else {
                        LOWER_WEIGHT
                    };
                    self.meshing_time_estimate = self.meshing_time_estimate
                        + (time_taken - self.meshing_time_estimate) * weight;
                    Some(())
                })();
            }
            if should_generate && request_queue.len() < request_count {
                //Having this chunk'd be pretty pog
                request_queue.push(pos);
            }
        }
        //Keep track of average meshes per sec
        if chunks_meshed > 0 {
            self.last_meshing = Instant::now();
        }
        unsafe {
            static mut LAST_UPDATE: std::time::SystemTime = std::time::UNIX_EPOCH;
            if chunks_meshed > 0 {
                LAST_UPDATE = std::time::SystemTime::now();
            }
            let since = LAST_UPDATE.elapsed().unwrap();
            if since > Duration::from_millis(500) {
                eprintln!("no chunks meshed");
                LAST_UPDATE = std::time::SystemTime::now();
            }
        }
        //Request missing chunks from generator
        self.generator.request(&request_queue);
        self.generator.swap_request_vec(request_queue);
        //measure_time!(end book_keep);
    }

    fn block_at(&self, pos: BlockPos) -> Option<BlockData> {
        self.chunks.get(&pos.to_chunk()).map(|chunk| {
            chunk.sub_get([
                pos[0].rem_euclid(CHUNK_SIZE),
                pos[1].rem_euclid(CHUNK_SIZE),
                pos[2].rem_euclid(CHUNK_SIZE),
            ])
        })
    }
}
impl Drop for Terrain {
    fn drop(&mut self) {
        eprintln!("dropping terrain");
        measure_time!(start drop_chunks);
        self.chunks = default();
        measure_time!(end drop_chunks);
    }
}

/// Keeps track of chunk meshes in an efficient grid structure.
struct MeshKeeper {
    corner_pos: ChunkPos,
    size_log2: u32,
    origin_idx: i32,
    meshes: Vec<ChunkMeshSlot>,
    pub render_order: Vec<i32>,
}
impl MeshKeeper {
    pub fn new(radius: f32, center: ChunkPos) -> Self {
        //Chunks right at the border are not renderable, because chunks need their neighbors
        //in order to be meshed
        //Therefore, must add 1 to the radius to make it effective
        let radius = radius + 1.;
        //Make sure length is a power of two
        let size = (radius.ceil() * 2.).max(2.) as i32;
        let size_log2 = (mem::size_of_val(&size) * 8) as u32 - (size - 1).leading_zeros();
        let size = 1 << size_log2;
        //Allocate space for meshes
        let total = (1 << (size_log2 * 3)) as usize;
        let mut meshes = Vec::with_capacity(total);
        meshes.resize_with(total, default);
        //Precalculate the render order
        let max_dist_sq = radius * radius;
        let flatten_factor = 1.;
        let mut distances = Vec::<(Sortf32, u32, i32)>::with_capacity(total);
        let mut rng = FastRng::seed_from_u64(0xadbcefabbd);
        let mut idx = 0;
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let pos = Vec3::new(x as f32, y as f32, z as f32);
                    let center = (size / 2) as f32 - 0.5;
                    let mut delta = pos - Vec3::broadcast(center);
                    delta.y *= flatten_factor;
                    let dist_sq = delta.mag_sq();
                    if dist_sq < max_dist_sq {
                        distances.push((Sortf32(dist_sq), rng.gen(), idx as i32));
                    }
                    idx += 1;
                }
            }
        }
        distances.sort_unstable();
        let render_order: Vec<_> = distances.into_iter().map(|(_, _, idx)| idx).collect();
        eprintln!("{} chunks to render", render_order.len());
        //Group em up
        Self {
            corner_pos: center.offset(
                -(1 << (size_log2 - 1)),
                -(1 << (size_log2 - 1)),
                -(1 << (size_log2 - 1)),
            ),
            size_log2,
            origin_idx: 0,
            meshes,
            render_order,
        }
    }

    fn center(&self) -> ChunkPos {
        self.corner_pos
            .offset(self.half_size(), self.half_size(), self.half_size())
    }

    fn size(&self) -> i32 {
        1 << self.size_log2
    }

    fn half_size(&self) -> i32 {
        1 << (self.size_log2 - 1)
    }

    fn total_len(&self) -> i32 {
        1 << (self.size_log2 * 3)
    }

    pub fn _get(&self, pos: ChunkPos) -> Option<&ChunkMeshSlot> {
        if pos[0] >= self.corner_pos[0]
            && pos[0] < self.corner_pos[0] + self.size()
            && pos[1] >= self.corner_pos[1]
            && pos[1] < self.corner_pos[1] + self.size()
            && pos[2] >= self.corner_pos[2]
            && pos[2] < self.corner_pos[2] + self.size()
        {
            Some(self._sub_get([
                pos[0] - self.corner_pos[0],
                pos[1] - self.corner_pos[1],
                pos[2] - self.corner_pos[2],
            ]))
        } else {
            None
        }
    }
    pub fn get_mut(&mut self, pos: ChunkPos) -> Option<&mut ChunkMeshSlot> {
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

    pub fn get_by_idx(&self, idx: i32) -> &ChunkMeshSlot {
        &self.meshes[(self.origin_idx + idx).rem_euclid(self.total_len()) as usize]
    }
    pub fn get_by_idx_mut(&mut self, idx: i32) -> &mut ChunkMeshSlot {
        let idx = (self.origin_idx + idx).rem_euclid(self.total_len()) as usize;
        &mut self.meshes[idx]
    }
    pub fn _sub_get(&self, pos: [i32; 3]) -> &ChunkMeshSlot {
        &self.meshes[(self.origin_idx
            + pos[0]
            + pos[1] * self.size()
            + pos[2] * (self.size() * self.size()))
        .rem_euclid(self.total_len()) as usize]
    }
    pub fn sub_get_mut(&mut self, pos: [i32; 3]) -> &mut ChunkMeshSlot {
        let size = self.size();
        let total_len = self.total_len();
        &mut self.meshes[(self.origin_idx + pos[0] + pos[1] * size + pos[2] * (size * size))
            .rem_euclid(total_len) as usize]
    }

    pub fn sub_idx_to_pos(&self, idx: i32) -> ChunkPos {
        let x = idx % self.size();
        let y = idx / self.size() % self.size();
        let z = idx / self.size() / self.size();
        self.corner_pos.offset(x, y, z)
    }

    /// Will slide chunks and remove chunks that went over the border.
    pub fn set_center(&mut self, new_center: ChunkPos) {
        let new_corner = new_center.offset(-self.half_size(), -self.half_size(), -self.half_size());
        let adj_x = new_corner[0] - self.corner_pos[0];
        let adj_y = new_corner[1] - self.corner_pos[1];
        let adj_z = new_corner[2] - self.corner_pos[2];
        let clear_range =
            |this: &mut MeshKeeper, x: ops::Range<i32>, y: ops::Range<i32>, z: ops::Range<i32>| {
                for z in z.clone() {
                    for y in y.clone() {
                        for x in x.clone() {
                            *this.sub_get_mut([x, y, z]) = default();
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

#[derive(Default)]
struct ChunkMeshSlot {
    mesh: Option<ChunkMesh>,
    generating: bool,
}

struct ChunkMesh {
    buf: Option<Buffer3d>,
}

pub struct Chunk {
    pub blocks: [BlockData; (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize],
}
impl Chunk {
    pub fn _new() -> Chunk {
        Chunk {
            //blocks: unsafe { Uninit::uninit().assume_init() },
            blocks: [BlockData { data: 0 }; (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize],
        }
    }

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
}

struct Transform {
    x: [i32; 3],
    y: [i32; 3],
    mov: [i32; 3],
    flip: bool,
    normal: u8,
}

struct Mesher {
    vert_cache: [VertIdx; Self::VERT_COUNT],
    block_buf: [BlockData; Self::BLOCK_COUNT * 2],
    front: i32,
    back: i32,
    mesh: Mesh,
}
impl Mesher {
    const VERT_COUNT: usize = ((CHUNK_SIZE + 1) * (CHUNK_SIZE + 1)) as usize;
    const BLOCK_COUNT: usize = ((CHUNK_SIZE + 2) * (CHUNK_SIZE + 2)) as usize;

    const ADV_X: i32 = 1;
    const ADV_Y: i32 = CHUNK_SIZE + 2;

    pub fn new() -> Self {
        Self {
            vert_cache: [0; Self::VERT_COUNT],
            block_buf: [BlockData { data: 0 }; Self::BLOCK_COUNT * 2],
            front: Self::BLOCK_COUNT as i32,
            back: 0,
            mesh: default(),
        }
    }

    fn flip_bufs(&mut self) {
        mem::swap(&mut self.front, &mut self.back);
    }

    fn front_mut(&mut self, idx: i32) -> &mut BlockData {
        &mut self.block_buf[(self.front + idx) as usize]
    }

    fn front(&self, idx: i32) -> BlockData {
        self.block_buf[(self.front + idx) as usize]
    }

    fn back(&self, idx: i32) -> BlockData {
        self.block_buf[(self.back + idx) as usize]
    }

    fn get_vert(&mut self, trans: &Transform, x: i32, y: i32, idx: i32) -> VertIdx {
        let cache_idx = (x + y * (CHUNK_SIZE + 1)) as usize;
        let mut cached = self.vert_cache[cache_idx];
        if cached == VertIdx::max_value() {
            //Create vertex
            //[NONE, BACK_ONLY, FRONT_ONLY, BOTH]
            const LIGHT_TABLE: [f32; 4] = [0.02, 0.0, -0.11, -0.11];
            let mut lightness = 1.;
            {
                let mut process = |idx| {
                    lightness += LIGHT_TABLE[(self.front(idx).is_solid() as usize) << 1
                        | self.back(idx).is_solid() as usize];
                };
                process(idx);
                process(idx - Self::ADV_X);
                process(idx - Self::ADV_Y);
                process(idx - Self::ADV_X - Self::ADV_Y);
            }
            let color_normal = [
                (128. * lightness) as u8,
                (128. * lightness) as u8,
                (128. * lightness) as u8,
                trans.normal,
            ];
            //Apply transform
            let vert = Vec3::new(
                (x & trans.x[0] | y & trans.y[0] | trans.mov[0]) as f32,
                (x & trans.x[1] | y & trans.y[1] | trans.mov[1]) as f32,
                (x & trans.x[2] | y & trans.y[2] | trans.mov[2]) as f32,
            );
            cached = self.mesh.add_vertex(vert, color_normal);
            self.vert_cache[cache_idx] = cached;
        }
        cached
    }

    fn layer(&mut self, trans: &Transform) {
        let mut idx = Self::ADV_X + Self::ADV_Y;
        for vidx in self.vert_cache.iter_mut() {
            *vidx = VertIdx::max_value();
        }
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                if self.back(idx).is_solid() && !self.front(idx).is_solid() {
                    //Place a face here
                    let v00 = self.get_vert(trans, x, y, idx);
                    let v01 = self.get_vert(trans, x, y + 1, idx + Self::ADV_Y);
                    let v10 = self.get_vert(trans, x + 1, y, idx + Self::ADV_X);
                    let v11 = self.get_vert(trans, x + 1, y + 1, idx + Self::ADV_X + Self::ADV_Y);
                    if trans.flip {
                        self.mesh.add_face(v01, v11, v00);
                        self.mesh.add_face(v00, v11, v10);
                    } else {
                        self.mesh.add_face(v01, v00, v11);
                        self.mesh.add_face(v00, v10, v11);
                    }
                }
                idx += 1;
            }
            idx += 2;
        }
    }

    pub fn make_mesh(&mut self, chunks: &[&Chunk; 3 * 3 * 3]) -> Mesh {
        let block_at = |pos: [i32; 3]| {
            let chunk_pos = [
                pos[0] as u32 / CHUNK_SIZE as u32,
                pos[1] as u32 / CHUNK_SIZE as u32,
                pos[2] as u32 / CHUNK_SIZE as u32,
            ];
            let sub_pos = [
                (pos[0] as u32 % CHUNK_SIZE as u32) as i32,
                (pos[1] as u32 % CHUNK_SIZE as u32) as i32,
                (pos[2] as u32 % CHUNK_SIZE as u32) as i32,
            ];
            chunks
                [(chunk_pos[0] + chunk_pos[1] * 3 as u32 + chunk_pos[2] * (3 * 3) as u32) as usize]
                .sub_get(sub_pos)
        };

        // X
        for x in CHUNK_SIZE - 1..2 * CHUNK_SIZE + 1 {
            let mut idx = 0;
            for z in CHUNK_SIZE - 1..2 * CHUNK_SIZE + 1 {
                for y in CHUNK_SIZE - 1..2 * CHUNK_SIZE + 1 {
                    *self.front_mut(idx) = block_at([x, y, z]);
                    idx += 1;
                }
            }
            if x > CHUNK_SIZE {
                //Facing `+`
                self.layer(&Transform {
                    x: [0, -1, 0],
                    y: [0, 0, -1],
                    mov: [x - CHUNK_SIZE, 0, 0],
                    flip: false,
                    normal: 0,
                });
            }
            self.flip_bufs();
            if x >= CHUNK_SIZE && x < 2 * CHUNK_SIZE {
                //Facing `-`
                self.layer(&Transform {
                    x: [0, -1, 0],
                    y: [0, 0, -1],
                    mov: [x - CHUNK_SIZE, 0, 0],
                    flip: true,
                    normal: 1,
                });
            }
        }

        // Y
        for y in CHUNK_SIZE - 1..2 * CHUNK_SIZE + 1 {
            let mut idx = 0;
            for z in CHUNK_SIZE - 1..2 * CHUNK_SIZE + 1 {
                for x in CHUNK_SIZE - 1..2 * CHUNK_SIZE + 1 {
                    *self.front_mut(idx) = block_at([x, y, z]);
                    idx += 1;
                }
            }
            if y > CHUNK_SIZE {
                //Facing `+`
                self.layer(&Transform {
                    x: [-1, 0, 0],
                    y: [0, 0, -1],
                    mov: [0, y - CHUNK_SIZE, 0],
                    flip: true,
                    normal: 2,
                });
            }
            self.flip_bufs();
            if y >= CHUNK_SIZE && y < 2 * CHUNK_SIZE {
                //Facing `-`
                self.layer(&Transform {
                    x: [-1, 0, 0],
                    y: [0, 0, -1],
                    mov: [0, y - CHUNK_SIZE, 0],
                    flip: false,
                    normal: 3,
                });
            }
        }

        // Z
        for z in CHUNK_SIZE - 1..2 * CHUNK_SIZE + 1 {
            let mut idx = 0;
            for y in CHUNK_SIZE - 1..2 * CHUNK_SIZE + 1 {
                for x in CHUNK_SIZE - 1..2 * CHUNK_SIZE + 1 {
                    *self.front_mut(idx) = block_at([x, y, z]);
                    idx += 1;
                }
            }
            if z > CHUNK_SIZE {
                //Facing `+`
                self.layer(&Transform {
                    x: [-1, 0, 0],
                    y: [0, -1, 0],
                    mov: [0, 0, z - CHUNK_SIZE],
                    flip: false,
                    normal: 4,
                });
            }
            self.flip_bufs();
            if z >= CHUNK_SIZE && z < 2 * CHUNK_SIZE {
                //Facing `-`
                self.layer(&Transform {
                    x: [-1, 0, 0],
                    y: [0, -1, 0],
                    mov: [0, 0, z - CHUNK_SIZE],
                    flip: true,
                    normal: 5,
                });
            }
        }

        mem::replace(&mut self.mesh, default())
    }
}

fn check_collisions(
    from: [f64; 3],
    mut delta: [f64; 3],
    size: [f64; 3],
    mut is_solid: impl FnMut(BlockPos, i32) -> bool,
    raycast: bool,
) -> [f64; 3] {
    if delta == [0.; 3] {
        return from;
    }

    let safe_gap = 1. / 128.;
    /*let safe_ratio = 63. / 64.;
    let size = [
        size[0] * safe_ratio,
        size[1] * safe_ratio,
        size[2] * safe_ratio,
    ];*/
    let binary = [
        (delta[0] > 0.) as i32,
        (delta[1] > 0.) as i32,
        (delta[2] > 0.) as i32,
    ];
    let dir = [
        if delta[0] > 0. { 1. } else { -1. },
        if delta[1] > 0. { 1. } else { -1. },
        if delta[2] > 0. { 1. } else { -1. },
    ];
    let mut frontier = [
        from[0] + size[0] * dir[0],
        from[1] + size[1] * dir[1],
        from[2] + size[2] * dir[2],
    ];
    let mut next_block = [
        (frontier[0] * dir[0]).ceil() * dir[0],
        (frontier[1] * dir[1]).ceil() * dir[1],
        (frontier[2] * dir[2]).ceil() * dir[2],
    ];
    let mut limit = [
        from[0] + size[0] * dir[0] + delta[0],
        from[1] + size[1] * dir[1] + delta[1],
        from[2] + size[2] * dir[2] + delta[2],
    ];

    while next_block
        .iter()
        .zip(&limit)
        .zip(&dir)
        .any(|((&next_block, &limit), &dir)| next_block * dir < limit * dir)
    {
        //Find closest axis
        let mut closest_axis = 0;
        let mut closest_dist = f64::INFINITY;
        for axis in 0..3 {
            if delta[axis] == 0. {
                continue;
            }
            let dist = (next_block[axis] - frontier[axis]) / delta[axis];
            if dist < closest_dist {
                closest_axis = axis;
                closest_dist = dist;
            }
        }

        //Advance on this axis
        let mut min_block = BlockPos([0; 3]);
        let mut max_block = BlockPos([0; 3]);
        for axis in 0..3 {
            if axis == closest_axis {
                min_block[axis] = next_block[axis] as i32 + (binary[axis] - 1);
                max_block[axis] = next_block[axis] as i32 + binary[axis];
                frontier[axis] = next_block[axis];
                next_block[axis] += dir[axis];
            } else {
                frontier[axis] += closest_dist * delta[axis];
                min_block[axis] =
                    (frontier[axis] - (2 * binary[axis]) as f64 * size[axis]).floor() as i32;
                max_block[axis] =
                    (frontier[axis] + (2 - 2 * binary[axis]) as f64 * size[axis]).ceil() as i32;
            }
        }

        //Check whether there is a collision
        let mut collides = false;
        'outer: for z in min_block[2]..max_block[2] {
            for y in min_block[1]..max_block[1] {
                for x in min_block[0]..max_block[0] {
                    if is_solid(BlockPos([x, y, z]), closest_axis as i32) {
                        collides = true;
                        break 'outer;
                    }
                }
            }
        }

        //If there is collision, start ignoring this axis
        if collides {
            if raycast {
                //Unless we're raycasting
                //In this case, abort immediately
                limit = frontier;
                break;
            }
            limit[closest_axis] = frontier[closest_axis] - safe_gap * dir[closest_axis];
            delta[closest_axis] = 0.;
            if delta == [0., 0., 0.] {
                break;
            }
        }
    }

    [
        limit[0] - size[0] * dir[0],
        limit[1] - size[1] * dir[1],
        limit[2] - size[2] * dir[2],
    ]
}

#[derive(Clone)]
pub struct TerrainRef {
    rc: AssertSync<Rc<RefCell<Terrain>>>,
}
impl TerrainRef {
    pub(crate) fn new(state: &Rc<State>, gen_cfg: crate::gen::GenConfig) -> TerrainRef {
        TerrainRef {
            rc: AssertSync(Rc::new(RefCell::new(Terrain::new(state, gen_cfg)))),
        }
    }
}
lua_type! {TerrainRef,
    fn book_keep(lua, this, (x, y, z, secs): (i32, i32, i32, f64)) {
        this.rc.borrow_mut().book_keep(BlockPos([x, y, z]), Duration::from_secs_f64(secs));
    }

    fn set_view_distance(lua, this, dist: f32) {
        this.rc.borrow_mut().set_view_radius(dist)
    }

    fn collide(lua, this, (x, y, z, dx, dy, dz, sx, sy, sz): (f64, f64, f64, f64, f64, f64, f64, f64, f64)) {
        let terrain = this.rc.borrow();
        let [fx, fy, fz] = check_collisions([x, y, z], [dx, dy, dz], [sx, sy, sz], |block_pos, _axis| {
            terrain.block_at(block_pos).map(|data| data.is_solid()).unwrap_or(true)
        }, false);
        (fx, fy, fz)
    }

    fn raycast(lua, this, (x, y, z, dx, dy, dz, sx, sy, sz): (f64, f64, f64, f64, f64, f64, f64, f64, f64)) {
        let terrain = this.rc.borrow();
        let [fx, fy, fz] = check_collisions([x, y, z], [dx, dy, dz], [sx, sy, sz], |block_pos, _axis| {
            terrain.block_at(block_pos).map(|data| data.is_solid()).unwrap_or(true)
        }, true);
        (fx, fy, fz)
    }

    fn draw(lua, this, (shader, uniforms, offset_uniform, params, x, y, z): (ShaderRef, UniformStorage, u32, LuaDrawParams, f64, f64, f64)) {
        //measure_time!(start draw_terrain);
        let this = this.rc.borrow();
        let mut frame = this.state.frame.borrow_mut();
        //Rendering in this order has the nice property that closer chunks are rendered first,
        //making better use of the depth buffer.
        for &idx in this.meshes.render_order.iter() {
            if let Some(buf) = &this.meshes.get_by_idx(idx).mesh.as_ref().and_then(|mesh| mesh.buf.as_ref()) {
                let pos = this.meshes.sub_idx_to_pos(idx).to_block_floor();
                let offset = Vec3::new((pos[0] as f64 - x) as f32, (pos[1] as f64 - y) as f32, (pos[2] as f64 - z) as f32);
                uniforms.vars.borrow_mut().get_mut(offset_uniform as usize).ok_or("offset uniform out of range").to_lua_err()?.1 = crate::UniformVal::Vec3(offset.into());
                frame.draw(&buf.vertex, &buf.index, &shader.program, &uniforms, &params.params).unwrap();
            }
        }
        //measure_time!(end draw_terrain);
    }
}

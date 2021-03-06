use std::f64::INFINITY;

use crate::{chunkmesh::MesherHandle, prelude::*};

/// Guaranteed to be a power of 2.
pub const CHUNK_SIZE: i32 = 32;

pub use self::chunk_arena::Box as ChunkBox;
make_arena!(pub chunk_arena, Chunk);

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
    pub fn is_solid(&self) -> bool {
        self.data != 0
    }
}

pub struct GridKeeper<T> {
    corner_pos: ChunkPos,
    size_log2: u32,
    origin_idx: i32,
    slots: Vec<T>,
}
impl<T: GridSlot> GridKeeper<T> {
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
    pub fn center(&self) -> ChunkPos {
        self.corner_pos
            .offset(self.half_size(), self.half_size(), self.half_size())
    }

    pub fn size(&self) -> i32 {
        1 << self.size_log2
    }

    pub fn half_size(&self) -> i32 {
        1 << (self.size_log2 - 1)
    }

    pub fn total_len(&self) -> i32 {
        1 << (self.size_log2 * 3)
    }

    pub fn get(&self, pos: ChunkPos) -> Option<&T> {
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

    pub fn get_by_idx(&self, idx: i32) -> &T {
        &self.slots[(self.origin_idx + idx).rem_euclid(self.total_len()) as usize]
    }
    pub fn _get_by_idx_mut(&mut self, idx: i32) -> &mut T {
        let idx = (self.origin_idx + idx).rem_euclid(self.total_len()) as usize;
        &mut self.slots[idx]
    }
    pub fn _sub_get(&self, pos: [i32; 3]) -> &T {
        &self.slots[(self.origin_idx
            + pos[0]
            + pos[1] * self.size()
            + pos[2] * (self.size() * self.size()))
        .rem_euclid(self.total_len()) as usize]
    }
    pub fn sub_get_mut(&mut self, pos: [i32; 3]) -> &mut T {
        let size = self.size();
        let total_len = self.total_len();
        &mut self.slots[(self.origin_idx + pos[0] + pos[1] * size + pos[2] * (size * size))
            .rem_euclid(total_len) as usize]
    }

    pub fn sub_idx_to_pos(&self, idx: i32) -> ChunkPos {
        let x = idx % self.size();
        let y = idx / self.size() % self.size();
        let z = idx / self.size() / self.size();
        self.corner_pos.offset(x, y, z)
    }
}

pub trait GridSlot {
    fn new() -> Self;
    fn reset(&mut self);
}

pub struct ChunkSlot {
    pub generating: AtomicCell<bool>,
    data: Option<ChunkBox>,
}
impl GridSlot for ChunkSlot {
    fn new() -> Self {
        Self {
            generating: false.into(),
            data: None,
        }
    }

    fn reset(&mut self) {
        self.generating = false.into();
        self.data = None;
    }
}
impl ChunkSlot {
    pub fn present(&self) -> bool {
        self.data.is_some()
    }

    pub fn as_ref(&self) -> Option<&Chunk> {
        self.data.as_deref()
    }

    pub fn _as_mut(&mut self) -> Option<&mut Chunk> {
        self.data.as_deref_mut()
    }
}

impl GridSlot for bool {
    fn new() -> bool {
        false
    }

    fn reset(&mut self) {
        *self = false;
    }
}

pub struct ChunkStorage {
    chunks: GridKeeper<ChunkSlot>,
    pub priority: Vec<i32>,
    pub center_hint: AtomicCell<ChunkPos>,
}
impl ChunkStorage {
    pub fn new() -> Self {
        //Calculate which chunks are more important based on distance to center
        let size = 32;
        let total = (size * size * size) as usize;
        let radius = 15.;
        let max_dist_sq = radius * radius;
        let mut distances = Vec::<(Sortf32, u32, i32)>::with_capacity(total);
        let mut rng = FastRng::seed_from_u64(0xadbcefabbd);
        let mut idx = 0;
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let pos = Vec3::new(x as f32, y as f32, z as f32);
                    let center = (size / 2) as f32 - 0.5;
                    let delta = pos - Vec3::broadcast(center);
                    let dist_sq = delta.mag_sq();
                    if dist_sq < max_dist_sq {
                        distances.push((Sortf32(dist_sq), rng.gen(), idx as i32));
                    }
                    idx += 1;
                }
            }
        }
        distances.sort_unstable();
        let priority: Vec<_> = distances.into_iter().map(|(_, _, idx)| idx).collect();
        Self {
            chunks: GridKeeper::new(size, ChunkPos([0, 0, 0])),
            priority,
            center_hint: AtomicCell::new(ChunkPos([0, 0, 0])),
        }
    }

    pub fn chunk_at(&self, pos: ChunkPos) -> Option<&Chunk> {
        self.chunks.get(pos).map(|opt| opt.as_ref()).unwrap_or(None)
    }

    pub fn _chunk_at_mut(&mut self, pos: ChunkPos) -> Option<&mut Chunk> {
        self.chunks
            .get_mut(pos)
            .map(|opt| opt._as_mut())
            .unwrap_or(None)
    }

    pub fn _chunk_slot_at(&self, pos: ChunkPos) -> Option<&ChunkSlot> {
        self.chunks.get(pos)
    }
    pub fn chunk_slot_at_mut(&mut self, pos: ChunkPos) -> Option<&mut ChunkSlot> {
        self.chunks.get_mut(pos)
    }

    pub fn block_at(&self, pos: BlockPos) -> Option<BlockData> {
        self.chunk_at(pos.to_chunk()).map(|chunk| {
            chunk.sub_get([
                pos[0].rem_euclid(CHUNK_SIZE),
                pos[1].rem_euclid(CHUNK_SIZE),
                pos[2].rem_euclid(CHUNK_SIZE),
            ])
        })
    }

    pub fn _block_at_mut(&mut self, pos: BlockPos) -> Option<&mut BlockData> {
        self._chunk_at_mut(pos.to_chunk()).map(|chunk| {
            chunk._sub_get_mut([
                pos[0].rem_euclid(CHUNK_SIZE),
                pos[1].rem_euclid(CHUNK_SIZE),
                pos[2].rem_euclid(CHUNK_SIZE),
            ])
        })
    }
}
impl ops::Deref for ChunkStorage {
    type Target = GridKeeper<ChunkSlot>;
    fn deref(&self) -> &Self::Target {
        &self.chunks
    }
}
impl ops::DerefMut for ChunkStorage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.chunks
    }
}
impl Drop for ChunkStorage {
    fn drop(&mut self) {
        self.chunks = GridKeeper::new(2, ChunkPos([0, 0, 0]));
    }
}

pub(crate) struct BookKeepHandle {
    pub generated_send: Sender<(ChunkPos, ChunkBox)>,
    close: Arc<AtomicCell<bool>>,
    thread: Option<JoinHandle<()>>,
}
impl BookKeepHandle {
    pub fn new(chunks: Arc<RwLock<ChunkStorage>>) -> Self {
        let close = Arc::new(AtomicCell::new(false));
        let (gen_send, gen_recv) = channel::bounded(64);
        let thread = {
            let close = close.clone();
            thread::spawn(move || {
                run_bookkeep(BookKeepState {
                    chunks,
                    close,
                    generated: gen_recv,
                });
            })
        };
        BookKeepHandle {
            generated_send: gen_send,
            close,
            thread: Some(thread),
        }
    }
}
impl Drop for BookKeepHandle {
    fn drop(&mut self) {
        self.close.store(true);
        if let Some(join) = self.thread.take() {
            join.thread().unpark();
            join.join().unwrap();
        }
    }
}

struct BookKeepState {
    close: Arc<AtomicCell<bool>>,
    chunks: Arc<RwLock<ChunkStorage>>,
    generated: Receiver<(ChunkPos, ChunkBox)>,
}

fn run_bookkeep(state: BookKeepState) {
    while !state.close.load() {
        //eprintln!("bookkeeping");
        let mut chunks = state.chunks.write();
        //Update center pos
        let new_center = chunks.center_hint.load();
        chunks.set_center(new_center);
        //Add queued generated chunks
        for (pos, chunk) in state.generated.try_iter() {
            if let Some(slot) = chunks.chunk_slot_at_mut(pos) {
                slot.data = Some(chunk);
                slot.generating = true.into();
            }
            //let _ = state.chunk_reuse.send(chunk);
        }
        //Wait some time or until the thread is unparked
        drop(chunks);
        if state.close.load() {
            break;
        }
        thread::park_timeout(Duration::from_millis(50));
    }
}

struct Terrain {
    mesher: MesherHandle,
    _bookkeeper: BookKeepHandle,
    _generator: GeneratorHandle,
    state: Rc<State>,
    chunks: Arc<RwLock<ChunkStorage>>,
    meshes: MeshKeeper,
}
impl Terrain {
    fn new(state: &Rc<State>, gen_cfg: crate::gen::GenConfig) -> Terrain {
        let chunks = Arc::new(RwLock::new(ChunkStorage::new()));
        let bookkeeper = BookKeepHandle::new(chunks.clone());
        Terrain {
            state: state.clone(),
            meshes: MeshKeeper::new(0., ChunkPos([0, 0, 0])),
            mesher: MesherHandle::new(state, chunks.clone()),
            _generator: GeneratorHandle::new(gen_cfg, chunks.clone(), &bookkeeper),
            _bookkeeper: bookkeeper,
            chunks,
        }
    }

    fn set_view_radius(&mut self, radius: f32) {
        self.meshes = MeshKeeper::new(radius / CHUNK_SIZE as f32, self.meshes.center());
    }

    fn hint_center(&mut self, center: BlockPos) {
        //eprintln!("hinting center");
        //Adjust center
        let center = center
            .offset(CHUNK_SIZE / 2, CHUNK_SIZE / 2, CHUNK_SIZE / 2)
            .to_chunk();
        self.chunks.read().center_hint.store(center);
        self.meshes.set_center(center);
        //Receive buffers from mesher thread
        for (pos, buf_pkg) in self.mesher.recv_bufs.try_iter() {
            let mesh = match buf_pkg {
                None => {
                    //A chunk with no geometry (ie. full air or full solid)
                    ChunkMesh { buf: None }
                }
                Some((vert, idx)) => unsafe {
                    //Deconstructed buffers
                    ChunkMesh {
                        buf: Some(Buffer3d {
                            vertex: VertexBuffer::from_raw_package(&self.state.display, vert),
                            index: IndexBuffer::from_raw_package(&self.state.display, idx),
                        }),
                    }
                },
            };
            if let Some(slot) = self.meshes.get_mut(pos) {
                slot.mesh = Some(mesh);
            }
        }
    }

    fn block_at(&self, pos: BlockPos) -> Option<BlockData> {
        self.chunks.read().block_at(pos)
    }
}

/// Keeps track of chunk meshes in an efficient grid structure.
struct MeshKeeper {
    pub radius: f32,
    pub meshes: GridKeeper<ChunkMeshSlot>,
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
        let meshes = GridKeeper::new(size, center);
        //Precalculate the render order, from closest to farthest
        //(Why? because rendering the nearest first makes better use of the depth buffer)
        let total = (1 << (size_log2 * 3)) as usize;
        let max_dist_sq = radius * radius;
        let mut distances = Vec::<(Sortf32, u32, i32)>::with_capacity(total);
        let mut rng = FastRng::seed_from_u64(0xadbcefabbd);
        let mut idx = 0;
        for z in 0..size {
            for y in 0..size {
                for x in 0..size {
                    let pos = Vec3::new(x as f32, y as f32, z as f32);
                    let center = (size / 2) as f32 - 0.5;
                    let delta = pos - Vec3::broadcast(center);
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
            radius: radius - 1.,
            meshes,
            render_order,
        }
    }
}
impl ops::Deref for MeshKeeper {
    type Target = GridKeeper<ChunkMeshSlot>;
    fn deref(&self) -> &Self::Target {
        &self.meshes
    }
}
impl ops::DerefMut for MeshKeeper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.meshes
    }
}

#[derive(Default)]
struct ChunkMeshSlot {
    mesh: Option<ChunkMesh>,
}
impl GridSlot for ChunkMeshSlot {
    fn new() -> Self {
        default()
    }
    fn reset(&mut self) {
        self.mesh = None;
    }
}

struct ChunkMesh {
    buf: Option<Buffer3d>,
}

#[derive(Clone)]
pub struct Chunk {
    pub blocks: [BlockData; (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize],
}
impl Chunk {
    pub fn _new() -> Self {
        Chunk {
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
    fn hint_center(lua, this, (x, y, z): (i32, i32, i32)) {
        this.rc.borrow_mut().hint_center(BlockPos([x, y, z]));
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

    fn visible_radius(lua, this, (x, y, z): (f64, f64, f64)) {
        let terrain = this.rc.borrow();
        for &idx in terrain.meshes.render_order.iter() {
            if terrain.meshes.get_by_idx(idx).mesh.is_none() {
                // This mesh is not visible
                let block = terrain.meshes.sub_idx_to_pos(idx).to_block_center();
                let dx = block[0] as f64 - x;
                let dy = block[1] as f64 - y;
                let dz = block[2] as f64 - z;
                let delta = Vec3::new(dx as f32, dy as f32, dz as f32);
                let radius = delta.mag() - (CHUNK_SIZE as f32 / 2.) * 3f32.cbrt();
                return Ok(radius.max(0.));
            }
        }
        terrain.meshes.radius * CHUNK_SIZE as f32
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

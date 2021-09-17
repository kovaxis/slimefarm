use std::f64::INFINITY;

use crate::{chunkmesh::MesherHandle, prelude::*};
use common::terrain::{GridKeeper, GridSlot};

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
    pub fn as_ref(&self) -> Option<ChunkRef> {
        self.data.as_ref().map(|chunk| chunk.as_ref())
    }
}

pub fn by_dist_up_to(radius: f32) -> impl FnMut(Vec3) -> Option<f32> {
    let max_sq = radius * radius;
    move |delta| {
        let dist_sq = delta.mag_sq();
        if dist_sq <= max_sq {
            Some(dist_sq)
        } else {
            None
        }
    }
}

pub fn gen_priorities(size: i32, mut cost: impl FnMut(Vec3) -> Option<f32>) -> Vec<i32> {
    let mut chunks = Vec::<(Sortf32, u32, i32)>::with_capacity((size * size * size) as usize);
    let mut rng = FastRng::seed_from_u64(0xadbcefabbd);
    let mut idx = 0;
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let pos = Vec3::new(x as f32, y as f32, z as f32);
                let center = (size / 2) as f32 - 0.5;
                let delta = pos - Vec3::broadcast(center);
                if let Some(cost) = cost(delta) {
                    chunks.push((Sortf32(cost), rng.gen(), idx as i32));
                }
                idx += 1;
            }
        }
    }
    chunks.sort_unstable();
    chunks.into_iter().map(|(_, _, idx)| idx).collect()
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
        let radius = 15.;
        Self {
            chunks: GridKeeper::new(size, ChunkPos([0, 0, 0])),
            priority: gen_priorities(size, by_dist_up_to(radius)),
            center_hint: AtomicCell::new(ChunkPos([0, 0, 0])),
        }
    }

    pub fn chunk_at(&self, pos: ChunkPos) -> Option<ChunkRef> {
        self.chunks.get(pos).map(|opt| opt.as_ref()).unwrap_or(None)
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

pub(crate) struct BookKeepHandle {
    pub generated_send: Sender<(ChunkPos, ChunkBox)>,
    close: Arc<AtomicCell<bool>>,
    thread: Option<JoinHandle<()>>,
}
impl BookKeepHandle {
    pub fn new(chunks: Arc<RwLock<ChunkStorage>>) -> Self {
        let close = Arc::new(AtomicCell::new(false));
        let (gen_send, gen_recv) = channel::bounded(1024);
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
    fn new(state: &Rc<State>, gen_cfg: &[u8]) -> Result<Terrain> {
        let chunks = Arc::new(RwLock::new(ChunkStorage::new()));
        let bookkeeper = BookKeepHandle::new(chunks.clone());
        let (chunkfill, blockcolor) = worldgen::new_generator(gen_cfg)?.split();
        Ok(Terrain {
            state: state.clone(),
            meshes: MeshKeeper::new(0., ChunkPos([0, 0, 0])),
            mesher: MesherHandle::new(state, blockcolor, chunks.clone()),
            _generator: unsafe {
                GeneratorHandle::new(gen_cfg, chunkfill, chunks.clone(), &bookkeeper)?
            },
            _bookkeeper: bookkeeper,
            chunks,
        })
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
        //Allocate space for meshes
        let meshes = GridKeeper::with_radius(radius, center);
        let size = meshes.size();
        //Precalculate the render order, from closest to farthest
        //(Why? because rendering the nearest first makes better use of the depth buffer)
        let render_order = gen_priorities(size, by_dist_up_to(radius));
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

struct ChunkMesh {
    buf: Option<Buffer3d>,
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
    pub(crate) fn new(state: &Rc<State>, gen_cfg: &[u8]) -> Result<TerrainRef> {
        Ok(TerrainRef {
            rc: AssertSync(Rc::new(RefCell::new(Terrain::new(state, gen_cfg)?))),
        })
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
    }

    fn chunk_gen_time(lua, this, ()) {
        this.rc.borrow()._generator.avg_gen_time.load()
    }
    fn chunk_mesh_time(lua, this, ()) {
        this.rc.borrow().mesher.avg_mesh_time.load()
    }
    fn chunk_mesh_upload_time(lua, this, ()) {
        this.rc.borrow().mesher.avg_upload_time.load()
    }
}

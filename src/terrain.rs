use std::f64::INFINITY;

use crate::{chunkmesh::MesherHandle, prelude::*};
use common::terrain::{GridKeeper, GridSlot};

pub struct ChunkSlot {
    pub generating: AtomicCell<bool>,
    data: Option<ChunkArc>,
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

    pub fn as_arc_ref(&self) -> Option<ChunkArc> {
        self.data.as_ref().cloned()
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

    pub fn chunk_slot_at(&self, pos: ChunkPos) -> Option<&ChunkSlot> {
        self.chunks.get(pos)
    }
    pub fn chunk_slot_at_mut(&mut self, pos: ChunkPos) -> Option<&mut ChunkSlot> {
        self.chunks.get_mut(pos)
    }

    pub fn block_at(&self, pos: BlockPos) -> Option<BlockData> {
        self.chunk_at(pos >> CHUNK_BITS)
            .map(|chunk| chunk.sub_get(pos.lowbits(CHUNK_BITS)))
    }

    pub fn portal_at(&self, pos: BlockPos, axis: usize) -> Option<&PortalData> {
        self.chunk_at(pos >> CHUNK_BITS)
            .and_then(|chunk| chunk.sub_portal_at(pos.lowbits(CHUNK_BITS), axis))
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
    pub generated_send: Sender<(ChunkPos, ChunkArc)>,
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
    generated: Receiver<(ChunkPos, ChunkArc)>,
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

struct PortalPlane {
    axis: usize,
    coord: i32,
    jump: Int3,
}

pub(crate) struct Terrain {
    pub _bookkeeper: BookKeepHandle,
    pub style: StyleTable,
    pub mesher: MesherHandle,
    pub generator: GeneratorHandle,
    pub state: Rc<State>,
    pub chunks: Arc<RwLock<ChunkStorage>>,
    pub meshes: MeshKeeper,
    tmp_bufs: Vec<RefCell<DynBuffer3d>>,
    tmp_colbuf: RefCell<Vec<PortalPlane>>,
}
impl Terrain {
    pub fn new(state: &Rc<State>, gen_cfg: &[u8]) -> Result<Terrain> {
        let chunks = Arc::new(RwLock::new(ChunkStorage::new()));
        let bookkeeper = BookKeepHandle::new(chunks.clone());
        let generator = GeneratorHandle::new(gen_cfg, &state.global, chunks.clone(), &bookkeeper)?;
        let tex = generator.take_block_textures()?;
        Ok(Terrain {
            style: StyleTable::new(&tex),
            state: state.clone(),
            meshes: MeshKeeper::new(0., ChunkPos([0, 0, 0])),
            mesher: MesherHandle::new(state, chunks.clone(), tex),
            tmp_bufs: {
                let max = 32;
                (0..max)
                    .map(|_| RefCell::new(DynBuffer3d::new(state)))
                    .collect()
            },
            tmp_colbuf: default(),
            generator: generator,
            _bookkeeper: bookkeeper,
            chunks,
        })
    }

    pub fn set_view_radius(&mut self, radius: f32) {
        self.meshes = MeshKeeper::new(radius / CHUNK_SIZE as f32, self.meshes.center());
        self.generator
            .reshape(self.meshes.size(), self.meshes.center());
    }

    pub fn hint_center(&mut self, center: BlockPos) {
        //Adjust center
        let center = (center + Int3::splat(CHUNK_SIZE / 2)) >> CHUNK_BITS;
        self.chunks.read().center_hint.store(center);
        self.generator.reshape(self.meshes.size(), center);
        self.meshes.set_center(center);

        //Receive buffers from mesher thread
        for buf_pkg in self.mesher.recv_bufs.try_iter() {
            let mesh = ChunkMesh {
                mesh: buf_pkg.mesh,
                buf: match buf_pkg.buf {
                    None => {
                        // A chunk with no geometry (ie. full air or full solid)
                        None
                    }
                    Some((vert, idx)) => unsafe {
                        // Deconstructed buffers
                        // Construct them back
                        Some(Buffer3d {
                            vertex: VertexBuffer::from_raw_package(&self.state.display, vert),
                            index: IndexBuffer::from_raw_package(&self.state.display, idx),
                        })
                    },
                },
                portals: buf_pkg.portals,
            };
            if let Some(slot) = self.meshes.get_mut(buf_pkg.pos) {
                slot.mesh = Some(mesh);
            }
        }
    }

    pub fn block_at(&self, pos: BlockPos) -> Option<BlockData> {
        self.chunks.read().block_at(pos)
    }

    /// Calculate the clipping planes of a framequad given by `fq`.
    /// The framequad must be properly behind the near clipping plane.
    /// This function receives and outputs world coordinates.
    /// If the framequad has strange coordinates that intersect the near plane, the 4 side clipping
    /// planes may be disabled.
    pub fn calc_clip_planes(mvp: &Mat4, fq: &[Vec3; 4]) -> [Vec4; 5] {
        let inv_mvp = mvp.inversed();
        let f000 = fq[0];
        let f100 = fq[1];
        let f110 = fq[2];
        let f010 = fq[3];

        let f100_clip = *mvp * f100.into_homogeneous_point();
        let mut f101_ndc = f100_clip.normalized_homogeneous_point().truncated();
        f101_ndc.z = 1.;
        let f101 = inv_mvp.transform_point3(f101_ndc);

        let f010_clip = *mvp * f010.into_homogeneous_point();
        let mut f011_ndc = f010_clip.normalized_homogeneous_point().truncated();
        f011_ndc.z = 1.;
        let f011 = inv_mvp.transform_point3(f011_ndc);

        let calc_plane = |p: [Vec3; 3]| {
            let n = (p[1] - p[0]).cross(p[2] - p[0]).normalized();
            Vec4::new(n.x, n.y, n.z, -p[0].dot(n))
        };

        if f100_clip.w > 0. && f010_clip.w > 0. {
            [
                calc_plane([f000, f010, f100]), // near plane
                calc_plane([f000, f011, f010]), // left plane
                calc_plane([f100, f110, f101]), // right plane
                calc_plane([f000, f100, f101]), // bottom plane
                calc_plane([f010, f011, f110]), // top plane
            ]
        } else {
            // Some coordinates fall outside the proper range
            // Give up on the 4 side clipping planes
            let disabled = Vec4::new(0., 0., 0., 1.);
            [
                calc_plane([f000, f010, f100]), // near plane
                disabled,
                disabled,
                disabled,
                disabled,
            ]
        }
    }

    pub fn draw(
        &self,
        shader: &Program,
        uniforms: crate::lua::gfx::UniformStorage,
        offset_uniform: u32,
        origin: [f64; 3],
        params: &DrawParameters,
        mvp: Mat4,
        framequad: [Vec3; 4],
        stencil: u8,
        subdraw: &dyn Fn(&[f64; 3], &[Vec3; 4], u8) -> Result<()>,
    ) -> Result<()> {
        let frame = &self.state.frame;

        // Calculate the frustum planes
        let clip_planes = Self::calc_clip_planes(&mvp, &framequad);

        // Get the position relative to the player and the chunk mesh of a chunk by its index
        // within the `GridKeeper`.
        // Also, apply some culling: If the chunk is outside the camera frustum or the chunk has no
        // mesh then just `continue` away.
        macro_rules! get_chunk {
            ($idx:expr) => {{
                let idx = $idx;
                let buf = match self.meshes.get_by_idx(idx).mesh.as_ref() {
                    Some(buf) => buf,
                    None => continue,
                };

                // Figure out chunk location
                let pos = (self.meshes.sub_idx_to_pos(idx) << CHUNK_BITS);
                let offset = Vec3::new(
                    (pos.x as f64 - origin[0]) as f32,
                    (pos.y as f64 - origin[1]) as f32,
                    (pos.z as f64 - origin[2]) as f32,
                );
                let center = offset + Vec3::broadcast(CHUNK_SIZE as f32 / 2.);

                // Cull chunks that are outside the worldview
                if clip_planes.iter().any(|&p| {
                    p.dot(center.into_homogeneous_point()) < -3f32.sqrt() / 2. * CHUNK_SIZE as f32
                }) {
                    // Cull chunk, it is too far away from the frustum
                    continue;
                }

                (offset, buf)
            }};
        }

        // Draw all visible chunks
        {
            let mut frame = frame.borrow_mut();
            let mut drawn = 0;
            let mut bytes_v = 0;
            let mut bytes_i = 0;
            for &idx in self.meshes.render_order.iter() {
                let (offset, chunk) = get_chunk!(idx);

                if let Some(buf) = &chunk.buf {
                    uniforms
                        .vars
                        .borrow_mut()
                        .get_mut(offset_uniform as usize)
                        .ok_or(anyhow!("offset uniform out of range"))?
                        .1 = crate::lua::gfx::UniformVal::Vec3(offset.into());
                    frame.draw(&buf.vertex, &buf.index, &shader, &uniforms, params)?;
                    drawn += 1;
                    bytes_v += chunk.mesh.vertices.len() * mem::size_of::<SimpleVertex>();
                    bytes_i += chunk.mesh.indices.len() * mem::size_of::<VertIdx>();
                }
            }
            if false {
                println!(
                    "drew {} chunks in {}KB of vertices and {}KB of indices",
                    drawn,
                    bytes_v / 1024,
                    bytes_i / 1024
                );
            }
        }

        // Draw portals
        for &idx in self.meshes.render_order.iter() {
            let chunk = match self.meshes.get_by_idx(idx).mesh.as_ref() {
                Some(chunk) => chunk,
                None => continue,
            };
            if !chunk.portals.is_empty() {
                let chunk_pos = self.meshes.sub_idx_to_pos(idx) << CHUNK_BITS;
                let chunk_offset = Vec3::new(
                    (chunk_pos.x as f64 - origin[0]) as f32,
                    (chunk_pos.y as f64 - origin[1]) as f32,
                    (chunk_pos.z as f64 - origin[2]) as f32,
                );
                'portal: for portal in chunk.portals.iter() {
                    // Make sure portal is visible
                    let subfq_world = [
                        chunk_offset + portal.bounds[0],
                        chunk_offset + portal.bounds[1],
                        chunk_offset + portal.bounds[2],
                        chunk_offset + portal.bounds[3],
                    ];
                    for clip_plane in &clip_planes {
                        if subfq_world
                            .iter()
                            .all(|p| p.into_homogeneous_point().dot(*clip_plane) <= 0.01)
                        {
                            // Portal is outside the parent frame view frustum
                            continue 'portal;
                        }
                    }
                    let subfq_clip = [
                        mvp * subfq_world[0].into_homogeneous_point(),
                        mvp * subfq_world[1].into_homogeneous_point(),
                        mvp * subfq_world[2].into_homogeneous_point(),
                        mvp * subfq_world[3].into_homogeneous_point(),
                    ];
                    //let proper = subfq_clip.iter().all(|p| -p.w < p.z && p.z < p.w);
                    let proper = subfq_clip.iter().all(|p| p.w > 0.);
                    if proper {
                        let xy = |v4: Vec4| Vec2::new(v4.x, v4.y) * v4.w.recip();
                        let subfq_2d = [
                            xy(subfq_clip[0]),
                            xy(subfq_clip[1]),
                            xy(subfq_clip[2]),
                            xy(subfq_clip[3]),
                        ];
                        if (subfq_2d[1] - subfq_2d[0])
                            .wedge(subfq_2d[3] - subfq_2d[0])
                            .xy
                            <= 0.
                        {
                            // Portal is backwards
                            continue 'portal;
                        }
                    }
                    // Get the portal shape into a buffer
                    let mut mesh = portal.mesh.take();
                    let mut portal_buf = self.tmp_bufs[stencil as usize].borrow_mut();
                    portal_buf.write(&self.state, &mut mesh.vertices, &mut mesh.indices);
                    portal.mesh.replace(mesh);
                    // Step 1: Figure out which parts of the portal are visible by drawing the
                    // portal shape (with depth testing) into the stencil buffer.
                    // TODO: Modify the draw parameters instead of creating a new one from scratch,
                    // using then the clip planes.
                    // OPTIMIZE: Use sample queries and conditional rendering to speed up occluded
                    // portals.
                    uniforms
                        .vars
                        .borrow_mut()
                        .get_mut(offset_uniform as usize)
                        .ok_or(anyhow!("offset uniform out of range"))?
                        .1 = crate::lua::gfx::UniformVal::Vec3(chunk_offset.into());
                    let (vert_buf, idx_buf) = portal_buf.bufs();
                    frame.borrow_mut().draw(
                        vert_buf,
                        idx_buf,
                        shader,
                        &uniforms,
                        &DrawParameters {
                            depth: glium::draw_parameters::Depth {
                                test: glium::draw_parameters::DepthTest::IfLess,
                                clamp: if proper {
                                    glium::draw_parameters::DepthClamp::NoClamp
                                } else {
                                    glium::draw_parameters::DepthClamp::Clamp
                                },
                                // It's not necessary to write to the depth buffer, since the next
                                // step will immediately reset the depth buffer to the maximum
                                // value.
                                write: false,
                                ..default()
                            },
                            stencil: glium::draw_parameters::Stencil {
                                test_counter_clockwise:
                                    glium::draw_parameters::StencilTest::IfEqual { mask: !0 },
                                reference_value_counter_clockwise: stencil as i32,
                                depth_pass_operation_counter_clockwise:
                                    glium::draw_parameters::StencilOperation::IncrementWrap,
                                ..default()
                            },
                            color_mask: (false, false, false, false),
                            polygon_offset: glium::draw_parameters::PolygonOffset {
                                factor: -1.,
                                units: -2.,
                                fill: true,
                                ..default()
                            },
                            ..default()
                        },
                    )?;
                    // Step 2: Draw what's on the other end of the portal.
                    // This entails drawing the skybox again, resetting the depth buffer.
                    let suborigin = [
                        origin[0] + portal.jump[0],
                        origin[1] + portal.jump[1],
                        origin[2] + portal.jump[2],
                    ];
                    subdraw(&suborigin, &subfq_world, stencil + 1)?;
                    // Step 3: Artificially replace the depth value for the portal shape.
                    // This allows the portal to occlude objects that are behind it and be occluded
                    // by objects in front.
                    // Also reset the stencil buffer to its original value.
                    uniforms
                        .vars
                        .borrow_mut()
                        .get_mut(offset_uniform as usize)
                        .ok_or(anyhow!("offset uniform out of range"))?
                        .1 = crate::lua::gfx::UniformVal::Vec3(chunk_offset.into());
                    let (vert_buf, idx_buf) = portal_buf.bufs();
                    frame.borrow_mut().draw(
                        vert_buf,
                        idx_buf,
                        shader,
                        &uniforms,
                        &DrawParameters {
                            depth: glium::draw_parameters::Depth {
                                test: glium::draw_parameters::DepthTest::Overwrite,
                                clamp: if proper {
                                    glium::draw_parameters::DepthClamp::NoClamp
                                } else {
                                    glium::draw_parameters::DepthClamp::Clamp
                                },
                                write: true,
                                ..default()
                            },
                            stencil: glium::draw_parameters::Stencil {
                                test_counter_clockwise:
                                    glium::draw_parameters::StencilTest::IfEqual { mask: !0 },
                                reference_value_counter_clockwise: (stencil + 1) as i32,
                                depth_pass_operation_counter_clockwise:
                                    glium::draw_parameters::StencilOperation::DecrementWrap,
                                ..default()
                            },
                            color_mask: (false, false, false, false),
                            ..default()
                        },
                    )?;
                }
            }
        }

        Ok(())
    }

    const SAFE_GAP: f64 = 1. / 128.;

    /// Move an integer block coordinate by the given offset, crossing through portals but going
    /// through solid blocks.
    /// First moves on the X axis, then on the Y axis and then the Z axis.
    /// The order could affect which portals are crossed!
    pub(crate) fn offset_coords(&self, from: BlockPos, delta: Int3) -> BlockPos {
        let mut cur = from;
        for i in 0..3 {
            cur = self.blockcast_ghost(cur, i, delta[i]);
        }
        cur
    }

    /// Move an integer block coordinate by the given amount of blocks in the given axis, crossing
    /// through portals but going through blocks.
    pub(crate) fn blockcast_ghost(&self, from: BlockPos, axis: usize, delta: i32) -> BlockPos {
        let binary = (delta > 0) as i32;
        let dir = if delta > 0 { 1 } else { -1 };
        let mut cur = from;
        for _ in 0..delta * dir {
            cur[axis] += dir;
            if self
                .block_at(cur)
                .map(|b| b.is_portal(&self.style))
                .unwrap_or(false)
            {
                let mut ppos = cur;
                ppos[axis] += 1 - binary;
                if let Some(portal) = self.chunks.read().portal_at(ppos, axis) {
                    cur += portal.jump;
                }
            }
        }
        cur
    }

    /// Move an integer block coordinate by the given amount of blocks in the given axis, crossing
    /// through portals and stopping at the first solid block (returning the last clear block).
    pub(crate) fn blockcast(&self, from: BlockPos, axis: usize, delta: i32) -> BlockPos {
        let binary = (delta > 0) as i32;
        let dir = if delta > 0 { 1 } else { -1 };
        let mut cur = from;
        for _ in 0..delta * dir {
            cur[axis] += dir;
            match self
                .block_at(cur)
                .map(|b| self.style.lookup(b))
                .unwrap_or(BlockStyle::Solid)
            {
                BlockStyle::Clear => {}
                BlockStyle::Solid => {
                    cur[axis] -= dir;
                    return cur;
                }
                BlockStyle::Portal => {
                    let mut ppos = cur;
                    ppos[axis] += 1 - binary;
                    if let Some(portal) = self.chunks.read().portal_at(ppos, axis) {
                        cur += portal.jump;
                    }
                }
                BlockStyle::Custom => unimplemented!(),
            }
        }
        cur
    }

    /// Place a virtual point particle at `from` and move it `delta` units in the `axis` axis.
    /// If there are any blocks in the way, the particle stops its movement.
    /// If there are any portals in the way, the particle crosses them.
    pub(crate) fn raycast_aligned(&self, from: [f64; 3], axis: usize, delta: f64) -> [f64; 3] {
        let dir = if delta > 0. { 1. } else { -1. };
        let binary = (delta > 0.) as i32;
        let mut cur = from;
        let mut limit = cur[axis] + delta;
        let mut next_block = [cur[0].floor(), cur[1].floor(), cur[2].floor()];
        next_block[axis] = (cur[axis] * dir).ceil() * dir;
        while next_block[axis] * dir < limit * dir {
            let col_int = Int3::from_f64(next_block);
            let mut block = col_int;
            block[axis] += binary - 1;
            match self
                .block_at(block)
                .map(|b| self.style.lookup(b))
                .unwrap_or(BlockStyle::Solid)
            {
                BlockStyle::Clear => next_block[axis] += dir,
                BlockStyle::Solid => {
                    let mut out = cur;
                    out[axis] = next_block[axis] - Self::SAFE_GAP * dir;
                    return out;
                }
                BlockStyle::Portal => {
                    if let Some(portal) = self.chunks.read().portal_at(col_int, axis) {
                        for i in 0..3 {
                            cur[i] += portal.jump[i] as f64;
                            next_block[i] += portal.jump[i] as f64;
                        }
                        limit += portal.jump[axis] as f64;
                    } else {
                        // Should never happen with proper portals!
                        next_block[axis] += dir;
                    }
                }

                BlockStyle::Custom => unimplemented!(),
            }
        }
        let mut out = cur;
        out[axis] = limit;
        out
    }

    /// Place a virtual cuboid at `from`, of dimensions `size * 2` (`size` being the "radius"), and
    /// move it by `delta`, checking for collisions against blocks and portals.
    /// If `eager` is true, the process stops as soon as a block is hit.
    /// If `eager` is false, when a block is hit the delta component in the direction of the hit
    /// block is killed, advancing no more in that direction.
    pub(crate) fn boxcast(
        &self,
        from: [f64; 3],
        mut delta: [f64; 3],
        size: [f64; 3],
        eager: bool,
    ) -> [f64; 3] {
        // TODO: Check that everything works alright with portals when objects have integer
        // positions.

        if delta == [0.; 3] {
            return from;
        }

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
        let idir = [
            if delta[0] > 0. { 1 } else { -1 },
            if delta[1] > 0. { 1 } else { -1 },
            if delta[2] > 0. { 1 } else { -1 },
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

        // Keep track of the planes of the portals that have been crossed
        // This way, we can work in virtual coordinates and only transfer to real absolute
        // coordinates when looking up a block, portal or at the very end when translating the
        // final virtual coordinates back to absolute world coordinates
        let mut portalplanes = self.tmp_colbuf.borrow_mut();
        portalplanes.clear();
        let get_warp = |planes: &[PortalPlane], pos: Int3| {
            let mut jump = Int3::zero();
            for plane in planes.iter() {
                if pos[plane.axis] * idir[plane.axis] < plane.coord * idir[plane.axis] {
                    break;
                }
                jump = plane.jump;
            }
            jump
        };
        let add_portal = |planes: &mut Vec<PortalPlane>, blockpos: Int3, axis: usize| {
            let mut pos = blockpos;
            pos[axis] += 1 - binary[axis];
            let real_pos = pos + get_warp(&planes, pos);
            if let Some(portal) = self.chunks.read().portal_at(real_pos, axis) {
                let last_jump = planes
                    .last()
                    .map(|plane| plane.jump)
                    .unwrap_or(Int3::zero());
                planes.push(PortalPlane {
                    axis,
                    coord: blockpos[axis],
                    jump: last_jump + portal.jump,
                });
            }
        };

        // Prebuild any portal planes that are currently intersecting the box
        {
            let src = Int3::from_f64(from);
            let dst = Int3::from_f64(frontier);
            for axis in 0..3 {
                let delta = dst[axis] - src[axis];
                let mut cur = src;
                for _ in 0..delta * idir[axis] {
                    cur[axis] += idir[axis];
                    let real_pos = cur + get_warp(&portalplanes, cur);
                    if self
                        .block_at(real_pos)
                        .map(|b| b.is_portal(&self.style))
                        .unwrap_or(false)
                    {
                        add_portal(&mut portalplanes, cur, axis);
                    }
                }
            }
        }

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
            let mut style = BlockStyle::Custom as u8;
            for z in min_block.z..max_block.z {
                for y in min_block.y..max_block.y {
                    for x in min_block.x..max_block.x {
                        let mut pos = Int3::new([x, y, z]);
                        let warp = get_warp(&portalplanes, pos);
                        pos += warp;
                        let s = self
                            .block_at(pos)
                            .map(|b| self.style.lookup(b))
                            .unwrap_or(BlockStyle::Solid);
                        style = style.min(s as u8);
                    }
                }
            }
            let style = BlockStyle::from_raw(style);

            match style {
                BlockStyle::Clear => {}
                BlockStyle::Solid => {
                    //If there is collision, start ignoring this axis
                    if eager {
                        //Unless we're raycasting
                        //In this case, abort immediately
                        limit = frontier;
                        break;
                    }
                    limit[closest_axis] =
                        frontier[closest_axis] - Self::SAFE_GAP * dir[closest_axis];
                    delta[closest_axis] = 0.;
                    if delta == [0., 0., 0.] {
                        break;
                    }
                }
                BlockStyle::Portal => {
                    add_portal(&mut portalplanes, min_block, closest_axis);
                }
                BlockStyle::Custom => unimplemented!(),
            }
        }

        let pos = [
            limit[0] - size[0] * dir[0],
            limit[1] - size[1] * dir[1],
            limit[2] - size[2] * dir[2],
        ];
        let warp = get_warp(&portalplanes, Int3::from_f64(pos)).to_f64();
        [pos[0] + warp[0], pos[1] + warp[1], pos[2] + warp[2]]
    }
}

/// Keeps track of chunk meshes in an efficient grid structure.
pub(crate) struct MeshKeeper {
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
pub(crate) struct ChunkMeshSlot {
    pub mesh: Option<ChunkMesh>,
}

pub(crate) struct ChunkMesh {
    pub mesh: Mesh,
    pub buf: Option<Buffer3d>,
    pub portals: Vec<PortalMesh>,
}

pub(crate) struct PortalMesh {
    pub mesh: Cell<Mesh>,
    pub bounds: [Vec3; 4],
    pub jump: [f64; 3],
}

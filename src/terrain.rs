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

pub(crate) struct Terrain {
    pub _bookkeeper: BookKeepHandle,
    pub solid: SolidTable,
    pub mesher: MesherHandle,
    pub generator: GeneratorHandle,
    pub state: Rc<State>,
    pub chunks: Arc<RwLock<ChunkStorage>>,
    pub tmp_bufs: Vec<RefCell<DynBuffer3d>>,
    pub meshes: MeshKeeper,
}
impl Terrain {
    pub fn new(state: &Rc<State>, gen_cfg: &[u8]) -> Result<Terrain> {
        let chunks = Arc::new(RwLock::new(ChunkStorage::new()));
        let bookkeeper = BookKeepHandle::new(chunks.clone());
        let generator = GeneratorHandle::new(gen_cfg, &state.global, chunks.clone(), &bookkeeper)?;
        let tex = generator.take_block_textures()?;
        Ok(Terrain {
            solid: SolidTable::new(&tex),
            state: state.clone(),
            meshes: MeshKeeper::new(0., ChunkPos([0, 0, 0])),
            mesher: MesherHandle::new(state, chunks.clone(), tex),
            tmp_bufs: {
                let max = 32;
                (0..max)
                    .map(|_| RefCell::new(DynBuffer3d::new(state)))
                    .collect()
            },
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
    /// The framequad is given in NDC, while the clipping planes are given in world coordinates.
    pub fn calc_clip_planes(mvp: &Mat4, fq: &[Vec3; 4]) -> [Vec4; 5] {
        let inv_mvp = mvp.inversed();
        let f000 = inv_mvp.transform_point3(fq[0]);
        let f100 = inv_mvp.transform_point3(fq[1]);
        let f110 = inv_mvp.transform_point3(fq[2]);
        let f010 = inv_mvp.transform_point3(fq[3]);
        let mut f101 = fq[1];
        f101.z = 1.;
        let f101 = inv_mvp.transform_point3(f101);
        let mut f011 = fq[3];
        f011.z = 1.;
        let f011 = inv_mvp.transform_point3(f011);

        let calc_plane = |p: [Vec3; 3]| {
            let n = (p[1] - p[0]).cross(p[2] - p[0]).normalized();
            Vec4::new(n.x, n.y, n.z, -p[0].dot(n))
        };
        [
            calc_plane([f000, f010, f100]), // near plane
            calc_plane([f000, f011, f010]), // left plane
            calc_plane([f100, f110, f101]), // right plane
            calc_plane([f000, f100, f101]), // bottom plane
            calc_plane([f010, f011, f110]), // top plane
        ]
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
                            .all(|p| p.into_homogeneous_point().dot(*clip_plane) <= 0.)
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
                    let subfq = if subfq_clip.iter().all(|p| -p.w < p.z && p.z < p.w) {
                        [
                            subfq_clip[0].truncated() * subfq_clip[0].w.recip(),
                            subfq_clip[1].truncated() * subfq_clip[1].w.recip(),
                            subfq_clip[2].truncated() * subfq_clip[2].w.recip(),
                            subfq_clip[3].truncated() * subfq_clip[3].w.recip(),
                        ]
                    } else {
                        // Give up and simply use the entire screen as the clipping framequad
                        // OPTIMIZE: Use a more accurate quad
                        [
                            Vec3::new(-1., -1., -1.),
                            Vec3::new(1., -1., -1.),
                            Vec3::new(1., 1., -1.),
                            Vec3::new(-1., 1., -1.),
                        ]
                    };
                    let xy = |v4: Vec3| Vec2::new(v4.x, v4.y);
                    if (xy(subfq[1]) - xy(subfq[0]))
                        .wedge(xy(subfq[3]) - xy(subfq[0]))
                        .xy
                        <= 0.
                    {
                        // Portal is backwards
                        continue 'portal;
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
                    subdraw(&suborigin, &subfq, stencil + 1)?;
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

pub(crate) fn check_collisions(
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

use std::f64::INFINITY;

use crate::{chunkmesh::MesherHandle, gen::GenArea, prelude::*};
use common::terrain::GridKeeper4;

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
    chunks: GridKeeper4<ChunkArc>,
    gc_interval: Duration,
    last_gc: Instant,
}
impl ChunkStorage {
    pub fn new() -> Self {
        Self {
            chunks: GridKeeper4::new(),
            gc_interval: Duration::from_millis(1894),
            last_gc: Instant::now(),
        }
    }

    pub fn chunk_at(&self, pos: ChunkPos) -> Option<ChunkRef> {
        self.chunks.get(pos).map(|cnk| cnk.as_ref())
    }

    pub fn chunk_arc_at(&self, pos: ChunkPos) -> Option<ChunkArc> {
        self.chunks.get(pos).map(|cnk| cnk.clone())
    }

    pub fn block_at(&self, pos: BlockPos) -> Option<BlockData> {
        self.chunk_at(pos.block_to_chunk())
            .map(|chunk| chunk.sub_get(pos.coords.lowbits(CHUNK_BITS)))
    }

    pub fn portal_at(&self, pos: BlockPos, axis: usize) -> Option<&PortalData> {
        self.chunk_at(pos.block_to_chunk())
            .and_then(|chunk| chunk.sub_portal_at(pos.coords.lowbits(CHUNK_BITS), axis))
    }

    /// Iterate over all chunk positions within a certain radius, without portal hopping or any
    /// funny business.
    pub fn iter_nearby_raw<F>(center: [f64; 3], range: f32, mut f: F) -> Result<()>
    where
        F: FnMut(Int3) -> Result<()>,
    {
        let center_block = Int3::from_f64(center);
        let center_chunk = center_block >> CHUNK_BITS;
        let mn = (center_block - Int3::splat(range.ceil() as i32)) >> CHUNK_BITS;
        let mx = (center_block + Int3::splat(range.ceil() as i32)) >> CHUNK_BITS;
        let range_chunk = range * (CHUNK_SIZE as f32).recip();
        let range_chunk_sq = (range_chunk * range_chunk).ceil() as i32;
        for z in mn.z..=mx.z {
            for y in mn.y..=mx.y {
                for x in mn.x..=mx.x {
                    let chunk_pos = Int3::new([x, y, z]);
                    let d = chunk_pos - center_chunk;
                    if d.mag_sq() <= range_chunk_sq {
                        f(chunk_pos)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Iterate through all the chunks nearby to a certain position, including those that are nearby
    /// through portals.
    /// Iterates over any given chunk at most once.
    /// Iterates over nonexisting chunk positions too.
    pub fn iter_nearby<F>(
        &self,
        seenbuf: &mut HashSet<ChunkPos>,
        center: BlockPos,
        range: f32,
        f: F,
    ) -> Result<()>
    where
        F: FnMut(ChunkPos, Option<ChunkRef>, (f32, i32)) -> Result<()>,
    {
        struct State<'a, F> {
            chunks: &'a ChunkStorage,
            seen: &'a mut HashSet<ChunkPos>,
            f: F,
        }
        fn explore<F>(s: &mut State<F>, center: BlockPos, range: f32, basedist: f32) -> Result<()>
        where
            F: FnMut(ChunkPos, Option<ChunkRef>, (f32, i32)) -> Result<()>,
        {
            let mn = (center.coords - Int3::splat(range.ceil() as i32)) >> CHUNK_BITS;
            let mx = (center.coords + Int3::splat(range.ceil() as i32)) >> CHUNK_BITS;
            let center_dist = center.coords - Int3::splat(CHUNK_SIZE / 2);
            let range_sq = (range * range).ceil() as i32;
            for z in mn.z..=mx.z {
                for y in mn.y..=mx.y {
                    for x in mn.x..=mx.x {
                        let chunk_pos = ChunkPos {
                            coords: Int3::new([x, y, z]),
                            dim: center.dim,
                        };
                        let pos_block = chunk_pos.coords << CHUNK_BITS;
                        let d2 = (pos_block - center_dist).mag_sq();
                        if d2 <= range_sq && s.seen.insert(chunk_pos) {
                            let chunk = s.chunks.chunk_at(chunk_pos);
                            (s.f)(chunk_pos, chunk, (basedist, d2))?;
                            if let Some(data) = chunk.and_then(|c| c.blocks()) {
                                for portal in data.portals() {
                                    let portal_center = portal.get_center();
                                    if !portal_center.is_within(Int3::splat(CHUNK_SIZE)) {
                                        continue;
                                    }
                                    let portal_center = pos_block + portal_center;
                                    let dist =
                                        ((portal_center - center.coords).mag_sq() as f32).sqrt();
                                    // TODO: Perhaps do something about large portals?
                                    // Approximating a portal as a point at its center only works
                                    // for smallish portals.
                                    explore(
                                        s,
                                        BlockPos {
                                            coords: portal_center + portal.jump,
                                            dim: portal.dim,
                                        },
                                        range - dist,
                                        basedist + dist,
                                    )?;
                                }
                            }
                        }
                    }
                }
            }
            Ok(())
        }
        seenbuf.clear();
        explore(
            &mut State {
                chunks: self,
                seen: seenbuf,
                f,
            },
            center,
            range,
            0.,
        )
    }

    pub fn maybe_gc(&mut self) {
        if self.last_gc.elapsed() > self.gc_interval {
            self.gc();
            self.last_gc = Instant::now();
        }
    }
}
impl ops::Deref for ChunkStorage {
    type Target = GridKeeper4<ChunkArc>;
    fn deref(&self) -> &Self::Target {
        &self.chunks
    }
}
impl ops::DerefMut for ChunkStorage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.chunks
    }
}

struct PortalPlane {
    axis: usize,
    coord: i32,
    jump: Int3,
    dim: u32,
}

/// Check whether a chunk is entirely within the given clip planes.
fn check_chunk_clip_planes(clip: &[Vec4; 5], chunk_center: Vec3) -> bool {
    let x = chunk_center.into_homogeneous_point();
    clip.iter()
        .all(|p| p.dot(x) >= -3f32.sqrt() / 2. * CHUNK_SIZE as f32)
}

pub(crate) struct Terrain {
    pub style: StyleTable,
    pub mesher: MesherHandle,
    pub generator: GeneratorHandle,
    pub state: Rc<State>,
    pub chunks: Arc<RwLock<ChunkStorage>>,
    pub meshes: MeshKeeper,
    pub view_radius: f32,
    pub gen_radius: f32,
    pub last_min_viewdist: f32,
    last_mesh_range: f32,
    last_mesh_finds: usize,
    tmp_bufs: Vec<RefCell<DynBuffer3d>>,
    tmp_colbuf: RefCell<Vec<PortalPlane>>,
    tmp_seenbuf: RefCell<HashSet<ChunkPos>>,
}
impl Terrain {
    pub fn new(state: &Rc<State>, gen_cfg: &[u8]) -> Result<Terrain> {
        let chunks = Arc::new(RwLock::new(ChunkStorage::new()));
        let generator = GeneratorHandle::new(gen_cfg, &state.global, chunks.clone())?;
        let tex = generator.take_block_textures()?;
        Ok(Terrain {
            style: StyleTable::new(&tex),
            state: state.clone(),
            meshes: MeshKeeper::new(),
            mesher: MesherHandle::new(state, chunks.clone(), tex),
            tmp_bufs: {
                let max = 32;
                (0..max)
                    .map(|_| RefCell::new(DynBuffer3d::new(state)))
                    .collect()
            },
            tmp_colbuf: default(),
            tmp_seenbuf: default(),
            last_min_viewdist: 0.,
            last_mesh_range: 0.,
            last_mesh_finds: 0,
            view_radius: 0.,
            gen_radius: 0.,
            generator: generator,
            chunks,
        })
    }

    pub fn set_view_radius(&mut self, view_radius: f32, gen_radius: f32) {
        self.view_radius = view_radius;
        self.gen_radius = gen_radius;
    }

    pub fn bookkeep(&mut self, center: BlockPos) {
        //Update generator genarea
        self.generator.set_gen_area(GenArea {
            center,
            gen_radius: self.gen_radius,
        });
        //Lock communication buffer with chunk mesher
        let mut request = self.mesher.request().lock();
        //Receive buffers from mesher thread
        let mut meshed_count = 0;
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
            self.meshes.insert(buf_pkg.pos, mesh);
            request.remove(buf_pkg.pos);
            meshed_count += 1;
        }
        //Request new meshes from mesher thread
        {
            request.unmark_all();
            let marked_goal = request.marked_goal();

            let sphere_factor = 4. / 3. * f32::PI;
            let last_range = self.last_mesh_range;
            let mut last_volume = last_range * last_range * last_range * sphere_factor;
            last_volume +=
                (marked_goal as i32 - request.marked_count() as i32 - self.last_mesh_finds as i32
                    + meshed_count) as f32
                    * (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as f32;
            self.last_mesh_range = (last_volume * sphere_factor.recip())
                .cbrt()
                .min(self.view_radius)
                .max(0.);

            let chunks = self.chunks.read();
            let mut mesh_finds = 0;
            let mut mindist = (self.last_mesh_range, 0., 0);
            chunks
                .iter_nearby(
                    &mut *self.tmp_seenbuf.borrow_mut(),
                    center,
                    self.last_mesh_range,
                    |pos, chunk, (df, d2)| {
                        if self.meshes.get(pos).is_none() {
                            if df == mindist.1 {
                                if d2 < mindist.2 {
                                    let dist = df + (d2 as f32).sqrt();
                                    mindist = (dist, df, d2);
                                }
                            } else {
                                let dist = df + (d2 as f32).sqrt();
                                if dist < mindist.0 {
                                    mindist = (dist, df, d2);
                                }
                            }
                            if chunk.is_some() {
                                mesh_finds += 1;
                                if request.marked_count() < marked_goal {
                                    request.mark(pos);
                                }
                            }
                        }
                        Ok(())
                    },
                )
                .unwrap();
            self.last_min_viewdist = mindist.0;
            self.last_mesh_finds = mesh_finds;
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

    /// Iterate through all the chunks that are visible from the given position, range and clip
    /// planes.
    /// No actual chunk data is accessed, and no portal hopping is done, so no dimension argument
    /// is received.
    pub fn iter_visible<F>(
        &self,
        center: [f64; 3],
        clip_planes: &[Vec4; 5],
        range: f32,
        mut f: F,
    ) -> Result<()>
    where
        F: FnMut(Int3) -> Result<()>,
    {
        let center_block = Int3::from_f64(center);
        let center_chunk = center_block >> CHUNK_BITS;
        let center_chunk_f64 =
            ((center_chunk << CHUNK_BITS) + Int3::splat(CHUNK_SIZE / 2)).to_f64();
        let d_off = Vec3::from([
            (center_chunk_f64[0] - center[0]) as f32,
            (center_chunk_f64[1] - center[1]) as f32,
            (center_chunk_f64[2] - center[2]) as f32,
        ]);
        let mn = (center_block - Int3::splat(range.ceil() as i32)) >> CHUNK_BITS;
        let mx = (center_block + Int3::splat(range.ceil() as i32)) >> CHUNK_BITS;
        let range_chunk = range * (CHUNK_SIZE as f32).recip();
        let range_chunk_sq = (range_chunk * range_chunk).ceil() as i32;
        for z in mn.z..=mx.z {
            for y in mn.y..=mx.y {
                for x in mn.x..=mx.x {
                    let chunk_pos = Int3::new([x, y, z]);
                    let d = chunk_pos - center_chunk;
                    if d.mag_sq() <= range_chunk_sq {
                        let dc = (d << CHUNK_BITS).to_f32() + d_off;
                        if check_chunk_clip_planes(clip_planes, dc) {
                            // Visible chunk!
                            f(chunk_pos)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn draw(
        &self,
        shader: &Program,
        uniforms: crate::lua::gfx::UniformStorage,
        offset_uniform: u32,
        origin: WorldPos,
        params: &DrawParameters,
        mvp: Mat4,
        framequad: [Vec3; 4],
        stencil: u8,
        subdraw: &dyn Fn(&WorldPos, &[Vec3; 4], u8) -> Result<()>,
    ) -> Result<()> {
        let frame = &self.state.frame;

        // Calculate the frustum planes
        let clip_planes = Self::calc_clip_planes(&mvp, &framequad);

        // Calculate some positioning
        let center_chunk = Int3::from_f64(origin.coords) >> CHUNK_BITS;
        let center_chunk_f64 = (center_chunk << CHUNK_BITS).to_f64();
        let off_d = Vec3::from([
            (center_chunk_f64[0] - origin.coords[0]) as f32,
            (center_chunk_f64[1] - origin.coords[1]) as f32,
            (center_chunk_f64[2] - origin.coords[2]) as f32,
        ]);

        // Draw all visible chunks
        {
            let mut frame = frame.borrow_mut();
            let mut drawn = 0;
            let mut bytes_v = 0;
            let mut bytes_i = 0;
            self.iter_visible(origin.coords, &clip_planes, self.view_radius, |pos| {
                let chunk = match self.meshes.get(ChunkPos {
                    coords: pos,
                    dim: origin.dim,
                }) {
                    Some(cnk) => cnk,
                    None => return Ok(()),
                };
                let buf = match chunk.buf.as_ref() {
                    Some(buf) => buf,
                    None => return Ok(()),
                };
                let offset = ((pos - center_chunk) << CHUNK_BITS).to_f32() + off_d;
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
                Ok(())
            })?;
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
        ChunkStorage::iter_nearby_raw(origin.coords, self.view_radius, |pos| {
            let chunk = match self.meshes.get(ChunkPos {
                coords: pos,
                dim: origin.dim,
            }) {
                Some(chunk) => chunk,
                None => return Ok(()),
            };
            if !chunk.portals.is_empty() {
                let offset = ((pos - center_chunk) << CHUNK_BITS).to_f32() + off_d;
                'portal: for portal in chunk.portals.iter() {
                    // Make sure portal is visible
                    let subfq_world = [
                        offset + portal.bounds[0],
                        offset + portal.bounds[1],
                        offset + portal.bounds[2],
                        offset + portal.bounds[3],
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
                    // TODO: Let Lua handle drawing the portal frame
                    // OPTIMIZE: Use sample queries and conditional rendering to speed up occluded
                    // portals.
                    uniforms
                        .vars
                        .borrow_mut()
                        .get_mut(offset_uniform as usize)
                        .ok_or(anyhow!("offset uniform out of range"))?
                        .1 = crate::lua::gfx::UniformVal::Vec3(offset.into());
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
                    let suborigin = WorldPos {
                        coords: [
                            origin.coords[0] + portal.jump[0],
                            origin.coords[1] + portal.jump[1],
                            origin.coords[2] + portal.jump[2],
                        ],
                        dim: portal.dim,
                    };
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
                        .1 = crate::lua::gfx::UniformVal::Vec3(offset.into());
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
            Ok(())
        })?;

        Ok(())
    }

    const SAFE_GAP: f64 = 1. / 128.;

    /// Move an integer block coordinate by the given offset, crossing through portals but going
    /// through solid blocks.
    /// First moves on the X axis, then on the Y axis and then the Z axis.
    /// The order could affect which portals are crossed!
    pub(crate) fn offset_coords(&self, abspos: &mut BlockPos, delta: Int3) {
        for i in 0..3 {
            self.blockcast_ghost(abspos, i, delta[i]);
        }
    }

    /// Move an integer block coordinate by the given amount of blocks in the given axis, crossing
    /// through portals but going through blocks.
    pub(crate) fn blockcast_ghost(&self, abspos: &mut BlockPos, axis: usize, delta: i32) {
        let binary = (delta > 0) as i32;
        let dir = if delta > 0 { 1 } else { -1 };
        for _ in 0..delta * dir {
            abspos.coords[axis] += dir;
            if self
                .block_at(*abspos)
                .map(|b| b.is_portal(&self.style))
                .unwrap_or(false)
            {
                let mut ppos = *abspos;
                ppos.coords[axis] += 1 - binary;
                if let Some(portal) = self.chunks.read().portal_at(ppos, axis) {
                    abspos.coords += portal.jump;
                    abspos.dim = portal.dim;
                }
            }
        }
    }

    /// Move an integer block coordinate by the given amount of blocks in the given axis, crossing
    /// through portals and stopping at the first solid block (returning the last clear block).
    pub(crate) fn blockcast(&self, abspos: &mut BlockPos, axis: usize, delta: i32) -> (i32, bool) {
        let binary = (delta > 0) as i32;
        let dir = if delta > 0 { 1 } else { -1 };
        let mut cur = *abspos;
        let mut last;
        for i in 0..delta * dir {
            last = cur;
            cur.coords[axis] += dir;
            match self
                .block_at(cur)
                .map(|b| self.style.lookup(b))
                .unwrap_or(BlockStyle::Solid)
            {
                BlockStyle::Clear => {}
                BlockStyle::Solid => {
                    *abspos = last;
                    return (i * dir, true);
                }
                BlockStyle::Portal => {
                    let mut ppos = cur;
                    ppos.coords[axis] += 1 - binary;
                    if let Some(portal) = self.chunks.read().portal_at(ppos, axis) {
                        cur.coords += portal.jump;
                        cur.dim = portal.dim;
                    }
                }
                BlockStyle::Custom => unimplemented!(),
            }
        }
        *abspos = cur;
        (delta, false)
    }

    /// Place a virtual point particle at `from` and move it `delta` units in the `axis` axis.
    /// If there are any blocks in the way, the particle stops its movement.
    /// If there are any portals in the way, the particle crosses them.
    pub(crate) fn raycast_aligned(
        &self,
        abspos: &mut WorldPos,
        axis: usize,
        delta: f64,
    ) -> (f64, bool) {
        let dir = if delta > 0. { 1. } else { -1. };
        let binary = (delta > 0.) as i32;
        let mut cur = *abspos;
        let mut limit = cur.coords[axis] + delta;
        let mut crashed = false;
        let mut next_block = [
            cur.coords[0].floor(),
            cur.coords[1].floor(),
            cur.coords[2].floor(),
        ];
        next_block[axis] = (cur.coords[axis] * dir).ceil() * dir;
        while next_block[axis] * dir < limit * dir {
            let col_int = BlockPos {
                coords: Int3::from_f64(next_block),
                dim: cur.dim,
            };
            let mut block = col_int;
            block.coords[axis] += binary - 1;
            match self
                .block_at(block)
                .map(|b| self.style.lookup(b))
                .unwrap_or(BlockStyle::Solid)
            {
                BlockStyle::Clear => next_block[axis] += dir,
                BlockStyle::Solid => {
                    limit = next_block[axis] - Self::SAFE_GAP * dir;
                    crashed = true;
                    break;
                }
                BlockStyle::Portal => {
                    if let Some(portal) = self.chunks.read().portal_at(col_int, axis) {
                        for i in 0..3 {
                            cur.coords[i] += portal.jump[i] as f64;
                            next_block[i] += portal.jump[i] as f64;
                        }
                        cur.dim = portal.dim;
                        limit += portal.jump[axis] as f64;
                    } else {
                        // Should never happen with proper portals!
                        next_block[axis] += dir;
                    }
                }

                BlockStyle::Custom => unimplemented!(),
            }
        }
        *abspos = cur;
        abspos.coords[axis] = limit;
        (limit - cur.coords[axis], crashed)
    }

    /// Place a virtual cuboid at `from`, of dimensions `size * 2` (`size` being the "radius"), and
    /// move it by `delta`, checking for collisions against blocks and portals.
    /// If `eager` is true, the process stops as soon as a block is hit.
    /// If `eager` is false, when a block is hit the delta component in the direction of the hit
    /// block is killed, advancing no more in that direction.
    pub(crate) fn boxcast(
        &self,
        abspos: &mut WorldPos,
        mut delta: [f64; 3],
        size: [f64; 3],
        eager: bool,
    ) -> ([f64; 3], [bool; 3]) {
        // TODO: Check that everything works alright with portals when objects have integer
        // positions.

        if delta == [0.; 3] {
            return ([0.; 3], [false; 3]);
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
            abspos.coords[0] + size[0] * dir[0],
            abspos.coords[1] + size[1] * dir[1],
            abspos.coords[2] + size[2] * dir[2],
        ];
        let mut next_block = [
            (frontier[0] * dir[0]).ceil() * dir[0],
            (frontier[1] * dir[1]).ceil() * dir[1],
            (frontier[2] * dir[2]).ceil() * dir[2],
        ];
        let mut limit = [
            frontier[0] + delta[0],
            frontier[1] + delta[1],
            frontier[2] + delta[2],
        ];
        let mut crashed = [false; 3];

        // Keep track of the planes of the portals that have been crossed
        // This way, we can work in virtual coordinates and only transfer to real absolute
        // coordinates when looking up a block, portal or at the very end when translating the
        // final virtual coordinates back to absolute world coordinates
        let mut portalplanes = self.tmp_colbuf.borrow_mut();
        portalplanes.clear();
        let get_warp = |planes: &[PortalPlane], pos: Int3| {
            let mut jump = Int3::zero();
            let mut dim = abspos.dim;
            for plane in planes.iter() {
                if pos[plane.axis] * idir[plane.axis] < plane.coord * idir[plane.axis] {
                    break;
                }
                jump = plane.jump;
                dim = plane.dim;
            }
            (jump, dim)
        };
        let to_real_pos = |planes: &[PortalPlane], pos: Int3| {
            let (warp, dim) = get_warp(planes, pos);
            BlockPos {
                coords: pos + warp,
                dim,
            }
        };
        let add_portal = |planes: &mut Vec<PortalPlane>, blockpos: Int3, axis: usize| {
            let mut pos = blockpos;
            pos[axis] += 1 - binary[axis];
            let real_pos = to_real_pos(planes, pos);
            if let Some(portal) = self.chunks.read().portal_at(real_pos, axis) {
                let last_jump = planes
                    .last()
                    .map(|plane| plane.jump)
                    .unwrap_or(Int3::zero());
                planes.push(PortalPlane {
                    axis,
                    coord: blockpos[axis],
                    jump: last_jump + portal.jump,
                    dim: portal.dim,
                });
            }
        };

        // Prebuild any portal planes that are currently intersecting the box
        {
            let src = Int3::from_f64(abspos.coords);
            let dst = Int3::from_f64(frontier);
            for axis in 0..3 {
                let delta = dst[axis] - src[axis];
                let mut cur = src;
                for _ in 0..delta * idir[axis] {
                    cur[axis] += idir[axis];
                    let real_pos = to_real_pos(&portalplanes, cur);
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
            let mut min_block = Int3::zero();
            let mut max_block = Int3::zero();
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
                        let pos = Int3::new([x, y, z]);
                        let real_pos = to_real_pos(&portalplanes, pos);
                        let s = self
                            .block_at(real_pos)
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
                    crashed[closest_axis] = true;
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
        let mv = [
            pos[0] - abspos.coords[0],
            pos[1] - abspos.coords[1],
            pos[2] - abspos.coords[2],
        ];
        let (warp, dim) = get_warp(&portalplanes, Int3::from_f64(pos));
        let warp = warp.to_f64();
        *abspos = WorldPos {
            coords: [pos[0] + warp[0], pos[1] + warp[1], pos[2] + warp[2]],
            dim,
        };
        (mv, crashed)
    }
}

/// Keeps track of chunk meshes.
pub(crate) struct MeshKeeper {
    pub meshes: GridKeeper4<ChunkMesh>,
}
impl MeshKeeper {
    pub fn new() -> Self {
        Self {
            meshes: GridKeeper4::new(),
        }
    }
}
impl ops::Deref for MeshKeeper {
    type Target = GridKeeper4<ChunkMesh>;
    fn deref(&self) -> &Self::Target {
        &self.meshes
    }
}
impl ops::DerefMut for MeshKeeper {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.meshes
    }
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
    pub dim: u32,
}

use std::f64::INFINITY;

use crate::{
    chunkmesh::{MesherHandle, RawBufPackage},
    gen::{GenArea, GenConfig},
    prelude::*,
};
use common::terrain::GridKeeper4;
use glium::buffer::RawBufferPackage;

#[derive(Default)]
struct SphereBuf {
    chunks: Vec<(Int3, f32)>,
    range: f32,
}

#[derive(Default)]
struct SphereBufs {
    bufs: Vec<Cell<SphereBuf>>,
    depth: usize,
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

    fn with_spherebuf<F, R>(f: F) -> R
    where
        F: FnOnce(&mut SphereBuf) -> R,
    {
        thread_local! {
            static SPHERE_BUFS: RefCell<SphereBufs> = default();
        }
        SPHERE_BUFS.with(|bufs_cell| {
            let mut bufs = bufs_cell.borrow_mut();
            while bufs.bufs.len() <= bufs.depth {
                bufs.bufs.push(default());
            }
            let mut buf = bufs.bufs[bufs.depth].take();
            bufs.depth += 1;
            drop(bufs);

            let r = f(&mut buf);

            let mut bufs = bufs_cell.borrow_mut();
            bufs.depth -= 1;
            bufs.bufs[bufs.depth].replace(buf);
            r
        })
    }

    fn get_sphere(sphere: &mut SphereBuf, radius: f32) {
        if sphere.range >= radius {
            return;
        }
        //Get up to the first chunk whose distance to the center is larger than `radius`
        println!("generating spherebuf");
        sphere.chunks.clear();
        let r_chunk = radius * (CHUNK_SIZE as f32).recip();
        let ir = r_chunk.ceil() as i32;
        let r2 = (r_chunk * r_chunk).floor() as i32;
        for z in -ir..=ir {
            for y in -ir..=ir {
                for x in -ir..=ir {
                    let pos = Int3::new([x, y, z]);
                    let d2 = pos.mag_sq();
                    if d2 <= r2 {
                        let dist = (d2 as f32).sqrt() * CHUNK_SIZE as f32;
                        sphere.chunks.push((pos, dist));
                    }
                }
            }
        }
        sphere.chunks.sort_by_key(|(_p, d)| Sortf32(*d));
        sphere.range = radius;
    }

    /// Iterate over all chunk positions within a certain radius, without portal hopping or any
    /// funny business.
    pub fn iter_nearby_raw<F>(center: [f64; 3], range: f32, mut f: F) -> Result<()>
    where
        F: FnMut(Int3) -> Result<()>,
    {
        ChunkStorage::with_spherebuf(|sphere| {
            ChunkStorage::get_sphere(sphere, range);
            let center_chunk = Int3::from_f64(center) >> CHUNK_BITS;
            for &(pos, dist) in sphere.chunks.iter() {
                if dist > range {
                    break;
                }
                let pos = center_chunk + pos;
                f(pos)?;
            }
            Ok(())
        })
    }

    /// Iterate through all the chunks nearby to a certain position, including those that are nearby
    /// through portals.
    /// Iterates over nonexisting chunk positions too.
    pub fn iter_nearby<F>(
        &self,
        seenbuf: &mut HashMap<ChunkPos, f32>,
        center: BlockPos,
        range: f32,
        f: F,
    ) -> Result<()>
    where
        F: FnMut(ChunkPos, Option<ChunkRef>, f32) -> Result<()>,
    {
        struct State<'a, F> {
            chunks: &'a ChunkStorage,
            sphere: &'a mut SphereBuf,
            seen: &'a mut HashMap<ChunkPos, f32>,
            f: F,
        }
        fn explore<F>(s: &mut State<F>, center: BlockPos, range: f32, basedist: f32) -> Result<()>
        where
            F: FnMut(ChunkPos, Option<ChunkRef>, f32) -> Result<()>,
        {
            use std::collections::hash_map::Entry;
            let epsilon = 4.;
            ChunkStorage::get_sphere(s.sphere, range);
            let center_chunk = center.coords >> CHUNK_BITS;
            let mut idx = 0;
            while idx < s.sphere.chunks.len() {
                let (pos, dist) = s.sphere.chunks[idx];
                let pos = center_chunk + pos;
                if dist > range {
                    break;
                }
                idx += 1;

                let chunk_pos = ChunkPos {
                    coords: pos,
                    dim: center.dim,
                };
                match s.seen.entry(chunk_pos) {
                    Entry::Occupied(mut prevdist) => {
                        if dist >= *prevdist.get() - epsilon {
                            continue;
                        }
                        prevdist.insert(dist);
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(dist);
                    }
                }
                let chunk = s.chunks.chunk_at(chunk_pos);
                (s.f)(chunk_pos, chunk, dist)?;
                if let Some(data) = chunk.and_then(|c| c.blocks()) {
                    for portal in data.portals() {
                        let portal_center = portal.get_center();
                        if !portal_center.is_within(Int3::splat(CHUNK_SIZE)) {
                            continue;
                        }
                        let portal_center = (chunk_pos.coords << CHUNK_BITS) + portal_center;
                        let pdist = ((portal_center - center.coords).mag_sq() as f32).sqrt();
                        // TODO: Perhaps do something about large portals?
                        // Approximating a portal as a point at its center only works
                        // for smallish portals.
                        explore(
                            s,
                            BlockPos {
                                coords: portal_center + portal.jump,
                                dim: portal.dim,
                            },
                            range - pdist,
                            basedist + pdist,
                        )?;
                    }
                }
            }
            Ok(())
        }
        Self::with_spherebuf(|sphere| {
            seenbuf.clear();
            explore(
                &mut State {
                    chunks: self,
                    sphere,
                    seen: seenbuf,
                    f,
                },
                center,
                range,
                0.,
            )
        })
    }

    pub fn maybe_gc(&mut self) {
        if self.last_gc.elapsed() > self.gc_interval {
            println!("doing gc on the chunks");
            let old = self.chunks.map.len();
            self.gc();
            let new = self.chunks.map.len();
            println!("  removed {} chunks", old - new);
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
    tmp_colbuf: RefCell<Vec<PortalPlane>>,
    tmp_seenbuf: RefCell<HashMap<ChunkPos, f32>>,
    tmp_seenbuf_simple: RefCell<HashSet<Int4>>,
    tmp_posbuf: RefCell<Vec<Int4>>,
    tmp_sortbuf: RefCell<Vec<(f32, Int4)>>,
}
impl Terrain {
    pub fn new(state: &Rc<State>, gen_cfg: GenConfig) -> Result<Terrain> {
        let chunks = Arc::new(RwLock::new(ChunkStorage::new()));
        let generator = GeneratorHandle::new(gen_cfg, &state.global, chunks.clone())?;
        let tex = generator.take_block_textures()?;
        Ok(Terrain {
            style: StyleTable::new(&tex),
            state: state.clone(),
            meshes: MeshKeeper::new(),
            mesher: MesherHandle::new(state, chunks.clone(), tex),
            tmp_colbuf: default(),
            tmp_seenbuf: default(),
            tmp_seenbuf_simple: default(),
            tmp_posbuf: default(),
            tmp_sortbuf: default(),
            last_min_viewdist: 0.,
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
        for buf_pkg in self.mesher.recv_bufs.try_iter() {
            let mesh = ChunkMesh {
                mesh: buf_pkg.mesh,
                buf: match buf_pkg.buf {
                    None => {
                        // A chunk with no geometry (ie. full air or full solid)
                        None
                    }
                    Some(raw) => unsafe {
                        // Deconstructed buffers
                        // Construct them back
                        Some(raw.unpack(&self.state.display))
                    },
                },
                portals: {
                    buf_pkg
                        .portals
                        .into_iter()
                        .map(|p| PortalMesh {
                            mesh: p.mesh.into(),
                            buf: unsafe { Rc::new(p.buf.unpack(&self.state.display)) },
                            bounds: p.bounds,
                            jump: p.jump,
                            dim: p.dim,
                        })
                        .collect()
                },
            };
            self.meshes.insert(buf_pkg.pos, mesh);
            request.remove(buf_pkg.pos);
        }
        //Request new meshes from mesher thread
        // OPTIMIZE: Dedicate a thread to exploring chunks and determining which ones to mesh.
        // This means sharing the mesh storage structure.
        // This secondary thread could wait for a signal from the main thread that means "i'm not
        // using the mesh storage"
        {
            request.unmark_all();
            let marked_goal = request.marked_goal();

            let mut sortbuf = self.tmp_sortbuf.borrow_mut();
            sortbuf.clear();

            let chunks = self.chunks.read();
            let mut visited = 0;
            let mut mindist = self.view_radius;
            chunks
                .iter_nearby(
                    &mut *self.tmp_seenbuf.borrow_mut(),
                    center,
                    self.view_radius,
                    |pos, chunk, dist| {
                        visited += 1;
                        if self.meshes.get(pos).is_none() {
                            if dist < mindist {
                                mindist = dist;
                            }
                            if chunk.is_some() {
                                sortbuf.push((dist, pos));
                            }
                        }
                        Ok(())
                    },
                )
                .unwrap();

            for &(dist, pos) in sortbuf
                .iter()
                .sorted_by(|a, b| Sortf32(a.0).cmp(&Sortf32(b.0)))
                .take(marked_goal)
            {
                request.mark(pos, dist);
            }

            self.last_min_viewdist = (mindist - (CHUNK_SIZE / 2) as f32 * 3f32.sqrt()).max(0.);
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
    pub fn calc_clip_planes(mvp: &Mat4, fq: &[Vec3; 4]) -> ([Vec4; 5], [bool; 4]) {
        let inv_mvp = mvp.inversed();

        let clip = [
            *mvp * fq[0].into_homogeneous_point(),
            *mvp * fq[1].into_homogeneous_point(),
            *mvp * fq[2].into_homogeneous_point(),
            *mvp * fq[3].into_homogeneous_point(),
        ];
        let mut ndc = [
            clip[0].normalized_homogeneous_point().truncated(),
            clip[1].normalized_homogeneous_point().truncated(),
            clip[2].normalized_homogeneous_point().truncated(),
            clip[3].normalized_homogeneous_point().truncated(),
        ];
        for i in 0..4 {
            ndc[i].z = 1.;
        }
        let fw = [
            inv_mvp.transform_point3(ndc[0]),
            inv_mvp.transform_point3(ndc[1]),
            inv_mvp.transform_point3(ndc[2]),
            inv_mvp.transform_point3(ndc[3]),
        ];

        let proper = [
            clip[0].w > 0.,
            clip[1].w > 0.,
            clip[2].w > 0.,
            clip[3].w > 0.,
        ];

        let calc_plane_raw = |p: [Vec3; 3]| {
            let n = (p[1] - p[0]).cross(p[2] - p[0]).normalized();
            Vec4::new(n.x, n.y, n.z, -p[0].dot(n))
        };
        let calc_plane = |i: usize, j: usize| {
            if proper[i] {
                calc_plane_raw([fq[i], fq[j], fw[i]])
            } else if proper[j] {
                calc_plane_raw([fq[i], fq[j], fw[j]])
            } else {
                Vec4::new(0., 0., 0., 1.)
            }
        };

        let planes = [
            calc_plane_raw([fq[2], fq[1], fq[0]]), // near plane
            calc_plane(3, 0),                      // left plane
            calc_plane(1, 2),                      // right plane
            calc_plane(0, 1),                      // bottom plane
            calc_plane(2, 3),                      // top plane
        ];

        (planes, proper)
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
        ChunkStorage::with_spherebuf(|sphere| {
            ChunkStorage::get_sphere(sphere, range);
            let center_chunk = Int3::from_f64(center) >> CHUNK_BITS;
            let center_chunk_f64 =
                ((center_chunk << CHUNK_BITS) + Int3::splat(CHUNK_SIZE / 2)).to_f64();
            let d_off = Vec3::from([
                (center_chunk_f64[0] - center[0]) as f32,
                (center_chunk_f64[1] - center[1]) as f32,
                (center_chunk_f64[2] - center[2]) as f32,
            ]);
            for &(dpos, dist) in sphere.chunks.iter() {
                if dist > range {
                    break;
                }
                let pos = center_chunk + dpos;
                let fpos = (dpos << CHUNK_BITS).to_f32() + d_off;
                if check_chunk_clip_planes(clip_planes, fpos) {
                    // Visible chunk!
                    f(pos)?;
                }
            }
            Ok(())
        })
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
        subdraw: &dyn Fn(&WorldPos, &[Vec3; 4], &Rc<Buffer3d>, Vec3, u8) -> Result<()>,
    ) -> Result<()> {
        let frame = &self.state.frame;

        // Calculate the frustum planes
        let (clip_planes, _) = Self::calc_clip_planes(&mvp, &framequad);

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
                    // Draw the portal and its interior
                    // But let Lua do it
                    let portal_mesh = &portal.buf;
                    let suborigin = WorldPos {
                        coords: [
                            origin.coords[0] + portal.jump[0],
                            origin.coords[1] + portal.jump[1],
                            origin.coords[2] + portal.jump[2],
                        ],
                        dim: portal.dim,
                    };
                    subdraw(&suborigin, &subfq_world, portal_mesh, offset, stencil + 1)?;

                    /*
                    // Get the portal shape into a buffer
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
                    subdraw(&suborigin, &subfq_world, portal_mesh, offset, stencil + 1)?;
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
                    */
                }
            }
            Ok(())
        })?;

        Ok(())
    }

    const SAFE_GAP: f64 = 1. / 128.;

    /// Move an integer block coordinate by the given amount of blocks in the given axis, crossing
    /// through portals and stopping at the first solid block (returning the last clear block).
    pub(crate) fn blockcast(&self, abspos: &mut BlockPos, axis: usize, delta: i32) -> (i32, bool) {
        let binary = (delta > 0) as i32;
        let dir = if delta > 0 { 1 } else { -1 };
        let mut cur = *abspos;
        let mut last = cur;
        cur.coords[axis] += dir;
        for i in 0..delta * dir {
            match self
                .block_at(cur)
                .map(|b| self.style.lookup(b))
                .unwrap_or(BlockStyle::Solid)
            {
                BlockStyle::Clear => {
                    last = cur;
                    cur.coords[axis] += dir;
                }
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

    /// When an entity or an object that is drawn in one piece is clipping a portal, it must be
    /// rendered twice (or as many times as portals it clips).
    ///
    /// This function calculates exactly how many copies and what offset should be applied to each
    /// copy (as well as in which dimension each copy ends up in).
    pub(crate) fn get_draw_positions(
        &self,
        abspos: WorldPos,
        size: [f64; 3],
        mut out_f: impl FnMut(usize, Int4),
    ) {
        let mut seen = self.tmp_seenbuf_simple.borrow_mut();
        let mut seen_up_to = 0;
        let mut out = self.tmp_posbuf.borrow_mut();
        out.clear();
        out.push(Int4 {
            coords: Int3::zero(),
            dim: abspos.dim,
        });
        seen.clear();
        let chunks = self.chunks.read();

        while seen_up_to < out.len() {
            let jump = out[seen_up_to];
            out_f(seen_up_to, jump);
            seen_up_to += 1;
            let pos = [
                abspos.coords[0] + jump.coords[0] as f64,
                abspos.coords[1] + jump.coords[1] as f64,
                abspos.coords[2] + jump.coords[2] as f64,
            ];
            // Minimum portal block coords (inclusive)
            let mn = Int3::new([
                (pos[0] - size[0]).ceil() as i32,
                (pos[1] - size[1]).ceil() as i32,
                (pos[2] - size[2]).ceil() as i32,
            ]);
            // Maximum portal block coords (inclusive)
            let mx = Int3::new([
                (pos[0] + size[0]).floor() as i32,
                (pos[1] + size[1]).floor() as i32,
                (pos[2] + size[2]).floor() as i32,
            ]);
            // Maxmin chunk coords (inclusive)
            let bounds = [mn >> CHUNK_BITS, mx >> CHUNK_BITS];
            for z in bounds[0].z..=bounds[1].z {
                for y in bounds[0].y..=bounds[1].y {
                    for x in bounds[0].x..=bounds[1].x {
                        let chunk_pos = Int3::new([x, y, z]);
                        let chunk = match chunks.chunk_at(ChunkPos {
                            coords: chunk_pos,
                            dim: jump.dim,
                        }) {
                            Some(cnk) => cnk,
                            None => continue,
                        };
                        let chunk = match chunk.blocks() {
                            Some(chunk) => chunk,
                            None => continue,
                        };
                        for portal in chunk.portals() {
                            let portal_mn = (chunk_pos << CHUNK_BITS)
                                + [
                                    portal.pos[0] as i32,
                                    portal.pos[1] as i32,
                                    portal.pos[2] as i32,
                                ];
                            let portal_mx = portal_mn
                                + [
                                    portal.size[0] as i32,
                                    portal.size[1] as i32,
                                    portal.size[2] as i32,
                                ];
                            if mn.x <= portal_mx.x
                                && portal_mn.x <= mx.x
                                && mn.y <= portal_mx.y
                                && portal_mn.y <= mx.y
                                && mn.z <= portal_mx.z
                                && portal_mn.z <= mx.z
                            {
                                // Entity collides with this portal
                                let subjump = Int4 {
                                    coords: jump.coords + portal.jump,
                                    dim: portal.dim,
                                };
                                if seen.insert(subjump) {
                                    out.push(subjump);
                                }
                            }
                        }
                    }
                }
            }
        }
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
    pub buf: Rc<Buffer3d>,
    pub bounds: [Vec3; 4],
    pub jump: [f64; 3],
    pub dim: u32,
}

pub(crate) struct RawPortalMesh {
    pub mesh: Mesh,
    pub buf: RawBufPackage,
    pub bounds: [Vec3; 4],
    pub jump: [f64; 3],
    pub dim: u32,
}

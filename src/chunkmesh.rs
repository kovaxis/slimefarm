use crate::{mesh::RawBufPackage, prelude::*, terrain::RawPortalMesh};
use rectpack::DensePacker;

pub(crate) struct ChunkMeshPkg {
    pub pos: ChunkPos,
    pub mesh: Mesh<VoxelVertex>,
    pub buf: Option<RawBufPackage<VoxelVertex>>,
    //pub atlas: usize,
    pub atlas: Option<RawTexturePackage>,
    pub portals: Vec<RawPortalMesh>,
}

pub struct MesherHandle {
    pub(crate) recv_bufs: Receiver<ChunkMeshPkg>,
    //pub(crate) recv_atlas: Receiver<(usize, RawTexturePackage)>,
    //pub(crate) send_atlas: Sender<(usize, RawTexturePackage)>,
    shared: Arc<SharedState>,
    thread: Option<JoinHandle<()>>,
}
impl MesherHandle {
    pub(crate) fn new(
        state: &Rc<State>,
        chunks: Arc<RwLock<ChunkStorage>>,
        textures: BlockTextures,
    ) -> Self {
        let shared = Arc::new(SharedState {
            request: default(),
            close: false.into(),
            avg_mesh_time: 0f32.into(),
            avg_upload_time: 0f32.into(),
        });
        let gl_ctx = state
            .sec_gl_ctx
            .take()
            .expect("no secondary opengl context available for mesher");
        let (send_bufs, recv_bufs) = channel::bounded(512);
        //let (send_atlas, recv_atlas) = channel::bounded(2);
        //let (send_recycleatlas, recv_recycleatlas) = channel::bounded(2);
        let thread = {
            let shared = shared.clone();
            thread::spawn(move || {
                let gl_ctx =
                    Display::from_gl_window(gl_ctx).expect("failed to create headless gl context");
                run_mesher(MesherState {
                    shared,
                    chunks: Some(chunks),
                    gl_ctx,
                    send_bufs,
                    //send_atlas: send_atlas,
                    //recv_atlas: recv_recycleatlas,
                    mesher: Box::new(Mesher2::new(textures)),
                });
            })
        };
        Self {
            thread: Some(thread),
            recv_bufs,
            //recv_atlas: recv_atlas,
            //send_atlas: send_recycleatlas,
            shared,
        }
    }

    pub(crate) fn request(&self) -> &Mutex<RequestBuf> {
        &self.shared.request
    }
}
impl ops::Deref for MesherHandle {
    type Target = SharedState;
    fn deref(&self) -> &SharedState {
        &self.shared
    }
}
impl Drop for MesherHandle {
    fn drop(&mut self) {
        self.recv_bufs = channel::never();
        self.shared.close.store(true);
        if let Some(join) = self.thread.take() {
            join.thread().unpark();
            join.join().unwrap();
        }
    }
}

pub struct SharedState {
    request: Mutex<RequestBuf>,
    close: AtomicCell<bool>,
    pub avg_mesh_time: AtomicCell<f32>,
    pub avg_upload_time: AtomicCell<f32>,
}

struct MesherState {
    shared: Arc<SharedState>,
    chunks: Option<Arc<RwLock<ChunkStorage>>>,
    gl_ctx: Display,
    /// Atlas chunks that have not yet been twice to the atlas texture.
    ///
    /// Because there are two atlas textures for each atlas index going around, each atlas chunk has
    /// to be written twice into two different textures.
    /// Therefore, whenever the alternative texture for an atlas is received, these atlas chunks are
    /// written and then sent back to the pool.
    //pending_atlas: Vec<Vec<AtlasChunk>>,
    mesher: Box<Mesher2>,
    send_bufs: Sender<ChunkMeshPkg>,
    //send_atlas: Sender<(usize, RawTexturePackage)>,
    //recv_atlas: Receiver<(usize, RawTexturePackage)>,
}
impl MesherState {
    fn try_mesh<'a>(
        &mut self,
        pos: ChunkPos,
        chunks: &'a RwLock<ChunkStorage>,
        chunks_store: &mut Option<RwLockReadGuard<'a, ChunkStorage>>,
    ) -> Option<ChunkMeshPkg> {
        // Make mesh
        time!(start mesh);
        self.mesher.make_mesh(pos, chunks, chunks_store)?;
        time!(store mesh self.shared.avg_mesh_time);

        time!(start upload);
        // Upload mesh to GPU
        let mesh = mem::take(&mut self.mesher.mesh);
        let buf_pkg = if mesh.indices.is_empty() {
            None
        } else {
            let buf = mesh.make_buffer(&self.gl_ctx);
            Some(RawBufPackage::pack(buf))
        };

        // Upload texture to GPU
        let tex_pkg = if mesh.indices.is_empty() {
            None
        } else {
            let (w, h) = (ATLAS_BIN, self.mesher.atlas_h);
            let raw_img = RawImage2d {
                data: Cow::Borrowed(&self.mesher.atlas[..(w * h) as usize]),
                width: w as u32,
                height: h as u32,
                format: glium::texture::ClientFormat::U8U8U8U8,
            };
            let tex = Texture2d::with_mipmaps(
                &self.gl_ctx,
                raw_img,
                glium::texture::MipmapsOption::NoMipmap,
            )
            .expect("failed to upload atlas texture for chunk");
            let tex = RawTexturePackage::pack(tex.into_any());
            Some(tex)
        };
        time!(store upload self.shared.avg_upload_time);

        // Package it all up
        Some(ChunkMeshPkg {
            pos,
            mesh,
            buf: buf_pkg,
            atlas: tex_pkg,
            // TODO: Re-implement portals
            portals: Vec::new(),
        })
    }
}

/*
fn try_mesh<'a>(
    state: &mut MesherState,
    pos: ChunkPos,
    chunks_raw: &'a RwLock<ChunkStorage>,
    chunks_store: &mut Option<RwLockReadGuard<'a, ChunkStorage>>,
) -> Option<BufPackage> {
    let mesh_start = Instant::now();
    let chunks = chunks_store.get_or_insert_with(|| chunks_raw.read());

    //Gather all of the adjacent chunk neighbors
    //TODO: Figure out occlusion with portals
    macro_rules! build_neighbors {
        (@$x:expr, $y:expr, $z:expr) => {{
            let mut subpos = pos;
            subpos.coords += [$x-1, $y-1, $z-1];
            chunks.chunk_arc_at(subpos)?
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

    // Release the chunk storage lock
    drop(chunks);
    chunks_store.take();

    //Mesh the chunk
    let mesh = state.mesher.make_mesh(pos, &neighbors);

    //Figure out portals
    let portals = state.mesher.mesh_portals(&neighbors[13], &state.gl_ctx);

    //Keep meshing statistics
    {
        //Dont care about data races here, after all it's just stats
        //Therefore, dont synchronize
        let time = mesh_start.elapsed().as_secs_f32();
        let old_time = state.shared.avg_mesh_time.load();
        let new_time = old_time + (time - old_time) * AVERAGE_WEIGHT;
        state.shared.avg_mesh_time.store(new_time);
    }

    //Upload the chunk buffer to GPU
    let buf_pkg = if mesh.indices.is_empty() {
        None
    } else {
        let upload_start = Instant::now();
        let buf = mesh.make_buffer(&state.gl_ctx);
        //Keep upload statistics
        //Dont care about data races here, after all it's just stats
        //Therefore, dont synchronize
        let time = upload_start.elapsed().as_secs_f32();
        let old_time = state.shared.avg_upload_time.load();
        let new_time = old_time + (time - old_time) * AVERAGE_WEIGHT;
        state.shared.avg_upload_time.store(new_time);
        Some(RawBufPackage::pack(buf))
    };

    //Package it all up
    Some(BufPackage {
        pos,
        mesh,
        buf: buf_pkg,
        portals,
    })
}
*/

/// Organizes chunk requests between threads.
/// Currently, coordinates:
/// main thread -> chunk mesher
/// main thread -> gen thread
#[derive(Default)]
pub struct RequestBuf {
    map: HashMap<ChunkPos, f32>,
}
impl RequestBuf {
    /// Request a chunk from the consumer thread.
    pub fn mark(&mut self, pos: ChunkPos, dist: f32) {
        self.map.entry(pos).or_insert(dist);
    }

    /// Unmark all requested chunks that the mesher has not yet started working on.
    pub fn unmark_all(&mut self) {
        self.map
            .retain(|_k, v| v.to_bits() == f32::INFINITY.to_bits());
    }

    /// Collect requested chunks from the producer thread.
    pub fn collect(&mut self, buf: &mut Vec<ChunkPos>, maxn: usize) {
        buf.clear();
        buf.extend(
            self.map
                .iter_mut()
                .filter(|(_p, d)| d.to_bits() != f32::INFINITY.to_bits())
                .sorted_by(|a, b| Sortf32(*a.1).cmp(&Sortf32(*b.1)))
                .take(maxn)
                .map(|(pos, dist)| {
                    *dist = f32::INFINITY;
                    *pos
                }),
        );
    }

    /// Remove a mark slot altogether.
    ///
    /// This method can be called from both threads:
    /// - From the producer thread, if producing a chunk failed.
    /// - From the consumer thread, once a chunk is received.
    pub fn remove(&mut self, pos: ChunkPos) {
        self.map.remove(&pos);
    }

    /// The ideal amount of marked chunks.
    /// This amount might adapt to the meshing speed.
    pub fn marked_goal(&self) -> usize {
        // TODO: Make adaptive
        32
    }
}

fn run_mesher(mut state: MesherState) {
    let mut last_stall_warning = Instant::now();

    const MESH_QUEUE: usize = 8;
    let mut mesh_queue = Vec::with_capacity(MESH_QUEUE);
    let mut fail_queue = Vec::with_capacity(MESH_QUEUE);
    let chunks = state.chunks.take().unwrap();
    let mut chunks_store = None;
    loop {
        // Get a list of candidate chunks from the main thread
        {
            let mut request = match chunks_store {
                Some(_) => match state.shared.request.try_lock() {
                    Some(req) => req,
                    None => {
                        chunks_store.take();
                        state.shared.request.lock()
                    }
                },
                None => state.shared.request.lock(),
            };
            for pos in fail_queue.drain(..) {
                request.remove(pos);
            }
            request.collect(&mut mesh_queue, MESH_QUEUE);
        }

        // Check candidates
        // It is important to drain the entire `mesh_queue`!
        // If it is not done, at least push it back to `fail_queue`
        let mut meshed_count = 0;
        for pos in mesh_queue.drain(..) {
            // Attempt to make this mesh
            let buf_pkg = match state.try_mesh(pos, &chunks, &mut chunks_store) {
                Some(buf) => buf,
                None => {
                    // Something failed
                    // Probably, a neighboring chunk was not available
                    fail_queue.push(pos);
                    continue;
                }
            };
            // Send the buffer back
            match state.send_bufs.try_send(buf_pkg) {
                Ok(()) => {}
                Err(err) => {
                    if err.is_full() {
                        //Channel is full, make sure to unlock chunks
                        let buf_pkg = err.into_inner();
                        let stall_start = Instant::now();
                        if let Some(chunks) = &mut chunks_store {
                            RwLockReadGuard::unlocked(chunks, || {
                                let _ = state.send_bufs.send(buf_pkg);
                            });
                        } else {
                            let _ = state.send_bufs.send(buf_pkg);
                        }
                        let now = Instant::now();
                        if now - last_stall_warning > Duration::from_millis(1500) {
                            last_stall_warning = now;
                            eprintln!(
                                "meshing thread stalled for {}ms",
                                (now - stall_start).as_millis()
                            );
                        }
                    } else {
                        //Channel closed, just discard it, we're gonna exit soon anyway
                    }
                }
            }
            //Count how many have we meshed
            meshed_count += 1;
        }

        if state.shared.close.load() {
            break;
        }
        if meshed_count <= 0 {
            //Pause for a while
            chunks_store.take();
            thread::park_timeout(Duration::from_millis(50));
            if state.shared.close.load() {
                break;
            }
        }
    }
}

const ATLAS_SIZE: i32 = 1024;
const ATLAS_BIN: i32 = 64;

const MARGIN: i32 = 4;
const BBUF_SIZE: i32 = 2 * MARGIN + CHUNK_SIZE;
const BBUF_LEN: usize = (BBUF_SIZE * BBUF_SIZE * BBUF_SIZE) as usize;
const ADVANCE: [i32; 3] = [1, BBUF_SIZE, BBUF_SIZE * BBUF_SIZE];

const NOISE_SIZE: i32 = CHUNK_SIZE + 1;
const NOISE_LEN: usize = (NOISE_SIZE * NOISE_SIZE * NOISE_SIZE) as usize;

/// A primitive with the same bitwidth as the chunk size.
type ChunkSizePrim = u32;

fn mask(w: u8) -> ChunkSizePrim {
    if w >= CHUNK_SIZE as u8 {
        !0
    } else {
        ((1 as ChunkSizePrim) << w).wrapping_sub(1)
    }
}

struct SurfaceBits {
    bits: [ChunkSizePrim; CHUNK_SIZE as usize],
}
impl SurfaceBits {
    fn new() -> Self {
        assert_eq!(mem::size_of::<ChunkSizePrim>() * 8, CHUNK_SIZE as usize);
        Self {
            bits: [0; CHUNK_SIZE as usize],
        }
    }

    fn clear(&mut self) {
        self.bits = [0; CHUNK_SIZE as usize];
    }

    fn get(&self, x: u8, y: u8) -> bool {
        (self.bits[y as usize] >> (x as usize)) & 1 != 0
    }

    fn set(&mut self, x: u8, y: u8) {
        self.bits[y as usize] |= 1 << x;
    }

    /// Returns the x position of the first unset bit to the right of `(x, y)`.
    fn march_right(&self, x: u8, y: u8) -> u8 {
        let mut row = self.bits[y as usize];
        row |= ((1 as ChunkSizePrim) << x).wrapping_sub(1);
        row.trailing_ones() as u8
    }

    /// The y position of the first row above `y` such that there is at least 1 bit unset from
    /// `(x, y)` to `(x+w, y)` (inclusive-exclusive range).
    fn march_up(&self, x: u8, mut y: u8, w: u8) -> u8 {
        let mask = mask(w) << x;
        y += 1;
        while y < CHUNK_SIZE as u8 {
            if mask & self.bits[y as usize] != mask {
                return y;
            }
            y += 1;
        }
        return CHUNK_SIZE as u8;
    }

    fn unset_rect(&mut self, x: u8, y: u8, w: u8, h: u8) {
        let mask = !(mask(w) << x);
        for y in y..y + h {
            self.bits[y as usize] &= mask;
        }
    }
}

struct AtlasChunk {
    /// To which atlas should this atlas chunk go.
    atlas_id: usize,
    data: Vec<(u8, u8, u8, u8)>,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
}
impl AtlasChunk {
    fn new() -> Self {
        Self {
            atlas_id: 0,
            data: vec![(0, 0, 0, 0); (ATLAS_BIN * ATLAS_SIZE) as usize],
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        }
    }
}

struct Mesher2 {
    block_buf: [BlockData; BBUF_LEN],
    corner_queue: Vec<(u8, u8)>,
    style: StyleTable,
    block_textures: [BlockTexture; 256],
    noise_buf: [f32; NOISE_LEN],
    mesh: Mesh<VoxelVertex>,
    //atlas: AtlasChunk,
    /// Garbage atlas chunks that can be reused.
    //free_atlas_pool: Vec<AtlasChunk>,
    packer: DensePacker,
    atlas: Vec<(u8, u8, u8, u8)>,
    atlas_h: i32,
}
impl Mesher2 {
    fn new(textures: BlockTextures) -> Self {
        Self {
            block_buf: [BlockData { data: 0 }; BBUF_LEN],
            corner_queue: Vec::with_capacity((CHUNK_SIZE * CHUNK_SIZE) as usize),
            style: StyleTable::new(&textures),
            block_textures: {
                let mut blocks: Uninit<[BlockTexture; 256]> = Uninit::uninit();
                for (src, dst) in textures.blocks.iter().zip(0..256) {
                    unsafe {
                        (blocks.as_mut_ptr() as *mut BlockTexture)
                            .offset(dst)
                            .write(src.take());
                    }
                }
                unsafe { blocks.assume_init() }
            },
            noise_buf: [0.; NOISE_LEN],
            mesh: default(),
            atlas: vec![(0, 0, 0, 0); (ATLAS_BIN * ATLAS_SIZE) as usize],
            atlas_h: 1,
            //atlas: AtlasChunk::new(),
            //free_atlas_pool: vec![],
            packer: DensePacker::new(1, 1),
        }
    }

    fn is_solid(&self, idx: i32) -> bool {
        self.block_buf[idx as usize].is_solid(&self.style)
    }
    fn is_clear(&self, idx: i32) -> bool {
        self.block_buf[idx as usize].is_clear(&self.style)
    }

    fn vertex(&mut self, pos: Vec3, uv: Vec2) -> VertIdx {
        self.mesh.add_vertex(VoxelVertex {
            pos: pos.into(),
            uv: uv.into(),
        })
    }

    fn color(&mut self, block: BlockData, blockpos: Int3, airpos: Int3) -> [u8; 4] {
        // Get the noise value at a particular location
        let noise_at = |pos: [i32; 3]| {
            self.noise_buf
                [(pos[0] + NOISE_SIZE * pos[1] + NOISE_SIZE * NOISE_SIZE * pos[2]) as usize]
        };
        // Snap to the nearest grid-aligned integer points
        let snap = |x: i32, i: usize| {
            let mask = (!0) << i;
            (x & mask, (x & mask) + (1 << i))
        };
        // Texture parameters for this block
        let tex = &self.block_textures[block.data as usize];
        // Color accumulator, starting with the base color
        let mut color = Vec4::from(tex.base);
        // Add the first layer of noise, which can be computed using a single noise lookup
        color += Vec4::from(tex.noise[0]) * noise_at(*blockpos);
        // Go through all noise layers
        for i in 1..BlockTexture::NOISE_LEVELS {
            // Get the 8 noise points
            let (x0, x1) = snap(blockpos.x, i);
            let (y0, y1) = snap(blockpos.y, i);
            let (z0, z1) = snap(blockpos.z, i);
            // Interpolate these 8 points using bilinear interpolation
            let f = blockpos.lowbits(i as i32).to_f32() * (1. / (1 << i) as f32);
            let s = Lerp::lerp(
                &Lerp::lerp(
                    &Lerp::lerp(&noise_at([x0, y0, z0]), noise_at([x1, y0, z0]), f.x),
                    Lerp::lerp(&noise_at([x0, y1, z0]), noise_at([x1, y1, z0]), f.x),
                    f.y,
                ),
                Lerp::lerp(
                    &Lerp::lerp(&noise_at([x0, y0, z1]), noise_at([x1, y0, z1]), f.x),
                    Lerp::lerp(&noise_at([x0, y1, z1]), noise_at([x1, y1, z1]), f.x),
                    f.y,
                ),
                f.z,
            );
            // Add the color offset
            color += Vec4::from(tex.noise[i]) * s;
        }
        // Quantize the color
        let q = |f: f32| (f * 255.) as u8;
        [q(color.x), q(color.y), q(color.z), q(color.w)]
    }

    fn quad(&mut self, blockpos: Int3, positive: i32, axes: [usize; 3], w: i32, h: i32) {
        // Allocate space in the terrain atlas
        let rect = loop {
            match self.packer.pack(w, h, true) {
                Some(rect) => break rect,
                None => {
                    // TODO: Grow atlas bin in this case
                    println!("ran out of atlas space for chunk!");
                    return;
                }
            }
        };
        self.atlas_h = self.atlas_h.max(rect.y + rect.height);

        // Figure out the mapping to block buffer indices
        let mut bidx1 = MARGIN * (ADVANCE[0] + ADVANCE[1] + ADVANCE[2])
            + blockpos.x * ADVANCE[0]
            + blockpos.y * ADVANCE[1]
            + blockpos.z * ADVANCE[2];
        let badv_x = ADVANCE[axes[0]];
        let badv_y = ADVANCE[axes[1]];

        // Figure out mapping to atlas texture indices
        let mut aidx1 = rect.x + rect.y * ATLAS_BIN;
        let (mut aadv_x, mut aadv_y) = (1, ATLAS_BIN);
        if rect.width != w {
            mem::swap(&mut aadv_x, &mut aadv_y);
        }

        // Actually write the terrain atlas
        let mut front_off = Int3::zero();
        front_off[axes[2]] = positive * 2 - 1;
        for y in 0..h {
            let mut bidx0 = bidx1;
            let mut aidx0 = aidx1;
            for x in 0..w {
                let mut bpos = blockpos;
                bpos[axes[0]] += x;
                bpos[axes[1]] += y;
                let color = self.color(self.block_buf[bidx0 as usize], bpos, bpos + front_off);
                self.atlas[aidx0 as usize] = (color[0], color[1], color[2], color[3]);
                bidx0 += badv_x;
                aidx0 += aadv_x;
            }
            bidx1 += badv_y;
            aidx1 += aadv_y;
        }

        // Figure out uv coordinates
        let uv00 = Vec2::new(rect.x as f32, rect.y as f32);
        let uv11 = Vec2::new((rect.x + rect.width) as f32, (rect.y + rect.height) as f32);

        let (mut uv10, mut uv01) = (Vec2::new(uv11.x, uv00.y), Vec2::new(uv00.x, uv11.y));
        if rect.width != w {
            mem::swap(&mut uv10, &mut uv01);
        }

        // Push the geometry out
        let mut p00 = blockpos;
        p00[axes[2]] += positive;
        let mut p10 = p00;
        p10[axes[0]] += w;
        let mut p01 = p00;
        p01[axes[1]] += h;
        let mut p11 = p10;
        p11[axes[1]] += h;

        let v = [
            self.vertex(p00.to_f32(), uv00),
            self.vertex(p10.to_f32(), uv10),
            self.vertex(p11.to_f32(), uv11),
            self.vertex(p01.to_f32(), uv01),
        ];
        self.mesh.add_face(v[0], v[1], v[2]);
        self.mesh.add_face(v[2], v[3], v[0]);
    }

    fn layer(&mut self, z: i32, dir: i32, axes: [usize; 3]) {
        // Stores which quads need meshing
        let mut pending = SurfaceBits::new();
        // Stores the potential coordinates for quad corners
        let mut corner_queue = mem::take(&mut self.corner_queue);
        corner_queue.clear();

        // Moves from the back block to the front block in index space
        let front = dir * ADVANCE[axes[2]];
        // 1 if pointing positively, 0 if negatively
        let positive = (dir + 1) / 2;

        // Get a flat 2d quadmap from this layer
        //let mut idx1 = (MARGIN + z) * ADVANCE[axes[2]] + MARGIN * ADVANCE[axes[1]] + MARGIN * ADVANCE[axes[0]];
        let mut idx1 = z * ADVANCE[axes[2]] + MARGIN * (ADVANCE[0] + ADVANCE[1] + ADVANCE[2]);
        for y in 0..CHUNK_SIZE as u8 {
            let mut idx0 = idx1;
            for x in 0..CHUNK_SIZE as u8 {
                if self.is_solid(idx0) && self.is_clear(idx0 + front) {
                    corner_queue.push((x, y));
                    pending.set(x, y);
                }
                idx0 += ADVANCE[axes[0]];
            }
            idx1 += ADVANCE[axes[1]];
        }

        // Mesh greedily
        for &(x0, y0) in corner_queue.iter() {
            if !pending.get(x0, y0) {
                // This block is contained in a previous quad
                continue;
            }
            // Greedily determine width then height
            let w = pending.march_right(x0, y0) - x0;
            let h = pending.march_up(x0, y0, w) - y0;
            pending.unset_rect(x0, y0, w, h);
            // Now build quad
            let posvirt = [x0 as i32, y0 as i32, z];
            let mut pos3d = Int3::zero();
            pos3d[axes[0]] = posvirt[0];
            pos3d[axes[1]] = posvirt[1];
            pos3d[axes[2]] = posvirt[2];
            self.quad(pos3d, positive, axes, w as i32, h as i32);
        }

        self.corner_queue = corner_queue;
    }

    fn visit_layers(&mut self) {
        // X+
        for x in 0..CHUNK_SIZE {
            self.layer(x, 1, [1, 2, 0]);
        }
        // X-
        for x in 0..CHUNK_SIZE {
            self.layer(x, -1, [2, 1, 0]);
        }
        // Y+
        for y in 0..CHUNK_SIZE {
            self.layer(y, 1, [2, 0, 1]);
        }
        // Y-
        for y in 0..CHUNK_SIZE {
            self.layer(y, -1, [0, 2, 1]);
        }
        // Z+
        for z in 0..CHUNK_SIZE {
            self.layer(z, 1, [0, 1, 2]);
        }
        // Z-
        for z in 0..CHUNK_SIZE {
            self.layer(z, -1, [1, 0, 2]);
        }
    }

    unsafe fn fetch_chunk(&mut self, chunk: ChunkRef, from: Int3, to: Int3, size: Int3) {
        let mut to_idx = ((to.z * BBUF_SIZE) + to.y) * BBUF_SIZE + to.x;
        if let Some(chunk) = chunk.blocks() {
            let mut from_idx = (((from.z << CHUNK_BITS) | from.y) << CHUNK_BITS) | from.x;
            for _z in 0..size.z {
                for _y in 0..size.y {
                    for _x in 0..size.x {
                        self.block_buf[to_idx as usize] = chunk.blocks[from_idx as usize];
                        from_idx += 1;
                        to_idx += 1;
                    }
                    from_idx += CHUNK_SIZE - size.x;
                    to_idx += BBUF_SIZE - size.x;
                }
                from_idx += CHUNK_SIZE * CHUNK_SIZE - CHUNK_SIZE * size.y;
                to_idx += BBUF_SIZE * BBUF_SIZE - BBUF_SIZE * size.y;
            }
        } else {
            let b = chunk.homogeneous_block_unchecked();
            for _z in 0..size.z {
                for _y in 0..size.y {
                    for _x in 0..size.x {
                        self.block_buf[to_idx as usize] = b;
                        to_idx += 1;
                    }
                    to_idx += BBUF_SIZE - size.x;
                }
                to_idx += BBUF_SIZE * BBUF_SIZE - BBUF_SIZE * size.y;
            }
        }
    }

    /// Fetch relevant chunk data and place it in a flat buffer.
    fn gather_chunks<'a>(&mut self, chunk_pos: ChunkPos, chunks: &ChunkStorage) -> Option<()> {
        let mut near_chunks = [ChunkRef::new_homogeneous(BlockData { data: 0 }); 27];
        {
            let mut idx = 0;
            for z in -1..=1 {
                for y in -1..=1 {
                    for x in -1..=1 {
                        let mut pos = chunk_pos;
                        pos.coords += [x, y, z];
                        near_chunks[idx] = chunks.chunk_at(pos)?;
                        idx += 1;
                    }
                }
            }
        }
        unsafe {
            let mut idx = 0;
            for z in 0..3 {
                for y in 0..3 {
                    for x in 0..3 {
                        const FROM: [i32; 3] = [CHUNK_SIZE - MARGIN, 0, 0];
                        const TO: [i32; 3] = [0, MARGIN, MARGIN + CHUNK_SIZE];
                        const SIZE: [i32; 3] = [MARGIN, CHUNK_SIZE, MARGIN];
                        self.fetch_chunk(
                            near_chunks[idx],
                            [FROM[x], FROM[y], FROM[z]].into(),
                            [TO[x], TO[y], TO[z]].into(),
                            [SIZE[x], SIZE[y], SIZE[z]].into(),
                        );
                        idx += 1;
                    }
                }
            }
        }
        Some(())
    }

    /// Generate texture noise.
    fn texture_noise(&mut self, chunk_pos: ChunkPos) {
        // OPTIMIZE: Generate noise rows by hashing hashes
        let mut idx = 0;
        let base_pos = chunk_pos.coords << CHUNK_BITS;
        for z in 0..=CHUNK_SIZE {
            for y in 0..=CHUNK_SIZE {
                for x in 0..=CHUNK_SIZE {
                    let rnd = fxhash::hash32(&(base_pos + [x, y, z]));
                    let val = 0x3f800000 | (rnd >> 9);
                    let val = f32::from_bits(val) * 2. - 3.;
                    self.noise_buf[idx] = val;
                    idx += 1;
                }
            }
        }
    }

    fn adjust_uv(&mut self) {
        self.atlas_h = (self.atlas_h as u32).next_power_of_two() as i32;
        let (w, h) = (ATLAS_BIN, self.atlas_h);
        let (adj_u, adj_v) = ((w as f32).recip(), (h as f32).recip());
        for v in self.mesh.vertices.iter_mut() {
            v.uv[0] *= adj_u;
            v.uv[1] *= adj_v;
        }
    }

    fn make_mesh<'a>(
        &mut self,
        chunk_pos: ChunkPos,
        chunks_raw: &'a RwLock<ChunkStorage>,
        chunks_store: &mut Option<RwLockReadGuard<'a, ChunkStorage>>,
    ) -> Option<()> {
        // Make sure the necessary chunks are available
        self.gather_chunks(
            chunk_pos,
            chunks_store.get_or_insert_with(|| chunks_raw.read()),
        )?;
        chunks_store.take();

        // Reset the atlas packer
        self.packer.reset(ATLAS_BIN, ATLAS_SIZE);
        //self.atlas.reset();
        self.atlas_h = 1;
        self.atlas.fill((0, 0, 0, 255));

        // Generate block texture noise
        self.texture_noise(chunk_pos);

        // Turn voxels into triangular meshes
        self.visit_layers();

        // Adjust uv coordinates to final texture size
        self.adjust_uv();

        Some(())
    }
}

struct LayerParams {
    x: [i32; 3],
    y: [i32; 3],
    mov: [i32; 3],
    flip: bool,
    normal: [i8; 3],
}

struct Mesher {
    /// Associate vertex positions with mesh vertex indices.
    /// Keep this info for two rows only, swapping them on every row change.
    /// At most 1 vertex can be stored per vertex position.
    /// If multiple block types coincide at a vertex, vertices are just duplicated.
    /// An empty cache position is signalled by a `VertIdx::max_value()`.
    vert_cache: Box<[(u8, VertIdx)]>,
    /// Store 2 layers of blocks, one being the front layer and one being the back layer.
    /// The meshing algorithm works on layers, so blocks are collected into this buffer and the
    /// algorithm is run for every layer in the chunk, in all 3 axes.
    block_buf: Box<[BlockData]>,
    /// Store a buffer of noise to give blocks texture.
    noise_buf: Box<[f32]>,
    /// Store the nature of all blocks.
    style: StyleTable,
    /// Store which blocks have skylight.
    skymap: [u8; Self::BLOCK_COUNT],
    /// Store the instructions to generate the color for every block type.
    block_textures: [BlockTexture; 256],
    /// The offset of the front block buffer within `block_buf`.
    front: i32,
    /// The offset of the back block buffer within `block_buf`.
    back: i32,
    /// The current position of the chunk being meshed.
    chunk_pos: ChunkPos,
    /// A temporary mesh buffer, storing vertex and index data.
    mesh: Mesh<SimpleVertex>,
}
impl Mesher {
    const EXTRA_BLOCKS: i32 = 1;

    const VERT_ROW: usize = (CHUNK_SIZE + 1) as usize;
    const BLOCK_COUNT: usize =
        ((CHUNK_SIZE + Self::EXTRA_BLOCKS * 2) * (CHUNK_SIZE + Self::EXTRA_BLOCKS * 2)) as usize;
    const NOISE_COUNT: usize = ((CHUNK_SIZE + 1) * (CHUNK_SIZE + 1) * (CHUNK_SIZE + 1)) as usize;

    const ADV_X: i32 = 1;
    const ADV_Y: i32 = CHUNK_SIZE + Self::EXTRA_BLOCKS * 2;

    pub fn new(textures: BlockTextures) -> Self {
        Self {
            vert_cache: vec![(0, 0); Self::VERT_ROW * 2].into_boxed_slice(),
            block_buf: vec![BlockData { data: 0 }; Self::BLOCK_COUNT * 2].into_boxed_slice(),
            noise_buf: vec![0.; Self::NOISE_COUNT].into_boxed_slice(),
            skymap: [0; Self::BLOCK_COUNT],
            style: StyleTable::new(&textures),
            block_textures: {
                let mut blocks: Uninit<[BlockTexture; 256]> = Uninit::uninit();
                for (src, dst) in textures.blocks.iter().zip(0..256) {
                    unsafe {
                        (blocks.as_mut_ptr() as *mut BlockTexture)
                            .offset(dst)
                            .write(src.take());
                    }
                }
                unsafe { blocks.assume_init() }
            },
            front: Self::BLOCK_COUNT as i32,
            back: 0,
            chunk_pos: ChunkPos {
                coords: Int3::zero(),
                dim: 0,
            },
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

    /// Expects a chunk-relative position.
    fn color_at(&mut self, id: u8, pos: Int3) -> Vec4 {
        let noise_at = |pos: [i32; 3]| {
            self.noise_buf[(pos[0]
                + pos[1] * (CHUNK_SIZE + 1)
                + pos[2] * ((CHUNK_SIZE + 1) * (CHUNK_SIZE + 1)))
                as usize]
        };
        let floorceil = |x: i32, i: usize| {
            let c = if x & ((1 << i) - 1) == 0 {
                x
            } else {
                x + (1 << i)
            };
            let m = (-1) << i;
            (x & m, c & m)
        };
        let tex = &self.block_textures[id as usize];
        let mut color = Vec4::from(tex.base);
        color += Vec4::from(tex.noise[0]) * noise_at(*pos);
        for i in 1..BlockTexture::NOISE_LEVELS {
            let (x0, x1) = floorceil(pos.x, i);
            let (y0, y1) = floorceil(pos.y, i);
            let (z0, z1) = floorceil(pos.z, i);
            let f = pos.lowbits(i as i32).to_f32() / (1 << i) as f32;
            let s = Lerp::lerp(
                &Lerp::lerp(
                    &Lerp::lerp(&noise_at([x0, y0, z0]), noise_at([x1, y0, z0]), f.x),
                    Lerp::lerp(&noise_at([x0, y1, z0]), noise_at([x1, y1, z0]), f.x),
                    f.y,
                ),
                Lerp::lerp(
                    &Lerp::lerp(&noise_at([x0, y0, z1]), noise_at([x1, y0, z1]), f.x),
                    Lerp::lerp(&noise_at([x0, y1, z1]), noise_at([x1, y1, z1]), f.x),
                    f.y,
                ),
                f.z,
            );
            color += Vec4::from(tex.noise[i]) * s;
        }
        color

        /*const BITS_PER_ELEM: usize = mem::size_of::<usize>() * 8;
        //Make sure buffer is created
        let buf = &mut self.color_bufs[id as usize];
        let ready_idx = id as usize / BITS_PER_ELEM;
        let ready_idx_bit = id as usize % BITS_PER_ELEM;
        if (self.ready_color_bufs[ready_idx] >> ready_idx_bit) & 1 == 0 {
            //Must create buffer
            // TODO: Fix color
            unsafe {
                self.colorizer.colorize(BlockColorArgs {
                    pos: self.chunk_pos.to_block_floor().offset(
                        pos[0] as i32,
                        pos[1] as i32,
                        pos[2] as i32,
                    ),
                    id,
                    out: mem::transmute(&mut *buf),
                });
            }
            self.ready_color_bufs[ready_idx] |= 1 << ready_idx_bit;
        }
        //Interpolate
        let pos0 = [
            pos[0] as u32 / CHUNK_COLOR_DOWNSCALE as u32,
            pos[1] as u32 / CHUNK_COLOR_DOWNSCALE as u32,
            pos[2] as u32 / CHUNK_COLOR_DOWNSCALE as u32,
        ];
        let pos1 = [
            (pos[0] as u32 + CHUNK_COLOR_DOWNSCALE as u32 - 1) / CHUNK_COLOR_DOWNSCALE as u32,
            (pos[1] as u32 + CHUNK_COLOR_DOWNSCALE as u32 - 1) / CHUNK_COLOR_DOWNSCALE as u32,
            (pos[2] as u32 + CHUNK_COLOR_DOWNSCALE as u32 - 1) / CHUNK_COLOR_DOWNSCALE as u32,
        ];
        let w = [
            (pos[0] as u32 % CHUNK_COLOR_DOWNSCALE as u32) as f32
                / (CHUNK_COLOR_DOWNSCALE - 1) as f32,
            (pos[1] as u32 % CHUNK_COLOR_DOWNSCALE as u32) as f32
                / (CHUNK_COLOR_DOWNSCALE - 1) as f32,
            (pos[2] as u32 % CHUNK_COLOR_DOWNSCALE as u32) as f32
                / (CHUNK_COLOR_DOWNSCALE - 1) as f32,
        ];
        macro_rules! at {
            (@ 0) => { pos0 };
            (@ 1) => { pos1 };
            ($x:tt, $y:tt, $z:tt) => {{
                Vec3::from(buf[
                    (at!(@ $x)[0]
                    + at!(@ $y)[1] * CHUNK_COLOR_BUF_WIDTH as u32
                    + at!(@ $z)[2] * (CHUNK_COLOR_BUF_WIDTH * CHUNK_COLOR_BUF_WIDTH) as u32) as usize
                ])
            }};
        }
        Lerp::lerp(
            &Lerp::lerp(
                &Lerp::lerp(&at!(0, 0, 0), at!(1, 0, 0), w[0]),
                Lerp::lerp(&at!(0, 1, 0), at!(1, 1, 0), w[0]),
                w[1],
            ),
            Lerp::lerp(
                &Lerp::lerp(&at!(0, 0, 1), at!(1, 0, 1), w[0]),
                Lerp::lerp(&at!(0, 1, 1), at!(1, 1, 1), w[0]),
                w[1],
            ),
            w[2],
        )
        .into()*/
    }

    fn get_vert(
        &mut self,
        params: &LayerParams,
        pos: Int2,
        blockpos: Int2,
        cache_offset: usize,
        idx: i32,
        id: u8,
    ) -> VertIdx {
        let tex = &self.block_textures[id as usize];
        let cache_idx = cache_offset as usize + pos.x as usize;
        let mut cached = self.vert_cache[cache_idx];
        if cached.1 == VertIdx::max_value() || !tex.smooth || id != cached.0 {
            //Create vertex
            //[NONE, BACK_ONLY, FRONT_ONLY, BOTH]
            const LIGHT_TABLE: [f32; 4] = [0.02, 0.0, -0.11, -0.11];
            let mut lightness = 1.;
            {
                let mut process = |idx| {
                    lightness += LIGHT_TABLE[(self.front(idx).is_solid(&self.style) as usize) << 1
                        | self.back(idx).is_solid(&self.style) as usize];
                };
                process(idx);
                process(idx - Self::ADV_X);
                process(idx - Self::ADV_Y);
                process(idx - Self::ADV_X - Self::ADV_Y);
            };
            let conv_2d_3d = |pos: Int2| {
                Int3::from([
                    pos.x & params.x[0] | pos.y & params.y[0] | params.mov[0],
                    pos.x & params.x[1] | pos.y & params.y[1] | params.mov[1],
                    pos.x & params.x[2] | pos.y & params.y[2] | params.mov[2],
                ])
            };
            let pos_3d = conv_2d_3d(pos);
            let blockpos_3d = conv_2d_3d(blockpos);
            let color_pos = if tex.smooth { pos_3d } else { blockpos_3d };
            let color = self.color_at(id, color_pos);
            const NTABLE: [i32; 3] = [0, 1, -1];
            let lightblock3d = blockpos_3d
                + [
                    NTABLE[(params.normal[0] as u8 >> 6) as usize],
                    NTABLE[(params.normal[1] as u8 >> 6) as usize],
                    NTABLE[(params.normal[2] as u8 >> 6) as usize],
                ];
            if self
                .skymap
                .get(
                    ((Self::EXTRA_BLOCKS + lightblock3d.y) * Self::ADV_Y
                        + lightblock3d.x
                        + Self::EXTRA_BLOCKS) as usize,
                )
                .map(|&z| lightblock3d.z < z as i32)
                .unwrap_or(true)
            {
                // In shadow
                lightness *= 0.15;
            }
            lightness *= 255.;
            let q = |f| (f * lightness) as u8;
            let color = [q(color[0]), q(color[1]), q(color[2]), q(color[3])];
            //Apply transform
            let vert = Vec3::new(pos_3d[0] as f32, pos_3d[1] as f32, pos_3d[2] as f32);
            cached = (id, self.mesh.add_vertex_simple(vert, params.normal, color));
            self.vert_cache[cache_idx] = cached;
        }
        cached.1
    }

    fn layer(&mut self, params: &LayerParams) {
        let mut idx = Self::ADV_X + Self::ADV_Y;
        for (_id, vidx) in self.vert_cache.iter_mut().take(Self::VERT_ROW) {
            *vidx = VertIdx::max_value();
        }
        let mut back = 0;
        let mut front = Self::VERT_ROW;
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                if self.back(idx).is_solid(&self.style) && !self.front(idx).is_solid(&self.style) {
                    //Place a face here
                    let pos = Int2::new([x, y]);
                    let id = self.back(idx).data;
                    let v00 = self.get_vert(params, pos + [0, 0], pos, back, idx, id);
                    let v01 =
                        self.get_vert(params, pos + [0, 1], pos, front, idx + Self::ADV_Y, id);
                    let v10 = self.get_vert(params, pos + [1, 0], pos, back, idx + Self::ADV_X, id);
                    let v11 = self.get_vert(
                        params,
                        pos + [1, 1],
                        pos,
                        front,
                        idx + Self::ADV_X + Self::ADV_Y,
                        id,
                    );
                    if params.flip {
                        self.mesh.add_face(v01, v11, v00);
                        self.mesh.add_face(v00, v11, v10);
                    } else {
                        self.mesh.add_face(v01, v00, v11);
                        self.mesh.add_face(v00, v10, v11);
                    }
                }
                idx += 1;
            }
            mem::swap(&mut front, &mut back);
            for (_id, vert) in self.vert_cache.iter_mut().skip(front).take(Self::VERT_ROW) {
                *vert = VertIdx::max_value();
            }
            idx += Self::ADV_Y - CHUNK_SIZE;
        }
    }

    pub fn make_mesh(
        &mut self,
        chunk_pos: ChunkPos,
        chunks: &[ChunkArc; 3 * 3 * 3],
    ) -> Mesh<SimpleVertex> {
        let chunk_at = |pos: Int3| &chunks[(pos[0] + pos[1] * 3 + pos[2] * (3 * 3)) as usize];
        let block_at = |pos: Int3| {
            let chunk_pos = pos >> CHUNK_BITS;
            let sub_pos = pos.lowbits(CHUNK_BITS);
            chunk_at(chunk_pos).sub_get(sub_pos)
        };
        let skymap_at = |pos: Int2| {
            let chunk_pos = pos >> CHUNK_BITS;
            let sub_pos = pos.lowbits(CHUNK_BITS);
            let z = chunk_at([chunk_pos.x, chunk_pos.y, 1].into()).sub_skymap(sub_pos);
            if z == CHUNK_SIZE as u8 {
                CHUNK_SIZE as u8
                    + chunk_at([chunk_pos.x, chunk_pos.y, 2].into()).sub_skymap(sub_pos)
            } else {
                z
            }
        };

        // Special case empty chunks
        if chunk_at([1, 1, 1].into()).is_clear() {
            //Empty chunks have no geometry
            return mem::take(&mut self.mesh);
        }

        // Special case solid chunks surrounded by solid chunks
        if chunk_at([1, 1, 1].into()).is_solid()
            && chunk_at([1, 1, 0].into()).is_solid()
            && chunk_at([1, 0, 1].into()).is_solid()
            && chunk_at([0, 1, 1].into()).is_solid()
            && chunk_at([2, 1, 1].into()).is_solid()
            && chunk_at([1, 2, 1].into()).is_solid()
            && chunk_at([1, 1, 2].into()).is_solid()
        {
            //Solid chunks surrounded by solid chunks have no visible geometry
            return mem::take(&mut self.mesh);
        }

        self.chunk_pos = chunk_pos;

        // Generate texture noise
        {
            let mut idx = 0;
            let base_pos = chunk_pos.coords << CHUNK_BITS;
            for z in 0..=CHUNK_SIZE {
                for y in 0..=CHUNK_SIZE {
                    for x in 0..=CHUNK_SIZE {
                        let rnd = fxhash::hash32(&(base_pos + [x, y, z]));
                        let val = 0x3f800000 | (rnd >> 9);
                        let val = f32::from_bits(val) * 2. - 3.;
                        self.noise_buf[idx] = val;
                        idx += 1;
                    }
                }
            }
        }

        // Fill skymap from chunk data
        {
            let mut sky_idx = 0;
            for y in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
                for x in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
                    self.skymap[sky_idx] = skymap_at([x, y].into());
                    sky_idx += 1;
                }
            }
        }

        // X
        time!(start xpass);
        for x in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
            time!(start gather);
            let mut idx = 0;
            for z in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
                for y in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
                    *self.front_mut(idx) = block_at([x, y, z].into());
                    idx += 1;
                }
            }
            time!(show gather);
            time!(start plus);
            if x > CHUNK_SIZE && x <= 2 * CHUNK_SIZE {
                //Facing `+`
                self.layer(&LayerParams {
                    x: [0, -1, 0],
                    y: [0, 0, -1],
                    mov: [x - CHUNK_SIZE, 0, 0],
                    flip: false,
                    normal: [i8::MAX, 0, 0],
                });
            }
            time!(show plus);
            time!(start minus);
            self.flip_bufs();
            if x >= CHUNK_SIZE && x < 2 * CHUNK_SIZE {
                //Facing `-`
                self.layer(&LayerParams {
                    x: [0, -1, 0],
                    y: [0, 0, -1],
                    mov: [x - CHUNK_SIZE, 0, 0],
                    flip: true,
                    normal: [i8::MIN, 0, 0],
                });
            }
            time!(show minus);
        }
        time!(show xpass);

        // Y
        for y in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
            let mut idx = 0;
            for z in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
                for x in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
                    *self.front_mut(idx) = block_at([x, y, z].into());
                    idx += 1;
                }
            }
            if y > CHUNK_SIZE {
                //Facing `+`
                self.layer(&LayerParams {
                    x: [-1, 0, 0],
                    y: [0, 0, -1],
                    mov: [0, y - CHUNK_SIZE, 0],
                    flip: true,
                    normal: [0, i8::MAX, 0],
                });
            }
            self.flip_bufs();
            if y >= CHUNK_SIZE && y < 2 * CHUNK_SIZE {
                //Facing `-`
                self.layer(&LayerParams {
                    x: [-1, 0, 0],
                    y: [0, 0, -1],
                    mov: [0, y - CHUNK_SIZE, 0],
                    flip: false,
                    normal: [0, i8::MIN, 0],
                });
            }
        }

        // Z
        for z in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
            let mut idx = 0;
            for y in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
                for x in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
                    *self.front_mut(idx) = block_at([x, y, z].into());
                    idx += 1;
                }
            }
            if z > CHUNK_SIZE {
                //Facing `+`
                self.layer(&LayerParams {
                    x: [-1, 0, 0],
                    y: [0, -1, 0],
                    mov: [0, 0, z - CHUNK_SIZE],
                    flip: false,
                    normal: [0, 0, i8::MAX],
                });
            }
            self.flip_bufs();
            if z >= CHUNK_SIZE && z < 2 * CHUNK_SIZE {
                //Facing `-`
                self.layer(&LayerParams {
                    x: [-1, 0, 0],
                    y: [0, -1, 0],
                    mov: [0, 0, z - CHUNK_SIZE],
                    flip: true,
                    normal: [0, 0, i8::MIN],
                });
            }
        }

        mem::take(&mut self.mesh)
    }

    fn mesh_portals(&mut self, chunk: &ChunkArc, gl_ctx: &Display) -> Vec<RawPortalMesh> {
        let mut portals = Vec::new();
        if let Some(chunk) = chunk.blocks() {
            for portal in chunk.portals() {
                // TODO: Merge multiple portals into a single mesh if they share the same jump.
                let pos = Int3::new([
                    portal.pos[0] as i32,
                    portal.pos[1] as i32,
                    portal.pos[2] as i32,
                ]);
                let size = Int3::new([
                    portal.size[0] as i32,
                    portal.size[1] as i32,
                    portal.size[2] as i32,
                ]);
                let center = pos + (size >> 1);
                if 0 <= center.x
                    && center.x < CHUNK_SIZE
                    && 0 <= center.y
                    && center.y < CHUNK_SIZE
                    && 0 <= center.z
                    && center.z < CHUNK_SIZE
                {
                    // The center of this portal is within this chunk, so add it to the chunk meshes
                    // Figure out the front side of the portal
                    let det = pos.max(Int3::zero());
                    let axis0 = portal.get_axis();
                    let (axis1, axis2) = if chunk.sub_get(det).is_portal(&self.style) {
                        // Positive side of this portal is `portal`
                        // Portal faces negative side
                        ((axis0 + 2) % 3, (axis0 + 1) % 3)
                    } else {
                        // Positive side of this portal is not `portal`
                        // Portal faces positive side
                        ((axis0 + 1) % 3, (axis0 + 2) % 3)
                    };

                    // Figure out all 4 portal corners
                    let v00 = pos;
                    let mut v10 = v00;
                    v10[axis1] += size[axis1];
                    let mut v01 = v00;
                    v01[axis2] += size[axis2];
                    let mut v11 = v10;
                    v11[axis2] += size[axis2];

                    // Make portal mesh
                    let mut mesh = Mesh::with_capacity(4, 2);
                    mesh.add_vertex_simple(v00.to_f32(), [0; 3], [0; 4]);
                    mesh.add_vertex_simple(v10.to_f32(), [0; 3], [0; 4]);
                    mesh.add_vertex_simple(v11.to_f32(), [0; 3], [0; 4]);
                    mesh.add_vertex_simple(v01.to_f32(), [0; 3], [0; 4]);
                    mesh.add_face(0, 1, 2);
                    mesh.add_face(0, 2, 3);

                    // Upload buffer to GPU
                    let buf_pkg = RawBufPackage::pack(mesh.make_buffer(gl_ctx));

                    // Pack it up
                    portals.push(RawPortalMesh {
                        mesh: mesh.into(),
                        buf: buf_pkg,
                        bounds: [
                            v00.to_f32(), // - x0 + x1 + x2,
                            v10.to_f32(), // - x0 - x1 + x2,
                            v11.to_f32(), // - x0 - x1 - x2,
                            v01.to_f32(), // - x0 + x1 - x2,
                        ],
                        jump: Int3::new(portal.jump).to_f64(),
                        dim: portal.dim,
                    });
                }
            }
        }
        portals
    }
}

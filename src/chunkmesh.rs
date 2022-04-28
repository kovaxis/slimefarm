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

#[derive(Clone, Deserialize)]
pub struct MesherCfg {
    /// Atlas width and height.
    /// The atlas width is fixed.
    /// However, the atlas height is only a maximum value, since smaller chunk atlases will be
    /// truncated to the next power-of-two size.
    /// Width and height should be a power of two.
    atlas_size: [i32; 2],
    /// Ambient occlusion exposure table.
    /// [0] = front clear, back clear
    /// [1] = back solid, front clear
    /// [2] = back clear, front solid
    /// [3] = back solid, front solid
    /// [4] = base exposure value
    exposure_table: [u8; 5],
    /// The offset that is applied to light uv coordinates.
    /// 0.5 means "the center of each pixel" and is usually the right value.
    light_uv_offset: f32,
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
        cfg: MesherCfg,
        info: WorldInfo,
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
                    mesher: Box::new(Mesher2::new(&cfg, TerrainMesherKind::new(info))),
                    _cfg: cfg,
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
    mesher: Box<Mesher2<TerrainMesherKind>>,
    send_bufs: Sender<ChunkMeshPkg>,
    _cfg: MesherCfg,
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
        self.mesher.make_chunk_mesh(pos, chunks, chunks_store)?;
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
            let tex = self.mesher.atlas.make_texture(&self.gl_ctx);
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

const MARGIN: i32 = 4;
const BBUF_SIZE: i32 = 2 * MARGIN + CHUNK_SIZE;
const BBUF_LEN: usize = (BBUF_SIZE * BBUF_SIZE * BBUF_SIZE) as usize;
const ADVANCE: [i32; 3] = [1, BBUF_SIZE, BBUF_SIZE * BBUF_SIZE];

const VERTSET_SIZE: i32 = CHUNK_SIZE + 1;
const VERTSET_LEN: usize = (VERTSET_SIZE * VERTSET_SIZE * VERTSET_SIZE) as usize;
const VERTSET_ADV: [i32; 3] = [1, VERTSET_SIZE, VERTSET_SIZE * VERTSET_SIZE];

const NOISE_SIZE: i32 = CHUNK_SIZE + 1;
const NOISE_LEN: usize = (NOISE_SIZE * NOISE_SIZE * NOISE_SIZE) as usize;

fn mask(w: u8) -> ChunkSizedInt {
    if w >= CHUNK_SIZE as u8 {
        !0
    } else {
        ((1 as ChunkSizedInt) << w).wrapping_sub(1)
    }
}

struct SurfaceBits {
    bits: [ChunkSizedInt; CHUNK_SIZE as usize],
}
impl SurfaceBits {
    fn new() -> Self {
        assert_eq!(mem::size_of::<ChunkSizedInt>() * 8, CHUNK_SIZE as usize);
        Self {
            bits: [0; CHUNK_SIZE as usize],
        }
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
        row |= ((1 as ChunkSizedInt) << x).wrapping_sub(1);
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

/// The atlas data for a single chunk.
pub struct AtlasChunk {
    data: Vec<(u8, u8, u8, u8)>,
    size: [i32; 2],
}
impl AtlasChunk {
    fn new(cfg: &MesherCfg) -> Self {
        Self {
            data: vec![(0, 0, 0, 0); (cfg.atlas_size[0] * cfg.atlas_size[1]) as usize],
            size: [0; 2],
        }
    }

    fn reset(&mut self, cfg: &MesherCfg) {
        // DEBUG: Fill with black so that wasted space can be easily seen on the atlas texture
        // Not necessary for production
        self.data.fill((0, 0, 0, 255));

        self.size = [cfg.atlas_size[0], 1];
    }

    fn round_size(&mut self) {
        self.size[0] = (self.size[0] as u32).next_power_of_two() as i32;
        self.size[1] = (self.size[1] as u32).next_power_of_two() as i32;
    }

    pub fn make_texture(&self, gl_ctx: &Display) -> Texture2d {
        let [w, h] = self.size;
        let raw_img = RawImage2d {
            data: Cow::Borrowed(&self.data[..(w * h) as usize]),
            width: w as u32,
            height: h as u32,
            format: glium::texture::ClientFormat::U8U8U8U8,
        };
        Texture2d::with_mipmaps(gl_ctx, raw_img, glium::texture::MipmapsOption::NoMipmap)
            .expect("failed to upload atlas texture for voxel data")
    }
}

struct PendingQuad {
    cuv: [u16; 2],
    luv: [u16; 2],
    pos: [u8; 3],
    axes: u8,
    size: [u8; 2],
}

bit_array! {
    BlocksetBits(BBUF_LEN);
}

bit_array! {
    VertsetBits(VERTSET_LEN);
}

struct Vertset {
    bits: VertsetBits,
}
impl Vertset {
    fn new() -> Self {
        Self {
            bits: VertsetBits::new(),
        }
    }

    fn clear(&mut self) {
        self.bits.clear();

        // Set all chunk boundaries to `true`, since we don't know where vertices are located in
        // the neighboring chunks
        // Too bad
        let mut idx = 0;
        for _y in 0..=CHUNK_SIZE {
            self.set_row(idx);
            idx += VERTSET_ADV[1];
        }
        for _z in 1..CHUNK_SIZE {
            self.set_row(idx);
            idx += VERTSET_ADV[1];
            for _y in 1..CHUNK_SIZE {
                self.set(idx);
                idx += VERTSET_ADV[1];
                self.set(idx - 1);
            }
            self.set_row(idx);
            idx += VERTSET_ADV[1];
        }
        for _y in 0..=CHUNK_SIZE {
            self.set_row(idx);
            idx += VERTSET_ADV[1];
        }
    }

    fn to_idx(pos: Int3) -> i32 {
        pos.x + VERTSET_SIZE * pos.y + VERTSET_SIZE * VERTSET_SIZE * pos.z
    }

    /// Set a `VERTSET_SIZE`-length row of vertex bits.
    fn set_row(&mut self, idx: i32) {
        self.bits.set_row(idx as usize, VERTSET_SIZE as usize);
    }

    fn set(&mut self, idx: i32) {
        self.bits.set(idx as usize);
    }

    fn get(&self, idx: i32) -> bool {
        self.bits.get(idx as usize)
    }
}

crate::terrain::light_spreader! {
    LightSpreader(BBUF_SIZE);
}

struct TerrainMesherKind {
    /// Knows which blocks are solid and which blocks are clear.
    style: StyleTable,
    /// Texture parameters for every block type.
    block_textures: [BlockTexture; 256],
    /// A flat buffer containing all blocks in the current chunk and some margin of the neighboring
    /// chunks.
    block_buf: [BlockData; BBUF_LEN],
    /// A buffer equivalent to `block_buf`, but containing light values.
    light_buf: [u8; BBUF_LEN],
    /// Stores the noise that is used to give block colors a little texture.
    noise_buf: [f32; NOISE_LEN],
    /// Subsystem responsible for spreading light correctly.
    light_spread: LightSpreader,
}
impl TerrainMesherKind {
    fn new(info: WorldInfo) -> Self {
        Self {
            block_buf: [BlockData { data: 0 }; BBUF_LEN],
            light_buf: [0; BBUF_LEN],
            style: StyleTable::new(&info),
            block_textures: arr![i => info.blocks[i].clone(); 256],
            noise_buf: [0.; NOISE_LEN],
            light_spread: LightSpreader::new(&info),
        }
    }
}
impl MesherKind for TerrainMesherKind {
    fn is_solid(&self, idx: i32) -> bool {
        self.block_buf[idx as usize].is_solid(&self.style)
    }

    fn is_clear(&self, idx: i32) -> bool {
        self.block_buf[idx as usize].is_clear(&self.style)
    }

    fn light_conf(&self) -> &LightingConf {
        &self.light_spread.light_modes[self.light_spread.mode]
    }

    fn light_value(&self, idx: i32) -> u8 {
        self.light_buf[idx as usize]
    }

    fn color(this: &mut Mesher2<Self>, blockidx: i32, blockpos: Int3, _airpos: Int3) -> [u8; 4] {
        let block = this.kind.block_buf[blockidx as usize];
        // Get the noise value at a particular location
        let noise_at = |pos: [i32; 3]| {
            this.kind.noise_buf
                [(pos[0] + NOISE_SIZE * pos[1] + NOISE_SIZE * NOISE_SIZE * pos[2]) as usize]
        };
        // Snap to the nearest grid-aligned integer points
        let snap = |x: i32, i: usize| {
            let mask = (!0) << i;
            (x & mask, (x & mask) + (1 << i))
        };
        // Texture parameters for this block
        let tex = &this.kind.block_textures[block.data as usize];
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
}

pub trait MesherKind: Sized {
    fn is_solid(&self, idx: i32) -> bool;
    fn is_clear(&self, idx: i32) -> bool;
    fn light_conf(&self) -> &LightingConf;
    fn light_value(&self, idx: i32) -> u8;
    fn color(this: &mut Mesher2<Self>, blockidx: i32, blockpos: Int3, airpos: Int3) -> [u8; 4];
}

pub struct Mesher2<D> {
    kind: D,

    /// A queue buffer for mesh quad corners.
    /// Used in the quad generation stage.
    corner_queue: Vec<(u8, u8)>,
    /// A queue buffer for mesh quads.
    /// This is the intermediate state between quad generation and triangle generation.
    quad_queue: Vec<PendingQuad>,
    /// A bitmask with 1 bit per vertex.
    /// The bit is set if a quad touches this vertex, and unset otherwise.
    /// Used to generate triangles from quads in a way that produces no T-junctions.
    vertex_set: Vertset,
    /// A rectangle packer, to associate mesh quads with texture coordinates in the chunk atlas.
    packer: DensePacker,
    /// The chunk atlas texture, storing color and lighting data for every quad.
    pub(crate) atlas: AtlasChunk,
    /// Stores the final mesh geometry.
    pub(crate) mesh: Mesh<VoxelVertex>,
    /// Configurable constants and behaviour.
    cfg: MesherCfg,
}
impl<D: MesherKind> Mesher2<D> {
    pub fn new(cfg: &MesherCfg, kind: D) -> Self {
        Self {
            kind,
            corner_queue: Vec::with_capacity((CHUNK_SIZE * CHUNK_SIZE) as usize),
            quad_queue: vec![],
            vertex_set: Vertset::new(),
            mesh: default(),
            atlas: AtlasChunk::new(cfg),
            packer: DensePacker::new(1, 1),
            cfg: cfg.clone(),
        }
    }

    /// Does not reset kind-specific data.
    fn reset_atlas(&mut self) {
        self.mesh.clear();
        self.packer
            .reset(self.cfg.atlas_size[0], self.cfg.atlas_size[1]);
        self.atlas.reset(&self.cfg);
    }

    /// Does not reset kind-specific data.
    fn reset_meshqueues(&mut self) {
        self.quad_queue.clear();
        self.vertex_set.clear();
    }

    #[inline]
    fn is_solid(&self, idx: i32) -> bool {
        self.kind.is_solid(idx)
    }
    #[inline]
    fn is_clear(&self, idx: i32) -> bool {
        self.kind.is_clear(idx)
    }
    #[inline]
    fn skylight(&self, idx: i32) -> u8 {
        self.kind.light_value(idx)
    }

    fn light(&mut self, _vertpos: Int3, blockidx: i32, axes: [usize; 3], positive: i32) -> [u8; 4] {
        // Index offsets to get from `blockidx` to the front and the back block
        let (front, back) = ((2 * positive - 1) * ADVANCE[axes[2]], 0);

        // Gather occlusion data for nearby blocks
        let mut exposure = self.cfg.exposure_table[4];
        let mut obs_count = 0;
        for y in -1..=0 {
            for x in -1..=0 {
                let b =
                    self.is_solid(blockidx + back + x * ADVANCE[axes[0]] + y * ADVANCE[axes[1]]);
                let f =
                    self.is_solid(blockidx + front + x * ADVANCE[axes[0]] + y * ADVANCE[axes[1]]);
                obs_count += f as usize;
                let occ = (b as usize) | ((f as usize) << 1);
                exposure = exposure.wrapping_add(self.cfg.exposure_table[occ]);
            }
        }

        // Check how much skylight reaches this block
        let mode = self.kind.light_conf();
        //let mut skylight = mode.light.base;
        let mut lacc = 0;
        for y in -1..=0 {
            for x in -1..=0 {
                let l =
                    self.skylight(blockidx + front + x * ADVANCE[axes[0]] + y * ADVANCE[axes[1]]);
                //lacc = lacc.max(l as u32);
                lacc += l as u32;
            }
        }
        // Compensate for obstructed blocks
        const SCALETABLE: [u32; 5] = [
            (4. / 4. * (1 << 20) as f64) as u32,
            (4. / 3. * (1 << 20) as f64) as u32,
            (4. / 2. * (1 << 20) as f64) as u32,
            (4. / 1. * (1 << 20) as f64) as u32,
            1 << 20,
        ];
        lacc = (lacc * SCALETABLE[obs_count]) >> 20;
        // Scale according to runtime parameters
        let skylight = mode.light.base + ((lacc * mode.light.mul) >> mode.light.shr);
        let skylight = skylight as u8;

        [exposure, skylight, 0, 255]
    }

    fn quad(&mut self, blockpos: Int3, positive: i32, axes: [usize; 3], w: i32, h: i32) {
        // Allocate space in the atlas' color texture
        let crect = loop {
            match self.packer.pack(w, h, true) {
                Some(rect) => break rect,
                None => {
                    // TODO: Grow atlas bin in this case
                    println!("ran out of color atlas space for chunk!");
                    return;
                }
            }
        };
        self.atlas.size[1] = self.atlas.size[1].max(crect.y + crect.height);

        // Allocate space in the atlas' light texture
        let lrect = loop {
            match self.packer.pack(w + 1, h + 1, true) {
                Some(rect) => break rect,
                None => {
                    // TODO: Grow atlas bin in this case
                    println!("ran out of light atlas space for chunk!");
                    return;
                }
            }
        };
        self.atlas.size[1] = self.atlas.size[1].max(lrect.y + lrect.height);

        // Figure out the mapping to block buffer indices
        let bidx_base = MARGIN * (ADVANCE[0] + ADVANCE[1] + ADVANCE[2])
            + blockpos.x * ADVANCE[0]
            + blockpos.y * ADVANCE[1]
            + blockpos.z * ADVANCE[2];
        let badv_x = ADVANCE[axes[0]];
        let badv_y = ADVANCE[axes[1]];

        // Figure out mapping to color atlas texture indices
        let mut cidx1 = crect.x + crect.y * self.cfg.atlas_size[0];
        let (mut cadv_x, mut cadv_y) = (1, self.cfg.atlas_size[0]);
        if crect.width != w {
            mem::swap(&mut cadv_x, &mut cadv_y);
        }

        // Figure out mapping to light atlas texture indices
        let mut lidx1 = lrect.x + lrect.y * self.cfg.atlas_size[0];
        let (mut ladv_x, mut ladv_y) = (1, self.cfg.atlas_size[0]);
        if lrect.width != w + 1 {
            mem::swap(&mut ladv_x, &mut ladv_y);
        }

        // Write the color atlas
        {
            let mut bidx1 = bidx_base;
            let mut front_off = Int3::zero();
            front_off[axes[2]] = positive * 2 - 1;
            for y in 0..h {
                let mut bidx0 = bidx1;
                let mut cidx0 = cidx1;
                for x in 0..w {
                    let mut bpos = blockpos;
                    bpos[axes[0]] += x;
                    bpos[axes[1]] += y;
                    let color = D::color(self, bidx0, bpos, bpos + front_off);
                    self.atlas.data[cidx0 as usize] = (color[0], color[1], color[2], color[3]);
                    bidx0 += badv_x;
                    cidx0 += cadv_x;
                }
                bidx1 += badv_y;
                cidx1 += cadv_y;
            }
        }

        // Write the light atlas
        {
            let mut bidx1 = bidx_base;
            let mut vpos_base = blockpos;
            vpos_base[axes[2]] += positive;
            for y in 0..=h {
                let mut bidx0 = bidx1;
                let mut lidx0 = lidx1;
                for x in 0..=w {
                    let mut vpos = vpos_base;
                    vpos[axes[0]] += x;
                    vpos[axes[1]] += y;
                    let light = self.light(vpos, bidx0, axes, positive);
                    self.atlas.data[lidx0 as usize] = (light[0], light[1], light[2], light[3]);
                    bidx0 += badv_x;
                    lidx0 += ladv_x;
                }
                bidx1 += badv_y;
                lidx1 += ladv_y;
            }
        }

        // Queue the geometry
        // We cannot push out geometry just now because of the damn t-junctions
        // We'll have to add vertices to the quads depending on which vertices are in use
        let mut quadpos = blockpos;
        quadpos[axes[2]] += positive;
        let cflip = ((crect.width != w) as u16) << 15;
        let lflip = ((lrect.width != w + 1) as u16) << 15;
        let quadaxes = axes[0] as u8 | ((axes[1] as u8) << 2);
        self.quad_queue.push(PendingQuad {
            cuv: [crect.x as u16 | cflip, crect.y as u16],
            luv: [lrect.x as u16 | lflip, lrect.y as u16],
            pos: [quadpos.x as u8, quadpos.y as u8, quadpos.z as u8],
            axes: quadaxes,
            size: [w as u8, h as u8],
        });

        // Mark that these vertices have quads relying on them
        let idx = Vertset::to_idx(quadpos);
        self.vertex_set.set(idx);
        self.vertex_set.set(idx + w * VERTSET_ADV[axes[0]]);
        self.vertex_set
            .set(idx + w * VERTSET_ADV[axes[0]] + h * VERTSET_ADV[axes[1]]);
        self.vertex_set.set(idx + h * VERTSET_ADV[axes[1]]);
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

    /// Generate triangles from the quad queue, avoiding T-junctions.
    fn produce_triangles(&mut self) {
        // Figure out atlas size to determine the proper UV coordinates of vertices
        self.atlas.round_size();
        let uv_scale = Vec2::new(
            (self.atlas.size[0] as f32).recip(),
            (self.atlas.size[1] as f32).recip(),
        );
        let luv_offset = uv_scale * self.cfg.light_uv_offset;

        // Go through all quads, generating the necessary triangles to not only cover the quads,
        // but also to make sure there are no T-junctions, ie. vertex to edge joints
        // To do this, the entire boundary of the quad is iterated, checking if there is queued
        // vertex at each point.
        // At the points marked to have vertices, a new vertex is created and the necessary
        // triangle is added.
        // The actual algorithm is a bit more complicated because many points are collinear, so
        // the points for the triangles must be picked carefully
        // OPTIMIZE: Make iteration through the vertset faster by using `count_zeros`, just like
        // the greedy meshing technique.
        // To make `count_zeros` efficient in all 3 axes, 3 different vertsets would need to be
        // kept track of, each organized in a different order (XYZ, YZX, ZXY).
        // Benchmark whether the read savings counteract the extra writes.
        let quad_queue = mem::take(&mut self.quad_queue);
        for quad in quad_queue.iter() {
            #[derive(Copy, Clone)]
            struct Vert {
                off: Int2,
                idx: i32,
            }

            // Extract data from the tiny `quad` struct
            let qpos = Int3::new([quad.pos[0] as i32, quad.pos[1] as i32, quad.pos[2] as i32]);
            let qvert = Vert {
                off: Int2::zero(),
                idx: Vertset::to_idx(qpos),
            };
            let axes = [
                (quad.axes & 0b11) as usize,
                ((quad.axes >> 2) & 0b11) as usize,
            ];
            let qcuv = Int2::new([(quad.cuv[0] & (u16::MAX >> 1)) as i32, quad.cuv[1] as i32]);
            let cuv_axes = if quad.cuv[0] >> 15 == 0 {
                [0, 1]
            } else {
                [1, 0]
            };
            let qluv = Int2::new([(quad.luv[0] & (u16::MAX >> 1)) as i32, quad.luv[1] as i32]);
            let luv_axes = if quad.luv[0] >> 15 == 0 {
                [0, 1]
            } else {
                [1, 0]
            };

            macro_rules! adv {
                ($vert:ident, $axis:expr, $dir:expr) => {{
                    $vert.off[$axis] += $dir;
                    $vert.idx += $dir * VERTSET_ADV[axes[$axis]];
                }};
            }

            let vertex = |this: &mut Self, vert: Vert| {
                let mut pos = qpos;
                pos[axes[0]] += vert.off.x;
                pos[axes[1]] += vert.off.y;
                let mut cuv = qcuv;
                cuv[cuv_axes[0]] += vert.off.x;
                cuv[cuv_axes[1]] += vert.off.y;
                let mut luv = qluv;
                luv[luv_axes[0]] += vert.off.x;
                luv[luv_axes[1]] += vert.off.y;
                this.mesh.add_vertex(VoxelVertex {
                    pos: [pos.x as u8, pos.y as u8, pos.z as u8, 1],
                    cuv: (cuv.to_f32() * uv_scale).into(),
                    luv: (luv.to_f32() * uv_scale + luv_offset).into(),
                    //uv: rand::random(),
                })
            };

            // Find first vertex along the first/bottom edge
            let (mut vbase0, mut xbase0) = (0, 0);
            {
                let mut vert = qvert;
                for x in 1..=quad.size[0] {
                    adv!(vert, 0, 1);
                    if self.vertex_set.get(vert.idx) {
                        xbase0 = x;
                        vbase0 = vertex(self, vert);
                        break;
                    }
                }
            }

            // Find first vertex along the third/top edge
            let (mut vbase1, mut xbase1) = (0, 0);
            {
                let mut vert = qvert;
                adv!(vert, 0, quad.size[0] as i32);
                adv!(vert, 1, quad.size[1] as i32);
                for x in (0..quad.size[0]).rev() {
                    adv!(vert, 0, -1);
                    if self.vertex_set.get(vert.idx) {
                        xbase1 = x;
                        vbase1 = vertex(self, vert);
                        break;
                    }
                }
            }

            // Do left side using base0
            {
                let mut vert = qvert;
                let mut vidx = 0;
                vertex(self, vert);
                // Iterate along the fourth/left edge
                for _y in 1..=quad.size[1] {
                    adv!(vert, 1, 1);
                    if self.vertex_set.get(vert.idx) {
                        vidx = vertex(self, vert);
                        self.mesh.add_face(vbase0, vidx, vidx - 1);
                    }
                }
                // Iterate along the third/top edge
                for _x in 1..xbase1 {
                    adv!(vert, 0, 1);
                    if self.vertex_set.get(vert.idx) {
                        vidx = vertex(self, vert);
                        self.mesh.add_face(vbase0, vidx, vidx - 1);
                    }
                }
                // Add final triangle to join both bases
                self.mesh.add_face(vbase0, vbase1, vidx);
            }

            // Do right side using base1
            {
                let mut vert = qvert;
                adv!(vert, 0, quad.size[0] as i32);
                adv!(vert, 1, quad.size[1] as i32);
                let mut vidx = 0;
                vertex(self, vert);
                // Iterate along the second/right edge
                for _y in (0..quad.size[1]).rev() {
                    adv!(vert, 1, -1);
                    if self.vertex_set.get(vert.idx) {
                        vidx = vertex(self, vert);
                        self.mesh.add_face(vbase1, vidx, vidx - 1);
                    }
                }
                // Iterate along the first/bottom edge
                for _x in (xbase0..quad.size[0]).rev() {
                    adv!(vert, 0, -1);
                    if self.vertex_set.get(vert.idx) {
                        vidx = vertex(self, vert);
                        self.mesh.add_face(vbase1, vidx, vidx - 1);
                    }
                }
                // Add final triangle to join both bases
                // This triangle complements the analogous triangle for base0
                self.mesh.add_face(vbase1, vbase0, vidx);
            }
        }
        self.quad_queue = quad_queue;
    }
}
impl Mesher2<TerrainMesherKind> {
    fn fetch_chunk(&mut self, chunk: ChunkRef, from: Int3, to: Int3, size: Int3) {
        let mut to_idx = ((to.z * BBUF_SIZE) + to.y) * BBUF_SIZE + to.x;
        if let Some(chunk) = chunk.blocks() {
            let mut from_idx = (((from.z << CHUNK_BITS) | from.y) << CHUNK_BITS) | from.x;
            for _z in 0..size.z {
                for _y in 0..size.y {
                    for _x in 0..size.x {
                        self.kind.block_buf[to_idx as usize] = chunk.blocks[from_idx as usize];
                        self.kind.light_buf[to_idx as usize] = chunk.skylight[from_idx as usize];
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
            let b = chunk.homogeneous_block();
            let l = chunk.homogeneous_skylight();
            for _z in 0..size.z {
                for _y in 0..size.y {
                    for _x in 0..size.x {
                        self.kind.block_buf[to_idx as usize] = b;
                        self.kind.light_buf[to_idx as usize] = l;
                        to_idx += 1;
                    }
                    to_idx += BBUF_SIZE - size.x;
                }
                to_idx += BBUF_SIZE * BBUF_SIZE - BBUF_SIZE * size.y;
            }
        }
    }

    /// Fetch relevant chunk data and place it in a flat buffer.
    fn gather_chunks<'a>(&mut self, chunk_pos: ChunkPos, chunks: &ChunkStorage) -> Option<bool> {
        // Fetch chunk references of the surrounding terrain
        let mut near_chunks = [ChunkRef::placeholder(); 27];
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

        // Check homogeneous chunk hints
        if near_chunks[13].is_homogeneous() {
            // If the center is clear, nothing to do
            if near_chunks[13]
                .homogeneous_block()
                .is_clear(&self.kind.style)
            {
                return Some(false);
            }
            // If the center and the 6 chunks around it are solid, nothing to do
            const SOLID_REQ: &[usize] = &[13, 4, 10, 12, 14, 16, 22];
            if SOLID_REQ.iter().all(|&idx| {
                near_chunks[idx].is_homogeneous()
                    && near_chunks[idx]
                        .homogeneous_block()
                        .is_solid(&self.kind.style)
            }) {
                return Some(false);
            }
        }

        // Copy block and light data
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

        // Initialize light spreader
        if let Some(data) = near_chunks[13].blocks() {
            let mut base_pos = chunk_pos.chunk_to_block();
            base_pos.coords -= [MARGIN; 3];
            self.kind.light_spread.reset(base_pos, data);
        }

        Some(true)
    }

    fn seed_light(&mut self, pos: Int3) {
        self.kind
            .light_spread
            .spread_from(&mut self.kind.light_buf, &self.kind.block_buf, pos);
    }

    fn spread_light(&mut self) {
        // Seed light from the 6 planes that separate chunks
        // OPTIMIZE: Only spread across chunks, reducing 65% of checks
        for y in 0..BBUF_SIZE {
            for x in 0..BBUF_SIZE {
                self.seed_light([x, y, MARGIN - 1].into());
                self.seed_light([x, y, MARGIN].into());
            }
        }
        for z in 0..BBUF_SIZE {
            for x in 0..BBUF_SIZE {
                self.seed_light([x, MARGIN - 1, z].into());
                self.seed_light([x, MARGIN, z].into());
            }
            for x in 0..BBUF_SIZE {
                self.seed_light([x, MARGIN + CHUNK_SIZE - 1, z].into());
                self.seed_light([x, MARGIN + CHUNK_SIZE, z].into());
            }
            for y in 0..BBUF_SIZE {
                self.seed_light([MARGIN - 1, y, z].into());
                self.seed_light([MARGIN, y, z].into());
            }
            for y in 0..BBUF_SIZE {
                self.seed_light([MARGIN + CHUNK_SIZE - 1, y, z].into());
                self.seed_light([MARGIN + CHUNK_SIZE, y, z].into());
            }
        }
        for y in 0..BBUF_SIZE {
            for x in 0..BBUF_SIZE {
                self.seed_light([x, y, MARGIN + CHUNK_SIZE - 1].into());
                self.seed_light([x, y, MARGIN + CHUNK_SIZE].into());
            }
        }

        // Spread initial light seeds
        self.kind
            .light_spread
            .spread_pending(&mut self.kind.light_buf, &self.kind.block_buf);
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
                    self.kind.noise_buf[idx] = val;
                    idx += 1;
                }
            }
        }
    }

    fn make_chunk_mesh<'a>(
        &mut self,
        chunk_pos: ChunkPos,
        chunks_raw: &'a RwLock<ChunkStorage>,
        chunks_store: &mut Option<RwLockReadGuard<'a, ChunkStorage>>,
    ) -> Option<()> {
        // Make sure the necessary chunks are available
        let nonempty = self.gather_chunks(
            chunk_pos,
            chunks_store.get_or_insert_with(|| chunks_raw.read()),
        )?;
        if !nonempty {
            return Some(());
        }
        chunks_store.take();

        // Reset everything
        self.reset_atlas();
        self.reset_meshqueues();

        // Update light using info from neighboring chunks
        // Commmented: debug hack to disable mesh-time light spreading
        /*
        if self.block_buf
            [(MARGIN * ADVANCE[0] + MARGIN * ADVANCE[1] + MARGIN * ADVANCE[2]) as usize]
            .data
            != 0
        {*/
        self.spread_light();
        // }

        // Generate block texture noise
        self.texture_noise(chunk_pos);

        // Turn voxels into a list of quads
        self.visit_layers();

        // Take these quads and produce triangles
        self.produce_triangles();

        Some(())
    }
}

#[derive(Deserialize)]
pub struct ModelMesherCfg {
    pub cfg: MesherCfg,
    pub transparency: [u8; 4],
    pub light_value: u8,
    pub lighting: LightingConf,
}

pub struct ModelMesherKind {
    /// The color used as transparency.
    transparency: [u8; 4],
    /// The uniform virtual light lighting everything up.
    uniform_light: u8,
    /// A flat buffer containing colors for all blocks and a little margin.
    color_buf: [[u8; 4]; BBUF_LEN],
    /// Configuration for how to do lighting.
    light_conf: LightingConf,
}
impl ModelMesherKind {
    pub fn new(cfg: &ModelMesherCfg) -> Self {
        Self {
            transparency: cfg.transparency,
            uniform_light: cfg.light_value,
            color_buf: [cfg.transparency; BBUF_LEN],
            light_conf: cfg.lighting.clone(),
        }
    }
}
impl MesherKind for ModelMesherKind {
    fn is_solid(&self, idx: i32) -> bool {
        self.color_buf[idx as usize] != self.transparency
    }

    fn is_clear(&self, idx: i32) -> bool {
        self.color_buf[idx as usize] == self.transparency
    }

    fn light_conf(&self) -> &LightingConf {
        &self.light_conf
    }

    fn light_value(&self, idx: i32) -> u8 {
        if self.color_buf[idx as usize] == self.transparency {
            self.uniform_light
        } else {
            0
        }
    }

    fn color(this: &mut Mesher2<Self>, blockidx: i32, _blockpos: Int3, _airpos: Int3) -> [u8; 4] {
        this.kind.color_buf[blockidx as usize]
    }
}
impl Mesher2<ModelMesherKind> {
    fn blit_color(&mut self, data: &[[u8; 4]], dsize: Int3, from: Int3, to: Int3, size: Int3) {
        let mut to_idx = (to.z * BBUF_SIZE + to.y) * BBUF_SIZE + to.x;
        let mut from_idx = (from.z * dsize.y + from.y) * dsize.x + from.x;
        for _z in 0..size.z {
            for _y in 0..size.y {
                for _x in 0..size.x {
                    self.kind.color_buf[to_idx as usize] = data[from_idx as usize];
                    from_idx += 1;
                    to_idx += 1;
                }
                from_idx += dsize.x - size.x;
                to_idx += BBUF_SIZE - size.x;
            }
            from_idx += dsize.x * dsize.y - dsize.x * size.y;
            to_idx += BBUF_SIZE * BBUF_SIZE - BBUF_SIZE * size.y;
        }
    }

    fn fill_transparency(&mut self, to: Int3, size: Int3) {
        let mut to_idx = ((to.z * BBUF_SIZE) + to.y) * BBUF_SIZE + to.x;
        for _z in 0..size.z {
            for _y in 0..size.y {
                for _x in 0..size.x {
                    self.kind.color_buf[to_idx as usize] = self.kind.transparency;
                    to_idx += 1;
                }
                to_idx += BBUF_SIZE - size.x;
            }
            to_idx += BBUF_SIZE * BBUF_SIZE - BBUF_SIZE * size.y;
        }
    }

    fn fetch_color(&mut self, data: &[[u8; 4]], dsize: Int3, mut from: Int3) {
        // Clip blit window
        from -= [MARGIN; 3];
        let mut to = Int3::zero();
        let mut size = Int3::splat(BBUF_SIZE);
        for axis in 0..3 {
            if from[axis] < 0 {
                to[axis] -= from[axis];
                size[axis] += from[axis];
                from[axis] = 0;
            }
        }
        for axis in 0..3 {
            let extra = from[axis] + size[axis] - dsize[axis];
            if extra > 0 {
                size[axis] -= extra;
            }
            if size[axis] <= 0 {
                return;
            }
        }

        // Blit color
        for axis in 0..3 {
            if to[axis] > 0 || to[axis] + size[axis] < BBUF_SIZE {
                self.fill_transparency(Int3::zero(), Int3::splat(BBUF_SIZE));
                break;
            }
        }
        self.blit_color(data, dsize, from, to, size);
    }

    fn make_submesh(&mut self) {
        // Reset only the mesh-producing machinery
        self.reset_meshqueues();

        // Turn voxels into a list of quads
        self.visit_layers();

        // Take these quads and produce triangles
        self.produce_triangles();
    }

    pub fn make_model_mesh(&mut self, data: &[[u8; 4]], dsize: Int3) {
        assert_eq!(data.len(), (dsize.x * dsize.y * dsize.z) as usize);

        self.reset_atlas();
        for z in (0..dsize.z).step_by(CHUNK_SIZE as usize) {
            for y in (0..dsize.z).step_by(CHUNK_SIZE as usize) {
                for x in (0..dsize.x).step_by(CHUNK_SIZE as usize) {
                    let offset = Int3::new([x, y, z]);
                    self.fetch_color(data, dsize, offset);
                    let top = self.mesh.vertices.len();
                    self.make_submesh();
                    for v in self.mesh.vertices.iter_mut().skip(top) {
                        for axis in 0..3 {
                            v.pos[axis] += offset[axis] as u8;
                        }
                    }
                }
            }
        }
    }
}

/*

// Old mesher implementation, still here for reference

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
*/

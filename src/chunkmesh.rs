use crate::prelude::*;
use common::terrain::GridKeeper;

pub(crate) struct BufPackage {
    pub pos: ChunkPos,
    pub mesh: Mesh,
    pub buf: Option<(RawVertexPackage<SimpleVertex>, RawIndexPackage<VertIdx>)>,
}

const AVERAGE_WEIGHT: f32 = 0.02;

pub struct MesherHandle {
    pub(crate) recv_bufs: Receiver<BufPackage>,
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
            close: false.into(),
            avg_mesh_time: 0f32.into(),
            avg_upload_time: 0f32.into(),
        });
        let gl_ctx = state
            .sec_gl_ctx
            .take()
            .expect("no secondary opengl context available for mesher");
        let (send_bufs, recv_bufs) = channel::bounded(512);
        let thread = {
            let shared = shared.clone();
            thread::spawn(move || {
                let gl_ctx =
                    Display::from_gl_window(gl_ctx).expect("failed to create headless gl context");
                run_mesher(
                    MesherState {
                        shared,
                        chunks,
                        gl_ctx,
                        send_bufs,
                    },
                    textures,
                );
            })
        };
        Self {
            thread: Some(thread),
            recv_bufs,
            shared,
        }
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
    close: AtomicCell<bool>,
    pub avg_mesh_time: AtomicCell<f32>,
    pub avg_upload_time: AtomicCell<f32>,
}

struct MesherState {
    shared: Arc<SharedState>,
    chunks: Arc<RwLock<ChunkStorage>>,
    gl_ctx: Display,
    send_bufs: Sender<BufPackage>,
}

fn run_mesher(state: MesherState, textures: BlockTextures) {
    // The maximum amount of chunks to mesh in a single walk.
    const MAX_MESH: i32 = 2;

    let mut mesher = Box::new(Mesher::new(textures));
    let mut meshed = GridKeeper::new(32, ChunkPos([0, 0, 0]));
    let mut last_stall_warning = Instant::now();
    loop {
        //Mission: find a meshable chunk
        let mut chunks_store = Some(state.chunks.read());
        meshed.set_center(chunks_store.as_ref().unwrap().center());
        //Look for candidate chunks
        let mut order_idx = 0;
        let mut meshed_count = 0;
        while meshed_count < MAX_MESH && order_idx < chunks_store.as_ref().unwrap().priority.len() {
            let idx = chunks_store.as_ref().unwrap().priority[order_idx];
            order_idx += 1;
            (|| {
                let chunks = chunks_store.as_ref().unwrap();
                let mesh_start = Instant::now();
                let chunk_slot = chunks.get_by_idx(idx);
                let chunk = chunk_slot.as_arc_ref()?;
                let pos = chunks.sub_idx_to_pos(idx);
                let meshed_mark = meshed.get_mut(pos)?;
                if *meshed_mark {
                    //Make sure not to duplicate work
                    return None;
                }
                //Get all of its adjacent neighbors
                macro_rules! build_neighbors {
                    (@_, _, _) => {{
                        chunk
                    }};
                    (@$x:expr, $y:expr, $z:expr) => {{
                        chunks.chunk_slot_at(pos + [$x-1, $y-1, $z-1])?.as_arc_ref()?
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
                    _, _, _;
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
                //Free the chunk lock, since we have `Arc` references to all necessary chunks
                drop(chunks);
                chunks_store.take();
                //Mesh the chunk
                let mesh = mesher.make_mesh(pos, &neighbors);
                {
                    //Keep meshing statistics
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
                    Some((buf.vertex.into_raw_package(), buf.index.into_raw_package()))
                };
                //Send the buffer back
                let buf_pkg = BufPackage {
                    pos,
                    mesh,
                    buf: buf_pkg,
                };
                match state.send_bufs.try_send(buf_pkg) {
                    Ok(()) => {}
                    Err(err) => {
                        if err.is_full() {
                            //Channel is full, make sure to unlock chunks
                            let stall_start = Instant::now();
                            //RwLockReadGuard::unlocked(&mut chunks, || {
                            let _ = state.send_bufs.send(err.into_inner());
                            //});
                            let now = Instant::now();
                            if now - last_stall_warning > Duration::from_millis(1500) {
                                last_stall_warning = now;
                                eprintln!(
                                    "meshing thread stalled for {}ms",
                                    (now - stall_start).as_millis()
                                );
                            }
                        } else {
                            //Channel closed, just discard it, we're gonna exit soon anyways
                        }
                    }
                }
                //Mark chunk as meshed
                *meshed_mark = true;
                //Acquire chunk lock again if there are still potential chunks to mesh
                meshed_count += 1;
                if meshed_count < MAX_MESH {
                    chunks_store = Some(state.chunks.read());
                }
                Some(())
            })();
        }
        chunks_store.take();
        if state.shared.close.load() {
            break;
        }
        if meshed_count > 0 {
            //Make sure the bookkeeper thread gets a chance to write on the chunks
            //RwLockReadGuard::bump(&mut chunks);
        } else {
            //Pause for a while
            thread::park_timeout(Duration::from_millis(50));
            if state.shared.close.load() {
                break;
            }
            //chunks = state.chunks.read();
        }
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
    vert_cache: [(u8, VertIdx); Self::VERT_ROW * 2],
    /// Store 2 layers of blocks, one being the front layer and one being the back layer.
    /// The meshing algorithm works on layers, so blocks are collected into this buffer and the
    /// algorithm is run for every layer in the chunk, in all 3 axes.
    block_buf: [BlockData; Self::BLOCK_COUNT * 2],
    /// Store a buffer of noise to give blocks texture.
    noise_buf: [f32; Self::NOISE_COUNT],
    /// Store which blocks are solid.
    solid: SolidTable,
    /// Store the instructions to generate the color for every block type.
    block_textures: [BlockTexture; 256],
    /// The offset of the front block buffer within `block_buf`.
    front: i32,
    /// The offset of the back block buffer within `block_buf`.
    back: i32,
    /// The current position of the chunk being meshed.
    chunk_pos: ChunkPos,
    /// A temporary mesh buffer, storing vertex and index data.
    mesh: Mesh,
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
            vert_cache: [(0, 0); Self::VERT_ROW * 2],
            block_buf: [BlockData { data: 0 }; Self::BLOCK_COUNT * 2],
            noise_buf: [0.; Self::NOISE_COUNT],
            solid: SolidTable::new(&textures),
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
            chunk_pos: ChunkPos([0, 0, 0]),
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
            let f = pos.lowbits(i as i32).to_f32_floor() / (1 << i) as f32;
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
                    lightness += LIGHT_TABLE[(self.front(idx).is_solid(&self.solid) as usize) << 1
                        | self.back(idx).is_solid(&self.solid) as usize];
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
            let color_pos = if tex.smooth {
                pos_3d
            } else {
                conv_2d_3d(blockpos)
            };
            lightness *= 255.;
            let color = self.color_at(id, color_pos);
            let q = |f| (f * lightness) as u8;
            let color = [q(color[0]), q(color[1]), q(color[2]), q(color[3])];
            //Apply transform
            let vert = Vec3::new(pos_3d[0] as f32, pos_3d[1] as f32, pos_3d[2] as f32);
            cached = (id, self.mesh.add_vertex(vert, params.normal, color));
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
                if self.back(idx).is_solid(&self.solid) && !self.front(idx).is_solid(&self.solid) {
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

    pub fn make_mesh(&mut self, chunk_pos: ChunkPos, chunks: &[ChunkArc; 3 * 3 * 3]) -> Mesh {
        let chunk_at = |pos: Int3| &chunks[(pos[0] + pos[1] * 3 + pos[2] * (3 * 3)) as usize];
        let block_at = |pos: Int3| {
            let chunk_pos = pos >> CHUNK_BITS;
            let sub_pos = pos.lowbits(CHUNK_BITS);
            chunk_at(chunk_pos).sub_get(sub_pos)
        };

        // Special case empty chunks
        if chunk_at([1, 1, 1].into()).is_nonsolid() {
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
            let base_pos = chunk_pos << CHUNK_BITS;
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

        // X
        for x in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
            let mut idx = 0;
            for z in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
                for y in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
                    *self.front_mut(idx) = block_at([x, y, z].into());
                    idx += 1;
                }
            }
            if x > CHUNK_SIZE {
                //Facing `+`
                self.layer(&LayerParams {
                    x: [0, -1, 0],
                    y: [0, 0, -1],
                    mov: [x - CHUNK_SIZE, 0, 0],
                    flip: false,
                    normal: [i8::MAX, 0, 0],
                });
            }
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
        }

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
}

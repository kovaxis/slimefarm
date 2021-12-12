use crate::prelude::*;
use common::{
    terrain::GridKeeper,
    worldgen::{
        BlockColorArgs, BlockColorizer, ChunkColorBuf, CHUNK_COLOR_BUF_LEN, CHUNK_COLOR_BUF_WIDTH,
        CHUNK_COLOR_DOWNSCALE,
    },
};

pub(crate) type BufPackage = (
    ChunkPos,
    Option<(RawVertexPackage<SimpleVertex>, RawIndexPackage<VertIdx>)>,
);

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
        colorizer: Box<dyn BlockColorizer>,
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
                    colorizer,
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

fn run_mesher(state: MesherState, colorizer: Box<dyn BlockColorizer>) {
    let mut mesher = Mesher::new(colorizer);
    let mut chunks = state.chunks.read();
    let mut meshed = GridKeeper::new(32, ChunkPos([0, 0, 0]));
    let mut last_stall_warning = Instant::now();
    loop {
        //Mission: find a meshable chunk
        meshed.set_center(chunks.center());
        //Look for candidate chunks
        let mut order_idx = 0;
        let mut did_mesh = false;
        while !did_mesh && order_idx < chunks.priority.len() {
            let idx = chunks.priority[order_idx];
            order_idx += 1;
            (|| {
                let mesh_start = Instant::now();
                let chunk_slot = chunks.get_by_idx(idx);
                let chunk = chunk_slot.as_ref()?;
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
                        chunks.chunk_at(pos.offset($x-1, $y-1, $z-1))?
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
                match state.send_bufs.try_send((pos, buf_pkg)) {
                    Ok(()) => {}
                    Err(err) => {
                        if err.is_full() {
                            //Channel is full, make sure to unlock chunks
                            let stall_start = Instant::now();
                            RwLockReadGuard::unlocked(&mut chunks, || {
                                let _ = state.send_bufs.send(err.into_inner());
                            });
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
                //Exit loop
                did_mesh = true;
                Some(())
            })();
        }
        if state.shared.close.load() {
            break;
        }
        if did_mesh {
            //Make sure the bookkeeper thread gets a chance to write on the chunks
            RwLockReadGuard::bump(&mut chunks);
        } else {
            //Pause for a while
            drop(chunks);
            thread::park_timeout(Duration::from_millis(50));
            if state.shared.close.load() {
                break;
            }
            chunks = state.chunks.read();
        }
    }
}

struct LayerParams {
    x: [i32; 3],
    y: [i32; 3],
    mov: [i32; 3],
    flip: bool,
    normal: u8,
}

struct Mesher {
    colorizer: Box<dyn BlockColorizer>,
    vert_cache: [VertIdx; Self::VERT_ROW * 2],
    block_buf: [BlockData; Self::BLOCK_COUNT * 2],
    front: i32,
    back: i32,
    chunk_pos: ChunkPos,
    ready_color_bufs: [usize; 256 / mem::size_of::<usize>() / 8],
    color_bufs: Box<[ChunkColorBuf]>,
    mesh: Mesh,
}
impl Mesher {
    const EXTRA_BLOCKS: i32 = 1;

    const VERT_ROW: usize = (CHUNK_SIZE + 1) as usize;
    const BLOCK_COUNT: usize =
        ((CHUNK_SIZE + Self::EXTRA_BLOCKS * 2) * (CHUNK_SIZE + Self::EXTRA_BLOCKS * 2)) as usize;

    const ADV_X: i32 = 1;
    const ADV_Y: i32 = CHUNK_SIZE + Self::EXTRA_BLOCKS * 2;

    pub fn new(colorizer: Box<dyn BlockColorizer>) -> Self {
        Self {
            colorizer,
            vert_cache: [0; Self::VERT_ROW * 2],
            block_buf: [BlockData { data: 0 }; Self::BLOCK_COUNT * 2],
            front: Self::BLOCK_COUNT as i32,
            back: 0,
            chunk_pos: ChunkPos([0, 0, 0]),
            ready_color_bufs: [0; 256 / mem::size_of::<usize>() / 8],
            color_bufs: vec![[[0., 0., 0.]; CHUNK_COLOR_BUF_LEN]; 256].into_boxed_slice(),
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
    fn color_at(&mut self, id: u8, pos: [i32; 3]) -> [f32; 3] {
        const BITS_PER_ELEM: usize = mem::size_of::<usize>() * 8;
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
        .into()
    }

    fn get_vert(
        &mut self,
        params: &LayerParams,
        x: i32,
        y: i32,
        cache_offset: usize,
        idx: i32,
    ) -> VertIdx {
        let cache_idx = cache_offset as usize + x as usize;
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
            };
            let pos_3d = [
                x & params.x[0] | y & params.y[0] | params.mov[0],
                x & params.x[1] | y & params.y[1] | params.mov[1],
                x & params.x[2] | y & params.y[2] | params.mov[2],
            ];
            let color = self.color_at(self.back(idx).data, pos_3d);
            lightness *= 256.;
            let color_normal = [
                (color[0] * lightness) as u8,
                (color[1] * lightness) as u8,
                (color[2] * lightness) as u8,
                params.normal,
            ];
            //Apply transform
            let vert = Vec3::new(pos_3d[0] as f32, pos_3d[1] as f32, pos_3d[2] as f32);
            cached = self.mesh.add_vertex(vert, color_normal);
            self.vert_cache[cache_idx] = cached;
        }
        cached
    }

    fn layer(&mut self, params: &LayerParams) {
        let mut idx = Self::ADV_X + Self::ADV_Y;
        for vidx in self.vert_cache.iter_mut().take(Self::VERT_ROW) {
            *vidx = VertIdx::max_value();
        }
        let mut back = 0;
        let mut front = Self::VERT_ROW;
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                if self.back(idx).is_solid() && !self.front(idx).is_solid() {
                    //Place a face here
                    let v00 = self.get_vert(params, x, y, back, idx);
                    let v01 = self.get_vert(params, x, y + 1, front, idx + Self::ADV_Y);
                    let v10 = self.get_vert(params, x + 1, y, back, idx + Self::ADV_X);
                    let v11 =
                        self.get_vert(params, x + 1, y + 1, front, idx + Self::ADV_X + Self::ADV_Y);
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
            for vert in self.vert_cache.iter_mut().skip(front).take(Self::VERT_ROW) {
                *vert = VertIdx::max_value();
            }
            idx += Self::ADV_Y - CHUNK_SIZE;
        }
    }

    pub fn make_mesh(&mut self, chunk_pos: ChunkPos, chunks: &[ChunkRef; 3 * 3 * 3]) -> &Mesh {
        let chunk_at = |pos: [u32; 3]| {
            &chunks[(pos[0] + pos[1] * 3 as u32 + pos[2] * (3 * 3) as u32) as usize]
        };
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
            chunk_at(chunk_pos).sub_get(sub_pos)
        };

        self.mesh.clear();

        // Special case empty chunks
        if chunk_at([1, 1, 1]).is_empty() {
            //Empty chunks have no geometry
            return &self.mesh;
        }

        // Special case solid chunks surrounded by solid chunks
        if chunk_at([1, 1, 1]).is_solid()
            && chunk_at([1, 1, 0]).is_solid()
            && chunk_at([1, 0, 1]).is_solid()
            && chunk_at([0, 1, 1]).is_solid()
            && chunk_at([2, 1, 1]).is_solid()
            && chunk_at([1, 2, 1]).is_solid()
            && chunk_at([1, 1, 2]).is_solid()
        {
            //Solid chunks surrounded by solid chunks have no visible geometry
            return &self.mesh;
        }

        self.chunk_pos = chunk_pos;
        for ready_bits in self.ready_color_bufs.iter_mut() {
            *ready_bits = 0;
        }

        // X
        for x in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
            let mut idx = 0;
            for z in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
                for y in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
                    *self.front_mut(idx) = block_at([x, y, z]);
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
                    normal: 0,
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
                    normal: 1,
                });
            }
        }

        // Y
        for y in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
            let mut idx = 0;
            for z in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
                for x in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
                    *self.front_mut(idx) = block_at([x, y, z]);
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
                    normal: 2,
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
                    normal: 3,
                });
            }
        }

        // Z
        for z in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
            let mut idx = 0;
            for y in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
                for x in CHUNK_SIZE - Self::EXTRA_BLOCKS..2 * CHUNK_SIZE + Self::EXTRA_BLOCKS {
                    *self.front_mut(idx) = block_at([x, y, z]);
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
                    normal: 4,
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
                    normal: 5,
                });
            }
        }

        &self.mesh
    }
}

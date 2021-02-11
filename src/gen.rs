use crate::{
    perlin::{NoiseScaler, PerlinNoise},
    prelude::*,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenConfig {
    pub seed: u64,
}

pub struct GeneratorHandle {
    last_send: Instant,
    last_send_count: usize,
    tmp_vec: Vec<ChunkPos>,
    request: Arc<Mutex<RequestPriority>>,
    pub chunks: Receiver<(ChunkPos, Box<Chunk>)>,
    threads: Vec<JoinHandle<()>>,
}
impl GeneratorHandle {
    pub fn new(cfg: GenConfig) -> Self {
        let request = Arc::new(Mutex::new(RequestPriority::new()));
        let (chunk_send, chunk_recv) = channel::bounded(64);
        let thread_count = (num_cpus::get() / 2).max(1);
        eprintln!("using {} worldgen threads", thread_count);
        let mut threads = Vec::with_capacity(thread_count);
        for _ in 0..thread_count {
            let request = request.clone();
            let chunk_send = chunk_send.clone();
            let cfg = cfg.clone();
            let join_handle = thread::spawn(move || {
                let state = GenState {
                    request,
                    chunks: chunk_send,
                };
                gen_thread(state, cfg)
            });
            threads.push(join_handle);
        }
        Self {
            request,
            last_send: Instant::now(),
            last_send_count: 16,
            tmp_vec: default(),
            chunks: chunk_recv,
            threads,
        }
    }

    pub fn swap_request_vec(&mut self, swap_in: Vec<ChunkPos>) -> Vec<ChunkPos> {
        mem::replace(&mut self.tmp_vec, swap_in)
    }

    /// Hint at how many chunks should be requested.
    pub fn request_hint(&self) -> usize {
        self.last_send_count
    }

    /// Request a list of chunks, from highest priority to lowest priority.
    /// Newer requests override older requests.
    pub fn request(&mut self, chunks: &[ChunkPos]) {
        let mut req = self.request.lock();
        let now = Instant::now();
        req.wanted.clear();
        req.wanted.extend(chunks);
        self.last_send_count = chunks.len();
        if req.last_idle > self.last_send {
            //The last request was too small, send more now
            self.last_send_count *= 2;
        } else if self.last_send - req.last_idle >= Duration::from_millis(1000) {
            //Too long without idling, try lowering the send count gradually
            self.last_send_count = (self.last_send_count as isize - 1).max(2) as usize;
        }
        self.last_send = now;
        drop(req);
        self.unpark_all();
    }

    fn unpark_all(&self) {
        for join in self.threads.iter() {
            join.thread().unpark();
        }
    }
}
impl Drop for GeneratorHandle {
    fn drop(&mut self) {
        self.request.lock().close = true;
        self.chunks = channel::never();
        self.unpark_all();
        for join in self.threads.drain(..) {
            join.join().unwrap();
        }
    }
}

pub struct RequestPriority {
    close: bool,
    wanted: VecDeque<ChunkPos>,
    last_idle: Instant,
}
impl RequestPriority {
    pub fn new() -> Self {
        Self {
            close: false,
            wanted: default(),
            last_idle: Instant::now(),
        }
    }
}

#[derive(Clone)]
struct GenState {
    request: Arc<Mutex<RequestPriority>>,
    chunks: Sender<(ChunkPos, Box<Chunk>)>,
}

fn gen_thread(gen: GenState, cfg: GenConfig) {
    const RECENT_COUNT: usize = 16;
    let mut recent_requests = VecDeque::<ChunkPos>::with_capacity(RECENT_COUNT);
    let noise_gen = PerlinNoise::new(
        cfg.seed,
        &[(128., 1.), (64., 0.5), (32., 0.25), (16., 0.125)],
    );
    let mut noise_buf = vec![0.; (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize];
    let mut noise_scaler = NoiseScaler::new(CHUNK_SIZE / 4, CHUNK_SIZE as f32);
    'outer: loop {
        //Acquire the next most prioritized chunk coordinate
        let mut req = gen.request.lock();
        let pos = 'inner: loop {
            match req.wanted.pop_front() {
                Some(pos) => {
                    if recent_requests.iter().any(|p| p == &pos) {
                        continue 'inner;
                    }
                    recent_requests.truncate(RECENT_COUNT - 1);
                    recent_requests.push_front(pos);
                    break 'inner pos;
                }
                None => {
                    if req.close {
                        break 'outer;
                    } else {
                        drop(req);
                        thread::park();
                        continue 'outer;
                    }
                }
            }
        };
        drop(req);
        //Generate this chunk
        //measure_time!(start gen_chunk);
        let mut chunk = Box::new(Chunk::new());
        //Generate noise in bulk for the entire chunk
        /*
        noise_gen.noise_block(
            [
                (pos[0] * CHUNK_SIZE) as f64,
                (pos[1] * CHUNK_SIZE) as f64,
                (pos[2] * CHUNK_SIZE) as f64,
            ],
            CHUNK_SIZE as f64,
            CHUNK_SIZE,
            &mut noise_buf,
            false,
        );
        // */
        //*
        noise_scaler.fill(
            &noise_gen,
            [
                (pos[0] * CHUNK_SIZE) as f64,
                (pos[1] * CHUNK_SIZE) as f64,
                (pos[2] * CHUNK_SIZE) as f64,
            ],
            CHUNK_SIZE as f64,
        );
        // */
        //Transform bulk noise into block ids
        let mut idx = 0;
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let real_y = pos[1] * CHUNK_SIZE + y;
                    //let noise = noise_buf[idx] - real_y as f32 * 0.04;
                    let noise = noise_scaler.get(Vec3::new(x as f32, y as f32, z as f32))
                        - real_y as f32 * 0.0; //0.021;
                    let normalized = noise / (0.4 + noise.abs());
                    if normalized > 0. {
                        chunk.blocks[idx].data = 15 + (normalized * 240.) as u8;
                    } else {
                        chunk.blocks[idx].data = 0;
                    }
                    idx += 1;
                }
            }
        }
        //measure_time!(end gen_chunk);
        //Send chunk back to main thread
        let _ = gen.chunks.send((pos, chunk));
    }
}

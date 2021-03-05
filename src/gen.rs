use crate::{
    perlin::{NoiseScaler, PerlinNoise},
    prelude::*,
    terrain::BookKeepHandle,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenConfig {
    pub seed: u64,
}

pub struct GeneratorHandle {
    pub close: Arc<AtomicCell<bool>>,
    threads: Vec<JoinHandle<()>>,
}
impl GeneratorHandle {
    pub(crate) fn new(
        cfg: GenConfig,
        chunks: Arc<RwLock<ChunkStorage>>,
        bookkeep: &BookKeepHandle,
    ) -> Self {
        let close = Arc::new(AtomicCell::new(false));
        let thread_count = (num_cpus::get() / 2).max(1);
        eprintln!("using {} worldgen threads", thread_count);
        let mut threads = Vec::with_capacity(thread_count);
        for i in 0..thread_count {
            let close = close.clone();
            let cfg = cfg.clone();
            let chunks = chunks.clone();
            let generated_send = bookkeep.generated_send.clone();
            let chunk_reuse = bookkeep.chunk_reuse.clone();

            let join_handle = thread::Builder::new()
                .name(format!("worldgen_{}", i))
                .spawn(move || {
                    let state = GenState {
                        chunks,
                        close,
                        generated_send,
                        chunk_reuse,
                    };
                    gen_thread(state, cfg)
                })
                .unwrap();
            threads.push(join_handle);
        }
        Self { close, threads }
    }

    fn unpark_all(&self) {
        for join in self.threads.iter() {
            join.thread().unpark();
        }
    }
}
impl Drop for GeneratorHandle {
    fn drop(&mut self) {
        measure_time!(start close_worldgen);
        self.close.store(true);
        self.unpark_all();
        for join in self.threads.drain(..) {
            join.join().unwrap();
        }
        measure_time!(end close_worldgen);
    }
}

#[derive(Clone)]
struct GenState {
    close: Arc<AtomicCell<bool>>,
    chunks: Arc<RwLock<ChunkStorage>>,
    generated_send: Sender<(ChunkPos, Box<Chunk>)>,
    chunk_reuse: Receiver<Box<Chunk>>,
}

fn gen_thread(gen: GenState, cfg: GenConfig) {
    let noise_gen = PerlinNoise::new(
        cfg.seed,
        &[(128., 1.), (64., 0.5), (32., 0.25), (16., 0.125)],
    );
    let mut noise_scaler = NoiseScaler::new(CHUNK_SIZE / 4, CHUNK_SIZE as f32);
    'outer: loop {
        //Acquire the next most prioritized chunk coordinate
        let pos = 'inner: loop {
            //eprintln!("acquiring chunk to generate");
            if gen.close.load() {
                break 'outer;
            }
            let chunks = gen.chunks.read();
            for &idx in chunks.priority.iter() {
                let slot = chunks.get_by_idx(idx);
                if !slot.present() {
                    //This chunk is not available
                    //Make sure it hasn't been taken by another generator thread
                    if !slot.generating.compare_and_swap(false, true) {
                        //Generate this chunk
                        break 'inner chunks.sub_idx_to_pos(idx);
                    }
                }
            }
            drop(chunks);
            if gen.close.load() {
                break 'outer;
            }
            thread::park_timeout(Duration::from_millis(50));
        };
        //Generate this chunk
        //measure_time!(start gen_chunk);
        let mut chunk = match gen.chunk_reuse.try_recv() {
            Ok(chunk) => chunk,
            Err(_) => unsafe {
                //Illegal shit in order to avoid unstable features and costly initialization
                let mut vec = Vec::with_capacity(1);
                vec.set_len(1);
                let mut boxed_slice: Box<[Chunk]> = vec.into_boxed_slice();
                let chunk = Box::from_raw(boxed_slice.as_mut_ptr());
                mem::forget(boxed_slice);
                chunk
            },
        };
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
                        - real_y as f32 * 0.008;
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
        if gen.generated_send.send((pos, chunk)).is_err() {
            break;
        }
    }
}

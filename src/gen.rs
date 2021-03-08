use crate::{prelude::*, terrain::BookKeepHandle};
use common::worldgen::{ChunkFillArgs, ChunkFillRet, ChunkGenerator};

const AVERAGE_WEIGHT: f32 = 0.02;

pub struct GeneratorHandle {
    shared: Arc<SharedState>,
    threads: Vec<JoinHandle<()>>,
}
impl GeneratorHandle {
    pub(crate) fn new(
        cfg: &[u8],
        chunks: Arc<RwLock<ChunkStorage>>,
        bookkeep: &BookKeepHandle,
    ) -> Result<Self> {
        let shared = Arc::new(SharedState {
            close: false.into(),
            avg_gen_time: 0f32.into(),
        });
        let thread_count = (num_cpus::get() / 2).max(1);
        eprintln!("using {} worldgen threads", thread_count);
        let mut threads = Vec::with_capacity(thread_count);
        let mut generators = (0..thread_count)
            .map(|_| worldgen::new_generator(cfg))
            .collect::<Result<Vec<_>>>()?;
        if let Some((first, rest)) = generators.split_first_mut() {
            for gen in rest {
                // SAFETY: Generators must have the same shared part type
                // This should happen because all of them were created with the same configstring.
                unsafe {
                    gen.share_with(first);
                }
            }
        }
        for (i, generator) in (0..thread_count).zip(generators) {
            let shared = shared.clone();
            let chunks = chunks.clone();
            let generated_send = bookkeep.generated_send.clone();

            let join_handle = thread::Builder::new()
                .name(format!("worldgen_{}", i))
                .spawn(move || {
                    let state = GenState {
                        chunks,
                        shared,
                        generated_send,
                    };
                    gen_thread(state, generator)
                })
                .unwrap();
            threads.push(join_handle);
        }
        Ok(Self { shared, threads })
    }

    fn unpark_all(&self) {
        for join in self.threads.iter() {
            join.thread().unpark();
        }
    }
}
impl ops::Deref for GeneratorHandle {
    type Target = SharedState;
    fn deref(&self) -> &SharedState {
        &self.shared
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

pub struct SharedState {
    close: AtomicCell<bool>,
    pub avg_gen_time: AtomicCell<f32>,
}

struct GenState {
    shared: Arc<SharedState>,
    chunks: Arc<RwLock<ChunkStorage>>,
    generated_send: Sender<(ChunkPos, ChunkBox)>,
}

fn gen_thread(gen: GenState, mut generator: ChunkGenerator) {
    let mut last_stall_warning = Instant::now();
    'outer: loop {
        //Acquire the next most prioritized chunk coordinate
        let center;
        let pos;
        'inner: loop {
            //eprintln!("acquiring chunk to generate");
            if gen.shared.close.load() {
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
                        center = chunks.center();
                        pos = chunks.sub_idx_to_pos(idx);
                        break 'inner;
                    }
                }
            }
            drop(chunks);
            if gen.shared.close.load() {
                break 'outer;
            }
            thread::park_timeout(Duration::from_millis(50));
        }
        //Generate this chunk
        let gen_start = Instant::now();
        let mut chunk: ChunkBox = unsafe {
            //Illegal shit in order to avoid costly initialization
            common::arena::alloc().assume_init()
            //common::arena::alloc().init_zero()
        };
        let result = generator.fill(ChunkFillArgs {
            center,
            pos,
            chunk: &mut *chunk,
        });
        {
            //Keep chunkgen statistics
            //Dont care about data races here, after all it's just stats
            //Therefore, dont synchronize
            let time = gen_start.elapsed().as_secs_f32();
            let old_time = gen.shared.avg_gen_time.load();
            let new_time = old_time + (time - old_time) * AVERAGE_WEIGHT;
            gen.shared.avg_gen_time.store(new_time);
        }
        if result.is_some() {
            //Send chunk back to main thread
            match gen.generated_send.try_send((pos, chunk)) {
                Ok(()) => {}
                Err(err) if err.is_full() => {
                    let stall_start = Instant::now();
                    if gen.generated_send.send(err.into_inner()).is_err() {
                        break;
                    }
                    let now = Instant::now();
                    if now - last_stall_warning > Duration::from_millis(1500) {
                        last_stall_warning = now;
                        eprintln!(
                            "worldgen thread stalled for {}ms",
                            (now - stall_start).as_millis()
                        );
                    }
                }
                Err(_) => {
                    //Disconnected
                    break;
                }
            }
        }
    }
}

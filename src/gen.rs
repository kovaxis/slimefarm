use std::num::NonZeroU32;

use crate::{
    prelude::*,
    terrain::{by_dist_up_to, gen_priorities, BookKeepHandle},
};
use common::{
    terrain::{GridKeeper, GridSlot},
    worldgen::{ChunkFiller, GenStore},
};

const AVERAGE_WEIGHT: f32 = 0.02;

pub struct GeneratorHandle {
    shared: Arc<SharedState>,
    thread: Option<JoinHandle<Result<()>>>,
}
impl GeneratorHandle {
    /// # Safety
    ///
    /// `share_with` must be of the same type that the `cfg` bytestring will generate.
    pub(crate) unsafe fn new(
        cfg: &[u8],
        chunks: Arc<RwLock<ChunkStorage>>,
        bookkeep: &BookKeepHandle,
    ) -> Result<Self> {
        let shared = Arc::new(SharedState {
            close: false.into(),
            avg_gen_time: 0f32.into(),
        });
        let cfg = cfg.to_vec();
        //let (colorizer_send, colorizer_recv) = channel::bounded(0);
        let join_handle = {
            let shared = shared.clone();
            let chunks = chunks.clone();
            let generated_send = bookkeep.generated_send.clone();
            thread::Builder::new()
                .name("worldgen".to_string())
                .spawn(move || {
                    let state = GenState {
                        chunks,
                        shared,
                        generated_send,
                    };
                    gen_thread(state, &cfg)
                })
                .unwrap()
        };
        /*let colorizer = match colorizer_recv.recv() {
            Ok(c) => c,
            Err(_) => {
                join_handle.join().unwrap()?;
                bail!("failed to get colorizer from world generator")
            }
        };*/
        Ok(Self {
            shared,
            thread: Some(join_handle),
        })
        /*
        let thread_count = (num_cpus::get() / 2).max(1).min(1);
        eprintln!("using {} worldgen threads", thread_count);
        let mut threads = Vec::with_capacity(thread_count);
        for i in 0..thread_count {
            let shared = shared.clone();
            let chunks = chunks.clone();
            let generated_send = bookkeep.generated_send.clone();
            let generator = worldgen::new_generator(cfg)?;
            let layer_count = generator.layer_count;
            ensure!(layer_count > 0, "no generator layers");
            let (mut generator, _) = generator.split();
            generator.share_with(&share_with);

            let join_handle = thread::Builder::new()
                .name(format!("worldgen_{}", i))
                .spawn(move || {
                    let state = GenState {
                        chunks,
                        shared,
                        generated_send,
                    };
                    gen_thread(state, layer_count, generator)
                })
                .unwrap();
            threads.push(join_handle);
        }
        */
    }

    fn unpark_all(&self) {
        if let Some(join) = self.thread.as_ref() {
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
        self.close.store(true);
        self.unpark_all();
        if let Some(join) = self.thread.take() {
            let _ = join.join().unwrap();
        }
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

struct GenChunk {
    data: Cell<ChunkBox>,
    layer: Cell<i32>,
}
impl GridSlot for GenChunk {
    fn new() -> Self {
        Self {
            layer: 0.into(),
            data: ChunkBox::new_empty().into(),
        }
    }
    fn reset(&mut self) {
        *self.layer.get_mut() = 0;
        self.data.get_mut().make_empty();
    }
}

#[derive(Default)]
struct GenStoreConcrete {
    capsules: RefCell<HashMap<Vec<u8>, ([usize; 2], unsafe fn([usize; 2]))>>,
}
impl GenStore for GenStoreConcrete {
    fn register_raw(&self, name: &[u8], obj: [usize; 2], destroy: unsafe fn([usize; 2])) {
        let mut caps = self.capsules.borrow_mut();
        let old = caps.insert(name.to_vec(), (obj, destroy));
        assert!(
            old.is_none(),
            "duplicate gen capsules with name `{}`",
            String::from_utf8_lossy(name)
        );
    }

    fn lookup_raw(&self, name: &[u8]) -> Option<[usize; 2]> {
        let caps = self.capsules.borrow();
        caps.get(name).map(|(obj, _destroy)| *obj)
    }
}
impl Drop for GenStoreConcrete {
    fn drop(&mut self) {
        for (cap, destroy) in self.capsules.borrow_mut().values() {
            unsafe {
                destroy(*cap);
            }
        }
    }
}

struct GenProvider {
    priority: Vec<i32>,
    provided: GridKeeper<bool>,

    center: ChunkPos,
    // OPTIMIZE: Instead of an option, use a dummy void chunkfill as the default value.
    fill: &'static dyn ChunkFiller,
}
impl GenProvider {
    fn new(store: &'static dyn GenStore) -> GenProvider {
        GenProvider {
            priority: vec![],
            provided: GridKeeper::new(2, ChunkPos([0, 0, 0])),

            center: ChunkPos([0, 0, 0]),
            fill: unsafe { store.lookup::<dyn ChunkFiller>("base.chunkfill") },
        }
    }

    fn reshape(&mut self, size: i32, center: ChunkPos) {
        if self.provided.size() != size {
            self.priority = gen_priorities(size, by_dist_up_to(size as f32));
            self.provided = GridKeeper::new(size, center);
        } else if self.center != center {
            self.provided.set_center(center);
        }
        self.center = center;
    }
}

fn gen_thread(gen: GenState, cfg: &[u8]) -> Result<()> {
    let raw_store = Box::new(GenStoreConcrete::default());
    let store: &'static dyn GenStore = unsafe { &*(&*raw_store as *const _) };

    let mut last_stall_warning = Instant::now();
    worldgen::new_generator(cfg, store)?;

    let mut provider = GenProvider::new(store);
    'outer: loop {
        //Make sure provider structure has the right size
        {
            let chunks = gen.chunks.read();
            provider.reshape(chunks.size(), chunks.center());
        }
        //Find a suitable chunk and generate it
        let mut priority_idx = 0;
        let mut generated = 0;
        while priority_idx < provider.priority.len() && generated < 4 {
            let gen_start = Instant::now();
            let idx = provider.priority[priority_idx];
            priority_idx += 1;
            let pos = provider.provided.sub_idx_to_pos(idx);
            let provided = provider.provided.get_by_idx_mut(idx);
            if *provided {
                continue;
            }
            let chunk = match provider.fill.fill(pos) {
                Some(chunk) => chunk,
                None => continue,
            };
            //Keep chunkgen timing statistics
            {
                //Dont care about data races here, after all it's just stats
                //Therefore, dont synchronize
                let time = gen_start.elapsed().as_secs_f32();
                let old_time = gen.shared.avg_gen_time.load();
                let new_time = old_time + (time - old_time) * AVERAGE_WEIGHT;
                gen.shared.avg_gen_time.store(new_time);
            }
            //Send chunk back to main thread
            *provided = true;
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
            generated += 1;
            if gen.shared.close.load() {
                break 'outer;
            }
        }
        //Sleep if no chunks were found
        if generated <= 0 {
            thread::park_timeout(Duration::from_millis(50));
            if gen.shared.close.load() {
                break 'outer;
            }
        }
    }
    Ok(())
}

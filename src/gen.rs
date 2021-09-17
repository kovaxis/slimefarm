use crate::{
    prelude::*,
    terrain::{by_dist_up_to, gen_priorities, BookKeepHandle},
};
use common::{
    terrain::{GridKeeper, GridSlot},
    worldgen::{AnyChunkFill, ChunkFillArgs, ChunkFillRet, ChunkView, ChunkViewOut},
};

const AVERAGE_WEIGHT: f32 = 0.02;

pub struct GeneratorHandle {
    shared: Arc<SharedState>,
    threads: Vec<JoinHandle<()>>,
}
impl GeneratorHandle {
    /// # Safety
    ///
    /// `share_with` must be of the same type that the `cfg` bytestring will generate.
    pub(crate) unsafe fn new(
        cfg: &[u8],
        share_with: AnyChunkFill,
        chunks: Arc<RwLock<ChunkStorage>>,
        bookkeep: &BookKeepHandle,
    ) -> Result<Self> {
        let shared = Arc::new(SharedState {
            close: false.into(),
            avg_gen_time: 0f32.into(),
        });
        let thread_count = (num_cpus::get() / 2).max(1).min(1);
        eprintln!("using {} worldgen threads", thread_count);
        let mut threads = Vec::with_capacity(thread_count);
        for i in 0..thread_count {
            let shared = shared.clone();
            let chunks = chunks.clone();
            let generated_send = bookkeep.generated_send.clone();
            let mut generator = worldgen::new_generator(cfg)?;
            let layer_ranges = mem::take(&mut generator.layers);
            ensure!(!layer_ranges.is_empty(), "no generator layers");
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
                    gen_thread(state, layer_ranges, generator)
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
        self.close.store(true);
        self.unpark_all();
        for join in self.threads.drain(..) {
            join.join().unwrap();
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

struct GenStage {
    counters: Vec<GridKeeper<u8>>,
    chunks: Vec<GridKeeper<Option<ChunkBox>>>,
    priority: Vec<i32>,
    tmp_vec: Vec<usize>,
    layer_ranges: Vec<i32>,
    center: ChunkPos,
}
impl GenStage {
    fn new(size: i32, layer_ranges: Vec<i32>) -> GenStage {
        GenStage {
            counters: layer_ranges
                .iter()
                .map(|_| GridKeeper::new(size, ChunkPos([0, 0, 0])))
                .collect(),
            chunks: layer_ranges
                .iter()
                .map(|_| GridKeeper::new(size, ChunkPos([0, 0, 0])))
                .collect(),
            priority: gen_priorities(size, by_dist_up_to(size as f32)),
            tmp_vec: Vec::new(),
            layer_ranges,
            center: ChunkPos([0, 0, 0]),
        }
    }

    fn set_center(&mut self, new_center: ChunkPos) {
        if new_center == self.center {
            return;
        }
        self.center = new_center;
        for grid in self.counters.iter_mut() {
            grid.set_center(new_center);
        }
        for grid in self.chunks.iter_mut() {
            grid.set_center(new_center);
        }
    }

    fn gen(&mut self, layer: i32, pos: ChunkPos, fill: &mut AnyChunkFill) -> Option<()> {
        let range = self.layer_ranges[layer as usize];
        let range_size = 1 + range * 2;
        if self.counters[layer as usize].get(pos)? & 0x80 != 0 {
            return Some(());
        }
        let base = pos.offset(-range, -range, -range);
        //Ensure that all substrate chunks are properly generated
        if layer > 0 {
            for z in 0..range_size {
                for y in 0..range_size {
                    for x in 0..range_size {
                        let sub_pos = base.offset(x, y, z);
                        self.gen(layer - 1, sub_pos, fill)?;
                    }
                }
            }
        }
        //Collect substrate and output chunks
        let empty_box = ChunkBox::new_empty();
        self.tmp_vec
            .resize((2 * range_size * range_size * range_size) as usize, 0);
        let (substrate, output) = self
            .tmp_vec
            .split_at_mut((range_size * range_size * range_size) as usize);
        if layer > 0 {
            let mut idx = 0;
            for z in 0..range_size {
                for y in 0..range_size {
                    for x in 0..range_size {
                        let sub_pos = base.offset(x, y, z);
                        let substrate_chunk = unsafe {
                            ChunkRef::from_raw(
                                self.chunks[(layer - 1) as usize]
                                    .get(sub_pos)
                                    .unwrap()
                                    .as_ref()
                                    .unwrap()
                                    .as_ref()
                                    .into_raw(),
                            )
                        };
                        substrate[idx] = substrate_chunk.into_raw() as usize;
                        let chunk_ref_mut = self.chunks[layer as usize]
                            .get_mut(sub_pos)
                            .unwrap()
                            .get_or_insert_with(|| substrate_chunk.clone_chunk());
                        output[idx] = chunk_ref_mut as *mut ChunkBox as usize;
                        idx += 1;
                    }
                }
            }
        } else {
            for sub in substrate.iter_mut() {
                *sub = empty_box.as_ref().into_raw() as usize;
            }
            let mut idx = 0;
            for z in 0..range_size {
                for y in 0..range_size {
                    for x in 0..range_size {
                        let sub_pos = base.offset(x, y, z);
                        output[idx] = self.chunks[layer as usize]
                            .get_mut(sub_pos)
                            .unwrap()
                            .get_or_insert_with(|| ChunkBox::new_empty())
                            as *mut ChunkBox as usize;
                        idx += 1;
                    }
                }
            }
        }
        //Generate this chunk
        fill.call(ChunkFillArgs {
            center: self.center,
            pos,
            layer,
            substrate: unsafe {
                ChunkView::new(
                    range_size,
                    mem::transmute::<&[usize], &'static [ChunkRef<'static>]>(substrate),
                )
            },
            output: unsafe {
                ChunkViewOut::new(
                    range_size,
                    mem::transmute::<&mut [usize], &'static mut [&'static mut ChunkBox]>(output),
                )
            },
        })?;
        //Mark all of the touching output chunks as "touched"
        for z in 0..range_size {
            for y in 0..range_size {
                for x in 0..range_size {
                    let sub_pos = base.offset(x, y, z);
                    *self.counters[layer as usize].get_mut(sub_pos).unwrap() += 1;
                }
            }
        }
        //Mark this layer as generated
        *self.counters[layer as usize].get_mut(pos).unwrap() |= 0x80;
        Some(())
    }

    fn take_ready_by_idx(&mut self, idx: i32) -> Option<ChunkBox> {
        let layer = self.layer_ranges.len() - 1;
        let range = self.layer_ranges[layer];
        let range_size = 1 + 2 * range;
        let total = range_size * range_size * range_size;
        if (*self.counters[layer].get_by_idx(idx) & 0x7F) as i32 >= total {
            self.chunks[layer].get_by_idx_mut(idx).take()
        } else {
            None
        }
    }
}

fn gen_thread(gen: GenState, layer_ranges: Vec<i32>, mut generator: AnyChunkFill) {
    let mut last_stall_warning = Instant::now();
    let mut stage = GenStage::new(2, layer_ranges.clone());
    'outer: loop {
        //Make sure stage has the right size
        {
            let chunks = gen.chunks.read();
            if stage.chunks[0].size() != chunks.size() {
                stage = GenStage::new(chunks.size(), layer_ranges.clone());
            }
            stage.set_center(chunks.center());
        }
        //Find a suitable chunk and generate it
        let mut priority_idx = 0;
        let mut generated = 0;
        while priority_idx < stage.priority.len() && generated < 4 {
            let gen_start = Instant::now();
            let idx = stage.priority[priority_idx];
            let pos = stage.chunks[0].sub_idx_to_pos(idx);
            priority_idx += 1;
            if stage
                .gen((layer_ranges.len() - 1) as i32, pos, &mut generator)
                .is_some()
            {
                if let Some(chunk) = stage.take_ready_by_idx(idx) {
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
}

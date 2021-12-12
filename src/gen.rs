use std::num::NonZeroU32;

use crate::{
    prelude::*,
    terrain::{by_dist_up_to, gen_priorities, BookKeepHandle},
};
use common::{
    impl_cast_to,
    terrain::{GridKeeper, GridSlot},
    worldgen::{
        BlockColorizer, ChunkFill, ChunkFillArgs, ChunkFillRet, ChunkView, ChunkViewOut,
        GenCapsule, GenStore, GenStoreRef, LandmarkBox, LandmarkHead, LandmarkId, LandmarkKind,
    },
};

const AVERAGE_WEIGHT: f32 = 0.02;

pub struct GeneratorHandle {
    shared: Arc<SharedState>,
    thread: Option<JoinHandle<Result<()>>>,
    colorizer: Option<Box<dyn BlockColorizer>>,
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
        let (colorizer_send, colorizer_recv) = channel::bounded(0);
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
                    gen_thread(state, &cfg, colorizer_send)
                })
                .unwrap()
        };
        let colorizer = match colorizer_recv.recv() {
            Ok(c) => c,
            Err(_) => {
                join_handle.join().unwrap()?;
                bail!("failed to get colorizer from world generator")
            }
        };
        Ok(Self {
            shared,
            thread: Some(join_handle),
            colorizer: Some(colorizer),
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

    pub fn take_colorizer(&mut self) -> Option<Box<dyn BlockColorizer>> {
        self.colorizer.take()
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
    capsules: RefCell<HashMap<String, Box<dyn GenCapsule<'static>>>>,
}
impl GenStoreConcrete {
    fn init_capsules(&self) {
        let caps = self.capsules.borrow();
        for cap in caps.values() {
            let cap: &dyn GenCapsule<'static> = &**cap;
            let cap: &dyn GenCapsule = unsafe { mem::transmute(cap) };
            cap.init(self);
        }
    }
}
impl GenStore for GenStoreConcrete {
    fn register<'a>(&'a self, name: &str, cap: Box<dyn 'a + GenCapsule<'a>>) {
        let mut caps = self.capsules.borrow_mut();
        // SAFETY: Self-referential types. During destruction, capsules must not reference each
        // other! Otherwise, a capsule might be destroyed, and later another capsule is destroyed
        // holding a reference to the destroyed capsule.
        // This is one of the safety tradeoffs for a usable game design.
        let extended_lifetime = unsafe {
            mem::transmute::<Box<dyn 'a + GenCapsule<'a>>, Box<dyn GenCapsule<'static>>>(cap)
        };
        let old = caps.insert(name.to_string(), extended_lifetime);
        assert!(old.is_none(), "duplicate gen capsules with name `{}`", name);
    }

    fn lookup_capsule<'a>(&'a self, name: &str) -> Option<&'a (dyn 'a + GenCapsule<'a>)> {
        let caps = self.capsules.borrow();
        match caps.get(name) {
            None => None,
            Some(b) => {
                let cap: &dyn GenCapsule<'static> = &**b;
                // SAFETY: The actual box must no longer be changed, since references are handed out
                // to the box contents.
                // No API is given to do this, so it's safe.
                let cap: &'a (dyn 'a + GenCapsule<'a>) = unsafe { mem::transmute(cap) };
                Some(cap)
            }
        }
    }
}

struct LandmarkNodeId(NonZeroU32);

struct LandmarkNode {
    next: Option<LandmarkNodeId>,
    id: LandmarkId,
}

struct GenStageWrap<'a> {
    cell: RefCell<GenStage<'a>>,
}

impl<'a> GenCapsule<'a> for GenStageWrap<'a> {
    impl_cast_to! {
        "base.genstage" => common::worldgen::GenStage
    }
    fn init(&'a self, store: GenStoreRef<'a>) {
        self.cell.borrow_mut().init(store)
    }
}

impl common::worldgen::GenStage for GenStageWrap<'_> {
    fn require(&self, min: ChunkPos, max: ChunkPos, layer: i32) -> Option<()> {
        self.cell.borrow().require(min, max, layer)
    }
    fn place(&self, min: ChunkPos, max: ChunkPos, landmark: LandmarkId) -> Option<()> {
        self.cell.borrow().place(min, max, landmark)
    }
    unsafe fn create_landmark(&self, landmark: LandmarkBox) -> LandmarkId {
        self.cell.borrow().create_landmark(landmark)
    }
    fn landmark_kind(&self, name: &str) -> LandmarkKind {
        self.cell.borrow_mut().landmark_name_to_kind(name)
    }
}

struct GenStage<'a> {
    priority: Vec<i32>,
    chunks: GridKeeper<GenChunk>,
    landmarks: GridKeeper<Option<LandmarkNodeId>>,
    landmark_nodes: RefCell<Vec<LandmarkNode>>,
    landmark_boxes: RefCell<SlotMap<LandmarkBox>>,

    landmark_kinds: HashMap<String, LandmarkKind>,
    landmark_next_kind: i32,
    center: ChunkPos,
    // OPTIMIZE: Instead of an option, use a dummy void chunkfill as the default value.
    fill: Option<&'a (dyn 'a + ChunkFill)>,
    layer_count: i32,
}
impl<'a> GenStage<'a> {
    fn register(store: GenStoreRef<'a>) {
        let stage = GenStage {
            priority: vec![],
            chunks: GridKeeper::new(2, ChunkPos([0, 0, 0])),
            landmarks: GridKeeper::new(2, ChunkPos([0, 0, 0])),
            landmark_nodes: Default::default(),
            landmark_boxes: Default::default(),

            landmark_kinds: HashMap::default(),
            landmark_next_kind: 1,
            layer_count: 0,
            center: ChunkPos([0, 0, 0]),
            fill: None,
        };
        let stage = Box::new(GenStageWrap {
            cell: RefCell::new(stage),
        });
        store.register("base.genstage", stage);
    }

    fn init(&mut self, store: GenStoreRef<'a>) {
        unsafe {
            self.fill = Some(store.lookup("base.chunkfill"));
            self.layer_count = self.fill.as_ref().unwrap().layer_count();
        }
    }

    fn reshape(&mut self, size: i32, center: ChunkPos) {
        if self.chunks.size() != size {
            self.priority = gen_priorities(size, by_dist_up_to(size as f32));
            self.chunks = GridKeeper::new(size, center);
            self.landmarks = GridKeeper::new(size, center);
            self.landmark_nodes.get_mut().clear();
            self.landmark_boxes.get_mut().clear();
        } else if self.center != center {
            self.chunks.set_center(center);
            self.landmarks.set_center(center);
        }
        self.center = center;
    }

    fn gen(&self, layer: i32, pos: ChunkPos) -> Option<()> {
        let chunk = self.chunks.get(pos)?;
        let mut cur_layer = chunk.layer.get();
        while cur_layer < layer {
            //Now generating this layer:
            cur_layer += 1;
            //Generate this chunk
            //This might entail any of the following:
            //  - Write into this chunk's block data.
            //  - Place landmarks on any chunk, including this one.
            //  - Generate _other_ chunks recursively, but at lower layers.
            //      This implies that this chunk might be recursively requested to generate, but
            //      those requests are trivial to satisfy and do not produce infinite recursion.
            let mut blocks = chunk.data.take();
            self.fill.as_ref().unwrap().fill(ChunkFillArgs {
                pos,
                layer: cur_layer,
                blocks: Cell::new(&mut blocks as *mut _),
            })?;
            chunk.layer.set(cur_layer);
            chunk.data.set(blocks);
        }
        Some(())
    }

    fn require(&self, min: ChunkPos, max: ChunkPos, layer: i32) -> Option<()> {
        for z in min[2]..max[2] {
            for y in min[1]..max[1] {
                for x in min[0]..max[0] {
                    self.gen(layer, ChunkPos([x, y, z]))?;
                }
            }
        }
        Some(())
    }

    fn place(&self, min: ChunkPos, max: ChunkPos, landmark: LandmarkId) -> Option<()> {
        todo!()
    }

    fn create_landmark(&self, landmark: LandmarkBox) -> LandmarkId {
        let mut boxes = self.landmark_boxes.borrow_mut();
        LandmarkId(boxes.insert(landmark).idx)
    }

    fn landmark_name_to_kind(&mut self, name: &str) -> LandmarkKind {
        let nxt = &mut self.landmark_next_kind;
        *self
            .landmark_kinds
            .entry(name.to_string())
            .or_insert_with(|| {
                let kind = LandmarkKind(*nxt);
                *nxt = nxt.checked_add(1).unwrap();
                kind
            })
    }
}

fn gen_thread(
    gen: GenState,
    cfg: &[u8],
    colorizer_send: Sender<Box<dyn BlockColorizer>>,
) -> Result<()> {
    let raw_store = Box::new(GenStoreConcrete::default());
    let store: &dyn GenStore = &*raw_store;

    let mut last_stall_warning = Instant::now();
    GenStage::register(store);
    worldgen::new_generator(cfg, store)?;

    raw_store.init_capsules();

    let stage: &GenStageWrap = unsafe { store.lookup_concrete("base.genstage") };
    let layer_count;
    unsafe {
        let chunkfill = store.lookup::<dyn ChunkFill>("base.chunkfill");
        layer_count = chunkfill.layer_count();
        colorizer_send
            .send(chunkfill.colorizer())
            .expect("generator handle is not expecting the colorizer");
        drop(colorizer_send);
    }
    'outer: loop {
        //Make sure stage has the right size
        {
            let chunks = gen.chunks.read();
            stage
                .cell
                .borrow_mut()
                .reshape(chunks.size(), chunks.center());
        }
        //Find a suitable chunk and generate it
        let stage = stage.cell.borrow();
        let mut priority_idx = 0;
        let mut generated = 0;
        while priority_idx < stage.priority.len() && generated < 4 {
            let gen_start = Instant::now();
            let idx = stage.priority[priority_idx];
            priority_idx += 1;
            let chunk = stage.chunks.get_by_idx(idx);
            if chunk.layer.get() > layer_count {
                continue;
            }
            let pos = stage.chunks.sub_idx_to_pos(idx);
            if stage.gen(layer_count, pos).is_some() {
                chunk.layer.set(layer_count + 1);
                let chunk = chunk.data.replace(ChunkBox::new_empty());
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

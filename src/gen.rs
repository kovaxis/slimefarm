use std::num::NonZeroU32;

use crate::prelude::*;
use common::{
    terrain::GridKeeper4,
    worldgen::{BlockIdAlloc, ChunkFiller, GenStore},
};

const AVERAGE_WEIGHT: f32 = 0.005;

pub struct GeneratorHandle {
    shared: Arc<GenSharedState>,
    tex_recv: Receiver<BlockTextures>,
    thread: Option<JoinHandle<Result<()>>>,
}
impl GeneratorHandle {
    pub(crate) fn new(
        cfg: &[u8],
        global: &Arc<GlobalState>,
        chunks: Arc<RwLock<ChunkStorage>>,
    ) -> Result<Self> {
        let shared = Arc::new(GenSharedState {
            gen_area: GenArea {
                center: BlockPos {
                    coords: [0; 3].into(),
                    dim: 0,
                },
                gen_radius: 0.,
            }
            .into(),
            close: false.into(),
            avg_gen_time: 0f32.into(),
        });
        let (tex_send, tex_recv) = channel::bounded(0);
        let cfg = cfg.to_vec();
        //let (colorizer_send, colorizer_recv) = channel::bounded(0);
        let join_handle = {
            let shared = shared.clone();
            let global = global.clone();
            let chunks = chunks.clone();
            thread::Builder::new()
                .name("worldgen".to_string())
                .spawn(move || {
                    let state = GenState {
                        chunks,
                        shared,
                        global,
                    };
                    let res = gen_thread(state, tex_send, &cfg);
                    if let Err(err) = &res {
                        eprintln!("fatal error initializing gen thread: {:#}", err);
                    }
                    res
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
            tex_recv,
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

    pub fn take_block_textures(&self) -> Result<BlockTextures> {
        self.tex_recv.recv().context("block texture channel closed")
    }

    pub fn set_gen_area(&self, genarea: GenArea) {
        *self.shared.gen_area.lock() = genarea;
    }
}
impl ops::Deref for GeneratorHandle {
    type Target = GenSharedState;
    fn deref(&self) -> &GenSharedState {
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

#[derive(Default)]
struct GenStoreConcrete {
    capsules: RefCell<HashMap<Vec<u8>, ([usize; 2], unsafe fn([usize; 2]))>>,
    events: RefCell<HashMap<Vec<u8>, Vec<Box<dyn FnMut(*const u8)>>>>,
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

    fn listen_raw(&self, name: &[u8], listener: Box<dyn FnMut(*const u8)>) {
        let mut evs = self.events.borrow_mut();
        match evs.get_mut(name) {
            Some(listeners) => {
                listeners.push(listener);
            }
            None => {
                evs.insert(name.to_vec(), vec![listener]);
            }
        }
    }

    unsafe fn trigger_raw(&self, name: &[u8], args: *const u8) {
        let mut evs = self.events.borrow_mut();
        if let Some(listeners) = evs.get_mut(name) {
            for l in listeners {
                l(args);
            }
        }
    }
}
impl Drop for GenStoreConcrete {
    fn drop(&mut self) {
        eprintln!("destroying GenStore capsules");
        for (cap, destroy) in self.capsules.borrow_mut().values() {
            unsafe {
                destroy(*cap);
            }
        }
    }
}

struct BlockIdAllocConcrete {
    adv: u8,
    nxt: Cell<u8>,
    nxt_seq: Cell<u8>,
    map: RefCell<HashMap<u64, u8>>,
}
impl BlockIdAllocConcrete {
    fn new() -> Self {
        let rnd = fxhash::hash64(&Instant::now());
        Self {
            adv: (rnd >> 8) as u8 | 1,
            nxt: (rnd as u8).into(),
            nxt_seq: 0.into(),
            map: default(),
        }
    }

    #[track_caller]
    fn alloc(&self) -> u8 {
        if self.nxt_seq.get() == u8::MAX {
            panic!("ran out of block ids!");
        }
        let id = self.nxt.get();
        self.nxt.set(id.wrapping_add(self.adv));
        self.nxt_seq.set(self.nxt_seq.get() + 1);
        id
    }

    #[track_caller]
    fn get_hash(&self, hash: u64) -> u8 {
        let mut map = self.map.borrow_mut();
        *map.entry(hash).or_insert_with(|| self.alloc())
    }
}

impl BlockIdAlloc for BlockIdAllocConcrete {
    #[track_caller]
    fn get_hash(&self, hash: u64) -> BlockData {
        BlockData {
            data: self.get_hash(hash),
        }
    }
}

struct GenProvider {
    fill: &'static dyn ChunkFiller,
}
impl GenProvider {
    fn new(store: &'static dyn GenStore) -> GenProvider {
        GenProvider {
            fill: unsafe { store.lookup::<dyn ChunkFiller>("base.chunkfill") },
        }
    }
}

#[derive(Clone)]
pub struct GenArea {
    pub center: BlockPos,
    pub gen_radius: f32,
}

pub struct GenSharedState {
    gen_area: Mutex<GenArea>,
    close: AtomicCell<bool>,
    pub avg_gen_time: AtomicCell<f32>,
}

struct GenState {
    shared: Arc<GenSharedState>,
    global: Arc<GlobalState>,
    chunks: Arc<RwLock<ChunkStorage>>,
}

fn empty_chunk_buf(
    chunks: &mut ChunkStorage,
    buf: &mut Vec<(ChunkPos, ChunkArc)>,
    unsent: &mut HashSet<ChunkPos>,
) {
    for (pos, chunk) in buf.drain(..) {
        unsent.remove(&pos);
        chunks.insert(pos, chunk);
    }
    chunks.maybe_gc();
}

fn gen_thread(gen: GenState, tex_send: Sender<BlockTextures>, cfg: &[u8]) -> Result<()> {
    let raw_store = Box::new(GenStoreConcrete::default());
    let store: &'static dyn GenStore = unsafe { &*(&*raw_store as *const _) };

    store.register(
        "base.blockregister",
        Box::new(BlockIdAllocConcrete::new()) as Box<dyn BlockIdAlloc>,
    );
    store.register("base.blocktextures", Box::new(BlockTextures::default()));

    let lua = Rc::new(Lua::new());
    lua.context(|lua| {
        crate::lua::open_generic_libs(&gen.global, lua);
        lua.load(&cfg)
            .set_name("worldgen.lua")
            .unwrap()
            .exec()
            .expect_lua("running worldgen.lua");
    });
    store.register("base.lua", Box::new(lua));

    let mut last_stall_warning = Instant::now();
    worldgen::new_generator(store)?;

    let tex = unsafe { store.lookup::<BlockTextures>("base.blocktextures").clone() };
    let style = StyleTable::new(&tex);
    tex_send
        .send(tex)
        .map_err(|_| Error::msg("failed to send block textures"))?;

    const GEN_QUEUE: usize = 32;
    const READY_QUEUE: usize = 64;
    let mut last_gc = Instant::now();
    let gc_interval = Duration::from_millis(2435);

    let mut seenbuf = default();
    let mut gen_sortbuf = Vec::new();

    let provider = GenProvider::new(store);
    let mut gen_queue = Vec::with_capacity(GEN_QUEUE);
    let mut ready_buf = Vec::with_capacity(READY_QUEUE);
    let mut unsent: HashSet<ChunkPos> = default();
    'outer: loop {
        //Collect unused data every once in a while
        if last_gc.elapsed() > gc_interval {
            unsafe {
                store.trigger("base.gc", &());
            }
            last_gc = Instant::now();
        }

        //Gather a set of chunks to generate by scanning available chunks
        {
            let gen_area = gen.shared.gen_area.lock().clone();
            let chunks = gen.chunks.read();
            gen_sortbuf.clear();
            chunks.iter_nearby(
                &mut seenbuf,
                gen_area.center,
                gen_area.gen_radius,
                |pos, chunk, dist| {
                    if chunk.is_none() && !unsent.contains(&pos) {
                        gen_sortbuf.push((dist, pos));
                    }
                    Ok(())
                },
            )?;
            drop(chunks);
            gen_queue.clear();
            gen_queue.extend(
                gen_sortbuf
                    .iter()
                    .sorted_by(|a, b| Sortf32(a.0).cmp(&Sortf32(b.0)))
                    .take(GEN_QUEUE)
                    .map(|(_dist, pos)| *pos),
            );
        }

        //Generate chunks from the request queue
        let mut gencount = 0;
        for pos in gen_queue.drain(..) {
            let gen_start = Instant::now();

            // Generate chunk from scratch
            let mut chunk = match provider.fill.fill(pos) {
                Some(chunk) => chunk,
                None => {
                    println!("failed to generate chunk at {:?}", pos);
                    continue;
                }
            };
            chunk.consolidate(&style);
            let chunk = ChunkArc::new(chunk);
            unsent.insert(pos);
            ready_buf.push((pos, chunk));
            gencount += 1;

            //Keep chunkgen timing statistics
            {
                //Dont care about data races here, after all it's just stats
                //Therefore, dont synchronize
                let time = gen_start.elapsed().as_secs_f32();
                let old_time = gen.shared.avg_gen_time.load();
                let new_time = old_time + (time - old_time) * AVERAGE_WEIGHT;
                gen.shared.avg_gen_time.store(new_time);
            }

            //Send chunks back to main thread
            if let Some(mut chunks) = gen.chunks.try_write() {
                empty_chunk_buf(&mut chunks, &mut ready_buf, &mut unsent);
            }

            // Close quickly if requested
            if gen.shared.close.load() {
                break 'outer;
            }
        }

        //Try harder to send back chunks
        if !ready_buf.is_empty() {
            let stall_start = Instant::now();
            let mut chunks = gen.chunks.write();
            empty_chunk_buf(&mut chunks, &mut ready_buf, &mut unsent);
            let now = Instant::now();
            if now - stall_start > Duration::from_millis(50)
                && now - last_stall_warning > Duration::from_millis(1500)
            {
                last_stall_warning = now;
                eprintln!(
                    "worldgen thread stalled for {}ms",
                    (now - stall_start).as_millis()
                );
            }
        }

        //Sleep if no chunks were found
        if gencount <= 0 {
            thread::park_timeout(Duration::from_millis(50));
            if gen.shared.close.load() {
                break 'outer;
            }
        }
    }
    Ok(())
}

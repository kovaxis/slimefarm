use std::num::NonZeroU32;

use crate::{
    prelude::*,
    terrain::{by_dist_up_to, gen_priorities, BookKeepHandle},
};
use common::{
    terrain::{GridKeeper, GridSlot},
    worldgen::{BlockIdAlloc, ChunkFiller, GenStore},
};

const AVERAGE_WEIGHT: f32 = 0.005;

pub struct GeneratorHandle {
    shared: Arc<GenSharedState>,
    reshape_send: Sender<(i32, ChunkPos)>,
    last_shape: Cell<(i32, ChunkPos)>,
    tex_recv: Receiver<BlockTextures>,
    thread: Option<JoinHandle<Result<()>>>,
}
impl GeneratorHandle {
    pub(crate) fn new(
        cfg: &[u8],
        global: &Arc<GlobalState>,
        chunks: Arc<RwLock<ChunkStorage>>,
        bookkeep: &BookKeepHandle,
    ) -> Result<Self> {
        let shared = Arc::new(GenSharedState {
            close: false.into(),
            avg_gen_time: 0f32.into(),
        });
        let (reshape_send, reshape_recv) = channel::bounded(64);
        let (tex_send, tex_recv) = channel::bounded(0);
        let cfg = cfg.to_vec();
        //let (colorizer_send, colorizer_recv) = channel::bounded(0);
        let join_handle = {
            let shared = shared.clone();
            let global = global.clone();
            let chunks = chunks.clone();
            let generated_send = bookkeep.generated_send.clone();
            thread::Builder::new()
                .name("worldgen".to_string())
                .spawn(move || {
                    let state = GenState {
                        _chunks: chunks,
                        shared,
                        global,
                        reshape_recv,
                        generated_send,
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
            reshape_send,
            last_shape: (0, ChunkPos([0, 0, 0])).into(),
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

    pub fn reshape(&self, size: i32, center: ChunkPos) {
        if self.last_shape.get() != (size, center) {
            self.last_shape.set((size, center));
            let _ = self.reshape_send.send((size, center));
        }
    }

    pub fn take_block_textures(&self) -> Result<BlockTextures> {
        self.tex_recv.recv().context("block texture channel closed")
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

pub struct GenSharedState {
    close: AtomicCell<bool>,
    pub avg_gen_time: AtomicCell<f32>,
}

struct GenState {
    shared: Arc<GenSharedState>,
    global: Arc<GlobalState>,
    _chunks: Arc<RwLock<ChunkStorage>>,
    reshape_recv: Receiver<(i32, ChunkPos)>,
    generated_send: Sender<(ChunkPos, ChunkBox)>,
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
        if self.nxt_seq.get() == u8::max_value() {
            panic!("ran out of block ids!");
        }
        let id = self.nxt.get();
        self.nxt.set(id.wrapping_add(self.adv));
        self.nxt_seq.set(self.nxt_seq.get() + 1);
        id
    }

    fn get_hash(&self, hash: u64) -> u8 {
        let mut map = self.map.borrow_mut();
        *map.entry(hash).or_insert_with(|| self.alloc())
    }
}

impl BlockIdAlloc for BlockIdAllocConcrete {
    fn get_hash(&self, hash: u64) -> BlockData {
        BlockData {
            data: self.get_hash(hash),
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
    let solid = SolidTable::new(&tex);
    tex_send
        .send(tex)
        .map_err(|_| Error::msg("failed to send block textures"))?;

    let mut provider = GenProvider::new(store);
    'outer: loop {
        //Make sure provider structure has the right size
        for (size, center) in gen.reshape_recv.try_iter() {
            eprintln!("recentering to {:?}", center);
            provider.reshape(size, center);
            unsafe {
                store.trigger("base.recenter", &center);
            }
            eprintln!("  finished recentering");
        }
        //Find a suitable chunk and generate it
        let mut priority_idx = 0;
        let mut generated = 0;
        let mut failed = 0;
        while priority_idx < provider.priority.len() && generated < 4 && failed < 8 {
            let gen_start = Instant::now();
            let idx = provider.priority[priority_idx];
            priority_idx += 1;
            let pos = provider.provided.sub_idx_to_pos(idx);
            let provided = provider.provided.get_by_idx_mut(idx);
            if *provided {
                continue;
            }
            let mut chunk = match provider.fill.fill(pos) {
                Some(chunk) => chunk,
                None => {
                    failed += 1;
                    continue;
                }
            };
            chunk.mark_solidity(&solid);
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

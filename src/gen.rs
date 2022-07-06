use std::num::NonZeroU32;

use crate::prelude::*;
use common::lua::LuaValueStatic;

pub struct GeneratorHandle {
    pub entity_evs: Receiver<EntityEv>,
    shared: Arc<GenSharedState>,
    info_recv: Receiver<Box<WorldInfo>>,
    thread: Option<JoinHandle<Result<()>>>,
}
impl GeneratorHandle {
    pub(crate) fn new(
        cfg: GenConfig,
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
            avg_light_time: 0f32.into(),
        });
        let (tex_send, tex_recv) = channel::bounded(0);
        let (ent_send, ent_recv) = channel::unbounded();
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
                        entity: EntityState::new(ent_send),
                    };
                    let res = gen_thread(state, tex_send, cfg);
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
            info_recv: tex_recv,
            entity_evs: ent_recv,
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

    pub fn take_world_info(&self) -> Result<Box<WorldInfo>> {
        self.info_recv.recv().context("world info channel closed")
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

#[derive(Clone)]
pub struct GenArea {
    pub center: BlockPos,
    pub gen_radius: f32,
}

#[derive(Serialize, Deserialize)]
pub struct GenConfig {
    pub lua_main: String,
    pub args: Vec<LuaValueStatic>,
}

pub struct GenSharedState {
    gen_area: Mutex<GenArea>,
    close: AtomicCell<bool>,
    pub avg_gen_time: AtomicCell<f32>,
    pub avg_light_time: AtomicCell<f32>,
}

struct GenState {
    shared: Arc<GenSharedState>,
    global: Arc<GlobalState>,
    chunks: Arc<RwLock<ChunkStorage>>,
    entity: EntityState,
}

struct EntityState {
    next_id: u64,
    id_stride: u64,
    id_map: HashMap<ChunkPos, u64>,
    send: Sender<EntityEv>,
}
impl EntityState {
    fn new(send: Sender<EntityEv>) -> Self {
        Self {
            // Go through the even ids to leave the odd ids to the main thread.
            next_id: rand::random::<u64>() & (!1),
            id_stride: rand::random::<u64>() & (!1),
            id_map: default(),
            send,
        }
    }

    fn next_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(self.id_stride);
        id
    }
}

pub enum EntityEv {
    Spawn {
        id: u64,
        pos: WorldPos,
        data: Vec<u8>,
    },
    Despawn {
        id: u64,
    },
}

/// Transfer chunks from the local buffer to the game chunkstore.
/// This is perhaps the only place in the entire codebase that `ChunkStorage` is modified.
fn empty_chunk_buf(
    chunks: &mut ChunkStorage,
    unsent: &mut HashMap<ChunkPos, ChunkBox>,
    entity: &mut EntityState,
) {
    for (pos, chunk) in unsent.drain() {
        // Trigger entity spawn events
        if let Some(data) = chunk.data() {
            if data.entity_len != 0 {
                entity.id_map.insert(pos, entity.next_id);
                let mut i = 0;
                while let Some((subpos, raw)) = data.next_entity(&mut i) {
                    // Make position absolute
                    let mut epos = pos.chunk_to_block().world_pos();
                    epos.coords[0] += subpos.x as f64;
                    epos.coords[1] += subpos.y as f64;
                    epos.coords[2] += subpos.z as f64;
                    // Generate a "unique" id for this entity
                    let id = entity.next_id();
                    // Send an open event to the main thread
                    let e = entity.send.try_send(EntityEv::Spawn {
                        id,
                        pos: epos,
                        data: raw.to_vec(),
                    });
                    if let Err(e) = e {
                        eprintln!("failed to send entity spawn event: {}", e);
                    }
                }
            }
        }
        // Insert the chunk directly into the chunk storage
        chunks.insert(pos, ChunkArc::new(chunk));
    }

    // Remove untouched chunks in the main chunk storage
    if chunks.should_gc() {
        let old = chunks.count();
        chunks.gc_with(|pos, chunk| {
            if let Some(data) = chunk.data() {
                if data.entity_len != 0 {
                    if let Some(mut base_id) = entity.id_map.remove(pos) {
                        let mut i = 0;
                        while let Some(_) = data.next_entity(&mut i) {
                            // Send entity despawn event
                            let e = entity.send.try_send(EntityEv::Despawn { id: base_id });
                            if let Err(e) = e {
                                eprintln!("failed to send entity despawn event: {}", e);
                            }
                            // Advance to next id
                            base_id = base_id.wrapping_add(entity.id_stride);
                        }
                    } else {
                        eprintln!("could not find base entity id for chunk at {:?}!", pos);
                    }
                }
            }
        });
        let new = chunks.count();
        println!("reclaimed {}/{} chunks", old - new, old);
    }
}

fn skylight_fall(above: ChunkRef, chunk: &mut ChunkBox, style: &StyleTable) {
    // Attempt to exploit the fact that some chunks are all-empty or all-full
    if chunk.is_homogeneous() {
        let block = chunk.homogeneous_block();
        if block.is_solid(style) {
            // All dark. Ready
            return;
        }
        match above.data() {
            Some(above) => {
                // Check whether the above lighting is uniform at the bottom layer
                if above.shinethrough.iter().all(|&l| l == 0) {
                    // All dark
                    return;
                }
                let abovelight = above.skylight[0];
                if above.shinethrough.iter().all(|&l| l == ChunkSizedInt::MAX)
                    && above.skylight[..(CHUNK_SIZE * CHUNK_SIZE) as usize]
                        .iter()
                        .all(|&l| l == abovelight)
                {
                    // All lighted
                    chunk.make_homogeneous(true, block, abovelight, chunk.homogeneous_lightmode());
                    return;
                }
                // We'll have to allocate memory for skylight, sadly
            }
            None => {
                let shinethrough = above.homogeneous_shinethrough();
                let light = above.homogeneous_skylight();
                if shinethrough {
                    // All lighted
                    chunk.make_homogeneous(true, block, light, chunk.homogeneous_lightmode());
                } else {
                    // All dark
                }
                return;
            }
        }
    }

    let chunk = chunk.data_mut();
    if let Some(above) = above.data() {
        let mut idx_2d = 0;
        for y in 0..CHUNK_SIZE as usize {
            for x in 0..CHUNK_SIZE as usize {
                if (above.shinethrough[y] >> x) & 1 != 0 {
                    // Propagate light down
                    let light = above.skylight[idx_2d];
                    let mut idx_3d = idx_2d + (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize;
                    for _z in (0..CHUNK_SIZE).rev() {
                        idx_3d -= (CHUNK_SIZE * CHUNK_SIZE) as usize;
                        if chunk.blocks[idx_3d].is_solid(style) {
                            break;
                        }
                        chunk.skylight[idx_3d] = light;
                    }
                    chunk.shinethrough[y] |= ((idx_3d == idx_2d) as ChunkSizedInt) << x;
                }
                idx_2d += 1;
            }
        }
    } else {
        let shinethrough = above.homogeneous_shinethrough();
        let light = above.homogeneous_skylight();
        if !shinethrough {
            // All dark
            return;
        }
        let mut idx_2d = 0;
        for y in 0..CHUNK_SIZE as usize {
            for x in 0..CHUNK_SIZE as usize {
                // Propagate light down
                let mut idx_3d = idx_2d + (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize;
                for _z in 0..CHUNK_SIZE {
                    idx_3d -= (CHUNK_SIZE * CHUNK_SIZE) as usize;
                    if chunk.blocks[idx_3d].is_solid(style) {
                        break;
                    }
                    chunk.skylight[idx_3d] = light;
                }
                chunk.shinethrough[y] |= ((idx_3d == idx_2d) as ChunkSizedInt) << x;
                idx_2d += 1;
            }
        }
    }
}

crate::terrain::light_spreader! {
    LightSpreader(CHUNK_SIZE);
}

fn process_lighting(
    shared: &GenSharedState,
    chunks: &ChunkStorage,
    unsent: &mut HashMap<ChunkPos, ChunkBox>,
    spreader: &mut LightSpreader,
    style: &StyleTable,
) {
    // This function does 2 things:
    // - Propagate skylight down for all unsent chunks
    // - After that, propagates light in all directions _within each chunk_
    // This means that at the time of meshing a final inter-chunk propagation pass must be done.

    // Propagate skylight from sky to ground, from highest chunks to lowest chunks
    let mut sortbuf = unsent.iter_mut().collect::<Vec<_>>();
    sortbuf.sort_unstable_by_key(|(p, _c)| [p.coords.x, p.coords.y, -p.coords.z]);
    let mut last_above: Option<(Int4, ChunkRef)> = None;
    for (&pos, chunk) in sortbuf {
        if !chunk.is_homogeneous() || !chunk.homogeneous_shinethrough() {
            let mut above_pos = pos;
            above_pos.coords.z += 1;
            let above = last_above
                .filter(|(abpos, _)| *abpos == above_pos)
                .map(|(_, ab)| ab)
                .or_else(|| chunks.chunk_at(above_pos));
            if let Some(above) = above {
                skylight_fall(above, chunk, style);
            } else {
                println!("no above-chunk found for chunk at {:?}!", pos);
            }
        }
        last_above = Some((pos, chunk.as_ref()));
    }

    // Spread skylight within each chunk
    for (pos, chunk) in unsent.iter_mut() {
        // Debug hack to disable gen-time light spreading
        /*if let Some(b) = chunk.blocks() {
            if b.blocks[1].data == 0 {
                continue;
            }
        }*/

        time!(start lightspread);

        // Get chunk lighting data
        let chunk = match chunk.try_data_mut() {
            Some(chunk) => chunk,
            None => continue,
        };
        spreader.reset(*pos, chunk);

        // Seed lightspread from every block
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    spreader.spread_from(&mut chunk.skylight, &chunk.blocks, [x, y, z].into());
                }
            }
        }

        // Finish spreading
        spreader.spread_pending(&mut chunk.skylight, &chunk.blocks);

        time!(store lightspread shared.avg_light_time);
    }
}

fn gen_thread(mut gen: GenState, info_send: Sender<Box<WorldInfo>>, cfg: GenConfig) -> Result<()> {
    // Load Lua state and get all gen-related functions from it
    let lua = Lua::new();
    let mut world_info = None;
    let mut gen_chunk = None;
    let mut lua_gc = None;
    lua.context(|lua| -> Result<()> {
        crate::lua::open_generic_libs(&gen.global, lua);
        lua.globals().set("gen", lua.create_table()?)?;
        let args = cfg
            .args
            .into_iter()
            .map(|arg| arg.to_lua(lua))
            .collect::<LuaResult<Vec<_>>>()?;
        lua.load(&cfg.lua_main)
            .set_name("worldgen.lua")
            .unwrap()
            .call(LuaMultiValue::from_vec(args))?;
        let luagen = lua.globals().get::<_, LuaTable>("gen")?;

        let mut info: Box<WorldInfo> = default();
        let luatex = luagen
            .get::<_, LuaFunction>("textures")?
            .call::<_, LuaTable>(())?;
        for res in luatex.pairs::<u8, LuaValue>() {
            let (k, v) = res?;
            let v = rlua_serde::from_value::<BlockTexture>(v)?;
            info.blocks[k as usize] = v;
        }
        let lualight = luagen
            .get::<_, LuaFunction>("lightmodes")?
            .call::<_, LuaTable>(())?;
        for res in lualight.pairs::<u8, LuaValue>() {
            let (k, v) = res?;
            let v = rlua_serde::from_value::<LightingConf>(v)?;
            info.light_modes[k as usize] = v;
        }
        world_info = Some(info);

        let gchunk = luagen.get::<_, LuaFunction>("chunk")?;
        let gchunk = lua.create_registry_value(gchunk)?;
        gen_chunk = Some(gchunk);

        let gc = luagen.get::<_, LuaFunction>("gc")?;
        let gc = lua.create_registry_value(gc)?;
        lua_gc = Some(gc);

        Ok(())
    })?;
    let world_info = world_info.unwrap();
    let gen_chunk = gen_chunk.unwrap();
    let lua_gc = lua_gc.unwrap();

    // Pass world info back to the main game
    let style = StyleTable::new(&world_info);
    let mut light_spreader = LightSpreader::new(&world_info);
    info_send
        .send(world_info)
        .map_err(|_| Error::msg("failed to send world info"))?;

    let mut last_stall_warning = Instant::now();

    const GEN_QUEUE: usize = 32;
    let mut last_gc = Instant::now();
    let gc_interval = Duration::from_millis(2435);

    let mut chunks = gen.chunks.read();

    let mut seenbuf = default();
    let mut gen_sortbuf = Vec::new();

    let mut gen_queue = Vec::with_capacity(GEN_QUEUE);
    let mut unsent: HashMap<ChunkPos, ChunkBox> = default();
    'outer: loop {
        //Collect unused data every once in a while
        if last_gc.elapsed() > gc_interval {
            lua.context(|lua| -> Result<()> {
                let gc = lua.registry_value::<LuaFunction>(&lua_gc)?;
                gc.call(())?;
                Ok(())
            })?;
            last_gc = Instant::now();
        }

        //Gather a set of chunks to generate by scanning available chunks
        {
            let gen_area = gen.shared.gen_area.lock().clone();
            gen_sortbuf.clear();
            chunks.iter_nearby(
                &mut seenbuf,
                gen_area.center,
                gen_area.gen_radius,
                |pos, dist, _delta| {
                    if chunks.get(pos).is_none() && !unsent.contains_key(&pos) {
                        gen_sortbuf.push((dist, pos));
                    }
                    Ok(())
                },
            )?;
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
        let mut queue_idx = 0;
        while queue_idx < gen_queue.len() {
            time!(start chunkgen);

            // Get the next chunk from the queue
            let pos = gen_queue[queue_idx];
            queue_idx += 1;

            // Fetch a new chunk from Lua
            let chunk = lua.context(|lua| -> Result<ChunkBox> {
                let gen = lua.registry_value::<LuaFunction>(&gen_chunk)?;
                let chunk = gen.call::<_, LuaLightUserData>((
                    pos.coords.x,
                    pos.coords.y,
                    pos.coords.z,
                    pos.dim,
                ))?;
                let chunk = unsafe { mem::transmute::<_, Option<ChunkBox>>(chunk.0) };
                Ok(chunk.ok_or(anyhow!("gen_chunk produced no chunk!"))?)
            });
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(err) => {
                    println!("failed to generate chunk at {:?}: {}", pos, err);
                    continue;
                }
            };

            // If the chunk is not shiny or solid, generate more chunks above it until the sky is
            // reached, so shadows can be computed
            if !(chunk.is_homogeneous()
                && (chunk.homogeneous_shinethrough() || chunk.homogeneous_block().is_solid(&style)))
            {
                let mut above = pos;
                above.coords.z += 1;
                if gen.chunks.read().chunk_at(above).is_none() && !unsent.contains_key(&above) {
                    gen_queue.push(above);
                }
            }

            // Add chunk to the queue
            // TODO: Sort portals and optimize lookups
            unsent.insert(pos, chunk);
            gencount += 1;

            // Keep chunkgen timing statistics
            time!(store chunkgen gen.shared.avg_gen_time);

            // Close quickly if requested
            if gen.shared.close.load() {
                break 'outer;
            }
        }

        // Process sky lighting
        process_lighting(
            &gen.shared,
            &chunks,
            &mut unsent,
            &mut light_spreader,
            &style,
        );

        // Try harder to send back chunks
        if !unsent.is_empty() {
            let stall_start = Instant::now();
            drop(chunks);
            {
                let mut chunks_w = gen.chunks.write();
                empty_chunk_buf(&mut chunks_w, &mut unsent, &mut gen.entity);
            }
            chunks = gen.chunks.read();
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

use std::num::NonZeroU32;

use crate::prelude::*;
use common::lua::LuaValueStatic;

const AVERAGE_WEIGHT: f32 = 0.005;

pub struct GeneratorHandle {
    shared: Arc<GenSharedState>,
    tex_recv: Receiver<BlockTextures>,
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
        });
        let (tex_send, tex_recv) = channel::bounded(0);
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
}

struct GenState {
    shared: Arc<GenSharedState>,
    global: Arc<GlobalState>,
    chunks: Arc<RwLock<ChunkStorage>>,
}

/// Transfer chunks from the local buffer to the game chunkstore.
fn empty_chunk_buf(chunks: &mut ChunkStorage, unsent: &mut HashMap<ChunkPos, ChunkBox>) {
    for (pos, chunk) in unsent.drain() {
        chunks.insert(pos, ChunkArc::new(chunk));
    }
    chunks.maybe_gc();
}

fn process_lighting(
    chunks: &ChunkStorage,
    unsent: &mut HashMap<ChunkPos, ChunkBox>,
    style: &StyleTable,
) {
    let full_light = |chunk: &mut ChunkBox| {
        if let Some(data) = chunk.try_blocks_mut() {
            data.skymap.fill(0);
        }
        chunk.mark_shiny();
    };
    let full_dark = |chunk: &mut ChunkBox| {
        if !chunk.is_solid() {
            let data = chunk.blocks_mut();
            data.skymap.fill(CHUNK_SIZE as u8);
        }
    };
    // Process chunk lighting from highest to lowest
    let mut sortbuf = unsent.iter_mut().collect::<Vec<_>>();
    sortbuf.sort_unstable_by_key(|(p, _c)| [p.coords.x, p.coords.y, -p.coords.z]);
    let mut tmp_skymap = [0; (CHUNK_SIZE * CHUNK_SIZE) as usize];
    let mut last_above = None;
    for (&pos, chunk) in sortbuf {
        if chunk.is_shiny() {
            // Shiny chunks are _the_ source of skylight
            full_light(chunk);
        } else if chunk.is_solid() {
            // Full darkness
        } else {
            // Get the chunk above
            let mut above_pos = pos;
            above_pos.coords.z += 1;
            let above = if let Some((abpos, above)) = last_above {
                if above_pos == abpos {
                    Some(above)
                } else {
                    None
                }
            } else {
                None
            }
            .or_else(|| chunks.chunk_at(above_pos));
            let above = match above {
                Some(above) => above,
                None => {
                    println!("could not find above-chunk for {:?}", pos);
                    continue;
                }
            };
            // Special case homogeneous chunks
            if above.is_homogeneous() && !above.is_shiny() {
                full_dark(chunk);
            } else if chunk.is_clear() {
                if let Some(abdata) = above.blocks() {
                    for idx in 0..(CHUNK_SIZE * CHUNK_SIZE) as usize {
                        tmp_skymap[idx] = if abdata.skymap[idx] == 0 {
                            0
                        } else {
                            CHUNK_SIZE as u8
                        };
                    }
                    if tmp_skymap.iter().all(|z| *z == 0) {
                        // Fully lighted
                        chunk.mark_shiny();
                    } else if tmp_skymap.iter().all(|z| *z != 0) {
                        // Fully shadowed
                    } else {
                        // Partial shadow
                        // Sad
                        let data = chunk.blocks_mut();
                        data.skymap = tmp_skymap;
                    }
                } else {
                    // The above-is-homogeneous-and-dark case was already handled
                    // The only option left is for it to be shiny
                    full_light(chunk);
                }
            } else {
                // Compute lighting for each column
                let data = chunk.blocks_mut();
                let has_light = |idx2d| {
                    above
                        .blocks()
                        .map(|abdata| abdata.skymap[idx2d] == 0)
                        .unwrap_or(true)
                };
                let mut col_idx = 0;
                let mut idx_3d;
                for y in 0..CHUNK_SIZE {
                    for x in 0..CHUNK_SIZE {
                        idx_3d = ((((CHUNK_SIZE - 1) << CHUNK_BITS) + y) << CHUNK_BITS) + x;
                        data.skymap[col_idx] = if has_light(col_idx) {
                            // TODO: Figure out portals
                            let mut skyz = 0;
                            for z in (0..CHUNK_SIZE as u8).rev() {
                                if data.blocks[idx_3d as usize].is_solid(style) {
                                    skyz = z;
                                    break;
                                }
                                idx_3d -= CHUNK_SIZE * CHUNK_SIZE;
                            }
                            skyz
                        } else {
                            CHUNK_SIZE as u8
                        };
                        col_idx += 1;
                    }
                }
            }
        }
        // Save this chunk as the above for the next chunk
        last_above = Some((pos, chunk.as_ref()));
    }
}

fn gen_thread(gen: GenState, tex_send: Sender<BlockTextures>, cfg: GenConfig) -> Result<()> {
    // Load Lua state and get all gen-related functions from it
    let lua = Lua::new();
    let mut textures = None;
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

        let tex = BlockTextures::default();
        let luatex = luagen
            .get::<_, LuaFunction>("textures")?
            .call::<_, LuaTable>(())?;
        for res in luatex.pairs::<u8, LuaValue>() {
            let (k, v) = res?;
            let v = rlua_serde::from_value::<BlockTexture>(v)?;
            tex.set(BlockData { data: k }, v);
        }
        textures = Some(tex);

        let gchunk = luagen.get::<_, LuaFunction>("chunk")?;
        let gchunk = lua.create_registry_value(gchunk)?;
        gen_chunk = Some(gchunk);

        let gc = luagen.get::<_, LuaFunction>("gc")?;
        let gc = lua.create_registry_value(gc)?;
        lua_gc = Some(gc);

        Ok(())
    })?;
    let textures = textures.unwrap();
    let gen_chunk = gen_chunk.unwrap();
    let lua_gc = lua_gc.unwrap();

    // Pass block textures back to the main game
    let style = StyleTable::new(&textures);
    tex_send
        .send(textures)
        .map_err(|_| Error::msg("failed to send block textures"))?;

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
                |pos, dist| {
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
            let gen_start = Instant::now();

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
            let mut chunk = match chunk {
                Ok(chunk) => chunk,
                Err(err) => {
                    println!("failed to generate chunk at {:?}: {}", pos, err);
                    continue;
                }
            };

            // If the chunk is homogeneous, mark the appropiate solidity (solid/nonsolid) tags.
            if let Some(block) = chunk.homogeneous_block() {
                if block.is_solid(&style) {
                    chunk.mark_solid();
                } else if block.is_clear(&style) {
                    chunk.mark_clear();
                }
            }

            // If the chunk is not shiny or solid, generate more chunks above it until the sky is
            // reached, so shadows can be computed
            if !chunk.is_shiny() && !chunk.is_solid() {
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
            {
                //Dont care about data races here, after all it's just stats
                //Therefore, dont synchronize
                let time = gen_start.elapsed().as_secs_f32();
                let old_time = gen.shared.avg_gen_time.load();
                let new_time = old_time + (time - old_time) * AVERAGE_WEIGHT;
                gen.shared.avg_gen_time.store(new_time);
            }

            // Close quickly if requested
            if gen.shared.close.load() {
                break 'outer;
            }
        }

        // Process sky lighting
        process_lighting(&chunks, &mut unsent, &style);

        // Try harder to send back chunks
        if !unsent.is_empty() {
            let stall_start = Instant::now();
            drop(chunks);
            {
                let mut chunks_w = gen.chunks.write();
                empty_chunk_buf(&mut chunks_w, &mut unsent);
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

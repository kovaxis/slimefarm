#![allow(unused_imports)]

use common::{
    noise2d::{Noise2d, NoiseScaler2d},
    noise3d::{Noise3d, NoiseScaler3d},
    prelude::*,
    spread2d::Spread2d,
    terrain::{BlockBuf, GridKeeper, GridKeeper2d},
    worldgen::{ChunkFiller, GenStore},
};

struct ClosureFill<F> {
    fill: RefCell<F>,
}
impl<F> ChunkFiller for ClosureFill<F>
where
    F: FnMut(ChunkPos) -> Option<ChunkBox>,
{
    fn fill(&self, pos: ChunkPos) -> Option<ChunkBox> {
        (self.fill.borrow_mut())(pos)
    }
}

fn wrap_gen<F>(store: &'static dyn GenStore, fill: F)
where
    F: FnMut(ChunkPos) -> Option<ChunkBox> + 'static,
{
    store.register(
        "base.chunkfill",
        Box::new(ClosureFill { fill: fill.into() }) as Box<dyn ChunkFiller>,
    );
}

trait Generator {
    fn fill(&mut self, _pos: ChunkPos) -> Option<ChunkBox> {
        Some(ChunkBox::new_empty())
    }

    fn recenter(&mut self, _center: ChunkPos) {}
}

fn register_gen<G>(store: &'static dyn GenStore, gen: G)
where
    G: Generator + 'static,
{
    struct Wrapper<G>(Rc<RefCell<G>>);
    impl<G> ChunkFiller for Wrapper<G>
    where
        G: Generator,
    {
        fn fill(&self, pos: ChunkPos) -> Option<ChunkBox> {
            self.0.borrow_mut().fill(pos)
        }
    }
    let w = Wrapper(Rc::new(RefCell::new(gen)));
    {
        let w = Wrapper(w.0.clone());
        store.register("base.chunkfill", Box::new(w) as Box<dyn ChunkFiller>);
    }
    unsafe {
        store.listen("base.recenter", move |center: &ChunkPos| {
            w.0.borrow_mut().recenter(*center);
        });
    }
}

struct ChunkWindow<'a> {
    chunk: &'a mut ChunkData,
    corner: BlockPos,
}
impl ChunkWindow<'_> {
    fn new(pos: ChunkPos, chunk: &mut ChunkBox) -> ChunkWindow {
        ChunkWindow {
            chunk: chunk.blocks_mut(),
            corner: pos.to_block_floor(),
        }
    }

    fn set(&mut self, pos: BlockPos, block: BlockData) {
        let pos = [
            (pos[0] - self.corner[0]),
            (pos[1] - self.corner[1]),
            (pos[2] - self.corner[2]),
        ];
        if (pos[0] as u32) < CHUNK_SIZE as u32
            && (pos[1] as u32) < CHUNK_SIZE as u32
            && (pos[2] as u32) < CHUNK_SIZE as u32
        {
            *self.chunk.sub_get_mut(pos) = block;
        }
    }
}

#[derive(Deserialize)]
struct Config {
    seed: u64,
    gen_radius: f32,
    kind: GenKind,
}

#[derive(Deserialize)]
enum GenKind {
    Void,
    Parkour(Parkour),
    Plains(Plains),
}

fn void(store: &'static dyn GenStore, _cfg: Config) {
    wrap_gen(store, |_pos| Some(ChunkBox::new_empty()))
}

#[derive(Deserialize, Clone)]
struct Parkour {
    z_offset: f32,
    delta: f32,
    color: [f32; 3],
}

fn parkour(store: &'static dyn GenStore, cfg: Config, k: Parkour) {
    let noise_gen = Noise3d::new(
        cfg.seed,
        &[(128., 1.), (64., 0.5), (32., 0.25), (16., 0.125)],
    );
    let noise_scaler = NoiseScaler3d::new(CHUNK_SIZE / 4, CHUNK_SIZE as f32);
    wrap_gen(store, move |pos| {
        let mut chunk = ChunkBox::new_quick();
        let blocks = chunk.blocks_mut();
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
            pos.to_block_floor().to_float_floor(),
            CHUNK_SIZE as f64,
        );
        // */
        //Transform bulk noise into block ids
        let mut idx = 0;
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let real_z = pos[2] * CHUNK_SIZE + z;
                    //let noise = noise_buf[idx] - real_z as f32 * 0.04;
                    let noise = noise_scaler.get(Vec3::new(x as f32, y as f32, z as f32))
                        - real_z as f32 * k.z_offset;
                    let normalized = noise / (k.delta + noise.abs());
                    if normalized > 0. {
                        blocks.set_idx(idx, BlockData { data: 1 });
                    } else {
                        blocks.set_idx(idx, BlockData { data: 0 });
                    }
                    idx += 1;
                }
            }
        }
        chunk.try_drop_blocks();
        Some(chunk)
    })
}

#[derive(Deserialize)]
struct Plains {
    xy_scale: f64,
    z_scale: f32,
    detail: i32,
    color: [f32; 3],
    log_color: [f32; 3],
}
fn plains(store: &'static dyn GenStore, cfg: Config, k: Plains) {
    const TREESPACING: i32 = 16;
    const TREESUBDIV: i32 = CHUNK_SIZE / TREESPACING;

    struct PlainsGen {
        cfg: Config,
        k: Plains,
        cols: GridKeeper2d<Option<(i16, i16, LoafBox<i16>)>>,
        trees: GridKeeper2d<Option<(BlockPos, BlockBuf)>>,
        noise2d: Noise2d,
        tree_spread: Spread2d,
    }
    impl PlainsGen {
        /// generate the heightmap of the given chunk.
        fn gen_hmap(&mut self, pos: [i32; 2]) -> Option<&(i16, i16, LoafBox<i16>)> {
            let k = &self.k;
            let noise2d = &self.noise2d;
            Some(self.cols.get_mut(pos)?.get_or_insert_with(|| {
                //Generate the height map for this column
                let mut hmap: LoafBox<i16> = unsafe { common::arena::alloc().assume_init() };
                let mut noise_buf = [0.; (CHUNK_SIZE * CHUNK_SIZE) as usize];
                noise2d.noise_block(
                    [(pos[0] * CHUNK_SIZE) as f64, (pos[1] * CHUNK_SIZE) as f64],
                    CHUNK_SIZE as f64,
                    CHUNK_SIZE,
                    &mut noise_buf[..],
                    false,
                );
                let mut min = i16::max_value();
                let mut max = i16::min_value();
                for (&noise, height) in noise_buf.iter().zip(hmap.iter_mut()) {
                    *height = (noise * k.z_scale) as i16;
                    min = min.min(*height);
                    max = max.max(*height);
                }
                (min, max, hmap)
            }))
        }

        /// generate the base terrain of a given chunk.
        fn gen_terrain(&mut self, pos: ChunkPos) -> Option<ChunkBox> {
            let (col_min, col_max, col) = self.cols.get(pos.xy())?.as_ref()?;
            if pos[2] * CHUNK_SIZE >= *col_max as i32 {
                //Chunk is high enough to be all-air
                return Some(ChunkBox::new_empty());
            } else if (pos[2] + 1) * CHUNK_SIZE <= *col_min as i32 {
                //Chunk is low enough to be all-ground
                return Some(ChunkBox::new_solid());
            }
            let mut chunk = ChunkBox::new_quick();
            let blocks = chunk.blocks_mut();
            let mut idx_3d = 0;
            for z in 0..CHUNK_SIZE {
                let mut idx_2d = 0;
                for _y in 0..CHUNK_SIZE {
                    for _x in 0..CHUNK_SIZE {
                        let real_z = pos[2] * CHUNK_SIZE + z;
                        blocks.set_idx(
                            idx_3d,
                            BlockData {
                                data: (real_z < col[idx_2d] as i32) as u8,
                            },
                        );
                        idx_2d += 1;
                        idx_3d += 1;
                    }
                }
            }
            Some(chunk)
        }

        /// generate the tree at the given tree-coord.
        /// (`TREESUBDIV` tree-units per chunk.)
        fn gen_tree(&mut self, tcoord: [i32; 2]) -> Option<&(BlockPos, BlockBuf)> {
            let chunkpos = [
                tcoord[0].div_euclid(TREESUBDIV),
                tcoord[1].div_euclid(TREESUBDIV),
            ];
            self.gen_hmap(chunkpos)?;
            let tree_spread = &self.tree_spread;
            let cols = &self.cols;
            let cfg = &self.cfg;
            Some(self.trees.get_mut(tcoord)?.get_or_insert_with(|| {
                // generate a tree at this tree-grid position
                let thorizpos = tree_spread.gen(tcoord) * TREESPACING as f32;
                let thorizpos = [
                    tcoord[0] * TREESPACING + thorizpos[0] as i32,
                    tcoord[1] * TREESPACING + thorizpos[1] as i32,
                ];
                let subchunkpos = [
                    thorizpos[0].rem_euclid(CHUNK_SIZE),
                    thorizpos[1].rem_euclid(CHUNK_SIZE),
                ];
                let (_, _, hmap) = cols.get(chunkpos).unwrap().as_ref().unwrap();
                let theight = hmap[(subchunkpos[0] + subchunkpos[1] * CHUNK_SIZE) as usize];
                let tpos = BlockPos([thorizpos[0], thorizpos[1], theight as i32]);
                let mut bbuf = BlockBuf::new();
                //let mut bbuf = BlockBuf::with_capacity([-8, -8, 0], [4, 4, 32]);

                // actually generate tree in the buffer
                let mut rng = FastRng::seed_from_u64(fxhash::hash64(&(cfg.seed, tcoord)));
                let wood = BlockData { data: 2 };
                let td = rng.gen_range(2. ..5.);
                let th: f32 = rng.gen_range(10. ..20.);
                let tbh = 5.;
                for z in 0..th.ceil() as i32 {
                    let d = td * ((th - z as f32) / (th - tbh)).powf(0.8);
                    let di = d.ceil() as i32;
                    let d2 = d * d;
                    for y in -di..=di {
                        for x in -di..=di {
                            if (x * x + y * y) as f32 <= d2 {
                                bbuf.set(BlockPos([x, y, z]), wood);
                            }
                        }
                    }
                }

                (tpos, bbuf)
            }))
        }
    }
    impl Generator for PlainsGen {
        fn fill(&mut self, pos: ChunkPos) -> Option<ChunkBox> {
            self.gen_hmap(pos.xy())?;
            let mut chunk = self.gen_terrain(pos)?;

            let xy = pos.xy();
            let xy = [xy[0] * TREESUBDIV, xy[1] * TREESUBDIV];
            for y in -2..=TREESUBDIV + 2 {
                for x in -2..=TREESUBDIV + 2 {
                    let tcoord = [xy[0] + x, xy[1] + y];
                    let (tpos, treebuf) = self.gen_tree(tcoord)?;
                    treebuf.transfer(*tpos, pos, &mut chunk);
                }
            }

            Some(chunk)

            /*
            let (min, max, hmap) = cols.get(args.pos.xz())?.as_ref()?;
            if args.pos[1] << CHUNK_BITS < *max as i32
                && (args.pos[1] + 1) << CHUNK_BITS > *min as i32
            {
                for _ in 0..1 {
                    let chunk_xz = args.pos.xz();
                    let mut counter: u32 = 0xfb4;
                    let mut rand = fxhash::hash64(&(chunk_xz, counter));
                    let mut bits_left: i32 = 64;
                    let mut rand_num = |bits| {
                        bits_left -= bits as i32;
                        if bits_left < 0 {
                            rand = fxhash::hash64(&(chunk_xz, counter));
                            bits_left = 64 - bits as i32;
                            counter = counter.wrapping_add(1);
                        }
                        let num = rand as i32 & ((1 << bits) - 1);
                        rand >>= bits;
                        num
                    };
                    let pos = [rand_num(CHUNK_BITS), rand_num(CHUNK_BITS)];
                    let y = hmap[(pos[0] | (pos[1] << CHUNK_BITS)) as usize];
                    if (y as i32) >> CHUNK_BITS == args.pos[1] {
                        //Generate a tree here
                        let base_y = y as i32 & CHUNK_MASK;
                        let pos_f = Vec2::new(
                            pos[0] as f32 + rand_num(3) as f32 / 8. - 0.5,
                            pos[1] as f32 + rand_num(3) as f32 / 8. - 0.5,
                        );
                        let n = (2 + rand_num(4) % 5) as usize;
                        let r = 2. + rand_num(4) as f32 * (2.4 / 16.);
                        let h = 5 + rand_num(5) * 7 / 32;
                        let angle_vel = (rand_num(4) as f32 + 0.5) * (0.44 / 8.) - 1.;
                        let mut attrs = [Vec2::broadcast(0.); 8];
                        let mut base_angle = rand_num(6) as f32 * (f32::PI / 64.);
                        let approach =
                            |attrs: &mut [Vec2], base_angle: f32, factor, radius| {
                                let mut angle = base_angle;
                                for i in 0..n {
                                    let target = pos_f
                                        + radius * Vec2::new(angle.cos(), angle.sin());
                                    attrs[i] += (target - attrs[i]) * factor;
                                    angle += 2. * f32::PI / n as f32;
                                }
                            };
                        approach(&mut attrs, base_angle, 1., r);
                        for dy in -2..h {
                            let y = base_y + dy;
                            let extra_r = r;
                            let r = if dy <= 0 {
                                ((-dy + 1) * (-dy + 1)) as f32
                            } else {
                                let s = (dy + 1) as f32 / h as f32;
                                s.powi(4) * r
                            };
                            let coef = {
                                let mut angle: f32 = 0.;
                                let mut sum = 0.;
                                for _ in 0..n {
                                    sum += (extra_r * extra_r
                                        + 2. * (extra_r + r) * r * (1. - angle.cos()))
                                    .sqrt()
                                    .sqrt();
                                    angle += 2. * f32::PI / n as f32;
                                }
                                sum
                            };
                            let r_int = (r + extra_r).ceil() as i32;
                            //Perturb attractors
                            base_angle += angle_vel;
                            approach(&mut attrs, base_angle, 0.8, r);
                            for i in 0..n {
                                let disturb = Vec2::new(
                                    (rand_num(5) as f32 + 0.5) / 16. - 1.,
                                    (rand_num(5) as f32 + 0.5) / 16. - 1.,
                                );
                                attrs[i] += disturb * 0.8;
                            }
                            eprintln!(
                                "setting layer [{}, {}] -> [{}, {}] on height {}",
                                pos[0] - r_int,
                                pos[1] - r_int,
                                pos[0] + r_int,
                                pos[1] + r_int,
                                y
                            );
                            for z in pos[1] - r_int..=pos[1] + r_int {
                                for x in pos[0] - r_int..=pos[0] + r_int {
                                    // TODO
                                    /*if args.substrate.get([x, y, z]).is_solid() {
                                        continue;
                                    }*/
                                    let block_pos = Vec2::new(x as f32, z as f32);
                                    if (block_pos - pos_f).mag_sq()
                                        <= (r + extra_r) * (r + extra_r)
                                    {
                                        //TODO
                                        //args.output.set([x, y, z], BlockData { data: 1 });
                                    }
                                    /*
                                    let mut sum = 0.;
                                    for i in 0..n {
                                        sum += (attrs[i] - block_pos).mag().sqrt();
                                    }
                                    if sum <= coef {
                                        args.output.set([x, y, z], BlockData { data: 1 });
                                    }*/
                                }
                            }
                        }
                    }
                }
            }
            */
        }

        fn recenter(&mut self, center: ChunkPos) {
            self.cols.set_center(center.xy());
        }
    }
    let gen = PlainsGen {
        cols: GridKeeper2d::with_radius(cfg.gen_radius, [0, 0]),
        trees: GridKeeper2d::with_radius(cfg.gen_radius * TREESUBDIV as f32, [0, 0]),
        noise2d: Noise2d::new_octaves(cfg.seed, k.xy_scale, k.detail),
        tree_spread: Spread2d::new(cfg.seed),
        cfg,
        k,
    };
    register_gen(store, gen);

    /*let gen_hmap = |cols: &mut GridKeeper2d<Option<(i16, i16, LoafBox<i16>)>>, pos: ChunkPos| {
        Some(cols.get_mut(pos.xy())?.get_or_insert_with(|| {
            //Generate the height map for this column
            let mut hmap: LoafBox<i16> = unsafe { common::arena::alloc().assume_init() };
            let mut noise_buf = [0.; (CHUNK_SIZE * CHUNK_SIZE) as usize];
            noise2d.noise_block(
                [(pos[0] * CHUNK_SIZE) as f64, (pos[1] * CHUNK_SIZE) as f64],
                CHUNK_SIZE as f64,
                CHUNK_SIZE,
                &mut noise_buf[..],
                false,
            );
            let mut min = i16::max_value();
            let mut max = i16::min_value();
            for (&noise, height) in noise_buf.iter().zip(hmap.iter_mut()) {
                *height = (noise * k.z_scale) as i16;
                min = min.min(*height);
                max = max.max(*height);
            }
            (min, max, hmap)
        }))
    };*/
}

pub fn new_generator<'a>(cfg: &[u8], store: &'static dyn GenStore) -> Result<()> {
    let mut cfg: Config = serde_json::from_slice(cfg)?;
    let kind = mem::replace(&mut cfg.kind, GenKind::Void);
    match kind {
        GenKind::Void => void(store, cfg),
        GenKind::Parkour(k) => parkour(store, cfg, k),
        GenKind::Plains(k) => plains(store, cfg, k),
    }
    Ok(())
}

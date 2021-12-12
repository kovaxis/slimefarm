#![allow(unused_imports)]

use common::{
    impl_cast_to,
    noise2d::{Noise2d, NoiseScaler2d},
    noise3d::{Noise3d, NoiseScaler3d},
    prelude::*,
    terrain::{GridKeeper, GridKeeper2d},
    worldgen::{
        BlockColorArgs, BlockColorRet, BlockColorizer, ChunkFill, ChunkFillArgs, ChunkFillRet,
        GenCapsule, GenStore, GenStoreRef,
    },
};

struct ClosureFill<I, S, T, F, C> {
    init: Cell<Option<I>>,
    init_state: RefCell<Option<(Arc<S>, T)>>,
    fill: F,
    colorizer: Cell<Option<Box<dyn BlockColorizer>>>,
    colorizer_raw: Cell<Option<C>>,
    layer_count: i32,
}
impl<'a, I, S, T, F, C> ChunkFill for ClosureFill<I, S, T, F, C>
where
    F: 'a + Fn(&S, &T, ChunkFillArgs) -> ChunkFillRet,
{
    fn fill(&self, args: ChunkFillArgs) -> ChunkFillRet {
        let st = self.init_state.borrow();
        let (s, t) = st.as_ref().expect("fill() called before init()");
        (self.fill)(s, t, args)
    }
    fn layer_count(&self) -> i32 {
        self.layer_count
    }
    fn colorizer(&self) -> Box<dyn BlockColorizer> {
        self.colorizer
            .take()
            .expect("colorizer taken twice or before calling init()")
    }
}
impl<'a, I, S, T, F, C> GenCapsule<'a> for ClosureFill<I, S, T, F, C>
where
    F: 'a + Fn(&S, &T, ChunkFillArgs) -> ChunkFillRet,
    S: 'static + Send + Sync,
    T: 'a,
    I: 'a + FnOnce(GenStoreRef<'a>) -> (S, T),
    C: 'static + Send + FnMut(&S, BlockColorArgs) -> BlockColorRet,
{
    impl_cast_to! {
        "base.chunkfill" => ChunkFill
    }
    fn init(&'a self, store: GenStoreRef<'a>) {
        let (s, t) = self.init.take().expect("init() called twice")(store);
        let s1 = Arc::new(s);
        let s2 = s1.clone();
        *self.init_state.borrow_mut() = Some((s1, t));
        self.colorizer.set(Some(Box::new(ClosureColorize {
            shared: s2,
            colorize: self.colorizer_raw.take().unwrap(),
        })));
    }
}

struct ClosureColorize<S, C> {
    shared: Arc<S>,
    colorize: C,
}
impl<S, C> BlockColorizer for ClosureColorize<S, C>
where
    S: Send + Sync,
    C: Send + FnMut(&S, BlockColorArgs) -> BlockColorRet,
{
    fn colorize(&mut self, args: BlockColorArgs) -> BlockColorRet {
        (self.colorize)(&self.shared, args)
    }
}

fn wrap_gen<'a, I, S, T, F, C>(
    store: GenStoreRef<'a>,
    layer_count: i32,
    init: I,
    fill: F,
    colorize: C,
) where
    I: 'a + FnOnce(GenStoreRef<'a>) -> (S, T),
    S: 'static + Send + Sync,
    T: 'a,
    F: 'a + Fn(&S, &T, ChunkFillArgs) -> ChunkFillRet,
    C: Send + 'static + FnMut(&S, BlockColorArgs) -> BlockColorRet,
{
    store.register(
        "base.chunkfill",
        Box::new(ClosureFill {
            init: Some(init).into(),
            init_state: None.into(),
            fill,
            colorizer: None.into(),
            colorizer_raw: Some(colorize).into(),
            layer_count,
        }),
    );
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

fn void<'a>(store: &'a (dyn 'a + GenStore), _cfg: Config) {
    wrap_gen(
        store,
        1,
        |_store| ((), ()),
        |(), (), _args| Some(()),
        |(), args| {
            for c in args.out {
                *c = [0.; 3];
            }
        },
    )
}

#[derive(Deserialize, Clone)]
struct Parkour {
    y_offset: f32,
    delta: f32,
    color: [f32; 3],
}

fn parkour(store: GenStoreRef, cfg: Config, k: Parkour) {
    let noise_gen = Noise3d::new(
        cfg.seed,
        &[(128., 1.), (64., 0.5), (32., 0.25), (16., 0.125)],
    );
    let noise_scaler = NoiseScaler3d::new(CHUNK_SIZE / 4, CHUNK_SIZE as f32);
    wrap_gen(
        store,
        1,
        move |_store| (k, ()),
        move |k, (), args| {
            let chunk = args.take_blocks();
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
                args.pos.to_block_floor().to_float_floor(),
                CHUNK_SIZE as f64,
            );
            // */
            //Transform bulk noise into block ids
            let mut idx = 0;
            for z in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE {
                    for x in 0..CHUNK_SIZE {
                        let real_y = args.pos[1] * CHUNK_SIZE + y;
                        //let noise = noise_buf[idx] - real_y as f32 * 0.04;
                        let noise = noise_scaler.get(Vec3::new(x as f32, y as f32, z as f32))
                            - real_y as f32 * k.y_offset;
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
            Some(())
        },
        move |k, args| {
            for color in args.out {
                *color = k.color;
            }
        },
    )
}

#[derive(Deserialize)]
struct Plains {
    xz_scale: f64,
    y_scale: f32,
    detail: i32,
    color: [f32; 3],
    log_color: [f32; 3],
}
/*
fn plains(cfg: Config, k: Plains) -> AnyGen {
    let mut cols =
        GridKeeper2d::<Option<(i16, i16, LoafBox<i16>)>>::with_radius(cfg.gen_radius, [0, 0]);
    let noise2d = Noise2d::new_octaves(cfg.seed, k.xz_scale, k.detail);
    wrap_gen(
        k,
        2,
        move |k, args: ChunkFillArgs| {
            match args.layer {
                1 => {
                    cols.set_center(args.pos.xz());
                    let (col_min, col_max, col) =
                        cols.get_mut(args.pos.xz())?.get_or_insert_with(|| {
                            //Generate the height map for this column
                            let mut hmap: LoafBox<i16> =
                                unsafe { common::arena::alloc().assume_init() };
                            let mut noise_buf = [0.; (CHUNK_SIZE * CHUNK_SIZE) as usize];
                            noise2d.noise_block(
                                [
                                    (args.pos[0] * CHUNK_SIZE) as f64,
                                    (args.pos[2] * CHUNK_SIZE) as f64,
                                ],
                                CHUNK_SIZE as f64,
                                CHUNK_SIZE,
                                &mut noise_buf[..],
                                false,
                            );
                            let mut min = i16::max_value();
                            let mut max = i16::min_value();
                            for (&noise, height) in noise_buf.iter().zip(hmap.iter_mut()) {
                                *height = (noise * k.y_scale) as i16;
                                min = min.min(*height);
                                max = max.max(*height);
                            }
                            (min, max, hmap)
                        });
                    let chunk = args.take_blocks();
                    if args.pos[1] * CHUNK_SIZE >= *col_max as i32 {
                        //Chunk is high enough to be all-air
                        chunk.make_empty();
                        return Some(());
                    } else if (args.pos[1] + 1) * CHUNK_SIZE <= *col_min as i32 {
                        //Chunk is low enough to be all-ground
                        chunk.make_solid();
                        return Some(());
                    }
                    let blocks = chunk.blocks_mut();
                    let mut idx_3d = 0;
                    let mut idx_2d = 0;
                    for _z in 0..CHUNK_SIZE {
                        for y in 0..CHUNK_SIZE {
                            let mut idx_2d = idx_2d;
                            for _x in 0..CHUNK_SIZE {
                                let real_y = args.pos[1] * CHUNK_SIZE + y;
                                blocks.set_idx(
                                    idx_3d,
                                    BlockData {
                                        data: (real_y < col[idx_2d] as i32) as u8,
                                    },
                                );
                                idx_2d += 1;
                                idx_3d += 1;
                            }
                        }
                        idx_2d += CHUNK_SIZE as usize;
                    }
                }
                2 => {
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
                }
                _ => {}
            };
            Some(())
        },
        |k, args| {
            let color = match args.id {
                0 => k.color,
                1 => k.color,
                2 => k.log_color,
                _ => [0.; 3],
            };
            for col in args.out {
                *col = color;
            }
        },
    )
}*/

pub fn new_generator<'a>(cfg: &[u8], store: &'a (dyn 'a + GenStore)) -> Result<()> {
    let mut cfg: Config = serde_json::from_slice(cfg)?;
    let kind = mem::replace(&mut cfg.kind, GenKind::Void);
    match kind {
        GenKind::Void => void(store, cfg),
        GenKind::Parkour(k) => parkour(store, cfg, k),
        GenKind::Plains(k) => todo!(), //plains(cfg, k),
    }
    Ok(())
}

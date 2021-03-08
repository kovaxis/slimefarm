#![allow(unused_imports)]

use common::{
    noise2d::{Noise2d, NoiseScaler2d},
    noise3d::{Noise3d, NoiseScaler3d},
    prelude::*,
    terrain::{GridKeeper, GridKeeper2d},
    worldgen::{ChunkFillArgs, ChunkFillRet, ChunkGen, ChunkGenerator},
};

struct ClosureGen<F>(F);
impl<F> ChunkGen for ClosureGen<F>
where
    F: FnMut(ChunkFillArgs) -> ChunkFillRet + Send,
{
    type Shared = ();
    type Local = F;
    fn split(self) -> ((), F) {
        ((), self.0)
    }
    fn fill(_shared: &(), local: &mut F, args: ChunkFillArgs) -> ChunkFillRet {
        local(args)
    }
}

type AnyGen = ChunkGenerator;
fn wrap_gen<F>(f: F) -> AnyGen
where
    F: FnMut(ChunkFillArgs) -> ChunkFillRet + Send + 'static,
{
    AnyGen::new(ClosureGen(f))
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

fn void(_cfg: Config) -> ChunkGenerator {
    wrap_gen(move |args| {
        for b in args.chunk.blocks.iter_mut() {
            *b = BlockData { data: 0 };
        }
        Some(())
    })
}

#[derive(Deserialize)]
struct Parkour {
    y_offset: f32,
    delta: f32,
}

fn parkour(cfg: Config, k: Parkour) -> ChunkGenerator {
    let noise_gen = Noise3d::new(
        cfg.seed,
        &[(128., 1.), (64., 0.5), (32., 0.25), (16., 0.125)],
    );
    let mut noise_scaler = NoiseScaler3d::new(CHUNK_SIZE / 4, CHUNK_SIZE as f32);
    wrap_gen(move |args| {
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
                        args.chunk.blocks[idx].data = 15 + (normalized * 235.) as u8;
                    } else {
                        args.chunk.blocks[idx].data = 0;
                    }
                    idx += 1;
                }
            }
        }
        Some(())
    })
}

#[derive(Deserialize)]
struct Plains {
    xz_scale: f64,
    y_scale: f32,
    detail: i32,
}

fn plains(cfg: Config, k: Plains) -> ChunkGenerator {
    let mut cols = GridKeeper2d::<Option<LoafBox<i32>>>::with_radius(cfg.gen_radius, [0, 0]);
    let noise2d = Noise2d::new_octaves(cfg.seed, k.xz_scale, k.detail);
    wrap_gen(move |args| {
        cols.set_center(args.center.xz());
        let col = cols.get_mut(args.pos.xz())?.get_or_insert_with(|| {
            //Generate the height map for this column
            let mut noise: LoafBox<i32> = unsafe { common::arena::alloc().assume_init() };
            noise2d.noise_block(
                [
                    (args.pos[0] * CHUNK_SIZE) as f64,
                    (args.pos[2] * CHUNK_SIZE) as f64,
                ],
                CHUNK_SIZE as f64,
                CHUNK_SIZE,
                unsafe { mem::transmute(&mut noise[..]) },
                false,
            );
            for y in noise.iter_mut() {
                *y = (f32::from_bits(*y as u32) * k.y_scale) as i32;
            }
            noise
        });
        let mut idx_3d = 0;
        let mut idx_2d = 0;
        for _z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                let mut idx_2d = idx_2d;
                for _x in 0..CHUNK_SIZE {
                    let real_y = args.pos[1] * CHUNK_SIZE + y;
                    args.chunk.blocks[idx_3d].data = (real_y < col[idx_2d]) as u8;
                    idx_2d += 1;
                    idx_3d += 1;
                }
            }
            idx_2d += CHUNK_SIZE as usize;
        }
        Some(())
    })
}

pub fn new_generator(cfg: &[u8]) -> Result<ChunkGenerator> {
    let mut cfg: Config = serde_json::from_slice(cfg)?;
    let kind = mem::replace(&mut cfg.kind, GenKind::Void);
    Ok(match kind {
        GenKind::Void => void(cfg),
        GenKind::Parkour(k) => parkour(cfg, k),
        GenKind::Plains(k) => plains(cfg, k),
    })
}

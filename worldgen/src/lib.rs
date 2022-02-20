#![allow(unused_imports)]

use crate::prelude::*;

mod prelude {
    pub(crate) use crate::{actionbuf::ActionBuf, blockbuf::BlockBuf};
    pub(crate) use common::{
        lua_assert, lua_bail, lua_func, lua_lib, lua_type,
        noise2d::{Noise2d, NoiseScaler2d},
        noise3d::{Noise3d, NoiseScaler3d},
        prelude::*,
        spread2d::Spread2d,
        terrain::{GridKeeper2, GridKeeper3, PortalData},
    };
}

mod actionbuf;
mod blockbuf;

struct LuaChunkBox {
    chunk: ChunkBox,
}
lua_type! {LuaChunkBox,
    mut fn into_raw(lua, this, ()) {
        let chunk = mem::replace(&mut this.chunk, ChunkBox::new_homogeneous(BlockData { data: 0 }));
        unsafe {
            mem::transmute::<ChunkBox, LuaLightUserData>(chunk)
        }
    }

    mut fn fill(lua, this, block: u8) {
        this.chunk.make_homogeneous(BlockData { data: block });
    }
}
impl From<ChunkBox> for LuaChunkBox {
    fn from(chunk: ChunkBox) -> Self {
        Self { chunk }
    }
}

#[derive(Deserialize)]
struct BufGridCfg {
    seed: i64,
    cell_size: i32,
    margin: i32,
}

struct BufGrid2 {
    grid: GridKeeper2<ActionBuf>,
    cellsize: i32,
    margin: i32,
    spread: Spread2d,
}
impl BufGrid2 {
    fn new(k: BufGridCfg) -> Self {
        Self {
            grid: default(),
            cellsize: k.cell_size,
            margin: k.margin,
            spread: Spread2d::new(k.seed as u64),
        }
    }

    fn gen_col(&mut self, gen: &LuaFunction, pos: Int2) -> LuaResult<&ActionBuf> {
        let cellsize = self.cellsize;
        let spread = &self.spread;
        Ok(&*self.grid.try_or_insert(pos, || -> LuaResult<_> {
            let cellfrac = spread.gen(pos) * cellsize as f32;
            let cellf64 = (pos * cellsize).to_f64();
            let realpos = [
                cellf64[0] + cellfrac[0] as f64,
                cellf64[1] + cellfrac[1] as f64,
            ];
            let tmpbuf = gen.call::<_, LuaAnyUserData>((realpos[0], realpos[1]))?;
            let mut tmpbuf = tmpbuf.borrow_mut::<ActionBuf>()?;
            Ok(tmpbuf.take())
        })?)
    }

    fn fill_chunk(
        &mut self,
        gen: &LuaFunction,
        chunk_pos: Int3,
        chunk: &mut ChunkBox,
    ) -> LuaResult<()> {
        // Minimum and maximum block positions (both inclusive)
        let mn = (chunk_pos.xy() << CHUNK_BITS) - Int2::splat(self.margin);
        let mx = (chunk_pos.xy() << CHUNK_BITS) + Int2::splat(CHUNK_SIZE - 1 + self.margin);
        // Minimum and maximum cell positions (both inclusive)
        let mn = mn / self.cellsize;
        let mx = mx / self.cellsize;
        // Apply actionbufs from all touching cells
        for y in mn.y..=mx.y {
            for x in mn.x..=mx.x {
                let cellpos = Int2::new([x, y]);
                let cellbuf = self.gen_col(gen, cellpos)?;
                cellbuf.transfer(chunk_pos, chunk);
            }
        }
        Ok(())
    }
}
lua_type! {BufGrid2,
    mut fn fill_chunk(lua, this, (x, y, z, chunk, gen): (i32, i32, i32, LuaAnyUserData, LuaFunction)) {
        let pos = Int3::new([x, y, z]);
        let mut chunk = chunk.borrow_mut::<LuaChunkBox>()?;
        this.fill_chunk(&gen, pos, &mut chunk.chunk)?;

    }

    fn cell_size(lua, this, ()) {
        this.cellsize
    }

    fn margin(lua, this, ()) {
        this.margin
    }
}

#[derive(Deserialize)]
struct HeightMapCfg {
    seed: i64,
    noise: Vec<(f64, f32)>,
    offset: f32,
    scale: f32,
    ground: BlockData,
    air: BlockData,
}

struct HeightMap {
    cols: GridKeeper2<(i16, i16, LoafBox<i16>)>,
    noise: Noise2d,
    k: HeightMapCfg,
}
impl HeightMap {
    fn new(k: HeightMapCfg) -> Self {
        Self {
            cols: default(),
            noise: Noise2d::new(k.seed as u64, &k.noise),
            k,
        }
    }

    fn gen_col(&mut self, pos: Int2) -> &(i16, i16, LoafBox<i16>) {
        let noise = &self.noise;
        let k = &self.k;
        self.cols.or_insert(pos, || {
            //Generate the height map for this chunk column
            let mut hmap: LoafBox<i16> = unsafe {
                let mut hmap = common::arena::alloc();
                ptr::write(
                    hmap.as_mut(),
                    Uninit::new([0i16; (CHUNK_SIZE * CHUNK_SIZE) as usize]),
                );
                hmap.assume_init()
            };
            let mut noise_buf = [0.; (CHUNK_SIZE * CHUNK_SIZE) as usize];
            noise.noise_block(
                [(pos[0] * CHUNK_SIZE) as f64, (pos[1] * CHUNK_SIZE) as f64],
                CHUNK_SIZE as f64,
                CHUNK_SIZE,
                &mut noise_buf[..],
                false,
            );
            let mut min = i16::max_value();
            let mut max = i16::min_value();
            for (&noise, height) in noise_buf.iter().zip(hmap.iter_mut()) {
                *height = (noise.mul_add(k.scale, k.offset)) as i16;
                min = min.min(*height);
                max = max.max(*height);
            }
            (min, max, hmap)
        })
    }

    fn height_at(&mut self, pos: Int2) -> i16 {
        let (_, _, cnk) = self.gen_col(pos >> CHUNK_BITS);
        cnk[pos.lowbits(CHUNK_BITS).to_index([CHUNK_BITS; 2].into())]
    }

    fn fill_chunk(&mut self, pos: Int3, chunk: &mut ChunkBox) {
        self.gen_col(pos.xy());
        let (col_min, col_max, col) = self.cols.get(pos.xy()).unwrap();
        let k = &self.k;
        if pos.z * CHUNK_SIZE >= *col_max as i32 {
            //Chunk is high enough to be all-air
            chunk.make_homogeneous(k.air);
            return;
        } else if (pos[2] + 1) * CHUNK_SIZE <= *col_min as i32 {
            //Chunk is low enough to be all-ground
            chunk.make_homogeneous(k.ground);
            return;
        }
        let blocks = chunk.blocks_mut();
        let mut idx_3d = 0;
        for z in 0..CHUNK_SIZE {
            let mut idx_2d = 0;
            for _y in 0..CHUNK_SIZE {
                for _x in 0..CHUNK_SIZE {
                    let real_z = (pos.z << CHUNK_BITS) + z;
                    blocks.set_idx(
                        idx_3d,
                        if real_z < col[idx_2d] as i32 {
                            k.ground
                        } else {
                            k.air
                        },
                    );
                    idx_2d += 1;
                    idx_3d += 1;
                }
            }
        }
    }
}
lua_type! {HeightMap,
    mut fn fill_chunk(lua, this, (x, y, z, chunk): (i32, i32, i32, LuaAnyUserData)) {
        let pos = Int3::new([x, y, z]);
        let mut chunk = chunk.borrow_mut::<LuaChunkBox>()?;
        this.fill_chunk(pos, &mut chunk.chunk);
    }

    // The Z coordinate of the lowest air block in the given column.
    mut fn height_at(lua, this, (x, y): (i32, i32)) {
        let pos = Int2::new([x, y]);
        this.height_at(pos)
    }
}

fn open_lib(lua: LuaContext) -> Result<LuaTable> {
    let state = ();
    let lib = lua_lib! {lua, state,
        fn chunk(block: u8) {
            LuaChunkBox::from(ChunkBox::new_homogeneous(BlockData { data: block }))
        }

        fn chunk_uninit(()) {
            unsafe {
                LuaChunkBox::from(ChunkBox::new_uninit())
            }
        }

        fn chunk_from_raw(raw: LuaLightUserData) {
            unsafe {
                LuaChunkBox {
                    chunk: mem::transmute(raw),
                }
            }
        }

        fn heightmap(cfg: LuaValue) {
            let cfg = rlua_serde::from_value(cfg).to_lua_err()?;
            HeightMap::new(cfg)
        }

        fn action_buf(()) {
            ActionBuf::new(Int3::zero())
        }

        fn structure_grid_2d(cfg: LuaValue) {
            let cfg = rlua_serde::from_value(cfg).to_lua_err()?;
            BufGrid2::new(cfg)
        }
    };
    Ok(lib)
}

#[no_mangle]
#[allow(improper_ctypes_definitions)]
extern "C" fn lua_open(lua: LuaContext, getfunc: fn(&[u8]) -> usize) -> Result<LuaValue> {
    unsafe {
        common::arena::init_impl(getfunc);
    }
    let lualib = open_lib(lua)?;
    Ok(lualib.to_lua(lua)?)
}

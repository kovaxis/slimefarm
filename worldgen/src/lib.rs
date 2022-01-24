#![allow(unused_imports)]

use crate::prelude::*;

mod prelude {
    pub(crate) use crate::serde::LuaFuncRef;
    pub(crate) use common::{
        lua::LuaRng,
        noise2d::{Noise2d, NoiseScaler2d},
        noise3d::{Noise3d, NoiseScaler3d},
        prelude::*,
        spread2d::Spread2d,
        terrain::{BlockBuf, GridKeeper, GridKeeper2d},
        worldgen::BlockIdAlloc,
        worldgen::{ChunkFiller, GenStore},
    };
}
mod serde;

fn get_lua(store: &'static dyn GenStore) -> Rc<Lua> {
    let lua = unsafe { store.lookup::<Rc<Lua>>("base.lua") };
    lua.clone()
}

fn get_blockreg(store: &'static dyn GenStore) -> &'static dyn BlockIdAlloc {
    unsafe { store.lookup::<dyn BlockIdAlloc>("base.blockregister") }
}

fn get_blocktextures(store: &'static dyn GenStore) -> &'static BlockTextures {
    unsafe { store.lookup::<BlockTextures>("base.blocktextures") }
}

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
    fn fill(&mut self, _pos: ChunkPos) -> Option<ChunkBox>;

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

fn register_block(store: &'static dyn GenStore, name: &str, tex: &BlockTexture) -> BlockData {
    let id = get_blockreg(store).get(name);
    get_blocktextures(store).set(id, tex.clone());
    id
}

fn lookup_block(store: &'static dyn GenStore, name: &str) -> BlockData {
    get_blockreg(store).get(name)
}

#[derive(Deserialize)]
struct Config {
    seed: u64,
    gen_radius: f32,
    kind: GenKind,
    air_tex: BlockTexture,
    void_tex: BlockTexture,
}

#[derive(Deserialize)]
enum GenKind {
    Void,
    Parkour(Parkour),
    Plains(Plains),
}

fn void(store: &'static dyn GenStore, _cfg: Config) {
    let air = lookup_block(store, "base.air");
    wrap_gen(store, move |_pos| Some(ChunkBox::new_nonsolid(air)))
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
            (pos << CHUNK_BITS).to_f64_floor(),
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
struct TreeCfg {
    /// Texture of the wood block.
    wood_tex: BlockTexture,
    /// Spacing between trees.
    /// Should be a power of two.
    spacing: i32,
    /// How many trees to generate from the nearby chunks.
    /// Usually 1 or 2 are enough, unless trees are very closely packed with huge foliages.
    extragen: i32,

    make: LuaFuncRef,
    /*
    /// Initial vertical inclination.
    /// 0 is directly upwards, PI/2 is directly horizontal.
    initial_incl: [f32; 2],
    /// Initial tree sectional area.
    initial_area: [f32; 2],
    /// Area loss per unit of branch length/depth.
    area_per_len: [f32; 2],
    /// Initial trunk length, serves as a base for the length of the rest of the branches.
    trunk_len: [f32; 2],
    /// How much does tree depth affect branch length.
    /// Every `halflen` units of depth, the branch length is halved.
    halflen: [f32; 2],
    /// How much to rotate the branching angle per branch-off.
    /// Seems to be a fibonacci ratio?
    /// 1, 1, 2, 3, 5, 8, 13
    /// Angles like 2PI * 1/2, 2PI * 1/3, 2PI * 2/5, 2PI * 3/8, etc...
    rot_angle: [f32; 2],
    /// How much sectional area of the branch should offshoots steal from the main branch.
    /// Eg. `0.2` means a fifth of the area goes to the offshoot.
    offshoot_area: [f32; 2],
    /// How much perturbation should the offshoot receive, in radians.
    /// `0` = directly forward
    /// `PI/2` = 90 degrees perpendicular from the main branch
    offshoot_perturb: [f32; 2],
    /// How much perturbation should the main branch receive from an offshoot in radians.
    /// Usually negative to compensate for the offshoot.
    /// `0` = no perturbation
    /// `-PI/2` = 90 degrees against from the offshoot
    main_perturb: [f32; 2],
    /// If less than this sectional area is attributed to a tree, stop generating branches.
    prune_area: f32,
    /// If tree depth (total distance to the root along the branches) is higher than this number,
    /// stop generating branches.
    prune_depth: f32,
    */
}

#[derive(Deserialize)]
struct Plains {
    xy_scale: f64,
    z_scale: f32,
    detail: i32,
    grass_tex: BlockTexture,

    tree: TreeCfg,
}
fn plains(store: &'static dyn GenStore, cfg: Config, k: Plains) {
    struct PlainsGen {
        lua: Rc<Lua>,
        cfg: Config,
        k: Plains,
        cols: GridKeeper2d<Option<(i16, i16, LoafBox<i16>)>>,
        trees: GridKeeper2d<Option<(BlockPos, BlockBuf)>>,
        noise2d: Noise2d,
        tree_spread: Spread2d,
        void: BlockData,
        air: BlockData,
        grass: BlockData,
        wood: BlockData,
    }
    #[derive(Deserialize)]
    struct Branch {
        yaw: f32,
        pitch: f32,
        len: f32,
        r0: f32,
        r1: f32,
        children: Vec<Branch>,
    }
    impl PlainsGen {
        /// generate the heightmap of the given chunk.
        fn gen_hmap(&mut self, pos: Int2) -> Option<&(i16, i16, LoafBox<i16>)> {
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
                return Some(ChunkBox::new_nonsolid(self.air));
            } else if (pos[2] + 1) * CHUNK_SIZE <= *col_min as i32 {
                //Chunk is low enough to be all-ground
                return Some(ChunkBox::new_solid(self.grass));
            }
            let mut chunk = ChunkBox::new_quick();
            let blocks = chunk.blocks_mut();
            let mut idx_3d = 0;
            for z in 0..CHUNK_SIZE {
                let mut idx_2d = 0;
                for _y in 0..CHUNK_SIZE {
                    for _x in 0..CHUNK_SIZE {
                        let real_z = pos.z * CHUNK_SIZE + z;
                        blocks.set_idx(
                            idx_3d,
                            if real_z < col[idx_2d] as i32 {
                                self.grass
                            } else {
                                self.air
                            },
                        );
                        idx_2d += 1;
                        idx_3d += 1;
                    }
                }
            }
            Some(chunk)
        }

        /*
        fn gen_branch(
            &self,
            rng: &mut FastRng,
            bbuf: &mut BlockBuf,
            pos: Vec3,
            up: Vec3,
            norm: Vec3,
            area: f32,
            depth: f32,
        ) {
            let wood = BlockData { data: 2 };
            let k = &self.k.tree;

            if area < k.prune_area || depth > k.prune_depth {
                return;
            }

            let offshoot_area = rng.gen_range(k.offshoot_area[0]..=k.offshoot_area[1]);
            let len = rng.gen_range(k.trunk_len[0]..=k.trunk_len[1])
                * (-depth / rng.gen_range(k.halflen[0]..=k.halflen[1])).exp2();
            let top = pos + up * len;
            let norm = norm.rotated_by(Rotor3::from_angle_plane(
                rng.gen_range(k.rot_angle[0]..=k.rot_angle[1]),
                Bivec3::from_normalized_axis(up),
            ));

            let newarea = area - rng.gen_range(k.area_per_len[0]..=k.area_per_len[1]) * len;
            let newdepth = depth + len;
            bbuf.fill_cylinder(pos, top, area.sqrt(), newarea.sqrt(), wood);

            let perturb_plane = up.wedge(norm);

            let subperturb = Rotor3::from_angle_plane(
                rng.gen_range(k.main_perturb[0]..=k.main_perturb[1]),
                perturb_plane,
            );
            let subup = up.rotated_by(subperturb);
            let subnorm = norm.rotated_by(subperturb);
            self.gen_branch(
                rng,
                bbuf,
                top,
                subup,
                subnorm,
                newarea * (1. - offshoot_area),
                newdepth,
            );

            let offperturb = Rotor3::from_angle_plane(
                rng.gen_range(k.offshoot_perturb[0]..=k.offshoot_perturb[1]),
                perturb_plane,
            );
            let offup = up.rotated_by(offperturb);
            let offnorm = norm.rotated_by(offperturb);
            self.gen_branch(
                rng,
                bbuf,
                top,
                offup,
                offnorm,
                newarea * offshoot_area,
                newdepth,
            );
        }
        */

        fn gen_branch(&self, bbuf: &mut BlockBuf, branch: Branch, pos: Vec3, up: Vec3, norm: Vec3) {
            let norm = norm.rotated_by(Rotor3::from_angle_plane(
                branch.yaw,
                Bivec3::from_normalized_axis(up),
            ));
            let perturb = Rotor3::from_angle_plane(branch.pitch, up.wedge(norm));
            let up = up.rotated_by(perturb);
            let norm = norm.rotated_by(perturb);
            // Maybe renormalize?

            let top = pos + up * branch.len;
            bbuf.fill_cylinder(pos, top, branch.r0, branch.r1, self.wood);

            for b in branch.children {
                self.gen_branch(bbuf, b, top, up, norm);
            }
        }

        /// generate the tree at the given tree-coord.
        /// (`TREESUBDIV` tree-units per chunk.)
        fn gen_tree(&mut self, tcoord: Int2) -> Option<()> {
            let k = &self.k.tree;
            if self.trees.get(tcoord)?.is_some() {
                return Some(());
            }
            // generate a tree at this tree-grid position
            let thorizpos = self.tree_spread.gen(tcoord) * k.spacing as f32;
            let tfracpos = thorizpos.map(f32::fract);
            let thorizpos = tcoord * k.spacing + Int2::from_f32(thorizpos);
            let chunkpos = thorizpos >> CHUNK_BITS;
            let subchunkpos = thorizpos.lowbits(CHUNK_BITS);
            let (_, _, hmap) = self.gen_hmap(chunkpos)?;
            let theight = hmap[subchunkpos.to_index([CHUNK_BITS; 2].into())];
            let tpos = thorizpos.with_z(theight as i32);
            let mut bbuf = BlockBuf::new(self.void);
            //let mut bbuf = BlockBuf::with_capacity([-32, -32, -16].into(), [64, 64, 128].into());

            // actually generate tree in the buffer
            let rng = FastRng::seed_from_u64(fxhash::hash64(&(self.cfg.seed, "trees", tcoord)));

            let res = self.lua.context(|lua| -> LuaResult<()> {
                let tree_root: LuaValue = self.k.tree.make.get(lua)?.call(LuaRng::new(rng))?;
                let tree_root: Branch = crate::serde::deserialize(lua, tree_root)
                    .map_err(|e| LuaError::RuntimeError(format!("{}", e)))?;
                self.gen_branch(
                    &mut bbuf,
                    tree_root,
                    Vec3::new(tfracpos.x, tfracpos.y, 0.),
                    [0., 0., 1.].into(),
                    [1., 0., 0.].into(),
                );
                Ok(())
            });
            if let Err(e) = res {
                eprintln!("error building tree: {}", e);
            }

            /*
            let k = &self.k.tree;
            let initial_area = rng.gen_range(k.initial_area[0]..=k.initial_area[1]);
            let up = Vec3::unit_x().rotated_by(
                Rotor3::from_rotation_xy(rng.gen_range(0. ..f32::PI * 2.))
                    * Rotor3::from_rotation_xz(
                        f32::PI / 2. + rng.gen_range(k.initial_incl[0]..=k.initial_incl[1]),
                    ),
            );
            let horiz = Vec3::unit_x()
                .rotated_by(Rotor3::from_rotation_xy(rng.gen_range(0. ..2. * f32::PI)));
            self.gen_branch(
                &mut rng,
                &mut bbuf,
                Vec3::new(tfracpos.x, tfracpos.y, 0.),
                up,
                horiz,
                initial_area,
                0.,
            );
            */

            /*
            let tr = rng.gen_range(k.tree_width[0]..k.tree_width[1]) / 2.;
            let th: f32 = rng.gen_range(k.tree_height[0]..k.tree_height[1]);
            let tbh = k.tree_height[2];
            for z in -k.tree_undergen..th.ceil() as i32 {
                let r = tr * ((th - z as f32) / (th - tbh)).powf(k.tree_taperpow);
                let ri = r.ceil() as i32;
                let r2 = r * r;
                for y in -ri..=ri {
                    for x in -ri..=ri {
                        if (x * x + y * y) as f32 <= r2 {
                            bbuf.set(BlockPos([x, y, z]), wood);
                        }
                    }
                }
            }
            */

            self.trees.get_mut(tcoord)?.get_or_insert((tpos, bbuf));
            Some(())
        }
    }
    impl Generator for PlainsGen {
        fn fill(&mut self, pos: ChunkPos) -> Option<ChunkBox> {
            self.gen_hmap(pos.xy())?;
            let mut chunk = self.gen_terrain(pos)?;

            let extragen = self.k.tree.extragen;
            let subdiv = (CHUNK_SIZE / self.k.tree.spacing).max(1);
            let xy = (pos.xy() << CHUNK_BITS) / self.k.tree.spacing;
            for y in -extragen..subdiv + extragen {
                for x in -extragen..subdiv + extragen {
                    let tcoord = xy + [x, y];
                    self.gen_tree(tcoord)?;
                    let (tpos, treebuf) = self.trees.get(tcoord)?.as_ref()?;
                    treebuf.transfer(*tpos, pos, &mut chunk);
                }
            }

            Some(chunk)
        }

        fn recenter(&mut self, center: ChunkPos) {
            self.cols.set_center(center.xy());
            self.trees
                .set_center((center.xy() << CHUNK_BITS) / self.k.tree.spacing);
        }
    }
    let gen = PlainsGen {
        void: lookup_block(store, "base.void"),
        air: lookup_block(store, "base.air"),
        grass: register_block(store, "base.grass", &k.grass_tex),
        wood: register_block(store, "base.wood", &k.tree.wood_tex),
        lua: get_lua(store),
        cols: GridKeeper2d::with_radius(cfg.gen_radius / CHUNK_SIZE as f32, Int2::zero()),
        trees: GridKeeper2d::with_radius(cfg.gen_radius / k.tree.spacing as f32, Int2::zero()),
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

pub fn new_generator<'a>(store: &'static dyn GenStore) -> Result<()> {
    let lua = get_lua(store);
    let mut cfg = lua.context(|lua| -> Result<Config> {
        let cfg = lua
            .globals()
            .get::<_, LuaFunction>("config")
            .context("failed to get worldgen config() global function")?
            .call::<_, LuaValue>(())
            .context("config() function errored out")?;
        Ok(crate::serde::deserialize(lua, cfg).context("failed to deserialize worldgen config")?)
    })?;
    register_block(store, "base.air", &cfg.air_tex);
    register_block(store, "base.void", &cfg.void_tex);
    /*let mut cfg: Config =
    serde_json::from_slice(cfg).context("failed to parse worldgen config string")?;*/
    let kind = mem::replace(&mut cfg.kind, GenKind::Void);
    match kind {
        GenKind::Void => void(store, cfg),
        GenKind::Parkour(k) => parkour(store, cfg, k),
        GenKind::Plains(k) => plains(store, cfg, k),
    }
    Ok(())
}

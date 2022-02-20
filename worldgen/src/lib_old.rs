#![allow(unused_imports)]

use common::lua_lib;

use crate::prelude::*;

mod prelude {
    pub(crate) use crate::{
        actionbuf::{actions, ActionBuf},
        blockbuf::BlockBuf,
    };
    pub(crate) use common::{
        lua::serde::LuaFuncRef,
        lua::LuaRng,
        noise2d::{Noise2d, NoiseScaler2d},
        noise3d::{Noise3d, NoiseScaler3d},
        prelude::*,
        spread2d::Spread2d,
        terrain::{GridKeeper2, GridKeeper3, PortalData},
    };
}

mod actionbuf;
mod blockbuf;

#[derive(Deserialize)]
struct Config {
    seed: u64,
    kind: GenKind,
    air_tex: BlockTexture,
    void_tex: BlockTexture,
    portal_tex: BlockTexture,
}

#[derive(Deserialize)]
enum GenKind {
    Void,
    Parkour(Parkour),
    Plains(Plains),
}

fn void(store: &'static dyn GenStore, _cfg: Config) {
    let air = lookup_block(store, "base.air");
    wrap_gen(store, move |_pos| Some(ChunkBox::new_homogeneous(air)))
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
            (pos.coords << CHUNK_BITS).to_f64(),
            CHUNK_SIZE as f64,
        );
        // */
        //Transform bulk noise into block ids
        let mut idx = 0;
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let real_z = pos.coords.z * CHUNK_SIZE + z;
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
    /// Texture of the leaf block.
    leaf_tex: BlockTexture,
    /// Average spacing between trees.
    /// More specifically, there is an average of 1 tree per `spacing x spacing` square.
    spacing: i32,
    /// How many trees to generate from the nearby chunks.
    /// Usually 1 or 2 are enough, unless trees are very closely packed with huge foliages.
    extragen: i32,

    /// The length of the underground root branch.
    root_len: f32,
    /// How much larger should the underground root branch be, compared to the trunk.
    root_grow: f32,

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
        cols: GridKeeper2<(i16, i16, LoafBox<i16>)>,
        trees: GridKeeper2<ActionBuf>,
        noise2d: Noise2d,
        tree_spread: Spread2d,
        void: BlockData,
        portal: BlockData,
        air: BlockData,
        grass: BlockData,
        wood: BlockData,
        leaf: BlockData,
    }
    #[derive(Deserialize)]
    struct Branch {
        yaw: f32,
        pitch: f32,
        len: f32,
        r0: f32,
        r1: f32,
        leaf: [f32; 2],
        children: Vec<Branch>,
    }
    impl PlainsGen {
        /// generate the heightmap of the given chunk.
        fn gen_hmap(&mut self, pos: Int2) -> Option<&(i16, i16, LoafBox<i16>)> {
            let k = &self.k;
            let noise2d = &self.noise2d;
            Some(self.cols.or_insert(pos, || {
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
        fn gen_terrain(&mut self, pos: Int3) -> Option<ChunkBox> {
            let (col_min, col_max, col) = self.cols.get(pos.xy())?;
            if pos[2] * CHUNK_SIZE >= *col_max as i32 {
                //Chunk is high enough to be all-air
                return Some(ChunkBox::new_homogeneous(self.air));
            } else if (pos[2] + 1) * CHUNK_SIZE <= *col_min as i32 {
                //Chunk is low enough to be all-ground
                return Some(ChunkBox::new_homogeneous(self.grass));
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

        fn gen_branch(
            &self,
            bbuf: &mut ActionBuf,
            branch: Branch,
            pos: Vec3,
            mut up: Vec3,
            mut norm: Vec3,
            depth: i32,
        ) {
            norm = norm.rotated_by(Rotor3::from_angle_plane(
                branch.yaw,
                Bivec3::from_normalized_axis(up),
            ));
            let perturb = Rotor3::from_angle_plane(branch.pitch, up.wedge(norm));
            up = up.rotated_by(perturb);
            norm = norm.rotated_by(perturb);
            // Maybe renormalize?

            let top = pos + up * branch.len;
            if depth == 0 {
                actions::Cylinder::paint(
                    bbuf,
                    pos,
                    pos - up * self.k.tree.root_len,
                    branch.r0,
                    branch.r0 * self.k.tree.root_grow,
                    self.wood,
                );
            }
            actions::Cylinder::paint(bbuf, pos, top, branch.r0, branch.r1, self.wood);
            //bbuf.fill_cylinder(pos, top, branch.r0, branch.r1, self.wood);

            if branch.leaf != [0.; 2] {
                actions::Oval::paint(
                    bbuf,
                    top,
                    [branch.leaf[0], branch.leaf[0], branch.leaf[1]].into(),
                    self.leaf,
                );
                //bbuf.fill_sphere(top, 4., self.leaf);
            }

            for b in branch.children {
                self.gen_branch(bbuf, b, top, up, norm, depth + 1);
            }
        }

        /// generate the tree at the given tree-coord.
        /// (`TREESUBDIV` tree-units per chunk.)
        fn gen_tree(&mut self, tcoord: Int2) -> Option<()> {
            let k = &self.k.tree;
            if self.trees.get(tcoord).is_some() {
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
            let mut bbuf = ActionBuf::new(tpos);
            //let mut bbuf = BlockBuf::new(tpos, self.void);
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
                    0,
                );
                Ok(())
            });
            if let Err(e) = res {
                eprintln!("error building tree: {}", e);
            }

            self.trees.insert(tcoord, bbuf);
            Some(())
        }

        fn gen_portal(&mut self, tcoord: Int2) -> Option<()> {
            let k = &self.k.tree;
            if self.trees.get(tcoord).is_some() {
                return Some(());
            }

            // Generate a portal
            let thorizpos = self.tree_spread.gen(tcoord) * k.spacing as f32;
            let thorizpos = tcoord * k.spacing + Int2::from_f32(thorizpos);
            let chunkpos = thorizpos >> CHUNK_BITS;
            let subchunkpos = thorizpos.lowbits(CHUNK_BITS);
            let (_, _, hmap) = self.gen_hmap(chunkpos)?;
            let theight = hmap[subchunkpos.to_index([CHUNK_BITS; 2].into())];
            let tpos = thorizpos.with_z(theight as i32);
            let mut bbuf = ActionBuf::new(tpos);

            // Actually generate portal
            if tcoord == [0, 0].into() {
                // Build outer cube
                actions::Cube::paint(
                    &mut bbuf,
                    Vec3::new(-4., -4., 0.),
                    Vec3::new(8., 8., 16.),
                    self.wood,
                );
                actions::Cube::paint(
                    &mut bbuf,
                    Vec3::new(-3., 3., 1.),
                    Vec3::new(6., 1., 14.),
                    self.air,
                );
                actions::Cube::paint(
                    &mut bbuf,
                    Vec3::new(-3., 2., 1.),
                    Vec3::new(6., 1., 14.),
                    self.portal,
                );

                // Build inner cube
                actions::Cube::paint(
                    &mut bbuf,
                    Vec3::new(-17., -17., -61.),
                    Vec3::new(34., 34., 34.),
                    self.wood,
                );
                actions::Cube::paint(
                    &mut bbuf,
                    Vec3::new(-16., -16., -60.),
                    Vec3::new(32., 32., 32.),
                    self.air,
                );
                actions::Cube::paint(
                    &mut bbuf,
                    Vec3::new(-3., 16., -60.),
                    Vec3::new(6., 1., 14.),
                    self.portal,
                );

                // Build portal between them
                actions::Portal::paint_pair(
                    &mut bbuf,
                    0, // TODO: Set up a dimension registry
                    [-3, 3, 1].into(),
                    [-3, 16, -60].into(),
                    [6, 0, 14].into(),
                );
            }

            self.trees.insert(tcoord, bbuf);
            Some(())
        }
    }
    impl Generator for PlainsGen {
        fn fill(&mut self, pos: ChunkPos) -> Option<ChunkBox> {
            let pos = pos.coords;
            self.gen_hmap(pos.xy())?;
            let mut chunk = self.gen_terrain(pos)?;

            let extragen = self.k.tree.extragen;
            let extragen = Int2::splat(extragen);
            let xy_min = ((pos.xy() << CHUNK_BITS) - extragen) / self.k.tree.spacing;
            let xy_max = (((pos.xy() + [1, 1]) << CHUNK_BITS) + [CHUNK_SIZE - 1; 2] + extragen)
                / self.k.tree.spacing;
            for y in xy_min.y..=xy_max.y {
                for x in xy_min.x..=xy_max.x {
                    let tcoord = Int2::new([x, y]);
                    //self.gen_tree(tcoord)?;
                    self.gen_portal(tcoord)?;
                    let treebuf = self.trees.get(tcoord)?;
                    treebuf.transfer(pos, &mut chunk);
                }
            }

            /*if pos.xy() == Int2::zero() {
                let (zmin, zmax, hmap) = self.gen_hmap(pos.xy())?;
                let abs_z = *zmax as i32 + 2;
                if pos.z == abs_z >> CHUNK_BITS {
                    let chunk = chunk.blocks_mut();

                    let z = abs_z & ((1 << CHUNK_BITS) - 1);
                    chunk.push_portal(PortalData {
                        pos: [0, 0, z as i16],
                        size: [0, 8, 8],
                        jump: [10, 10, 0],
                    });
                    println!("placed portal at z = {}", abs_z);
                }
            }*/

            Some(chunk)
        }

        fn gc(&mut self) {
            self.cols.gc();
            self.trees.gc();
        }
    }
    let gen = PlainsGen {
        void: lookup_block(store, "base.void"),
        portal: lookup_block(store, "base.portal"),
        air: lookup_block(store, "base.air"),
        grass: register_block(store, "base.grass", &k.grass_tex),
        wood: register_block(store, "base.wood", &k.tree.wood_tex),
        leaf: register_block(store, "base.leaf", &k.tree.leaf_tex),
        lua: get_lua(store),
        cols: GridKeeper2::new(),
        trees: GridKeeper2::new(),
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

fn new_generator<'a>(store: &'static dyn GenStore) -> Result<()> {
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
    register_block(store, "base.portal", &cfg.portal_tex);
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

use crate::{
    lua::gfx::{LuaDrawParams, ShaderRef, UniformStorage, UniformVal},
    prelude::*,
};
use common::{lua::LuaRng, lua_assert, lua_bail, lua_func, lua_lib, lua_type};
use notify::{DebouncedEvent as WatcherEvent, Watcher as _};

pub(crate) mod gen;
pub(crate) mod gfx;

#[derive(Clone)]
struct MatrixStack {
    stack: AssertSync<Rc<RefCell<(Vec<Mat4>, Mat4)>>>,
}
impl From<Mat4> for MatrixStack {
    fn from(mat: Mat4) -> MatrixStack {
        MatrixStack {
            stack: AssertSync(Rc::new(RefCell::new((Vec::new(), mat)))),
        }
    }
}
lua_type! {MatrixStack,
    fn reset(lua, this, ()) {
        let (stack, top) = &mut *this.stack.borrow_mut();
        stack.clear();
        *top = Mat4::identity();
    }

    fn reset_from(lua, this, other: MatrixStack) {
        let (stack, top) = &mut *this.stack.borrow_mut();
        let (_, other) = &*other.stack.borrow();
        stack.clear();
        *top = *other;
    }

    fn identity(lua, this, ()) {
        let (_, top) = &mut *this.stack.borrow_mut();
        *top = Mat4::identity();
    }

    fn mul_right(lua, this, other: MatrixStack) {
        let (_, this) = &mut *this.stack.borrow_mut();
        let (_, other) = &*other.stack.borrow();
        *this = *this * *other;
    }
    fn mul_left(lua, this, other: MatrixStack) {
        let (_, this) = &mut *this.stack.borrow_mut();
        let (_, other) = &*other.stack.borrow();
        *this = *other * *this;
    }

    fn push(lua, this, ()) {
        let (stack, top) = &mut *this.stack.borrow_mut();
        stack.push(top.clone());
    }

    fn pop(lua, this, ()) {
        let (stack, top) = &mut *this.stack.borrow_mut();
        if let Some(new_top) = stack.pop() {
            *top = new_top;
        }else{
            *top = Mat4::identity();
        }
    }

    fn translate(lua, this, (x, y, z): (f32, f32, f32)) {
        let (_, top) = &mut *this.stack.borrow_mut();
        *top = *top * Mat4::from_translation(Vec3::new(x, y, z));
    }

    fn scale(lua, this, (x, y, z): (f32, Option<f32>, Option<f32>)) {
        let (_, top) = &mut *this.stack.borrow_mut();
        match (y, z) {
            (Some(y), Some(z)) => {
                *top = *top * Mat4::from_nonuniform_scale(Vec3::new(x, y, z));
            }
            (None, None) => {
                *top = *top * Mat4::from_scale(x)
            }
            _ => lua_bail!("expected 1 or 3 arguments, not 2")
        }
    }

    fn rotate_x(lua, this, angle: f32) {
        let (_, top) = &mut *this.stack.borrow_mut();
        *top = *top * Mat4::from_rotation_x(angle);
    }
    fn rotate_y(lua, this, angle: f32) {
        let (_, top) = &mut *this.stack.borrow_mut();
        *top = *top * Mat4::from_rotation_y(angle);
    }
    fn rotate_z(lua, this, angle: f32) {
        let (_, top) = &mut *this.stack.borrow_mut();
        *top = *top * Mat4::from_rotation_z(angle);
    }

    fn invert(lua, this, ()) {
        let (_, top) = &mut *this.stack.borrow_mut();
        top.inverse();
    }

    fn perspective(lua, this, (fov, aspect, near, far): (f32, f32, f32, f32)) {
        let (_, top) = &mut *this.stack.borrow_mut();
        *top = *top * uv::projection::perspective_gl(fov, aspect, near, far) * Mat4::from_rotation_x(-f32::PI/2.);
    }
    fn orthographic(lua, this, (xleft, xright, ydown, yup, znear, zfar): (f32, f32, f32, f32, f32, f32)) {
        let (_, top) = &mut *this.stack.borrow_mut();
        *top = *top * uv::projection::orthographic_gl(xleft, xright, ydown, yup, znear, zfar);
    }

    fn transform_vec(lua, this, (x, y, z): (f32, f32, f32)) {
        let (_, top) = &*this.stack.borrow();
        let (x, y, z) = top.transform_vec3(Vec3::new(x, y, z)).into();
        (x, y, z)
    }
    fn transform_point(lua, this, (x, y, z): (f32, f32, f32)) {
        let (_, top) = &*this.stack.borrow();
        let (x, y, z) = top.transform_point3(Vec3::new(x, y, z)).into();
        (x, y, z)
    }

    fn set_col(lua, this, (i, x, y, z, w): (usize, f32, f32, f32, f32)) {
        let (_, top) = &mut *this.stack.borrow_mut();
        lua_assert!(i < 4, "invalid row index");
        top[i] = Vec4::new(x, y, z, w);
    }
    fn set_row(lua, this, (i, x, y, z, w): (usize, f32, f32, f32, f32)) {
        let (_, top) = &mut *this.stack.borrow_mut();
        lua_assert!(i < 4, "invalid row index");
        let mat = top.as_mut_slice();
        mat[4 * i + 0] = x;
        mat[4 * i + 1] = y;
        mat[4 * i + 2] = z;
        mat[4 * i + 3] = w;
    }
}

#[derive(Clone)]
struct LuaWorldPos {
    pos: Cell<WorldPos>,
}
impl LuaWorldPos {
    fn get(&self) -> WorldPos {
        self.pos.get()
    }
    fn set(&self, pos: WorldPos) {
        self.pos.set(pos);
    }
}
lua_type! {LuaWorldPos,
    fn raw_difference(lua, this, other: LuaWorldPos) {
        let lhs = this.get();
        let rhs = other.get();
        (
            lhs.coords[0] - rhs.coords[0],
            lhs.coords[1] - rhs.coords[1],
            lhs.coords[2] - rhs.coords[2],
            lhs.dim.wrapping_sub(rhs.dim),
        )
    }

    mut fn copy_from(lua, this, other: LuaWorldPos) {
        this.pos.set(other.get());
    }

    mut fn r#move(lua, this, (terrain, dx, dy, dz): (TerrainRef, f64, f64, f64)) {
        let terrain = terrain.rc.borrow();
        let mut pos = this.get();
        let (mv, crash) = if dx == 0. && dy == 0. {
            terrain.raycast_aligned(&mut pos, 2, dz)
        } else if dx == 0. && dz == 0. {
            terrain.raycast_aligned(&mut pos, 1, dy)
        } else if dy == 0. && dz == 0. {
            terrain.raycast_aligned(&mut pos, 0, dx)
        } else {
            // TODO: Implement particle raycasting
            todo!("unaligned particle raycasting is not implemented yet")
        };
        this.pos.set(pos);
        (mv, crash)
    }

    mut fn move_box(lua, this, (terrain, dx, dy, dz, sx, sy, sz, slide): (TerrainRef, f64, f64, f64, f64, f64, f64, Option<bool>)) {
        let terrain = terrain.rc.borrow();
        let mut pos = this.get();
        let (mv, crash) = terrain.boxcast(&mut pos, [dx, dy, dz], [sx, sy, sz], !slide.unwrap_or(false));
        this.pos.set(pos);
        (mv[0], mv[1], mv[2], crash[0], crash[1], crash[2])
    }
}

#[derive(Clone)]
pub struct TerrainRef {
    rc: AssertSync<Rc<RefCell<Terrain>>>,
}
impl TerrainRef {
    pub(crate) fn new(state: &Rc<State>, gen_cfg: &[u8]) -> Result<TerrainRef> {
        Ok(TerrainRef {
            rc: AssertSync(Rc::new(RefCell::new(Terrain::new(state, gen_cfg)?))),
        })
    }
}
lua_type! {TerrainRef,
    fn bookkeep(lua, this, pos: LuaWorldPos) {
        this.rc.borrow_mut().bookkeep(pos.pos.get().block_pos());
    }

    fn set_view_distance(lua, this, (view, gen): (f32, f32)) {
        this.rc.borrow_mut().set_view_radius(view, gen)
    }

    fn visible_radius(lua, this, ()) {
        this.rc.borrow().last_min_viewdist
    }

    fn draw(lua, this, (
        shader,
        uniforms,
        offset_uniform,
        params,
        mvp,
        camstack_raw,
        subdraw_callback
    ): (
        ShaderRef,
        UniformStorage,
        u32,
        LuaDrawParams,
        MatrixStack,
        LuaAnyUserData,
        LuaFunction
    )) {
        let this = this.rc.borrow();
        let mvp = mvp.stack.borrow().1;
        let camstack = camstack_raw.borrow::<CameraStack>()?;
        let subdraw = |origin: &WorldPos, framequad: &[Vec3; 4], depth: u8| -> Result<()> {
            camstack.push(&mvp, *origin, *framequad);
            ensure!(camstack.depth() == depth, "invalid CameraStack depth");
            subdraw_callback.call(camstack_raw.clone())?;
            camstack.pop();
            Ok(())
        };
        this.draw(
            &shader.program,
            uniforms,
            offset_uniform,
            camstack.origin(),
            &params.params,
            mvp,
            camstack.framequad(),
            camstack.depth(),
            &subdraw,
        ).to_lua_err()?;
    }

    fn get_draw_positions(lua, this, (entpos, sx, sy, sz, camstack, out): (LuaWorldPos, f64, f64, f64, LuaAnyUserData, LuaTable)) {
        let this = this.rc.borrow();
        let entpos = entpos.get();
        let camstack = camstack.borrow::<CameraStack>()?;
        let origin = camstack.origin();

        let mut idx: usize = 0;
        this.get_draw_positions(entpos, [sx, sy, sz], |_, jump| {
            if origin.dim == jump.dim {
                let pos = [
                    entpos.coords[0] + jump.coords[0] as f64 - origin.coords[0],
                    entpos.coords[1] + jump.coords[1] as f64 - origin.coords[1],
                    entpos.coords[2] + jump.coords[2] as f64 - origin.coords[2],
                ];
                idx += 1;
                let _ = out.raw_set(idx, pos[0]);
                idx += 1;
                let _ = out.raw_set(idx, pos[1]);
                idx += 1;
                let _ = out.raw_set(idx, pos[2]);
            }
        });
        for i in idx + 1 ..= out.raw_len() as usize {
            out.raw_set(i, LuaValue::Nil)?;
        }
    }

    fn chunk_gen_time(lua, this, ()) {
        this.rc.borrow().generator.avg_gen_time.load()
    }
    fn chunk_mesh_time(lua, this, ()) {
        this.rc.borrow().mesher.avg_mesh_time.load()
    }
    fn chunk_mesh_upload_time(lua, this, ()) {
        this.rc.borrow().mesher.avg_upload_time.load()
    }
}

// TODO: Include portal/screen buffer to offload portal drawing to lua
struct CameraFrame {
    /// Where should the graphics-world-coordinates origin be, in absolute world coordinates.
    origin: WorldPos,
    /// A quad that contains all that is being currently rendered to the screen, in relative world
    /// coordinates (ie. relative to `origin`).
    framequad: [Vec3; 4],
    /// Clipping planes for the current framequad.
    clip_planes: [Vec4; 5],
}
impl CameraFrame {
    fn new() -> Self {
        Self {
            origin: WorldPos {
                coords: [0.; 3],
                dim: 0,
            },
            framequad: default(),
            clip_planes: default(),
        }
    }
}

/// A camera stack, representing portal cameras.
struct CameraStack {
    /// A stack of cameras.
    stack: RefCell<Vec<CameraFrame>>,
}
impl CameraStack {
    fn new() -> Self {
        Self {
            stack: vec![CameraFrame::new()].into(),
        }
    }

    fn origin(&self) -> WorldPos {
        self.stack.borrow().last().unwrap().origin
    }

    fn framequad(&self) -> [Vec3; 4] {
        self.stack.borrow().last().unwrap().framequad
    }

    fn clip_planes(&self) -> [Vec4; 5] {
        self.stack.borrow().last().unwrap().clip_planes
    }

    fn depth(&self) -> u8 {
        (self.stack.borrow().len() - 1) as u8
    }

    fn push(&self, mvp: &Mat4, origin: WorldPos, fq: [Vec3; 4]) {
        self.stack.borrow_mut().push(CameraFrame {
            origin,
            framequad: fq,
            clip_planes: Terrain::calc_clip_planes(mvp, &fq),
        });
    }

    fn pop(&self) {
        self.stack.borrow_mut().pop();
    }
}
lua_type! {CameraStack,
    fn reset(lua, this, (origin, mvp): (LuaWorldPos, MatrixStack)) {
        this.stack.borrow_mut().clear();
        let mvp = mvp.stack.borrow().1;
        let inv_mvp = mvp.inversed();
        this.push(
            &mvp,
            origin.get(),
            [
                inv_mvp.transform_point3(Vec3::new(-1., -1., -1.)),
                inv_mvp.transform_point3(Vec3::new(1., -1., -1.)),
                inv_mvp.transform_point3(Vec3::new(1., 1., -1.)),
                inv_mvp.transform_point3(Vec3::new(-1., 1., -1.)),
            ],
        );
    }

    fn set_origin(lua, this, pos: LuaWorldPos) {
        this.stack.borrow_mut().last_mut().unwrap().origin = pos.get();
    }

    fn set_framequad(lua, this, (i, x, y, z): (usize, f32, f32, f32)) {
        match this.stack.borrow_mut().last_mut().unwrap().framequad.get_mut(i) {
            Some(v) => *v = Vec3::new(x, y, z),
            None => lua_bail!("invalid framequad vertex index {}", i),
        }
    }

    fn origin(lua, this, pos: LuaAnyUserData) {
        let pos = pos.borrow::<LuaWorldPos>()?;
        pos.set(this.origin());
    }

    fn framequad(lua, this, i: usize) {
        lua_assert!(i < 4, "invalid framequad vertex index");
        let v = this.framequad()[i];
        (v.x, v.y, v.z)
    }

    fn clip_planes(lua, this, out: LuaTable) {
        let p = this.clip_planes();
        for i in 0..5 {
            for j in 0..4 {
                out.raw_set(i*4+j+1, p[i][j])?;
            }
        }
    }

    fn can_view(lua, this, (x, y, z, r): (f32, f32, f32, f32)) {
        let u = Vec4::new(x, y, z, 1.);
        this.clip_planes().iter()
            .all(|p| p.dot(u) >= -r)
    }

    fn depth(lua, this, ()) {
        this.depth()
    }
}

struct Watcher {
    _raw: notify::RecommendedWatcher,
    rx: std::sync::mpsc::Receiver<WatcherEvent>,
    modified: Cell<Instant>,
    checked: Cell<Instant>,
}
impl Watcher {
    fn new(path: &str, debounce: Duration) -> Result<Self> {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut raw = notify::watcher(tx, debounce)?;
        raw.watch(path, notify::RecursiveMode::NonRecursive)?;
        let now = Instant::now();
        Ok(Self {
            _raw: raw,
            rx,
            modified: now.into(),
            checked: now.into(),
        })
    }

    fn tick(&self) {
        let now = Instant::now();
        for _ev in self.rx.try_iter() {
            self.modified.set(now);
        }
    }
}

#[derive(Clone)]
struct WatcherRef {
    rc: AssertSync<Rc<Watcher>>,
}
lua_type! {WatcherRef,
    // Check whether the file being watched was changed between the last call to `check` and this
    // call.
    fn changed(lua, this, ()) {
        this.rc.tick();
        let modified = this.rc.modified > this.rc.checked;
        this.rc.checked.set(this.rc.modified.get());
        modified
    }
}

pub(crate) fn modify_std_lib(state: &Arc<GlobalState>, lua: LuaContext) {
    let os = lua.globals().get::<_, LuaTable>("os").unwrap();
    os.set(
        "sleep",
        lua_func!(lua, state, fn(secs: f64) {
            thread::sleep(Duration::from_secs_f64(secs))
        }),
    )
    .unwrap();
    os.set(
        "clock",
        lua_func!(lua, state, fn(()) {
            (Instant::now() - state.base_time).as_secs_f64()
        }),
    )
    .unwrap();

    let math = lua.globals().get::<_, LuaTable>("math").unwrap();
    math.set(
        "rng",
        lua_func!(lua, state, fn(seed: LuaMultiValue) {
            use std::hash::{Hash, Hasher};
            let mut hasher = fxhash::FxHasher64::default();
            for val in seed {
                match val {
                    LuaValue::Nil => (0u8).hash(&mut hasher),
                    LuaValue::Boolean(b) => (1u8, b as u8).hash(&mut hasher),
                    LuaValue::Integer(int) => (2u8, int).hash(&mut hasher),
                    LuaValue::Number(num) => if num as i64 as f64 == num {
                        (2u8, num as i64).hash(&mut hasher)
                    }else{
                        (3u8, num.to_bits()).hash(&mut hasher)
                    }
                    LuaValue::String(s) => s.as_bytes().hash(&mut hasher),
                    _ => {
                        Err(LuaError::RuntimeError(format!("cannot hash type {}", "unknown"/*val.type_name()*/)))?;
                    }
                }
            }
            LuaRng::seed(hasher.finish())
        }),
    )
    .unwrap();
}

pub(crate) fn open_vec3_lib(_state: &Arc<GlobalState>, lua: LuaContext) {
    let state = ();
    lua.globals()
        .set(
            "vec3",
            lua_lib! {lua, state,
                // Dot product between two 3D vectors.
                fn dot((x0, y0, z0, x1, y1, z1): (f64, f64, f64, f64, f64, f64)) {
                    x0 * x1 + y0 * y1 + z0 * z1
                }

                // Cross product between two 3D vectors, resulting in a third vector.
                fn cross((x0, y0, z0, x1, y1, z1): (f64, f64, f64, f64, f64, f64)) {
                    (y0 * z1 -z0 * y1, z0 * x1 -x0 * z1, x0 * y1 - y0 * x1)
                }

                // The squared magnitude of a vector.
                fn magsq((x, y, z): (f64, f64, f64)) {
                    x * x + y * y + z * z
                }

                // The magnitude of a vector.
                fn mag((x, y, z): (f64, f64, f64)) {
                    (x * x + y * y + z * z).sqrt()
                }

                // Normalize a vector.
                fn normalize((x, y, z): (f64, f64, f64)) {
                    let f = (x * x + y * y + z * z).sqrt().recip();
                    (x * f, y * f, z * f)
                }

                // Rotate a vector around an axis (normalized) given a certain angle.
                fn rotate((angle, x, y, z): (f64, f64, f64, f64)) {
                    let (sin, cos) = angle.sin_cos();
                    let mul = 1. - cos;

                    let x_sin = x * sin;
                    let y_sin = y * sin;
                    let z_sin = z * sin;

                    let xy_mul = x * y * mul;
                    let xz_mul = x * z * mul;
                    let yz_mul = y * z * mul;

                    let m00 = (x * x).mul_add(mul, cos);
                    let m10 = xy_mul + z_sin;
                    let m20 = xz_mul - y_sin;
                    let m01 = xy_mul - z_sin;
                    let m11 = (y * y).mul_add(mul, cos);
                    let m21 = yz_mul + x_sin;
                    let m02 = xz_mul + y_sin;
                    let m12 = yz_mul - x_sin;
                    let m22 = (z * z).mul_add(mul, cos);

                    (
                        m00 * x + m01 * y + m02 * z,
                        m10 * x + m11 * y + m12 * z,
                        m20 * x + m21 * y + m22 * z
                    )
                }
            },
        )
        .unwrap();
}

pub(crate) fn open_fs_lib(_state: &Arc<GlobalState>, lua: LuaContext) {
    let state = ();
    lua.globals()
        .set(
            "fs",
            lua_lib! {lua, state,
                fn watch((path, debounce): (LuaString, Option<f64>)) {
                    let debounce = Duration::from_secs_f64(debounce.unwrap_or(1.));
                    WatcherRef {
                        rc: AssertSync(Rc::new(
                            Watcher::new(path.to_str()?, debounce).to_lua_err()?
                        )),
                    }
                }
            },
        )
        .unwrap();
}

pub(crate) fn open_system_lib(state: &Rc<State>, lua: LuaContext) {
    lua.globals()
        .set(
            "system",
            lua_lib! {lua, state,
                fn matrix(()) {
                    MatrixStack::from(Mat4::identity())
                }

                fn terrain(cfg: LuaValue) {
                    let cfg = match &cfg {
                        LuaValue::String(s) => {
                            s.as_bytes()
                        },
                        _ => return Err("expected string").to_lua_err()
                    };
                    TerrainRef::new(state, cfg).to_lua_err()?
                }

                fn world_pos((raw_x, raw_y, raw_z, raw_dim): (Option<f64>, Option<f64>, Option<f64>, Option<u32>)) {
                    LuaWorldPos {
                        pos: WorldPos {
                            coords: [
                                raw_x.unwrap_or(0.),
                                raw_y.unwrap_or(0.),
                                raw_z.unwrap_or(0.),
                            ],
                            dim: raw_dim.unwrap_or(0),
                        }.into(),
                    }
                }
            },
        )
        .unwrap();
}

pub(crate) fn open_generic_libs(state: &Arc<GlobalState>, lua: LuaContext) {
    modify_std_lib(state, lua);
    open_fs_lib(state, lua);
    open_vec3_lib(state, lua);
}

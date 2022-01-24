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
    fn hint_center(lua, this, (x, y, z): (i32, i32, i32)) {
        this.rc.borrow_mut().hint_center(BlockPos([x, y, z]));
    }

    fn set_view_distance(lua, this, dist: f32) {
        this.rc.borrow_mut().set_view_radius(dist)
    }

    fn collide(lua, this, (x, y, z, dx, dy, dz, sx, sy, sz): (f64, f64, f64, f64, f64, f64, f64, f64, f64)) {
        let terrain = this.rc.borrow();
        let [fx, fy, fz] = crate::terrain::check_collisions([x, y, z], [dx, dy, dz], [sx, sy, sz], |block_pos, _axis| {
            terrain.block_at(block_pos).map(|data| data.is_solid(&terrain.solid)).unwrap_or(true)
        }, false);
        (fx, fy, fz)
    }

    fn raycast(lua, this, (x, y, z, dx, dy, dz, sx, sy, sz): (f64, f64, f64, f64, f64, f64, f64, f64, f64)) {
        let terrain = this.rc.borrow();
        let [fx, fy, fz] = crate::terrain::check_collisions([x, y, z], [dx, dy, dz], [sx, sy, sz], |block_pos, _axis| {
            terrain.block_at(block_pos).map(|data| data.is_solid(&terrain.solid)).unwrap_or(true)
        }, true);
        (fx, fy, fz)
    }

    fn visible_radius(lua, this, (x, y, z): (f64, f64, f64)) {
        let terrain = this.rc.borrow();
        for &idx in terrain.meshes.render_order.iter() {
            if terrain.meshes.get_by_idx(idx).mesh.is_none() {
                // This mesh is not visible
                let block = (terrain.meshes.sub_idx_to_pos(idx) << CHUNK_BITS) + Int3::splat(CHUNK_SIZE/2);
                let dx = block[0] as f64 - x;
                let dy = block[1] as f64 - y;
                let dz = block[2] as f64 - z;
                let delta = Vec3::new(dx as f32, dy as f32, dz as f32);
                let radius = delta.mag() - (CHUNK_SIZE as f32 / 2.) * 3f32.cbrt();
                return Ok(radius.max(0.));
            }
        }
        terrain.meshes.radius * CHUNK_SIZE as f32
    }

    fn draw(lua, this, (shader, uniforms, offset_uniform, params, x, y, z): (ShaderRef, UniformStorage, u32, LuaDrawParams, f64, f64, f64)) {
        let this = this.rc.borrow();
        let mut frame = this.state.frame.borrow_mut();
        //Rendering in this order has the nice property that closer chunks are rendered first,
        //making better use of the depth buffer.
        for &idx in this.meshes.render_order.iter() {
            if let Some(buf) = &this.meshes.get_by_idx(idx).mesh.as_ref().and_then(|mesh| mesh.buf.as_ref()) {
                let pos = this.meshes.sub_idx_to_pos(idx) << CHUNK_BITS;
                let offset = Vec3::new((pos.x as f64 - x) as f32, (pos.y as f64 - y) as f32, (pos.z as f64 - z) as f32);
                uniforms.vars.borrow_mut().get_mut(offset_uniform as usize).ok_or("offset uniform out of range").to_lua_err()?.1 = UniformVal::Vec3(offset.into());
                frame.draw(&buf.vertex, &buf.index, &shader.program, &uniforms, &params.params).unwrap();
            }
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
                        Err(LuaError::RuntimeError(format!("cannot hash type {}", val.type_name())))?;
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
            },
        )
        .unwrap();
}

pub(crate) fn open_generic_libs(state: &Arc<GlobalState>, lua: LuaContext) {
    modify_std_lib(state, lua);
    open_fs_lib(state, lua);
    open_vec3_lib(state, lua);
}

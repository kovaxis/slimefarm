//! Game about enslaving slimes in a voxel world.
//!
//! Nice smoothed approximation of x/tan(x) for the range [-2, 2]:
//!     1 - 0.212834*min(abs(x),2)^2 - 0.287166*min(abs(x),2)^3 + 0.134292*min(abs(x),2)^4
//!
//! The idea is that instead of using min(x, pi/2) to calculate an angle from a mouse distance,
//! x/f(x) can be used instead (where f(x) is the above approximation).

#![allow(unused_imports)]

use glium::uniforms::SamplerWrapFunction;

use crate::prelude::*;

#[macro_use]
pub mod prelude {
    pub(crate) use crate::{
        gen::GeneratorHandle,
        mesh::Mesh,
        terrain::{BlockData, BlockPos, Chunk, ChunkPos, ChunkStorage, CHUNK_SIZE},
        Buffer3d, LuaDrawParams, ShaderRef, SimpleVertex, State, UniformStorage,
    };
    pub use anyhow::{anyhow, bail, ensure, Context, Error, Result};
    pub use crossbeam::{
        atomic::AtomicCell,
        channel::{self, Receiver, Sender},
        sync::{Parker, Unparker},
    };
    pub use fxhash::{FxHashMap as HashMap, FxHashSet as HashSet};
    pub use glium::{
        glutin::{
            dpi::PhysicalSize,
            event::{DeviceEvent, Event, KeyboardInput, WindowEvent},
            event_loop::{ControlFlow, EventLoop},
            window::WindowBuilder,
            ContextBuilder,
        },
        implement_vertex,
        index::{PrimitiveType, RawIndexPackage},
        program,
        uniforms::{
            MagnifySamplerFilter, MinifySamplerFilter, SamplerBehavior, UniformValue, Uniforms,
        },
        vertex::RawVertexPackage,
        Display, DrawParameters, Frame, IndexBuffer, Program, Surface, Texture2d, VertexBuffer,
    };
    pub use glium_text_rusttype::{FontTexture, TextDisplay, TextSystem};
    pub use parking_lot::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
    pub use rand::{Rng, SeedableRng};
    pub use rand_xoshiro::Xoshiro128Plus as FastRng;
    pub use rlua::prelude::*;
    pub use serde::{Deserialize, Serialize};
    pub use std::{
        cell::{Cell, RefCell},
        cmp,
        collections::VecDeque,
        f32::consts as f32,
        f64::consts as f64,
        fs::{self, File},
        mem::{self, MaybeUninit as Uninit},
        ops, ptr,
        rc::Rc,
        sync::Arc,
        thread::{self, JoinHandle},
        time::{Duration, Instant},
    };
    pub use uv::{Mat2, Mat3, Mat4, Vec2, Vec3, Vec4};

    pub type VertIdx = u16;

    pub fn default<T>() -> T
    where
        T: Default,
    {
        T::default()
    }

    #[derive(Copy, Clone, Default, Debug)]
    pub struct Sortf32(pub f32);
    impl Ord for Sortf32 {
        fn cmp(&self, rhs: &Self) -> cmp::Ordering {
            match (self.0.is_nan(), rhs.0.is_nan()) {
                (false, false) => {
                    if self.0 < rhs.0 {
                        cmp::Ordering::Less
                    } else if self.0 > rhs.0 {
                        cmp::Ordering::Greater
                    } else {
                        cmp::Ordering::Equal
                    }
                }
                (false, true) => cmp::Ordering::Greater,
                (true, false) => cmp::Ordering::Less,
                (true, true) => cmp::Ordering::Equal,
            }
        }
    }
    impl PartialOrd for Sortf32 {
        fn partial_cmp(&self, rhs: &Self) -> Option<cmp::Ordering> {
            Some(self.cmp(rhs))
        }
    }
    impl PartialEq for Sortf32 {
        fn eq(&self, rhs: &Self) -> bool {
            self.cmp(rhs) == cmp::Ordering::Equal
        }
    }
    impl Eq for Sortf32 {}

    #[allow(unused_macros)]
    macro_rules! measure_time {
        (@start $now:ident $name:ident) => {
            let $name = $now;
        };
        (@end $now:ident $name:ident) => {{
            let elapsed: Duration = $now - $name;
            eprintln!("{}: {}ms", stringify!($name), (elapsed.as_micros() as f32/100.).round() / 10.);
        }};
        ($($method:ident $name:ident),*) => {
            let now = Instant::now();
            $(
                measure_time!(@$method now $name);
            )*
        };
    }

    macro_rules! lua_bail {
        ($err:expr, $($rest:tt)*) => {{
            return Err(format!($err, $($rest)*)).to_lua_err();
        }};
        ($err:expr) => {{
            return Err($err).to_lua_err();
        }};
    }

    macro_rules! lua_assert {
        ($cond:expr, $($err:tt)*) => {{
            if !$cond {
                lua_bail!($($err)*);
            }
        }};
    }

    macro_rules! lua_type {
        (@$m:ident fn $fn_name:ident ($lua:ident, $this:ident, $($args:tt)*) $fn_code:block $($rest:tt)*) => {{
            $m.add_method(stringify!($fn_name), |lua, this, $($args)*| {
                #[allow(unused_variables)]
                let $lua = lua;
                #[allow(unused_variables)]
                let $this = this;
                Ok($fn_code)
            });
            lua_type!(@$m $($rest)*);
        }};
        (@$m:ident mut fn $fn_name:ident ($lua:ident, $this:ident, $($args:tt)*) $fn_code:block $($rest:tt)*) => {{
            $m.add_method_mut(stringify!($fn_name), |lua, this, $($args)*| {
                #[allow(unused_variables)]
                let $lua = lua;
                #[allow(unused_variables)]
                let $this = this;
                Ok($fn_code)
            });
            lua_type!(@$m $($rest)*);
        }};
        (@$m:ident) => {};
        ($ty:ty, $($rest:tt)*) => {
            impl LuaUserData for $ty {
                fn add_methods<'lua, M: LuaUserDataMethods<'lua, Self>>(m: &mut M) {
                    lua_type!(@m $($rest)*);
                }
            }
        };
    }

    macro_rules! lua_func {
        ($lua:ident, $state:ident, fn($($fn_args:tt)*) $fn_code:block) => {{
            let state = AssertSync($state.clone());
            $lua.create_function(move |ctx, $($fn_args)*| {
                #[allow(unused_variables)]
                let $lua = ctx;
                #[allow(unused_variables)]
                let $state = &*state;
                Ok($fn_code)
            }).unwrap()
        }};
    }

    macro_rules! lua_lib {
        ($lua:ident, $state:ident, $( fn $fn_name:ident($($fn_args:tt)*) $fn_code:block )*) => {{
            let lib = $lua.create_table().unwrap();
            $(
                lib.set(stringify!($fn_name), lua_func!($lua, $state, fn($($fn_args)*) $fn_code)).unwrap();
            )*
            lib
        }};
    }

    pub trait ResultExt {
        type Inner;
        fn unwrap_lua(self) -> Self::Inner;
        fn expect_lua(self, msg: &str) -> Self::Inner;
    }
    impl<T> ResultExt for LuaResult<T> {
        type Inner = T;
        fn unwrap_lua(self) -> T {
            match self {
                Ok(t) => t,
                Err(err) => {
                    panic!("{}", err);
                }
            }
        }
        fn expect_lua(self, msg: &str) -> T {
            match self {
                Ok(t) => t,
                Err(err) => {
                    panic!("{}: {}", msg, err);
                }
            }
        }
    }

    #[derive(Copy, Clone, Debug, Default)]
    pub(crate) struct AssertSync<T>(pub T);
    unsafe impl<T> Send for AssertSync<T> {}
    unsafe impl<T> Sync for AssertSync<T> {}
    impl<T> ops::Deref for AssertSync<T> {
        type Target = T;
        fn deref(&self) -> &T {
            &self.0
        }
    }
    impl<T> ops::DerefMut for AssertSync<T> {
        fn deref_mut(&mut self) -> &mut T {
            &mut self.0
        }
    }
}

/// The pinnacle of stupid design: `ElementState`.
fn elem_state_to_bool(elem_state: glium::glutin::event::ElementState) -> bool {
    use glium::glutin::event::ElementState::*;
    match elem_state {
        Pressed => true,
        Released => false,
    }
}

/// Runs the game on the dedicated GPU in Nvidia systems.
#[cfg(target_os = "windows")]
#[no_mangle]
pub static NvOptimusEnablement: u32 = 1;

/// Runs the game on the dedicated GPU in AMD systems.
#[cfg(target_os = "windows")]
#[no_mangle]
pub static AmdPowerXpressRequestHighPerformance: u32 = 1;

mod chunkmesh;
mod gen;
mod mesh;
mod perlin;
mod terrain;

struct State {
    display: Display,
    text_sys: TextSystem,
    frame: RefCell<Frame>,
    base_time: Instant,
    sec_gl_ctx: Cell<Option<glium::glutin::RawContext<glium::glutin::NotCurrent>>>,
    _sec_win: glium::glutin::window::Window,
}
impl Drop for State {
    fn drop(&mut self) {
        self.frame.borrow_mut().set_finish().unwrap();
    }
}

#[derive(Clone)]
struct ShaderRef {
    program: AssertSync<Rc<Program>>,
}
impl LuaUserData for ShaderRef {}

#[derive(Copy, Clone, Debug)]
struct SimpleVertex {
    pos: [f32; 3],
    color: u32,
}
implement_vertex!(SimpleVertex, pos, color);

struct Buffer3d {
    vertex: VertexBuffer<SimpleVertex>,
    index: IndexBuffer<VertIdx>,
}

#[derive(Copy, Clone, Debug)]
struct TexturedVertex {
    pos: [f32; 2],
    tex: [f32; 2],
}
implement_vertex!(TexturedVertex, pos, tex);

struct Buffer2d {
    vertex: VertexBuffer<TexturedVertex>,
    index: IndexBuffer<VertIdx>,
}

enum AnyBuffer {
    Buf2d(Buffer2d),
    Buf3d(Buffer3d),
}

#[derive(Clone)]
struct BufferRef {
    rc: AssertSync<Rc<AnyBuffer>>,
}
impl LuaUserData for BufferRef {}

#[derive(Clone)]
struct TextureRef {
    tex: AssertSync<Rc<Texture2d>>,
    sampling: SamplerBehavior,
}
impl TextureRef {
    fn new(tex: Texture2d) -> Self {
        use glium::uniforms::{MagnifySamplerFilter, MinifySamplerFilter, SamplerWrapFunction};
        Self {
            tex: AssertSync(Rc::new(tex)),
            sampling: SamplerBehavior {
                minify_filter: MinifySamplerFilter::Nearest,
                magnify_filter: MagnifySamplerFilter::Nearest,
                wrap_function: (
                    SamplerWrapFunction::Repeat,
                    SamplerWrapFunction::Repeat,
                    SamplerWrapFunction::Repeat,
                ),
                depth_texture_comparison: None,
                max_anisotropy: 1,
            },
        }
    }
}
lua_type! {TextureRef,
    fn dimensions(lua, this, ()) {
        (this.tex.width(), this.tex.height())
    }

    mut fn set_min(lua, this, filter: LuaString) {
        use glium::uniforms::MinifySamplerFilter::*;
        this.sampling.minify_filter = match filter.as_bytes() {
            b"linear" => Linear,
            b"nearest" => Nearest,
            _ => lua_bail!("unknown minify filter '{}'", filter.to_str().unwrap_or_default())
        };
    }

    mut fn set_mag(lua, this, filter: LuaString) {
        use glium::uniforms::MagnifySamplerFilter::*;
        this.sampling.magnify_filter = match filter.as_bytes() {
            b"linear" => Linear,
            b"nearest" => Nearest,
            _ => lua_bail!("unknown magnify filter '{}'", filter.to_str().unwrap_or_default())
        };
    }

    mut fn set_wrap(lua, this, wrap: LuaString) {
        use glium::uniforms::SamplerWrapFunction::*;
        let func = match wrap.as_bytes() {
            b"repeat" => Repeat,
            b"mirror" => Mirror,
            b"clamp" => Clamp,
            _ => lua_bail!("unknown wrap function '{}'", wrap.to_str().unwrap_or_default())
        };
        this.sampling.wrap_function = (func, func, func);
    }
}

#[derive(Clone, Default)]
struct LuaDrawParams {
    params: AssertSync<DrawParameters<'static>>,
}
lua_type! {LuaDrawParams,
    mut fn set_depth(lua, this, (test, write): (LuaString, bool)) {
        use glium::draw_parameters::DepthTest::*;
        let test = match test.as_bytes() {
            b"ignore" => Ignore,
            b"overwrite" => Overwrite,
            b"if_equal" => IfEqual,
            b"if_not_equal" => IfNotEqual,
            b"if_more" => IfMore,
            b"if_more_or_equal" => IfMoreOrEqual,
            b"if_less" => IfLess,
            b"if_less_or_equal" => IfLessOrEqual,
            _ => lua_bail!("unknown depth test"),
        };
        this.params.depth.test = test;
        this.params.depth.write = write;
    }

    mut fn set_color_blend(lua, this, (func, src, dst): (LuaString, LuaString, LuaString)) {
        use glium::draw_parameters::{BlendingFunction::*, LinearBlendingFactor::{self, *}};
        fn map_factor(s: LuaString) -> LuaResult<LinearBlendingFactor> {
            Ok(match s.as_bytes() {
                b"zero" => Zero,
                b"one" => One,
                b"src_color" => SourceColor,
                b"one_minus_src_color" => OneMinusSourceColor,
                b"dst_color" => DestinationColor,
                b"one_minus_dst_color" => OneMinusDestinationColor,
                b"src_alpha" => SourceAlpha,
                b"src_alpha_saturate" => SourceAlphaSaturate,
                b"one_minus_src_alpha" => OneMinusSourceAlpha,
                b"dst_alpha" => DestinationAlpha,
                b"one_minus_dst_alpha" => OneMinusDestinationAlpha,
                b"constant_color" => ConstantColor,
                b"one_minus_constant_color" => OneMinusConstantColor,
                b"constant_alpha" => ConstantAlpha,
                b"one_minus_constant_alpha" => OneMinusConstantAlpha,
                _ => lua_bail!("unknown blending factor"),
            })
        }
        let source = map_factor(src)?;
        let destination = map_factor(dst)?;
        let func = match func.as_bytes() {
            b"replace" => AlwaysReplace,
            b"min" => Min,
            b"max" => Max,
            b"add" => Addition{source,destination},
            b"sub" => Subtraction{source,destination},
            b"reverse_sub" => ReverseSubtraction{source,destination},
            _ => lua_bail!("unknown depth test"),
        };
        this.params.blend.color = func;
    }

    mut fn set_alpha_blend(lua, this, (func, src, dst): (LuaString, LuaString, LuaString)) {
        use glium::draw_parameters::{BlendingFunction::*, LinearBlendingFactor::{self, *}};
        fn map_factor(s: LuaString) -> LuaResult<LinearBlendingFactor> {
            Ok(match s.as_bytes() {
                b"zero" => Zero,
                b"one" => One,
                b"src_color" => SourceColor,
                b"one_minus_src_color" => OneMinusSourceColor,
                b"dst_color" => DestinationColor,
                b"one_minus_dst_color" => OneMinusDestinationColor,
                b"src_alpha" => SourceAlpha,
                b"src_alpha_saturate" => SourceAlphaSaturate,
                b"one_minus_src_alpha" => OneMinusSourceAlpha,
                b"dst_alpha" => DestinationAlpha,
                b"one_minus_dst_alpha" => OneMinusDestinationAlpha,
                b"constant_color" => ConstantColor,
                b"one_minus_constant_color" => OneMinusConstantColor,
                b"constant_alpha" => ConstantAlpha,
                b"one_minus_constant_alpha" => OneMinusConstantAlpha,
                _ => lua_bail!("unknown blending factor"),
            })
        }
        let source = map_factor(src)?;
        let destination = map_factor(dst)?;
        let func = match func.as_bytes() {
            b"replace" => AlwaysReplace,
            b"min" => Min,
            b"max" => Max,
            b"add" => Addition{source,destination},
            b"sub" => Subtraction{source,destination},
            b"reverse_sub" => ReverseSubtraction{source,destination},
            _ => lua_bail!("unknown depth test"),
        };
        this.params.blend.alpha = func;
    }

    mut fn set_cull(lua, this, winding: LuaString) {
        use glium::draw_parameters::BackfaceCullingMode::*;
        let cull = match winding.as_bytes() {
            b"cw" => CullClockwise,
            b"none" => CullingDisabled,
            b"ccw" => CullCounterClockwise,
            _ => lua_bail!("unknown cull winding")
        };
        this.params.backface_culling = cull;
    }
}

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
        *top = *top * uv::projection::perspective_gl(fov, aspect, near, far);
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

enum UniformVal {
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Mat4([[f32; 4]; 4]),
    Texture2d(TextureRef, SamplerBehavior),
}
impl UniformVal {
    fn as_uniform(&self) -> UniformValue {
        match self {
            &Self::Float(v) => UniformValue::Float(v),
            &Self::Vec2(v) => UniformValue::Vec2(v),
            &Self::Vec3(v) => UniformValue::Vec3(v),
            &Self::Vec4(v) => UniformValue::Vec4(v),
            &Self::Mat4(v) => UniformValue::Mat4(v),
            Self::Texture2d(tex, sampling) => UniformValue::Texture2d(&tex.tex, Some(*sampling)),
        }
    }
}

#[derive(Clone)]
struct UniformStorage {
    vars: AssertSync<Rc<RefCell<Vec<(String, UniformVal)>>>>,
}
impl Uniforms for UniformStorage {
    fn visit_values<'a, F: FnMut(&str, UniformValue<'a>)>(&self, mut visit: F) {
        for (name, val) in self.vars.borrow().iter() {
            let as_uniform: UniformValue<'a> = unsafe {
                let as_uniform: UniformValue = val.as_uniform();
                mem::transmute(as_uniform)
            };
            visit(name, as_uniform);
        }
    }
}
lua_type! {UniformStorage,
    fn add(lua, this, name: String) {
        let mut vars = this.vars.0.borrow_mut();
        let idx = vars.len();
        vars.push((name, UniformVal::Float(0.)));
        idx
    }

    fn set_float(lua, this, (idx, val): (usize, f32)) {
        let mut vars = this.vars.borrow_mut();
        vars
            .get_mut(idx)
            .ok_or("index out of range")
            .to_lua_err()?
            .1 = UniformVal::Float(val);
    }
    fn set_vec2(lua, this, (idx, x, y): (usize, f32, f32)) {
        let mut vars = this.vars.borrow_mut();
        vars
            .get_mut(idx)
            .ok_or("index out of range")
            .to_lua_err()?
            .1 = UniformVal::Vec2([x, y]);
    }
    fn set_vec3(lua, this, (idx, x, y, z): (usize, f32, f32, f32)) {
        let mut vars = this.vars.borrow_mut();
        vars
            .get_mut(idx)
            .ok_or("index out of range")
            .to_lua_err()?
            .1 = UniformVal::Vec3([x, y, z]);
    }
    fn set_vec4(lua, this, (idx, x, y, z, w): (usize, f32, f32, f32, f32)) {
        let mut vars = this.vars.borrow_mut();
        vars
            .get_mut(idx)
            .ok_or("index out of range")
            .to_lua_err()?
            .1 = UniformVal::Vec4([x, y, z, w]);
    }

    fn set_matrix(lua, this, (idx, mat): (usize, MatrixStack)) {
        let (_, top) = *mat.stack.borrow();
        let mut vars = this.vars.borrow_mut();
        vars
            .get_mut(idx)
            .ok_or("index out of range")
            .to_lua_err()?
            .1 = UniformVal::Mat4(top.into());
    }

    fn set_texture(lua, this, (idx, tex): (usize, TextureRef)) {
        let mut vars = this.vars.borrow_mut();
        let sampling = tex.sampling;
        vars
            .get_mut(idx)
            .ok_or("index out of range")
            .to_lua_err()?
            .1 = UniformVal::Texture2d(tex, sampling);
    }
}

struct Font {
    state: Rc<State>,
    text: RefCell<TextDisplay<Rc<FontTexture>>>,
}
impl Font {
    fn new(state: &Rc<State>, font_data: &[u8], size: u32) -> Result<Font> {
        let state = state.clone();
        let tex = Rc::new(FontTexture::new(
            &state.display,
            font_data,
            size,
            (0..0x250).filter_map(|i| std::char::from_u32(i)),
        )?);
        let text = RefCell::new(TextDisplay::new(&state.text_sys, tex, ""));
        Ok(Font { state, text })
    }

    fn draw(&self, text: &str, mvp: Mat4, color: [f32; 4], draw_params: &DrawParameters) {
        let mut frame = self.state.frame.borrow_mut();
        let mut text_disp = self.text.borrow_mut();
        text_disp.set_text(text);
        glium_text_rusttype::draw_with_params(
            &text_disp,
            &self.state.text_sys,
            &mut *frame,
            mvp,
            (color[0], color[1], color[2], color[3]),
            SamplerBehavior {
                minify_filter: MinifySamplerFilter::Nearest,
                magnify_filter: MagnifySamplerFilter::Nearest,
                ..default()
            },
            &draw_params,
        )
        .unwrap();
    }
}

#[derive(Clone)]
struct FontRef {
    rc: AssertSync<Rc<Font>>,
}
lua_type! {FontRef,
    fn draw(lua, this, (text, mvp, draw_params, r, g, b, a): (LuaString, MatrixStack, LuaDrawParams, f32, f32, f32, Option<f32>)) {
        let (_, mvp) = &*mvp.stack.borrow();
        this.rc.draw(text.to_str()?, *mvp, [r, g, b, a.unwrap_or(1.)], &draw_params.params);
    }
}

fn modify_std_lib(state: &Rc<State>, lua: LuaContext) {
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
}
fn open_gfx_lib(state: &Rc<State>, lua: LuaContext) {
    lua.globals()
            .set(
                "gfx",
                lua_lib! {lua, state,
                    fn shader((vertex, fragment): (String, String)) {
                        let shader = program!{&state.display,
                            110 => {
                                vertex: &*vertex,
                                fragment: &*fragment,
                            }
                        }.to_lua_err()?;
                        ShaderRef{program: AssertSync(Rc::new(shader))}
                    }

                    fn buffer_3d((pos, color, indices): (Vec<f32>, Vec<f32>, Vec<VertIdx>)) {
                        lua_assert!(pos.len() % 3 == 0, "positions not multiple of 3");
                        lua_assert!(color.len() % 4 == 0, "colors not multiple of 4");
                        lua_assert!(pos.len() / 3 == color.len() / 4, "not the same amount of positions as colors");
                        let vertices = pos.chunks_exact(3).zip(color.chunks_exact(4)).map(|(pos, color)| {
                            let q = |f| (f*255.) as u8 as u32;
                            SimpleVertex {
                                pos: [pos[0], pos[1], pos[2]],
                                color: (q(color[0]) << 24) | (q(color[1]) << 16) | (q(color[2]) << 8) | q(color[3]),
                            }
                        }).collect::<Vec<_>>();
                        BufferRef {
                            rc: AssertSync(Rc::new(AnyBuffer::Buf3d(Buffer3d {
                                vertex: VertexBuffer::new(&state.display, &vertices[..]).unwrap(),
                                index: IndexBuffer::new(&state.display, PrimitiveType::TrianglesList, &indices[..]).unwrap(),
                            })))
                        }
                    }

                    fn buffer_2d((pos, tex, indices): (Vec<f32>, Vec<f32>, Vec<VertIdx>)) {
                        lua_assert!(pos.len() % 2 == 0, "positions not multiple of 2");
                        lua_assert!(tex.len() % 2 == 0, "texcoords not multiple of 4");
                        lua_assert!(pos.len() == tex.len(), "not the same amount of positions as texcoords");
                        let vertices = pos.chunks_exact(2).zip(tex.chunks_exact(2)).map(|(pos, tex)| {
                            TexturedVertex {pos: [pos[0], pos[1]], tex: [tex[0], tex[1]]}
                        }).collect::<Vec<_>>();
                        BufferRef {
                            rc: AssertSync(Rc::new(AnyBuffer::Buf2d(Buffer2d {
                                vertex: VertexBuffer::new(&state.display, &vertices[..]).unwrap(),
                                index: IndexBuffer::new(&state.display, PrimitiveType::TrianglesList, &indices[..]).unwrap(),
                            })))
                        }
                    }

                    fn texture(path: String) {
                        use glium::texture::RawImage2d;

                        let img = image::io::Reader::open(&path)
                            .with_context(|| format!("image file \"{}\" not found", path))
                            .to_lua_err()?
                            .decode()
                            .with_context(|| format!("image file \"{}\" invalid or corrupted", path))
                            .to_lua_err()?
                            .to_rgba8();
                        let (w, h) = img.dimensions();
                        let tex = Texture2d::new(
                            &state.display,
                            RawImage2d::from_raw_rgba(img.into_vec(), (w, h))
                        ).unwrap();
                        TextureRef::new(tex)
                    }

                    fn uniforms(()) {
                        UniformStorage{vars: AssertSync(Rc::new(RefCell::new(Vec::new())))}
                    }

                    fn draw_params(()) {
                        LuaDrawParams::default()
                    }

                    fn font((font_data, size): (LuaString, u32)) {
                        FontRef{
                            rc: AssertSync(Rc::new(Font::new(state, font_data.as_bytes(), size).to_lua_err()?)),
                        }
                    }

                    fn dimensions(()) {
                        state.frame.borrow().get_dimensions()
                    }

                    fn clear(()) {
                        state.frame.borrow_mut().clear_color_and_depth((0., 0., 0., 0.), 1.);
                    }

                    fn draw((buf, shader, uniforms, params): (BufferRef, ShaderRef, UniformStorage, LuaDrawParams)) {
                        let mut frame = state.frame.borrow_mut();
                        match &**buf.rc {
                            AnyBuffer::Buf2d(buf) => {
                                frame.draw(&buf.vertex, &buf.index, &shader.program, &uniforms, &params.params)
                            },
                            AnyBuffer::Buf3d(buf) => {
                                frame.draw(&buf.vertex, &buf.index, &shader.program, &uniforms, &params.params)
                            },
                        }.unwrap();
                    }

                    fn finish(()) {
                        state.frame.borrow_mut().set_finish().unwrap();
                        *state.frame.borrow_mut() = state.display.draw();
                    }

                    fn toggle_fullscreen(exclusive: bool) {
                        use glium::glutin::window::Fullscreen;
                        let win = state.display.gl_window();
                        let win = win.window();
                        if win.fullscreen().is_some() {
                            win.set_fullscreen(None);
                        }else{
                            if exclusive {
                                if let Some(mon) = win.current_monitor() {
                                    let mode = mon.video_modes().max_by_key(|mode| {
                                        (mode.bit_depth(), mode.size().width * mode.size().height, mode.refresh_rate())
                                    });
                                    if let Some(mode) = mode {
                                        win.set_fullscreen(Some(Fullscreen::Exclusive(mode)));
                                        return Ok(());
                                    }
                                }
                            }
                            win.set_fullscreen(Some(Fullscreen::Borderless(None)));
                        }
                    }
                },
            )
            .unwrap();
}
fn open_algebra_lib(state: &Rc<State>, lua: LuaContext) {
    lua.globals()
        .set(
            "algebra",
            lua_lib! {lua, state,
                fn matrix(()) {
                    MatrixStack::from(Mat4::identity())
                }
            },
        )
        .unwrap();
}
fn open_terrain_lib(state: &Rc<State>, lua: LuaContext) {
    lua.globals()
        .set(
            "terrain",
            lua_lib! {lua, state,
                fn new(cfg: LuaValue) {
                    let cfg = rlua_serde::from_value(cfg)?;
                    terrain::TerrainRef::new(state, cfg)
                }
            },
        )
        .unwrap();
}

fn main() {
    //Initialize window
    let evloop = EventLoop::new();
    let wb = WindowBuilder::new()
        .with_resizable(true)
        .with_title("Slime Farm")
        .with_visible(true);
    let cb = ContextBuilder::new().with_vsync(false);
    let display = Display::new(wb, cb, &evloop).expect("failed to initialize OpenGL");

    //Create secondary context for parallel uploading
    let sec_ctxwin = ContextBuilder::new()
        .with_shared_lists(&display.gl_window())
        .build_windowed(
            WindowBuilder::new()
                .with_inner_size(PhysicalSize::new(1, 1))
                .with_visible(false),
            &evloop,
        )
        .expect("failed to initialize secondary OpenGL context");
    let (sec_ctx, sec_win) = unsafe { sec_ctxwin.split() };

    //Pack it all up
    let state = Rc::new(State {
        frame: RefCell::new(display.draw()),
        text_sys: TextSystem::new(&display),
        display,
        sec_gl_ctx: Cell::new(Some(sec_ctx)),
        _sec_win: sec_win,
        base_time: Instant::now(),
    });

    //Load main.lua
    std::env::set_current_dir("lua").unwrap();
    let lua_main = fs::read("main.lua").expect("could not find main.lua");

    //Initialize lua environment
    let lua = Lua::new();
    let mut main_reg_key = None;
    lua.context(|lua| {
        modify_std_lib(&state, lua);
        open_gfx_lib(&state, lua);
        open_algebra_lib(&state, lua);
        open_terrain_lib(&state, lua);
        let main_chunk = lua
            .load(&lua_main)
            .set_name("main.lua")
            .unwrap()
            .into_function()
            .expect_lua("parsing main.lua");
        let main_co = lua.create_thread(main_chunk).unwrap();
        main_reg_key = Some(lua.create_registry_value(main_co).unwrap());
    });
    let main_reg_key = main_reg_key.unwrap();

    //Run game
    macro_rules! run_event {
        ($flow:ident, $($arg:tt)*) => {{
            let mut exit = false;
            lua.context(|lua| {
                let co: LuaThread = lua.registry_value(&main_reg_key).unwrap();
                exit = co.resume($($arg)*).unwrap_lua();
            });
            if exit {
                *$flow = ControlFlow::Exit;
            }
        }};
    }
    state.display.gl_window().window().set_cursor_visible(false);
    let _ = state.display.gl_window().window().set_cursor_grab(true);
    evloop.run(move |ev, _, flow| {
        *flow = ControlFlow::Poll;
        match ev {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            scancode, state, ..
                        },
                    ..
                } => run_event!(flow, ("key", scancode, elem_state_to_bool(state))),
                WindowEvent::MouseInput { button, state, .. } => {
                    use glium::glutin::event::MouseButton::*;
                    let button = match button {
                        Left => 0,
                        Right => 1,
                        Middle => 2,
                        Other(int) => int,
                    };
                    run_event!(flow, ("click", button, elem_state_to_bool(state)));
                }
                WindowEvent::CloseRequested => run_event!(flow, "quit"),
                WindowEvent::Focused(focus) => {
                    if focus {
                        state.display.gl_window().window().set_cursor_visible(false);
                        let _ = state.display.gl_window().window().set_cursor_grab(true);
                    }
                }
                _ => {}
            },
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                    run_event!(flow, ("mousemove", dx, dy))
                }
                _ => {}
            },
            Event::MainEventsCleared => run_event!(flow, "update"),
            _ => {}
        }
    });
}

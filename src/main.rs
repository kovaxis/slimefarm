#![allow(unused_imports)]

use anyhow::{anyhow, bail, ensure, Context, Error, Result};
use glium::{
    glutin::{
        event::{DeviceEvent, Event, KeyboardInput, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::WindowBuilder,
        ContextBuilder,
    },
    implement_vertex,
    index::PrimitiveType,
    program,
    uniforms::{UniformValue, Uniforms},
    Display, DrawParameters, Frame, IndexBuffer, Program, Surface, VertexBuffer,
};
use rlua::prelude::*;
use std::{
    cell::{Cell, RefCell},
    fs, mem, ops,
    rc::Rc,
};
use uv::{Mat4, Vec2, Vec3, Vec4};
pub fn default<T>() -> T
where
    T: Default,
{
    T::default()
}

#[derive(Copy, Clone, Debug, Default)]
struct AssertSync<T>(pub T);
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

/// The pinnacle of stupid design: `ElementState`.
fn elem_state_to_bool(elem_state: glium::glutin::event::ElementState) -> bool {
    use glium::glutin::event::ElementState::*;
    match elem_state {
        Pressed => true,
        Released => false,
    }
}

macro_rules! lua_assert {
    ($cond:expr, $err:expr) => {{
        if !$cond {
            return Err($err).to_lua_err();
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

macro_rules! lua_lib {
    ($lua:ident, $state:ident, $( fn $fn_name:ident($($fn_args:tt)*) $fn_code:block )*) => {{
        let lib = $lua.create_table().unwrap();
        $(
            let state = AssertSync($state.clone());
            lib.set(stringify!($fn_name), $lua.create_function(move |ctx, $($fn_args)*| {
                #[allow(unused_variables)]
                let $lua = ctx;
                #[allow(unused_variables)]
                let $state = &*state.0;
                Ok($fn_code)
            }).unwrap()).unwrap();
        )*
        lib
    }};
}

struct State {
    display: Display,
    frame: RefCell<Frame>,
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
    color: [u8; 4],
}
implement_vertex!(SimpleVertex, pos normalize(false), color normalize(true));

struct Buffer {
    vertex: VertexBuffer<SimpleVertex>,
    index: IndexBuffer<u16>,
}

#[derive(Clone)]
struct BufferRef {
    rc: AssertSync<Rc<Buffer>>,
}
impl LuaUserData for BufferRef {}

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
            _ => {
                *top = *top * Mat4::from_scale(x);
            }
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

    fn perspective(lua, this, (fov, aspect, near, far): (f32, f32, f32, f32)) {
        let (_, top) = &mut *this.stack.borrow_mut();
        *top = *top * uv::projection::perspective_gl(fov, aspect, near, far);
    }
}

#[derive(Clone)]
struct UniformStorage {
    vars: AssertSync<Rc<RefCell<Vec<(String, UniformValue<'static>)>>>>,
}
impl Uniforms for UniformStorage {
    fn visit_values<'a, F: FnMut(&str, UniformValue<'a>)>(&self, mut visit: F) {
        for (name, val) in self.vars.0.borrow().iter() {
            visit(name, *val);
        }
    }
}
lua_type! {UniformStorage,
    fn add(lua, this, name: String) {
        let mut vars = this.vars.0.borrow_mut();
        let idx = vars.len();
        vars.push((name, UniformValue::Float(0.)));
        idx
    }

    fn set_float(lua, this, (idx, val): (usize, f32)) {
        let mut vars = this.vars.0.borrow_mut();
        vars
            .get_mut(idx)
            .ok_or("index out of range")
            .to_lua_err()?
            .1 = UniformValue::Float(val);
    }
    fn set_vec2(lua, this, (idx, x, y): (usize, f32, f32)) {
        let mut vars = this.vars.0.borrow_mut();
        vars
            .get_mut(idx)
            .ok_or("index out of range")
            .to_lua_err()?
            .1 = UniformValue::Vec2([x, y]);
    }
    fn set_vec3(lua, this, (idx, x, y, z): (usize, f32, f32, f32)) {
        let mut vars = this.vars.0.borrow_mut();
        vars
            .get_mut(idx)
            .ok_or("index out of range")
            .to_lua_err()?
            .1 = UniformValue::Vec3([x, y, z]);
    }
    fn set_vec4(lua, this, (idx, x, y, z, w): (usize, f32, f32, f32, f32)) {
        let mut vars = this.vars.0.borrow_mut();
        vars
            .get_mut(idx)
            .ok_or("index out of range")
            .to_lua_err()?
            .1 = UniformValue::Vec4([x, y, z, w]);
    }

    fn set_matrix(lua, this, (idx, mat): (usize, MatrixStack)) {
        let (_, top) = *mat.stack.borrow();
        let mut vars = this.vars.0.borrow_mut();
        vars
            .get_mut(idx)
            .ok_or("index out of range")
            .to_lua_err()?
            .1 = UniformValue::Mat4(top.into());
    }
}

fn main() {
    let evloop = EventLoop::new();
    let wb = WindowBuilder::new()
        .with_resizable(true)
        .with_title("Slime Farm")
        .with_visible(true);
    let cb = ContextBuilder::new();
    let display = Display::new(wb, cb, &evloop).expect("failed to initialize OpenGL");
    let state = Rc::new(State {
        frame: RefCell::new(display.draw()),
        display,
    });

    std::env::set_current_dir("lua").unwrap();
    let lua_main = fs::read("main.lua").expect("failed to read main.lua");

    let lua = Lua::new();
    let mut main_reg_key = None;
    lua.context(|lua| {
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

                    fn buffer((pos, color, indices): (Vec<f32>, Vec<f32>, Vec<u16>)) {
                        lua_assert!(pos.len() % 3 == 0, "positions not multiple of 3");
                        lua_assert!(color.len() % 4 == 0, "colors not multiple of 4");
                        lua_assert!(pos.len() / 3 == color.len() / 4, "not the same amount of positions as colors");
                        let vertices = pos.chunks_exact(3).zip(color.chunks_exact(4)).map(|(pos, color)| {
                            let q = |f| (f*255.) as u8;
                            SimpleVertex {pos: [pos[0], pos[1], pos[2]], color: [q(color[0]), q(color[1]), q(color[2]), q(color[3])]}
                        }).collect::<Vec<_>>();
                        BufferRef {
                            rc: AssertSync(Rc::new(Buffer {
                                vertex: VertexBuffer::new(&state.display, &vertices[..]).unwrap(),
                                index: IndexBuffer::new(&state.display, PrimitiveType::TrianglesList, &indices[..]).unwrap(),
                            }))
                        }
                    }

                    fn uniforms(()) {
                        UniformStorage{vars: AssertSync(Rc::new(RefCell::new(Vec::new())))}
                    }

                    fn dimensions(()) {
                        state.frame.borrow().get_dimensions()
                    }

                    fn clear(()) {
                        state.frame.borrow_mut().clear_color_and_depth((0., 0., 0., 0.), 0.);
                    }

                    fn draw((buf, shader, uniforms): (BufferRef, ShaderRef, UniformStorage)) {
                        let draw_params = DrawParameters {
                            backface_culling: glium::draw_parameters::BackfaceCullingMode::CullClockwise,
                            ..default()
                        };
                        state.frame.borrow_mut().draw(&buf.rc.vertex, &buf.rc.index, &shader.program, &uniforms, &draw_params).unwrap();
                    }

                    fn finish(()) {
                        state.frame.borrow_mut().set_finish().unwrap();
                        *state.frame.borrow_mut() = state.display.draw();
                    }
                },
            )
            .unwrap();
        lua.globals().set("algebra", lua_lib!{lua, state,
            fn matrix(()) {
                MatrixStack::from(Mat4::identity())
            }
        }).unwrap();
        let main_chunk = lua.load(&lua_main).set_name("main.lua").unwrap().into_function().expect("parsing main.lua failed");
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
                exit = co.resume($($arg)*).expect("lua error");
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

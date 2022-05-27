//! Game about enslaving slimes in a voxel world.
//!
//! Nice smoothed approximation of x/tan(x) for the range [-2, 2]:
//!     1 - 0.212834*min(abs(x),2)^2 - 0.287166*min(abs(x),2)^3 + 0.134292*min(abs(x),2)^4
//!
//! The idea is that instead of using min(x, pi/2) to calculate an angle from a mouse distance,
//! x/f(x) can be used instead (where f(x) is the above approximation).

#![allow(unused_imports)]

use std::path::Path;

use glium::uniforms::SamplerWrapFunction;

use crate::prelude::*;

#[macro_use]
pub mod prelude {
    pub(crate) use crate::{
        gen::GeneratorHandle,
        mesh::Mesh,
        terrain::{ChunkStorage, Terrain},
        GlobalState, GpuBuffer, SimpleVertex, State, TexturedVertex, VoxelVertex,
    };
    pub use common::prelude::*;
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
        texture::{RawImage2d, RawTexturePackage, SrgbTexture2d, Texture2d},
        uniforms::{
            MagnifySamplerFilter, MinifySamplerFilter, SamplerBehavior, UniformValue, Uniforms,
        },
        vertex::RawVertexPackage,
        Display, DrawParameters, Frame, IndexBuffer, Program, Surface, VertexBuffer,
    };
    pub use glium_text_rusttype::{FontTexture, TextDisplay, TextSystem};
    pub use lazysort::{Sorted, SortedBy};

    pub type VertIdx = u32;

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
    macro_rules! time {
        (@print $name:ident $time:expr) => {{
            let time = $time;
            let (scale, rnd, unit) = {
                if time < 0.002 {
                    (1e-6, 1., "us")
                }else if time < 0.020 {
                    (1e-3, 0.01, "ms")
                }else if time < 0.200 {
                    (1e-3, 0.1, "ms")
                }else if time < 2. {
                    (1e-3, 1., "ms")
                }else if time < 20. {
                    (1., 0.01, "s")
                }else{
                    (1., 0.1, "s")
                }
            };
            println!(concat!(stringify!($name), ": {}{}"), (time / (scale * rnd)).round() * rnd, unit);
        }};
        (start $name:ident) => {
            let $name = Instant::now();
        };
        (store $name:ident $place:expr) => {{
            const AVERAGE_WEIGHT: f32 = 0.01;
            let time = $name.elapsed().as_secs_f32();
            let old_time = $place.load();
            let new_time = old_time + (time - old_time) * AVERAGE_WEIGHT;
            $place.store(new_time);
        }};
        (show $name:ident) => {{
            const AVERAGE_WEIGHT: f32 = 0.01;
            static AVG: Mutex<Option<(f32, Instant)>> = parking_lot::const_mutex(None);
            let mut avg = AVG.lock();
            let time = $name.elapsed().as_secs_f32();
            match &mut *avg {
                Some((avg, last)) => {
                    let new_avg = *avg + (time - *avg) * AVERAGE_WEIGHT;
                    *avg = new_avg;
                    if last.elapsed() > Duration::from_secs(2) {
                        time!(@print $name *avg);
                        *last = Instant::now();
                    }
                },
                None => *avg = Some((time, Instant::now())),
            }
        }};
        (show_now $name:ident) => {{
            time!(@print $name $name.elapsed().as_secs_f32());
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
mod lua;
mod magicavox;
mod mesh;
mod terrain;

/// State shared by all threads.
struct GlobalState {
    base_time: Instant,
}

/// Main thread state.
struct State {
    global: Arc<GlobalState>,
    display: Display,
    text_sys: TextSystem,
    frame: RefCell<Frame>,
    sec_gl_ctx: Cell<Option<glium::glutin::WindowedContext<glium::glutin::NotCurrent>>>,
}
impl Drop for State {
    fn drop(&mut self) {
        self.frame.borrow_mut().set_finish().unwrap();
    }
}

#[derive(Copy, Clone, Debug)]
struct VoxelVertex {
    pos: [u8; 4],
    cuv: [f32; 2],
    luv: [f32; 2],
}
implement_vertex!(VoxelVertex, pos normalize(false), cuv normalize(false), luv normalize(false));

#[derive(Copy, Clone, Debug)]
struct SimpleVertex {
    pos: [f32; 3],
    normal: [i8; 4],
    color: [u8; 4],
}
implement_vertex!(SimpleVertex, pos normalize(false), normal normalize(true), color normalize(true));

struct GpuBuffer<V: Copy, I: glium::index::Index = VertIdx> {
    vertex: VertexBuffer<V>,
    index: IndexBuffer<I>,
}

/*
struct DynBuffer3d {
    buf: Buffer3d<SimpleVertex>,
    vert_len: usize,
    idx_len: usize,
}
impl DynBuffer3d {
    fn new(state: &Rc<State>) -> Self {
        Self {
            buf: Buffer3d {
                vertex: VertexBuffer::empty_dynamic(&state.display, 16).unwrap(),
                index: IndexBuffer::empty_dynamic(&state.display, PrimitiveType::TrianglesList, 64)
                    .unwrap(),
            },
            vert_len: 0,
            idx_len: 0,
        }
    }

    fn write(&mut self, state: &Rc<State>, vert: &[SimpleVertex], idx: &[VertIdx]) {
        // Write vertex data
        if self.buf.vertex.len() < vert.len() {
            println!("reallocating vertex buffer");
            self.buf.vertex =
                VertexBuffer::empty_dynamic(&state.display, vert.len().next_power_of_two())
                    .unwrap();
        }
        self.buf.vertex.slice(..vert.len()).unwrap().write(&vert);
        self.vert_len = vert.len();

        // Write index data
        if self.buf.index.len() < idx.len() {
            println!("reallocating index buffer");
            self.buf.index = IndexBuffer::empty_dynamic(
                &state.display,
                PrimitiveType::TrianglesList,
                idx.len().next_power_of_two(),
            )
            .unwrap();
        }
        self.buf.index.slice(..idx.len()).unwrap().write(&idx);
        self.idx_len = idx.len();
    }

    fn bufs(
        &self,
    ) -> (
        glium::vertex::VertexBufferSlice<SimpleVertex>,
        glium::index::IndexBufferSlice<VertIdx>,
    ) {
        (
            self.buf.vertex.slice(0..self.vert_len).unwrap(),
            self.buf.index.slice(0..self.idx_len).unwrap(),
        )
    }
}
*/

#[derive(Copy, Clone, Debug)]
struct TexturedVertex {
    pos: [f32; 2],
    tex: [f32; 2],
}
implement_vertex!(TexturedVertex, pos, tex);

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    unsafe {
        common::staticinit::static_init();
    }

    //Initialize window
    let evloop = EventLoop::new();
    let wb = WindowBuilder::new()
        .with_resizable(true)
        .with_title("Slime Farm")
        .with_visible(true);
    let cb = ContextBuilder::new().with_vsync(false);
    let display = Display::new(wb, cb, &evloop).expect("failed to initialize OpenGL");

    //Pack it all up
    let shared = Arc::new(GlobalState {
        base_time: Instant::now(),
    });
    let state = Rc::new(State {
        global: shared,
        frame: RefCell::new(display.draw()),
        text_sys: TextSystem::new(&display),
        display,
        sec_gl_ctx: Cell::new(None),
    });

    //Load main.lua
    std::env::set_current_dir("lua").unwrap();
    let lua_main = fs::read("main.lua").expect("could not find main.lua");

    //Initialize lua environment
    let lua = Lua::new();
    let mut main_reg_key = None;
    lua.context(|lua| {
        crate::lua::open_generic_libs(&state.global, lua);
        crate::lua::open_system_lib(&state, lua);
        crate::lua::gfx::open_gfx_lib(&state, lua);
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
    evloop.run(move |ev, evloop, flow| {
        *flow = ControlFlow::Poll;
        {
            let sec_ctx = state.sec_gl_ctx.take();
            if sec_ctx.is_some() {
                state.sec_gl_ctx.set(sec_ctx);
            } else {
                //Create secondary context for parallel uploading
                let sec_ctx = ContextBuilder::new()
                    .with_shared_lists(&state.display.gl_window())
                    .build_windowed(
                        WindowBuilder::new()
                            .with_inner_size(PhysicalSize::new(1, 1))
                            .with_visible(false),
                        &evloop,
                    )
                    .expect("failed to initialize secondary OpenGL context");
                state.sec_gl_ctx.set(Some(sec_ctx));
            }
        }
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
                    run_event!(flow, ("focus", focus));
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

//! Rendering infrastructure.
//!
//! The logic thread runs separately from the main thread. However, rendering takes place using
//! information from the logic thread. Therefore, data must be sent from the logic thread to the
//! main thread. Additionally, the logic thread runs at a fixed 64 ticks per second, while the
//! main thread loops at any arbitrary FPS, depending on the monitor refresh rate, GPU capacity,
//! system load, etc...
//!
//! Therefore, the logic thread issues "render commands" into two separate "command queues".
//! The first command queue is the static queue, also referred to as "the queue". Commands that are
//! put in this queue once are executed exactly once some time when the main thread is free.
//!
//! The second queue is the "render snapshot". This queue is re-built every tick and re-run every
//! frame, and may include graphic transforms that depend on the specific time that the frame is
//! drawn, so as to interpolate between ticks.

/// A command to be run once.
pub enum GlobalCommand {
    Object(ObjectCommand),
    ToggleFullscreen { exclusive: bool },
}

/// Execute a function on a render object.
pub struct ObjectCommand {
    func: Box<dyn FnOnce(&Rc<State>, Uninit<RenderObj<u8>>) + Send>,
    obj: Uninit<RenderObj<u8>>,
}

/// From the logic thread side, this is an opaque reference type.
/// From the render thread side, this is a container for OpenGL objects.
pub struct RenderObj<T>
where
    T: 'static,
{
    inner: ManuallyDrop<Arc<RenderObjInner<T>>>,
}
impl<T> RenderObj<T> {
    /// Create a new render object using the given initialization command that runs in the main
    /// thread with access to the OpenGL context.
    pub(crate) fn new<F>(handle: Arc<RenderHandle>, init: F) -> Self
    where
        F: FnOnce(&Rc<State>) -> T + Send + 'static,
    {
        let obj = Self {
            inner: ManuallyDrop::new(Arc::new(RenderObjInner {
                handle,
                obj: RefCell::new(None),
            })),
        };
        obj.inner
            .handle
            .shared
            .queue_send
            .try_send(GlobalCommand::Object(ObjectCommand {
                func: Box::new(move |state, obj| {
                    let obj = unsafe { mem::transmute::<Uninit<RenderObj<u8>>, RenderObj<T>>(obj) };
                    *obj.inner.obj.borrow_mut() = Some(init(state));
                }),
                obj: unsafe { mem::transmute::<RenderObj<T>, Uninit<RenderObj<u8>>>(obj.clone()) },
            }))
            .expect("failed to queue render object initialization");
        obj
    }

    /// Carry out an action on this render object in the main thread with access to the OpenGL
    /// context.
    pub(crate) fn action<F>(&self, action: F)
    where
        F: FnOnce(&Rc<State>, &mut T) + Send + 'static,
    {
        self.inner
            .handle
            .shared
            .queue_send
            .try_send(GlobalCommand::Object(ObjectCommand {
                func: Box::new(move |state, obj| {
                    let obj = unsafe { mem::transmute::<Uninit<RenderObj<u8>>, RenderObj<T>>(obj) };
                    action(
                        state,
                        obj.inner
                            .obj
                            .borrow_mut()
                            .as_mut()
                            .expect("attempt to perform action on uninitialized render object"),
                    );
                }),
                obj: unsafe { mem::transmute::<RenderObj<T>, Uninit<RenderObj<u8>>>(self.clone()) },
            }))
            .expect("failed to queue render object command");
    }

    /// Get the render state handle associated with this object.
    ///
    /// unsafe !
    /// SAFETY: Must be called from the logic thread.
    pub unsafe fn handle(&self) -> &Arc<RenderHandle> {
        &self.inner.handle
    }

    /// Get a reference to the inner render object.
    ///
    /// Panics if the render object has not been initialized.
    ///
    /// SAFETY: Must be called from the render thread.
    pub unsafe fn inner(&self) -> core::cell::Ref<T> {
        std::cell::Ref::map(self.inner.obj.borrow(), |obj| {
            obj.as_ref()
                .expect("attempt to access uninitialized render object")
        })
    }

    /// Get a mutable reference to the inner render object.
    ///
    /// Panics if the render object has not been initialized.
    ///
    /// SAFETY: Must be called from the render thread.
    pub unsafe fn inner_mut(&self) -> core::cell::RefMut<T> {
        std::cell::RefMut::map(self.inner.obj.borrow_mut(), |obj| {
            obj.as_mut()
                .expect("attempt to access uninitialized render object")
        })
    }
}
// SAFETY: Because `T` is created/accessed/destroyed only from the render thread, `RenderObj`
// itself is `Send`.
// Additionally, the non-sync type `RenderHandle` is only accessed from the logic thread, so
// everything is ok.
unsafe impl<T> Send for RenderObj<T> where T: 'static {}
impl<T> Clone for RenderObj<T>
where
    T: 'static,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}
impl<T> Drop for RenderObj<T>
where
    T: 'static,
{
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) == 1 {
            // If this is the last reference, keep it alive and send it to be destroyed to the main thread
            let this = unsafe {
                Self {
                    inner: ManuallyDrop::new(ManuallyDrop::take(&mut self.inner)),
                }
            };
            let e = self
                .inner
                .handle
                .shared
                .queue_send
                .try_send(GlobalCommand::Object(ObjectCommand {
                    // zero-allocation box
                    func: Box::new(|_, obj| {
                        let mut obj =
                            unsafe { mem::transmute::<Uninit<RenderObj<u8>>, RenderObj<T>>(obj) };
                        unsafe {
                            ManuallyDrop::drop(&mut obj.inner);
                            mem::forget(obj);
                        }
                    }),
                    obj: unsafe { mem::transmute::<RenderObj<T>, Uninit<RenderObj<u8>>>(this) },
                }));
            if let Err(e) = e {
                eprintln!("leaking resources, could not send render object back to renderthread for cleanup: {}", e);
            }
        } else {
            unsafe {
                ManuallyDrop::drop(&mut self.inner);
            }
        }
    }
}

struct RenderObjInner<T>
where
    T: 'static,
{
    handle: Arc<RenderHandle>,
    obj: RefCell<Option<T>>,
}

#[derive(Default)]
pub struct RenderSnap {
    cmds: Vec<RenderCommand>,
}
impl RenderSnap {
    pub fn reset(&mut self) {
        self.cmds.clear();
    }
}

struct Shared {
    queue_send: Arc<Sender<GlobalCommand>>,
    new_snap: Mutex<(RenderSnap, bool)>,
    last_dims: AtomicCell<(u16, u16)>,
}

pub struct RenderState {
    state: Rc<State>,
    snap: RenderSnap,
    queue: Receiver<GlobalCommand>,
    shared: Arc<Shared>,
}
impl RenderState {
    pub fn new(state: Rc<State>) -> RenderState {
        let (queue_send, queue_recv) = channel::unbounded();
        Self {
            state,
            snap: default(),
            queue: queue_recv,
            shared: Arc::new(Shared {
                queue_send: Arc::new(queue_send),
                new_snap: default(),
                last_dims: AtomicCell::new((1, 1)),
            }),
        }
    }

    pub fn handle(&self) -> RenderHandle {
        RenderHandle {
            snap: default(),
            shared: self.shared.clone(),
        }
    }

    pub(crate) fn draw(&mut self) {
        // Start drawing on screen
        let mut frame = self.state.display.draw();

        // Set frame dimensions
        let (w, h) = frame.get_dimensions();
        self.shared.last_dims.store((w as u16, h as u16));

        // Check for new rendersnaps
        {
            let mut new = self.shared.new_snap.lock();
            if new.1 {
                mem::swap(&mut self.snap, &mut new.0);
                new.1 = false;
            }
        }

        // Process queued object commands
        while let Ok(cmd) = self.queue.try_recv() {
            self.process_global_command(cmd);
        }

        // Process render snapshot
        let cmds = mem::take(&mut self.snap.cmds);
        for cmd in cmds.iter() {
            self.process_command(&mut frame, &cmd);
        }
        self.snap.cmds = cmds;

        // Swap buffers
        frame.finish().unwrap();
    }

    pub fn process_global_command(&mut self, cmd: GlobalCommand) {
        match cmd {
            GlobalCommand::Object(cmd) => {
                (cmd.func)(&self.state, cmd.obj);
            }
            GlobalCommand::ToggleFullscreen { exclusive } => {
                self.toggle_fullscreen(exclusive);
            }
        }
    }

    pub fn process_command(&mut self, frame: &mut Frame, cmd: &RenderCommand) {
        match cmd {
            RenderCommand::Clear {
                color,
                depth,
                stencil,
            } => match (*color, *depth, *stencil) {
                (Some((r, g, b, a)), None, None) => frame.clear_color(r, g, b, a),
                (Some(c), Some(d), None) => frame.clear_color_and_depth(c, d),
                (Some(c), None, Some(s)) => frame.clear_color_and_stencil(c, s),
                (Some(c), Some(d), Some(s)) => frame.clear_all(c, d, s),
                (None, Some(d), None) => frame.clear_depth(d),
                (None, None, Some(s)) => frame.clear_stencil(s),
                (None, Some(d), Some(s)) => frame.clear_depth_and_stencil(d, s),
                (None, None, None) => {}
            },
            _ => todo!(),
        }
    }

    fn toggle_fullscreen(&self, exclusive: bool) {
        use glium::glutin::window::Fullscreen;
        let win = self.state.display.gl_window();
        let win = win.window();
        if win.fullscreen().is_some() {
            win.set_fullscreen(None);
        } else {
            if exclusive {
                if let Some(mon) = win.current_monitor() {
                    let mode = mon.video_modes().max_by_key(|mode| {
                        (
                            mode.bit_depth(),
                            mode.size().width * mode.size().height,
                            mode.refresh_rate(),
                        )
                    });
                    if let Some(mode) = mode {
                        win.set_fullscreen(Some(Fullscreen::Exclusive(mode)));
                        return;
                    }
                }
            }
            win.set_fullscreen(Some(Fullscreen::Borderless(None)));
        }
    }
}
impl Drop for RenderState {
    fn drop(&mut self) {
        // Dropping in-flight object commands will leak memory, so run them all before dropping
        for cmd in self.queue.try_iter() {
            if let GlobalCommand::Object(cmd) = cmd {
                (cmd.func)(&self.state, cmd.obj);
            }
        }
    }
}

pub enum RenderCommand {
    Clear {
        color: Option<(f32, f32, f32, f32)>,
        depth: Option<f32>,
        stencil: Option<i32>,
    },
    Matrix(MatrixStack, MatrixOp),
    Uniform(LuaUniforms, usize, StaticUniform),
    Text {
        text: String,
        mvp: MatrixStack,
        params: LuaDrawParams,
        color: [f32; 4],
    },
    Draw {
        buf: LuaBuffer,
        shader: LuaShader,
        uniforms: LuaUniforms,
        params: LuaDrawParams,
    },
    Chunk {
        shader: LuaShader,
        params: LuaDrawParams,
        uniforms: LuaUniforms,
        offset: Vec3,
        delta: Vec3,
        chunk: RenderObj<GpuChunk>,
        color_sampling: SamplerBehavior,
        light_sampling: SamplerBehavior,
    },
}

pub enum MatrixOp {
    Reset,
    Copy(MatrixStack),
    Identity,
    MulRight(MatrixStack),
    MulLeft(MatrixStack),
    Push,
    Pop,
    Translate {
        x: Vec3,
        dx: Vec3,
    },
    Scale {
        x: Vec3,
        dx: Vec3,
    },
    RotX(f32, f32),
    RotY(f32, f32),
    RotZ(f32, f32),
    Rotate {
        a: f32,
        da: f32,
        axis: Vec3,
    },
    Invert,
    Perspective {
        fov: f32,
        aspect: f32,
        near: f32,
        far: f32,
    },
    Orthographic {
        xleft: f32,
        xright: f32,
        ydown: f32,
        yup: f32,
        znear: f32,
        zfar: f32,
    },
}

pub struct RenderHandle {
    pub snap: RefCell<RenderSnap>,
    shared: Arc<Shared>,
}
impl RenderHandle {
    /// Create a render object using the given initialization function.
    pub(crate) fn new_obj<T, F>(self: &Arc<Self>, init: F) -> RenderObj<T>
    where
        T: 'static,
        F: FnOnce(&Rc<State>) -> T + Send + 'static,
    {
        RenderObj::new(self.clone(), init)
    }

    /// Add a new render command to the snapqueue.
    pub fn push(&self, cmd: RenderCommand) {
        self.snap.borrow_mut().cmds.push(cmd);
    }

    /// Transfer the current rendersnap into the render thread.
    /// Clears the current rendersnap.
    pub fn finish(&self) {
        let mut snap = self.snap.borrow_mut();
        {
            let mut inner = self.shared.new_snap.lock();
            mem::swap(&mut *snap, &mut inner.0);
            inner.1 = true;
        }
        snap.reset();
    }

    pub fn dimensions(&self) -> (u32, u32) {
        let (w, h) = self.shared.last_dims.load();
        (w as u32, h as u32)
    }

    /// Add a command to the global command queue.
    pub fn global_cmd(&self, cmd: GlobalCommand) {
        let _ = self.shared.queue_send.try_send(cmd);
    }
}

use std::mem::ManuallyDrop;

use crate::{
    lua::gfx::{LuaBuffer, LuaDrawParams, LuaShader, LuaUniforms, MatrixStack, StaticUniform},
    prelude::*,
    terrain::GpuChunk,
};

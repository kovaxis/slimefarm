use crate::prelude::*;

type ChunkMask = [ChunkSizedInt; (CHUNK_SIZE * CHUNK_SIZE) as usize];

type Actor = Rc<dyn Fn(&ActorArg, &mut BufState, &mut ChunkBox)>;

struct Action {
    /// Bounding box of the action, in virtual buffer coordinates.
    bbox: [Int3; 2],
    /// Actor function.
    actor: Actor,
}

#[derive(Clone)]
struct BufState {
    mask: ChunkMask,
    bbox: [Int3; 2],
}

struct ActorArg {
    /// Origin of the virtual buffer space, in local chunk coordinates.
    origin: Int3,
    /// Bounding box of the current action, in local chunk coordinates.
    bbox: [Int3; 2],
    /// Chunk coordinates of the current chunk.
    chunkpos: ChunkPos,
}

/// Stores actions in virtual space for later repeatable application in actual chunks.
///
/// Each action consists of two parts: a *bounding box* and an *actor*.
///
/// The bounding box indicates where on the buffer virtual space does the action affect.
/// This is used as an optimization, to not process every block with each action.
///
/// The actor is a function that works **not in virtual space**.
/// Each time the actor function is called it is in the context of a chunk, and it works in local
/// chunk space.
/// It receives a few parameters:
///     - Action parameters, encoded in `ActorArg`.
///     - A mutable shared state, used to share data between actions.
///     - A reference to the data for the current chunk.
pub struct ActionBuf {
    origin: Int3,
    state: RefCell<BufState>,

    /// Storse an action for later application in a chunk.
    /// The action consists of two parts: a bounding box and a chunk application function.
    /// The bounding box is composed of a minimum and a maximum coordinate, in actionbuf-relative
    /// coordinates.
    /// The minimum coordinate is inclusive, the maximum coordinate is exclusive.
    ///
    /// The application function is called for every chunk that is in contact with the bounding box,
    /// and is called with three arguments: position, bounding box and chunk box.
    /// The position is the position of the chunk minimum corner _relative to the actionbuf origin_.
    /// The bounding box is basically the same as the original bounding box, but clipped to fit
    /// within the chunk. It is also relative to the actionbuf origin.
    /// The chunk box is just a mutable reference the data for the chunk in question.
    actions: Vec<Action>,
}
impl fmt::Debug for ActionBuf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ActionBuf")
    }
}
impl ActionBuf {
    pub fn new(origin: Int3) -> Self {
        Self {
            origin,
            state: BufState {
                mask: [0; (CHUNK_SIZE * CHUNK_SIZE) as usize],
                bbox: [Int3::zero(); 2],
            }
            .into(),
            actions: vec![],
        }
    }

    pub fn transfer(&self, chunkpos: ChunkPos, chunk: &mut ChunkBox) {
        let chunk_mn = (chunkpos.coords << CHUNK_BITS) - self.origin;
        let chunk_mx = chunk_mn + [CHUNK_SIZE; 3];
        let mut state = self.state.borrow_mut();
        let mut arg = ActorArg {
            origin: -chunk_mn,
            bbox: [Int3::zero(); 2],
            chunkpos,
        };
        for action in &self.actions {
            let [mn, mx] = action.bbox;
            if mx.x > chunk_mn.x
                && mx.y > chunk_mn.y
                && mx.z > chunk_mn.z
                && mn.x < chunk_mx.x
                && mn.y < chunk_mx.y
                && mn.z < chunk_mx.z
            {
                let mn = mn - chunk_mn;
                let mx = mx - chunk_mn;
                arg.bbox = [mn.max(Int3::zero()), mx.min(Int3::splat(CHUNK_SIZE))];
                (action.actor)(&arg, &mut state, chunk);
            }
        }
    }

    pub fn take(&mut self) -> ActionBuf {
        Self {
            origin: self.origin,
            state: self.state.clone(),
            actions: mem::take(&mut self.actions),
        }
    }
}

trait InsertAction {
    type Args;
    fn insert(args: Self::Args, put: &mut ActionBuf) -> LuaResult<()>;
}

trait BuildPainter {
    type Args;
    fn build(args: Self::Args) -> LuaResult<PainterRef>;
}

macro_rules! action {
    {
        fn $name:ident(($($argpat:pat),*) : $argty:ty) -> $bounds1:ident { $($code1:tt)* }
        fn apply($arg:pat, $state:pat, $chunk:pat) $code2:block
    } => {
        #[allow(non_camel_case_types)]
        pub struct $name;
        impl InsertAction for $name {
            type Args = $argty;
            #[allow(unused_parens)]
            fn insert(($($argpat),*): $argty, put: &mut ActionBuf) -> LuaResult<()> {
                let $bounds1;
                $($code1)*
                put.actions.push(Action {
                    bbox: $bounds1,
                    actor: Rc::new(move |$arg: &ActorArg, $state: &mut BufState, $chunk: &mut ChunkBox| $code2),
                });
                put.state.borrow_mut().bbox = $bounds1;
                Ok(())
            }
        }
    };
    {
        fn $name:ident(($($argpat:pat),*) : $argty:ty) { $($code1:tt)* }
        fn place() $code2:block
        $(
            fn $act:ident ($arg:pat, $state:pat, $chunk:pat, $($argpat3:tt)*) $(-> $setrange:ident)? $code3:block
        )*
    } => {
        #[allow(non_camel_case_types)]
        pub struct $name;
        impl InsertAction for $name {
            type Args = $argty;
            fn insert(($($argpat),*): $argty, put: &mut ActionBuf) -> LuaResult<()> {
                $($code1)*
                let tmp_put = Cell::new(mem::take(&mut put.actions));
                let mut setrange = 0;
                $(
                    #[allow(unused_mut)]
                    let mut $act = |bounds: [Int3; 2], $($argpat3)*| {
                        let mut list = tmp_put.take();
                        list.push(Action {
                            bbox: bounds,
                            actor: Rc::new(move |$arg: &ActorArg, $state: &mut BufState, $chunk: &mut ChunkBox| $code3),
                        });
                        tmp_put.replace(list);
                        $(
                            action!(@match_setrange $setrange);
                            put.state.borrow_mut().bbox = bounds;
                            setrange += 1;
                        )?
                    };
                )*
                $code2;
                put.actions = tmp_put.take();
                debug_assert!(setrange == 1, "bounding box must be set exactly once");
                Ok(())
            }
        }
    };
    {
        fn $name:ident(($($argpat:pat),*) : $argty:ty) { $($code1:tt)* }
        fn paint($arg:pat, $state:pat, $chunk:pat) $code2:block
    } => {
        #[allow(non_camel_case_types)]
        pub struct $name;
        impl BuildPainter for $name {
            type Args = $argty;
            #[allow(unused_parens)]
            fn build(($($argpat),*): $argty) -> LuaResult<PainterRef> {
                $($code1)*
                let painter = PainterRef {
                    actor: Rc::new(move |$arg: &ActorArg, $state: &mut BufState, $chunk: &mut ChunkBox| $code2),
                };
                Ok(painter)
            }
        }
    };
    (@match_setrange setrange) => {};
}

#[derive(Clone)]
struct PainterRef {
    actor: Actor,
}
lua_type! {PainterRef, lua, this,

}

mod actions;

macro_rules! actionbuf_lua {
    {
        shapers {
            $($sname:ident,)*
        }

        painters {
            $($pname:ident,)*
        }
    } => {
        lua_type! {ActionBuf, lua, this,
            mut fn reset((x, y, z): (i32, i32, i32)) {
                this.origin = [x, y, z].into();
                this.actions.clear();
            }

            fn transfer((x, y, z, w, chunk): (i32, i32, i32, u32, LuaAnyUserData)) {
                let chunkpos = ChunkPos {
                    coords: [x, y, z].into(),
                    dim: w,
                };
                let mut chunk = chunk.borrow_mut::<LuaChunkBox>()?;
                this.transfer(chunkpos, &mut chunk.chunk);
            }

            mut fn paint(painter: PainterRef) {
                this.actions.push(Action {
                    bbox: this.state.borrow().bbox,
                    actor: painter.actor.clone(),
                })
            }

            $(
                mut fn $sname(args: <actions::$sname as InsertAction>::Args) {
                    actions::$sname::insert(args, this)?;
                }
            )*
        }

        pub(crate) fn painter_constructors(lua: LuaContext) -> LuaResult<LuaTable> {
            let _state = ();
            let paint = lua_lib! {lua, _state,
                $(
                    fn $pname(args: <actions::$pname as BuildPainter>::Args) {
                        actions::$pname::build(args)?
                    }
                )*
            };
            Ok(paint)
        }
    };
}

actionbuf_lua! {
    shapers {
        portal,
        cube,
        sphere,
        oval,
        cylinder,
        blobs,
        cloud,
    }

    painters {
        solid,
        gradient,
        noisy,
    }
}

use crate::prelude::*;

type Painter = ([Int3; 2], Box<dyn Fn(Int3, [Int3; 2], &mut ChunkBox)>);

pub struct ActionBuf {
    origin: Int3,
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
    actions: Vec<Painter>,
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
            actions: vec![],
        }
    }

    pub fn transfer(&self, chunkpos: Int3, chunk: &mut ChunkBox) {
        let chunk_mn = (chunkpos << CHUNK_BITS) - self.origin;
        let chunk_mx = chunk_mn + [CHUNK_SIZE; 3];
        for &([mn, mx], ref action) in &self.actions {
            if mx.x > chunk_mn.x
                && mx.y > chunk_mn.y
                && mx.z > chunk_mn.z
                && mn.x < chunk_mx.x
                && mn.y < chunk_mx.y
                && mn.z < chunk_mx.z
            {
                let bbox = [mn.max(chunk_mn), mx.min(chunk_mx)];
                action(chunk_mn, bbox, chunk);
            }
        }
    }

    pub fn take(&mut self) -> ActionBuf {
        Self {
            origin: self.origin,
            actions: mem::take(&mut self.actions),
        }
    }
}

trait Action {
    type Args;
    fn make(args: Self::Args, put: &mut Vec<Painter>) -> LuaResult<()>;
}

macro_rules! action {
    {
        fn $name:ident(($($argpat:pat),*) : $argty:ty) -> $bounds1:ident { $($code1:tt)* }
        fn apply($pos:pat, $bounds2:pat, $chunk:pat) $code2:block
    } => {
        #[allow(non_camel_case_types)]
        pub struct $name;
        impl Action for $name {
            type Args = $argty;
            fn make(($($argpat),*): $argty, put: &mut Vec<Painter>) -> LuaResult<()> {
                let $bounds1;
                $($code1)*
                put.push(($bounds1, Box::new(move |pos: Int3, bounds: [Int3; 2], chunk: &mut ChunkBox| {
                    let $pos = pos;
                    let $bounds2 = bounds;
                    let $chunk = chunk;
                    $code2
                })));
                Ok(())
            }
        }
    };
    {
        fn $name:ident(($($argpat:pat),*) : $argty:ty) { $($code1:tt)* }
        fn place() $code2:block
        $(
            fn $act:ident ($pos:pat, $bounds3:pat, $chunk:pat, $($argpat3:tt)*) $code3:block
        )*
    } => {
        #[allow(non_camel_case_types)]
        pub struct $name;
        impl Action for $name {
            type Args = $argty;
            fn make(($($argpat),*): $argty, put: &mut Vec<Painter>) -> LuaResult<()> {
                $($code1)*
                let tmp_put = Cell::new(mem::take(put));
                $(
                    let $act = |bounds: [Int3; 2], $($argpat3)*| {
                        let mut put = tmp_put.take();
                        put.push((bounds, Box::new(move |pos: Int3, bounds: [Int3; 2], chunk: &mut ChunkBox| {
                            let $pos = pos;
                            let $bounds3 = bounds;
                            let $chunk = chunk;
                            $code3
                        })));
                        tmp_put.replace(put);
                    };
                )*
                $code2;
                *put = tmp_put.take();
                Ok(())
            }
        }
    };
}

mod actions;

macro_rules! actionbuf_lua {
    ($($name:ident,)*) => {
        lua_type! {ActionBuf, lua, this,
            mut fn reset((x, y, z): (i32, i32, i32)) {
                this.origin = [x, y, z].into();
                this.actions.clear();
            }

            fn transfer((x, y, z, chunk): (i32, i32, i32, LuaAnyUserData)) {
                let mut chunk = chunk.borrow_mut::<LuaChunkBox>()?;
                this.transfer([x, y, z].into(), &mut chunk.chunk);
            }

            $(
                mut fn $name(args: <actions::$name as Action>::Args) {
                    actions::$name::make(args, &mut this.actions)?;
                }
            )*
        }
    };
}

actionbuf_lua! {
    portal,
    entity,
    cube,
    sphere,
    ellipsoid,
    cylinder,
    blobs,
    cloud,
    rock,
}

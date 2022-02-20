use crate::prelude::*;

pub struct ActionBuf {
    origin: Int3,
    actions: Vec<(
        [Int3; 2],
        Box<dyn Fn(Int3, [Int3; 2], ChunkBox) -> ChunkBox + Send>,
    )>,
}
impl ActionBuf {
    pub fn new(origin: Int3) -> Self {
        Self {
            origin,
            actions: vec![],
        }
    }

    /// Apply this action onto a chunk.
    /// The coordinates given are the block coordinates of the chunk floor in action-local
    /// coordinates.
    /// The bounding box given is in action-local coordinates, and is within the chunk, otherwise
    /// `apply` will not be called.
    pub fn act(
        &mut self,
        action: (
            [Int3; 2],
            Box<dyn Fn(Int3, [Int3; 2], ChunkBox) -> ChunkBox + Send>,
        ),
    ) {
        self.actions.push(action);
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
                let mut c = mem::replace(chunk, ChunkBox::new_homogeneous(BlockData { data: 0 }));
                c = action(chunk_mn, bbox, c);
                *chunk = c;
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
    fn make(
        args: Self::Args,
    ) -> (
        [Int3; 2],
        Box<dyn Fn(Int3, [Int3; 2], ChunkBox) -> ChunkBox + Send>,
    );
}

macro_rules! action {
    {
        fn $name:ident($argpat:pat, $argty:ty) -> $bounds1:ident { $($code1:tt)* }
        fn apply($pos:pat, $bounds2:pat, $chunk:pat) $code2:block
    } => {
        #[allow(non_camel_case_types)]
        pub struct $name;
        impl Action for $name {
            type Args = $argty;
            fn make($argpat: $argty) -> ([Int3; 2], Box<dyn Fn(Int3, [Int3; 2], ChunkBox) -> ChunkBox + Send>) {
                let $bounds1;
                $($code1)*
                ($bounds1, Box::new(move |pos: Int3, bounds: [Int3; 2], mut chunk: ChunkBox| {
                    let $pos = pos;
                    let $bounds2 = bounds;
                    let $chunk = &mut chunk;
                    $code2;
                    chunk
                }))
            }
        }
    };
}

mod actions;

macro_rules! actionbuf_lua {
    ($($name:ident,)*) => {
        lua_type! {ActionBuf,
            mut fn reset(lua, this, (x, y, z): (i32, i32, i32)) {
                this.origin = [x, y, z].into();
                this.actions.clear();
            }

            $(
                mut fn $name(lua, this, args: <actions::$name as Action>::Args) {
                    this.act(actions::$name::make(args));
                }
            )*
        }
    };
}

actionbuf_lua! {
    cube,
    sphere,
}

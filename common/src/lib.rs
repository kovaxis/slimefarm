#![allow(unused_imports)]

extern crate alloc;

use crate::prelude::*;

pub mod prelude {
    pub use crate::{
        arena::{Box as ArenaBox, BoxUninit as ArenaBoxUninit},
        ivec::{Int2, Int3},
        slotmap::{SlotId, SlotMap},
        terrain::{
            BlockData, BlockPos, BlockTexture, BlockTextures, ChunkArc, ChunkBox, ChunkData,
            ChunkPos, ChunkRef, LoafBox, PortalData, SolidTable, CHUNK_BITS, CHUNK_MASK,
            CHUNK_SIZE,
        },
    };
    pub use anyhow::{anyhow, bail, ensure, Context, Error, Result};
    pub use crossbeam::{
        atomic::AtomicCell,
        channel::{self, Receiver, Sender},
        sync::{Parker, Unparker},
    };
    pub use fxhash::{FxHashMap as HashMap, FxHashSet as HashSet};
    pub use parking_lot::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
    pub use rand::{Rng, SeedableRng};
    pub use rand_xoshiro::Xoshiro128Plus as FastRng;
    pub use rlua::prelude::*;
    pub use serde::{Deserialize, Serialize};
    pub use serde_derive::{Deserialize, Serialize};
    pub use std::{
        any::Any,
        cell::{Cell, RefCell},
        cmp,
        collections::VecDeque,
        convert::{TryFrom, TryInto},
        error::Error as StdError,
        f32::consts as f32,
        f64::consts as f64,
        fmt::{self, Write as _},
        fs::{self, File},
        marker::PhantomData,
        mem::{self, MaybeUninit as Uninit},
        ops,
        ptr::{self, NonNull},
        rc::Rc,
        sync::Arc,
        thread::{self, JoinHandle},
        time::{Duration, Instant},
    };
    pub use uv::{Bivec2, Bivec3, Lerp, Mat2, Mat3, Mat4, Rotor2, Rotor3, Vec2, Vec3, Vec4};

    /// Unsafe as fuck, but whatever.
    #[derive(Copy, Clone, Debug, Default)]
    pub struct AssertSync<T>(pub T);
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

    pub fn default<T>() -> T
    where
        T: Default,
    {
        T::default()
    }

    /// Stupid workaround for serde.
    #[inline]
    pub fn default_true() -> bool {
        true
    }
}

#[macro_use]
pub mod arena;
pub mod actionbuf;
pub mod blockbuf;
pub mod ivec;
pub mod lua;
pub mod noise2d;
pub mod noise3d;
pub mod slotmap;
pub mod spread2d;
pub mod terrain;
pub mod worldgen;

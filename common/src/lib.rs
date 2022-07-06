#![allow(unused_imports)]

extern crate alloc;

use crate::prelude::*;

#[macro_use]
pub mod prelude {
    pub use crate::{
        arena::{Box as ArenaBox, BoxUninit as ArenaBoxUninit},
        arr, bit_array,
        ivec::{Int2, Int3},
        lua::LuaBytes,
        slotmap::{SlotId, SlotMap},
        terrain::{
            BlockData, BlockPos, BlockStyle, BlockTexture, ChunkArc, ChunkBox, ChunkData, ChunkPos,
            ChunkRef, ChunkSizedInt, Int4, LightingConf, LoafBox, PortalData, StyleTable,
            WorldInfo, WorldPos, CHUNK_BITS, CHUNK_MASK, CHUNK_SIZE,
        },
    };
    pub use anyhow::{anyhow, bail, ensure, Context, Error, Result};
    pub use crossbeam::{
        atomic::AtomicCell,
        channel::{self, Receiver, Sender},
        sync::{Parker, Unparker},
    };
    pub use parking_lot::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
    pub use rand::{Rng, SeedableRng};
    pub use rand_xoshiro::Xoshiro128Plus as FastRng;
    pub use rlua::prelude::*;
    pub use serde::{Deserialize, Serialize};
    pub use std::{
        any::Any,
        borrow::Cow,
        cell::{Cell, RefCell},
        cmp,
        collections::VecDeque,
        convert::{TryFrom, TryInto},
        error::Error as StdError,
        f32::consts as f32,
        f64::consts as f64,
        fmt::{self, Write as _},
        fs::{self, File},
        hash::Hash,
        marker::PhantomData,
        mem::{self, MaybeUninit as Uninit},
        ops,
        ptr::{self, NonNull},
        rc::Rc,
        result::Result as StdResult,
        sync::Arc,
        thread::{self, JoinHandle},
        time::{Duration, Instant},
    };
    pub use uv::{
        Bivec2, Bivec3, DVec2, DVec3, DVec4, Lerp, Mat2, Mat3, Mat4, Rotor2, Rotor3, Vec2, Vec3,
        Vec4,
    };

    // Very simple fxhash implementation.
    // About 20% faster than aHash, but definitely lower quality.
    // The issue is that this is basically multiplication by a constant.
    // This means that the lower bits of the hash are never affected by the higher bits.
    // However, the highest bits are affected by most of the bits in the data.
    // Because `hashbrown` uses the top bits of the hash, it's not too bad
    use std::{
        hash::{BuildHasherDefault, Hasher},
        ops::BitXor,
    };
    pub struct CustomHasher {
        hash: u64,
    }
    impl Default for CustomHasher {
        #[inline]
        fn default() -> Self {
            Self { hash: 0 }
        }
    }
    impl Hasher for CustomHasher {
        #[inline]
        fn write(&mut self, mut b: &[u8]) {
            while b.len() >= 8 {
                let mut n = [0; 8];
                n.copy_from_slice(&b[..8]);
                self.write_u64(u64::from_le_bytes(n));
                b = &b[8..];
            }
            if !b.is_empty() {
                let mut n = [0; 8];
                n[..b.len()].copy_from_slice(b);
                self.write_u64(u64::from_le_bytes(n));
            }
        }

        #[inline]
        fn finish(&self) -> u64 {
            self.hash
        }

        #[inline]
        fn write_u64(&mut self, x: u64) {
            const ROTATE: u32 = 5;
            const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
            self.hash = self.hash.rotate_left(ROTATE).bitxor(x).wrapping_mul(SEED);
        }

        #[inline]
        fn write_u32(&mut self, x: u32) {
            self.write_u64(x as u64);
        }

        #[inline]
        fn write_u16(&mut self, x: u16) {
            self.write_u64(x as u64);
        }

        #[inline]
        fn write_u8(&mut self, x: u8) {
            self.write_u64(x as u64);
        }

        #[inline]
        fn write_u128(&mut self, x: u128) {
            self.write_u64(x as u64);
            self.write_u64((x >> 64) as u64);
        }
    }
    pub type HashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<CustomHasher>>;
    pub type HashSet<V> = std::collections::HashSet<V, BuildHasherDefault<CustomHasher>>;

    pub fn default<T>() -> T
    where
        T: Default,
    {
        T::default()
    }

    #[macro_export]
    macro_rules! arr {
        [$init:expr; $n:expr] => {
            $crate::init_array(|_| $init)
        };
        [$i:ident => $init:expr; $n:expr] => {
            $crate::init_array(|$i| $init)
        };
    }

    /// Stupid workaround for serde.
    #[inline]
    pub fn default_true() -> bool {
        true
    }
}

pub fn init_array<T, F: FnMut(usize) -> T, const N: usize>(mut init: F) -> [T; N] {
    assert!(N.checked_mul(mem::size_of::<T>()).unwrap() <= isize::MAX as usize);
    let mut arr: Uninit<[T; N]> = Uninit::uninit();
    for i in 0..N {
        unsafe {
            (arr.as_mut_ptr() as *mut T)
                .offset(i as isize)
                .write(init(i));
        }
    }
    unsafe { arr.assume_init() }
}

#[macro_use]
pub mod arena;
pub mod bitset;
pub mod ivec;
pub mod lua;
pub mod noise2d;
pub mod noise3d;
pub mod slotmap;
pub mod spread2d;
pub mod staticinit;
pub mod terrain;

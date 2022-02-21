use crate::prelude::*;
use commonmem_consts::{ALIGN_LOG2, MAX_SIZE_CLASS};
use std::ptr::NonNull;

// OPTIMIZE: If allocation becomes a bottleneck, notice that the ownership pf recycled blocks of
// data can be sent around at no cost.
// Therefore, some sharding with shard balancing could be done.

pub static mut ARENA_ALLOC: Uninit<unsafe fn(usize) -> *mut u8> = Uninit::uninit();
pub static mut ARENA_DEALLOC: Uninit<unsafe fn(usize, *mut u8)> = Uninit::uninit();

const fn size_class_of<T>() -> usize {
    let raw_size = mem::size_of::<T>();
    mem::size_of::<usize>() * 8
        - (raw_size
            .saturating_sub(1)
            .leading_zeros()
            .saturating_sub(ALIGN_LOG2 as u32)
            + ALIGN_LOG2 as u32) as usize
}

pub fn alloc<T>() -> BoxUninit<T> {
    let size_class = size_class_of::<T>();
    if mem::align_of::<T>() > 1 << ALIGN_LOG2 {
        panic!("cannot arena-allocate objects with alignment higher than 16");
    }
    if mem::size_of::<T>() == 0 {
        return BoxUninit {
            ptr: NonNull::dangling(),
        };
    }
    if size_class > MAX_SIZE_CLASS {
        panic!(
            "cannot arena-allocate objects larger than {} bytes",
            1 << MAX_SIZE_CLASS
        );
    }
    unsafe {
        let b = ARENA_ALLOC.assume_init()(size_class);
        BoxUninit {
            ptr: NonNull::new_unchecked(b as *mut Uninit<T>),
        }
    }
}

pub struct BoxUninit<T> {
    ptr: ptr::NonNull<Uninit<T>>,
}
unsafe impl<T: Send> Send for BoxUninit<T> {}
unsafe impl<T: Sync> Sync for BoxUninit<T> {}
impl<T> Drop for BoxUninit<T> {
    fn drop(&mut self) {
        unsafe {
            let size_class = size_class_of::<T>();
            // OPTIMIZE: An easy optimization is to remove this `unwrap` at the cost of "safety"
            // ("safety" because once set up correctly there is virtually no chance to invoke this
            // unsafety, and even if it happens it is almost guaranteed to produce a crash, since
            // it is a call to null)
            ARENA_DEALLOC.assume_init()(size_class, self.ptr.as_ptr() as *mut u8);
        }
    }
}
impl<T> BoxUninit<T> {
    #[inline]
    pub fn new() -> BoxUninit<T> {
        alloc()
    }

    #[inline]
    pub unsafe fn init_zero(mut self) -> Box<T> {
        ptr::write_bytes(self.as_mut(), 0, 1);
        Box { inner: self }
    }

    #[inline]
    pub fn init(mut self, val: T) -> Box<T> {
        *self.as_mut() = Uninit::new(val);
        Box { inner: self }
    }

    #[inline]
    pub unsafe fn assume_init(self) -> Box<T> {
        Box { inner: self }
    }

    #[inline]
    pub fn as_ref(&self) -> &Uninit<T> {
        unsafe { &*(self.ptr.as_ptr() as *const Uninit<T>) }
    }

    #[inline]
    pub fn as_mut(&mut self) -> &mut Uninit<T> {
        unsafe { &mut *(self.ptr.as_ptr() as *mut Uninit<T>) }
    }

    #[inline]
    pub unsafe fn as_ref_init(&self) -> &T {
        &*(self.ptr.as_ptr() as *const T)
    }

    #[inline]
    pub unsafe fn as_mut_init(&mut self) -> &mut T {
        &mut *(self.ptr.as_ptr() as *mut T)
    }
}

pub struct Box<T> {
    inner: BoxUninit<T>,
}
impl<T> Box<T> {
    #[inline]
    pub fn new(val: T) -> Box<T> {
        BoxUninit::new().init(val)
    }

    #[inline]
    pub fn into_raw(this: Self) -> NonNull<T> {
        let ptr = this.inner.ptr;
        mem::forget(this);
        ptr.cast()
    }

    #[inline]
    pub unsafe fn from_raw(raw: NonNull<T>) -> Box<T> {
        Box {
            inner: BoxUninit { ptr: raw.cast() },
        }
    }
}
impl<T> ops::Deref for Box<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        unsafe { self.inner.as_ref_init() }
    }
}
impl<T> ops::DerefMut for Box<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { self.inner.as_mut_init() }
    }
}
impl<T> Drop for Box<T> {
    fn drop(&mut self) {
        unsafe {
            // Drop the inner T
            ptr::drop_in_place::<T>(&mut **self);
            // The inner BoxUninit will be dropped automatically
        }
    }
}

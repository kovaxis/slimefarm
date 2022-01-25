use crate::prelude::*;
use std::ptr::NonNull;

/// What size of chunk to allocate in one go.
/// 20 = 1MB
/// 22 = 4MB
/// 24 = 16MB
/// 26 = 64MB
const BLOCK_SIZE_LOG2: usize = 24;

struct SizeClass {
    available: Vec<AssertSync<*mut u8>>,
}

const fn class() -> Mutex<SizeClass> {
    parking_lot::const_mutex(SizeClass {
        available: Vec::new(),
    })
}

static STATE: [Mutex<SizeClass>; 24] = [
    class(),
    class(),
    class(),
    class(),
    class(),
    class(),
    class(),
    class(),
    class(),
    class(),
    class(),
    class(),
    class(),
    class(),
    class(),
    class(),
    class(),
    class(),
    class(),
    class(),
    class(),
    class(),
    class(),
    class(),
];

#[repr(align(16))]
struct Unit([u8; 16]);
const ALIGN_LOG2: usize = 4;

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
    let mut state = STATE[size_class].lock();
    match state.available.pop() {
        Some(b) => BoxUninit {
            ptr: unsafe { NonNull::new_unchecked(*b as *mut Uninit<T>) },
        },
        None => {
            //Allocate a new segment
            let (first, count);
            {
                let alloc_size = 1 << (BLOCK_SIZE_LOG2.max(size_class) - ALIGN_LOG2);
                let seg = Vec::<Unit>::with_capacity(alloc_size);
                first = seg.as_ptr() as *mut u8;
                count = 1 << BLOCK_SIZE_LOG2.saturating_sub(size_class);
                mem::forget(seg);
            }
            //Place the allocated slots in the available state
            unsafe {
                for i in 1..count {
                    state
                        .available
                        .push(AssertSync(first.offset((i << size_class) as isize)));
                }
                BoxUninit {
                    ptr: NonNull::new_unchecked(first as *mut Uninit<T>),
                }
            }
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
        let size_class = size_class_of::<T>();
        let mut state = STATE[size_class].lock();
        state
            .available
            .push(AssertSync(self.ptr.as_ptr() as *mut u8));
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
    fn deref(&self) -> &T {
        unsafe { self.inner.as_ref_init() }
    }
}
impl<T> ops::DerefMut for Box<T> {
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

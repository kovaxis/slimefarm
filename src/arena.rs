use crate::prelude::*;

macro_rules! make_arena {
    (@ [$($mod:tt)*], $t:ty, $segment_size:expr) => {
        $($mod)* {
            use super::*;
            use parking_lot::Mutex;
            use std::{
                mem::{self, MaybeUninit as Uninit},
                ops,
                ptr::{self, NonNull},
            };

            type T = $t;
            static AVAILABLE: Mutex<Vec<AssertSync<*mut Uninit<T>>>> = parking_lot::const_mutex(Vec::new());

            fn alloc_segment() -> (*mut Uninit<T>, usize) {
                const SEGMENT_SIZE: usize = $segment_size;

                let seg = Vec::<Uninit<T>>::with_capacity(SEGMENT_SIZE);
                let ptr = seg.as_ptr() as *mut Uninit<T>;
                let cap = seg.capacity();
                mem::forget(seg);
                (ptr, cap)
            }

            pub fn alloc() -> BoxUninit {
                let mut avail = AVAILABLE.lock();
                match avail.pop() {
                    Some(b) => BoxUninit {
                        ptr: unsafe { NonNull::new_unchecked(*b) },
                    },
                    None => {
                        //Allocate a new segment
                        let (first, count) = alloc_segment();
                        unsafe {
                            for i in 1..count {
                                avail.push(AssertSync(first.offset(i as isize)));
                            }
                            BoxUninit {
                                ptr: NonNull::new_unchecked(first),
                            }
                        }
                    }
                }
            }

            #[allow(dead_code)]
            fn assert_types() {
                fn assert_sendsync<U: Send + Sync>() {}
                assert_sendsync::<T>();
            }

            pub struct BoxUninit {
                ptr: ptr::NonNull<Uninit<T>>,
            }
            unsafe impl Send for BoxUninit {}
            unsafe impl Sync for BoxUninit {}
            impl Drop for BoxUninit {
                fn drop(&mut self) {
                    let mut avail = AVAILABLE.lock();
                    avail.push(AssertSync(self.ptr.as_ptr()));
                }
            }
            impl BoxUninit {
                #[allow(dead_code)]
                #[inline]
                pub fn new() -> BoxUninit {
                    alloc()
                }

                #[allow(dead_code)]
                #[inline]
                pub fn init(mut self, val: T) -> Box {
                    *self.as_mut() = Uninit::new(val);
                    Box { inner: self }
                }

                #[allow(dead_code)]
                #[inline]
                pub unsafe fn assume_init(self) -> Box {
                    Box { inner: self }
                }

                #[allow(dead_code)]
                #[inline]
                pub fn as_ref(&self) -> &Uninit<T> {
                    unsafe { &*(self.ptr.as_ptr() as *const Uninit<T>) }
                }

                #[allow(dead_code)]
                #[inline]
                pub fn as_mut(&mut self) -> &mut Uninit<T> {
                    unsafe { &mut *(self.ptr.as_ptr() as *mut Uninit<T>) }
                }

                #[allow(dead_code)]
                #[inline]
                pub unsafe fn as_ref_init(&self) -> &T {
                    &*(self.ptr.as_ptr() as *const T)
                }

                #[allow(dead_code)]
                #[inline]
                pub unsafe fn as_mut_init(&mut self) -> &mut T {
                    &mut *(self.ptr.as_ptr() as *mut T)
                }
            }

            pub struct Box {
                inner: BoxUninit,
            }
            impl Box {
                #[allow(dead_code)]
                #[inline]
                pub fn new(val: T) -> Box {
                    BoxUninit::new().init(val)
                }
            }
            impl ops::Deref for Box {
                type Target = T;
                fn deref(&self) -> &T {
                    unsafe { self.inner.as_ref_init() }
                }
            }
            impl ops::DerefMut for Box {
                fn deref_mut(&mut self) -> &mut T {
                    unsafe { self.inner.as_mut_init() }
                }
            }
            impl Drop for Box {
                fn drop(&mut self) {
                    unsafe {
                        ptr::drop_in_place(&mut **self);
                    }
                }
            }
        }
    };
    (@ [$($mod:tt)*], $t:ty) => {
        make_arena!(@ [$($mod)*], $t, 1024 * 1024 / std::mem::size_of::<$t>());
    };
    (pub $name:ident, $t:ty) => {
        make_arena!(@ [pub mod $name], $t);
    };
    ($name:ident, $t:ty) => {
        make_arena!(@ [mod $name], $t);
    };
    (pub $name:ident, $t:ty, $ss:expr) => {
        make_arena!(@ [pub mod $name], $t, $ss);
    };
    ($name:ident, $t:ty, $ss:expr) => {
        make_arena!(@ [mod $name], $t, $ss);
    };
}

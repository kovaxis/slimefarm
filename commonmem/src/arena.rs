use commonmem_consts::{ALIGN_LOG2, MAX_SIZE_CLASS};
use parking_lot::Mutex;
use std::mem;

/// What size of chunk to allocate in one go.
/// 20 = 1MB
/// 22 = 4MB
/// 24 = 16MB
/// 26 = 64MB
const BLOCK_SIZE_LOG2: usize = 24;

struct SizeClass {
    available: Vec<*mut u8>,
}
unsafe impl Send for SizeClass {}

const fn class() -> Mutex<SizeClass> {
    parking_lot::const_mutex(SizeClass {
        available: Vec::new(),
    })
}

static STATE: [Mutex<SizeClass>; MAX_SIZE_CLASS + 1] = [
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

// OPTIMIZE: If allocation becomes a bottleneck, notice that the ownership pf recycled blocks of
// data can be sent around at no cost.
// Therefore, some sharding with shard balancing could be done.

#[no_mangle]
pub unsafe fn game_arena_alloc(size_class: usize) -> *mut u8 {
    let mut state = STATE[size_class].lock();
    match state.available.pop() {
        Some(b) => b,
        None => {
            //Allocate a new segment
            let (first, count);
            {
                let alloc_size = 1 << (BLOCK_SIZE_LOG2.max(size_class) - ALIGN_LOG2);
                let seg = Vec::<Unit>::with_capacity(alloc_size);
                first = seg.as_ptr() as *mut u8;
                count = 1usize << BLOCK_SIZE_LOG2.saturating_sub(size_class);
                mem::forget(seg);
            }
            //Place the allocated slots in the available state
            for i in 1..count {
                state
                    .available
                    .push(first.offset((i << size_class) as isize));
            }
            first
        }
    }
}

#[no_mangle]
pub unsafe fn game_arena_dealloc(size_class: usize, ptr: *mut u8) {
    let mut state = STATE[size_class].lock();
    state.available.push(ptr);
}

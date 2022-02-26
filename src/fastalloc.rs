//! A general algorithm to allocate blocks of data from a contiguous space quickly.
//! The trade-off is, we care basically nothing about fragmentation.
//! This means that memory usage will be much worse than optimal.

use crate::prelude::*;

type BlockId = u32;

#[derive(Copy, Clone)]
struct Block {
    start: u32,
    size: u32,
    left: BlockId,
    right: BlockId,
}

const FIRST_BLOCK: usize = 0;
const LAST_BLOCK: usize = 0;

fn size_class_of(size: u32) -> usize {
    mem::size_of::<u32>() * 8 - 1 - size.leading_zeros() as usize
}

pub struct FastAlloc {
    /// A simple pool of blocks, which are referenced by index.
    blocks: Vec<Block>,
    /// Free block indices.
    free_ids: Vec<BlockId>,
    /// For each size class, store a list of free blocks.
    /// Size class `n` contains blocks with sizes in the range `[2^n, 2^(n+1))`
    /// (inclusive-exclusive).
    /// Each list is not in any particular order.
    free_list: Vec<Vec<BlockId>>,
    /// The virtual maximum size of the address space.
    /// Setting to a giant value will not affect performance.
    max_offset: u32,
}
impl FastAlloc {
    pub fn new() -> Self {
        Self {
            blocks: vec![
                Block {
                    start: 0,
                    size: 0,
                    left: 0,
                    right: 1,
                },
                Block {
                    start: 0,
                    size: 0,
                    left: 0,
                    right: 1,
                },
            ],
            free_ids: vec![],
            free_list: vec![],
            max_offset: u32::MAX,
        }
    }

    pub fn set_max(&mut self, mx: Option<u32>) {
        self.max_offset = mx.unwrap_or(u32::MAX);
    }

    pub fn size(&self) -> u32 {
        let last = &self.blocks[LAST_BLOCK];
        last.start + last.size
    }

    pub fn grow(&mut self, new_size: u32) {
        let old_size = self.size();
        assert!(new_size >= old_size);
        let grow_by = new_size - old_size;
        self.blocks[LAST_BLOCK].size += grow_by;
    }

    fn remove_block(&mut self, id: BlockId) {
        let b = &self.blocks[id as usize];
        let left = b.left;
        let right = b.right;
        self.blocks[left as usize].right = right;
        self.blocks[right as usize].left = left;
        self.free_ids.push(id);
    }

    fn add_block(&mut self, block: Block) -> BlockId {
        if let Some(id) = self.free_ids.pop() {
            self.blocks[id as usize] = block;
            id
        } else {
            let len = self.blocks.len();
            self.blocks.resize(
                2 * len,
                Block {
                    start: 0,
                    size: 0,
                    left: 0,
                    right: 0,
                },
            );
            len as BlockId
        }
    }

    pub fn alloc(&mut self, size: u32) -> Option<u32> {
        if size == 0 {
            return Some(0);
        }
        let cl = size_class_of(size);
        if self.free_list.len() <= cl {
            self.free_list.resize(cl + 1, vec![]);
        }
        for cl in cl + 1..self.free_list.len() {
            if let Some(id) = self.free_list[cl].pop() {
                let block = &mut self.blocks[id as usize];
                let start = block.start;
                if block.size != size {
                    // Move the current block to the leftover data and push back in the free list
                    block.start += size;
                    block.size -= size;
                    self.free_list[cl].push(id);
                    // Add the used
                    let used = self.add_block(Block {
                        start,
                        size,
                        left: block.left,
                        right: id,
                    });
                }
                return Some(start);
            }
        }
        // Found no recycled block of this size
        // Take a chunk off the last block, growing it if necessary
        let block = &mut self.blocks[LAST_BLOCK];
        if block.size < size {
            let new_max = block.start + size;
            if new_max > self.max_offset {
                return None;
            }
            block.size = size;
        }
        let start = block.start;
        block.start += size;
        block.size -= size;
        Some(start)
    }

    pub fn dealloc(&mut self, offset: u32, size: u32) {
        if size == 0 {
            return;
        }
        let cl = size_class_of(size);
    }
}

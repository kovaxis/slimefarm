use crate::prelude::*;

#[derive(PartialEq, Eq)]
pub struct SlotId<T> {
    pub idx: u32,
    pub _marker: PhantomData<T>,
}
impl<T> SlotId<T> {
    pub fn from_idx(idx: u32) -> Self {
        Self {
            idx,
            _marker: PhantomData,
        }
    }
}

pub struct SlotMap<T> {
    slots: Vec<Option<T>>,
    avail: Vec<u32>,
}
impl<T> Default for SlotMap<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T> SlotMap<T> {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            avail: Vec::new(),
        }
    }

    pub fn insert(&mut self, t: T) -> SlotId<T> {
        let idx = match self.avail.pop() {
            Some(idx) => {
                self.slots[idx as usize] = Some(t);
                idx
            }
            None => {
                assert!(self.slots.len() <= u32::MAX as usize);
                let idx = self.slots.len() as u32;
                self.slots.push(Some(t));
                idx
            }
        };
        SlotId::from_idx(idx)
    }

    pub fn get(&self, id: SlotId<T>) -> &T {
        self.slots[id.idx as usize].as_ref().unwrap()
    }

    pub fn get_mut(&mut self, id: SlotId<T>) -> &mut T {
        self.slots[id.idx as usize].as_mut().unwrap()
    }

    pub fn take(&mut self, id: SlotId<T>) -> T {
        let t = self.slots[id.idx as usize].take().unwrap();
        self.avail.push(id.idx);
        t
    }

    pub fn clear(&mut self) {
        self.slots.clear();
        self.avail.clear();
    }
}

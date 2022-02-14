use crate::{prelude::*, terrain::PortalData};

pub trait PaintAction {
    /// Apply this action onto a chunk.
    /// The coordinates given are the block coordinates of the chunk floor in action-local
    /// coordinates.
    /// The bounding box given is in action-local coordinates, and is within the chunk, otherwise
    /// `apply` will not be called.
    fn apply(&self, pos: BlockPos, bbox: [Int3; 2], chunk: &mut ChunkBox);
}

pub struct ActionBuf {
    origin: BlockPos,
    actions: Vec<([Int3; 2], Box<dyn PaintAction>)>,
}
impl ActionBuf {
    pub fn new(origin: BlockPos) -> Self {
        Self {
            origin,
            actions: vec![],
        }
    }

    pub fn act(&mut self, action: ([Int3; 2], Box<dyn PaintAction>)) {
        self.actions.push(action);
    }

    pub fn transfer(&self, chunkpos: ChunkPos, chunk: &mut ChunkBox) {
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
                action.apply(chunk_mn, bbox, chunk);
            }
        }
    }
}

pub mod actions;

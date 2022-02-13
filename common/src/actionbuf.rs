use crate::prelude::*;

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

pub struct ActionSphere {
    u: Vec3,
    r2: f32,
    block: BlockData,
}
impl ActionSphere {
    pub fn new(u: Vec3, r: f32, block: BlockData) -> ([Int3; 2], Box<dyn PaintAction>) {
        let mn = Int3::from_f32((u - Vec3::broadcast(r)).map(f32::floor));
        let mx = Int3::from_f32((u + Vec3::broadcast(r)).map(f32::floor)) + [1; 3];
        (
            [mn, mx],
            Box::new(Self {
                u,
                r2: r * r,
                block,
            }),
        )
    }

    pub fn paint(buf: &mut ActionBuf, u: Vec3, r: f32, block: BlockData) {
        buf.act(Self::new(u, r, block));
    }
}
impl PaintAction for ActionSphere {
    fn apply(&self, pos: Int3, [mn, mx]: [Int3; 2], chunk: &mut ChunkBox) {
        let chunk = chunk.blocks_mut();
        for z in mn.z..mx.z {
            for y in mn.y..mx.y {
                // OPTIMIZE: Use sqrt and math to figure out exactly where does this row start and
                // end, and write all the row blocks in one go.
                for x in mn.x..mx.x {
                    let d = Vec3::new(x as f32, y as f32, z as f32) - self.u;
                    if d.mag_sq() <= self.r2 {
                        chunk.sub_set([x, y, z] - pos, self.block);
                    }
                }
            }
        }
    }
}

pub struct ActionOval {
    u: Vec3,
    inv_r: Vec3,
    block: BlockData,
}
impl ActionOval {
    pub fn new(u: Vec3, r: Vec3, block: BlockData) -> ([Int3; 2], Box<dyn PaintAction>) {
        let mn = Int3::from_f32((u - r).map(f32::floor));
        let mx = Int3::from_f32((u + r).map(f32::floor)) + [1; 3];
        (
            [mn, mx],
            Box::new(Self {
                u,
                inv_r: r.map(|r| r.recip()),
                block,
            }),
        )
    }

    pub fn paint(buf: &mut ActionBuf, u: Vec3, r: Vec3, block: BlockData) {
        buf.act(Self::new(u, r, block));
    }
}
impl PaintAction for ActionOval {
    fn apply(&self, pos: Int3, [mn, mx]: [Int3; 2], chunk: &mut ChunkBox) {
        let chunk = chunk.blocks_mut();
        for z in mn.z..mx.z {
            for y in mn.y..mx.y {
                // OPTIMIZE: Use sqrt and math to figure out exactly where does this row start and
                // end, and write all the row blocks in one go.
                for x in mn.x..mx.x {
                    let d = Vec3::new(x as f32, y as f32, z as f32) - self.u;
                    if (d * self.inv_r).mag_sq() <= 1. {
                        chunk.sub_set([x, y, z] - pos, self.block);
                    }
                }
            }
        }
    }
}

pub struct ActionCylinder {
    u0: Vec3,
    u1: Vec3,
    r0: f32,
    r1: f32,
    block: BlockData,
}
impl ActionCylinder {
    pub fn new(
        u0: Vec3,
        u1: Vec3,
        r0: f32,
        r1: f32,
        block: BlockData,
    ) -> ([Int3; 2], Box<dyn PaintAction>) {
        let mn = Int3::from_f32(
            (u0 - Vec3::broadcast(r0))
                .min_by_component(u1 - Vec3::broadcast(r1))
                .map(f32::floor),
        );
        let mx = Int3::from_f32(
            (u0 + Vec3::broadcast(r0))
                .max_by_component(u1 + Vec3::broadcast(r1))
                .map(f32::floor),
        ) + [1; 3];
        (
            [mn, mx],
            Box::new(Self {
                u0,
                u1,
                r0,
                r1,
                block,
            }),
        )
    }

    pub fn paint(buf: &mut ActionBuf, u0: Vec3, u1: Vec3, r0: f32, r1: f32, block: BlockData) {
        buf.act(Self::new(u0, u1, r0, r1, block));
    }
}
impl PaintAction for ActionCylinder {
    fn apply(&self, pos: Int3, [mn, mx]: [Int3; 2], chunk: &mut ChunkBox) {
        let chunk = chunk.blocks_mut();
        let n = self.u1 - self.u0;
        let inv = n.mag_sq().recip();
        for z in mn.z..mx.z {
            for y in mn.y..mx.y {
                for x in mn.x..mx.x {
                    let p = Vec3::new(x as f32, y as f32, z as f32) - self.u0;
                    let s = (n.dot(p) * inv).min(1.).max(0.);
                    let r = self.r0 + (self.r1 - self.r0) * s;
                    let d2 = (n * s - p).mag_sq();
                    if d2 <= r * r {
                        chunk.sub_set([x, y, z] - pos, self.block);
                    }
                }
            }
        }
    }
}

use crate::{
    actionbuf::{ActionBuf, PaintAction},
    prelude::*,
};

pub struct Portal {
    pos: Int3,
    size: Int3,
    jump: Int3,
}
impl Portal {
    pub fn new(pos: Int3, size: Int3, target: Int3) -> ([Int3; 2], Box<dyn PaintAction>) {
        (
            [pos, pos + size],
            Box::new(Self {
                pos,
                size,
                jump: target - pos,
            }),
        )
    }

    pub fn paint(buf: &mut ActionBuf, pos: Int3, size: Int3, target: Int3) {
        buf.act(Self::new(pos, size, target));
    }
}
impl PaintAction for Portal {
    fn apply(&self, pos: Int3, [_mn, _mx]: [Int3; 2], chunk: &mut ChunkBox) {
        let chunk = chunk.blocks_mut();
        let pos = self.pos - pos;
        chunk.push_portal(PortalData {
            pos: [pos.x as i16, pos.y as i16, pos.z as i16],
            size: [self.size.x as i16, self.size.y as i16, self.size.z as i16],
            jump: self.jump.into(),
        });
    }
}

pub struct Cube {
    block: BlockData,
}
impl Cube {
    pub fn new(u: Vec3, s: Vec3, block: BlockData) -> ([Int3; 2], Box<dyn PaintAction>) {
        let mn = Int3::from_f32(u.map(f32::round));
        let mx = Int3::from_f32((u + s).map(f32::round));
        ([mn, mx], Box::new(Self { block }))
    }

    pub fn paint(buf: &mut ActionBuf, u: Vec3, s: Vec3, block: BlockData) {
        buf.act(Self::new(u, s, block));
    }
}
impl PaintAction for Cube {
    fn apply(&self, pos: Int3, [mn, mx]: [Int3; 2], chunk: &mut ChunkBox) {
        let chunk = chunk.blocks_mut();
        for z in mn.z..mx.z {
            for y in mn.y..mx.y {
                // OPTIMIZE: Use `memset`
                for x in mn.x..mx.x {
                    chunk.sub_set([x, y, z] - pos, self.block);
                }
            }
        }
    }
}

pub struct Sphere {
    u: Vec3,
    r2: f32,
    block: BlockData,
}
impl Sphere {
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
impl PaintAction for Sphere {
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

pub struct Oval {
    u: Vec3,
    inv_r: Vec3,
    block: BlockData,
}
impl Oval {
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
impl PaintAction for Oval {
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

pub struct Cylinder {
    u0: Vec3,
    u1: Vec3,
    r0: f32,
    r1: f32,
    block: BlockData,
}
impl Cylinder {
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
impl PaintAction for Cylinder {
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

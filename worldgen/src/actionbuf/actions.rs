use crate::{
    actionbuf::{Action, ActionBuf},
    prelude::*,
};

fn block(b: u8) -> BlockData {
    BlockData { data: b }
}

action! {
    fn portal((x, y, z, sx, sy, sz, jx, jy, jz, jw): (i32, i32, i32, i32, i32, i32, i32, i32, i32, u32)) -> bd {
        let u = Int3::new([x, y, z]);
        let s = Int3::new([sx, sy, sz]);
        let axes = [sx == 0, sy == 0, sz == 0];
        if axes[0] as i32 + axes[1] as i32 + axes[2] as i32 != 1 {
            panic!("exactly 1 portal axis must have 0 size, but size is {:?}", s);
        }
        let axis = if axes[0] {0}else if axes[1] {1} else {2};
        let mut s2 = s;
        s2[axis] = 1;
        bd = [u, u + s2];
    }
    fn apply(pos, _, chunk) {
        let chunk = chunk.blocks_mut();
        let p = u - pos;
        chunk.push_portal(PortalData {
            pos: [p.x as i16, p.y as i16, p.z as i16],
            size: [s.x as i16, s.y as i16, s.z as i16],
            jump: [jx, jy, jz],
            dim: jw,
        });
    }
}

action! {
    fn cube((ux, uy, uz, sx, sy, sz, b): (f32, f32, f32, f32, f32, f32, u8)) -> bd {
        let u = Vec3::from([ux, uy, uz]);
        let s = Vec3::from([sx, sy, sz]);
        let b = block(b);
        let mn = Int3::from_f32(u.map(f32::round));
        let mx = Int3::from_f32((u + s).map(f32::round));
        bd = [mn, mx];
    }
    fn apply(pos, [mn, mx], chunk) {
        let chunk = chunk.blocks_mut();
        for z in mn.z..mx.z {
            for y in mn.y..mx.y {
                // OPTIMIZE: Use `memset`
                for x in mn.x..mx.x {
                    chunk.sub_set([x, y, z] - pos, b);
                }
            }
        }
    }
}

action! {
    fn sphere((ux, uy, uz, r, b): (f32, f32, f32, f32, u8)) -> bd {
        let u = Vec3::new(ux, uy, uz);
        let b = block(b);
        let r2 = r * r;
        let mn = Int3::from_f32((u - Vec3::broadcast(r)).map(f32::floor));
        let mx = Int3::from_f32((u + Vec3::broadcast(r)).map(f32::floor)) + [1; 3];
        bd = [mn, mx];
    }
    fn apply(pos, [mn, mx], chunk) {
        let chunk = chunk.blocks_mut();
        for z in mn.z..mx.z {
            for y in mn.y..mx.y {
                // OPTIMIZE: Use sqrt and math to figure out exactly where does this row start and
                // end, and write all the row blocks in one go.
                for x in mn.x..mx.x {
                    let d = Vec3::new(x as f32, y as f32, z as f32) - u;
                    if d.mag_sq() <= r2 {
                        chunk.sub_set([x, y, z] - pos, b);
                    }
                }
            }
        }
    }
}

action! {
    fn oval((ux, uy, uz, rx, ry, rz, b): (f32, f32, f32, f32, f32, f32, u8)) -> bd {
        let u = Vec3::new(ux, uy, uz);
        let r = Vec3::new(rx, ry, rz);
        let b = block(b);
        let inv_r = r.map(f32::recip);
        let mn = Int3::from_f32((u - r).map(f32::floor));
        let mx = Int3::from_f32((u + r).map(f32::floor)) + [1; 3];
        bd = [mn, mx];
    }
    fn apply(pos, [mn, mx], chunk) {
        let chunk = chunk.blocks_mut();
        for z in mn.z..mx.z {
            for y in mn.y..mx.y {
                // OPTIMIZE: Use sqrt and math to figure out exactly where does this row start and
                // end, and write all the row blocks in one go.
                for x in mn.x..mx.x {
                    let d = Vec3::new(x as f32, y as f32, z as f32) - u;
                    if (d * inv_r).mag_sq() <= 1. {
                        chunk.sub_set([x, y, z] - pos, b);
                    }
                }
            }
        }
    }
}

action! {
    fn cylinder((x0, y0, z0, r0, x1, y1, z1, r1, b): (f32, f32, f32, f32, f32, f32, f32, f32, u8)) -> bd {
        let u0 = Vec3::new(x0, y0, z0);
        let u1 = Vec3::new(x1, y1, z1);
        let b = block(b);
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
        bd = [mn, mx];
    }
    fn apply(pos, [mn, mx], chunk) {
        let chunk = chunk.blocks_mut();
        let n = u1 - u0;
        let inv = n.mag_sq().recip();
        for z in mn.z..mx.z {
            for y in mn.y..mx.y {
                for x in mn.x..mx.x {
                    let p = Vec3::new(x as f32, y as f32, z as f32) - u0;
                    let s = (n.dot(p) * inv).min(1.).max(0.);
                    let r = r0 + (r1 - r0) * s;
                    let d2 = (n * s - p).mag_sq();
                    if d2 <= r * r {
                        chunk.sub_set([x, y, z] - pos, b);
                    }
                }
            }
        }
    }
}

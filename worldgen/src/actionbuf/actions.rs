use crate::{
    actionbuf::{Action, ActionBuf},
    prelude::*,
};

action! {
    fn cube((ux, uy, uz, sx, sy, sz, b), (f32, f32, f32, f32, f32, f32, u8)) -> bounds {
        let u = Vec3::from([ux, uy, uz]);
        let s = Vec3::from([sx, sy, sz]);
        let b = BlockData { data: b };
        let mn = Int3::from_f32(u.map(f32::round));
        let mx = Int3::from_f32((u + s).map(f32::round));
        bounds = [mn, mx];
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
    fn sphere((ux, uy, uz, r, b), (f32, f32, f32, f32, u8)) -> bounds {
        let u = Vec3::new(ux, uy, uz);
        let b = BlockData { data: b };
        let r2 = r * r;
        let mn = Int3::from_f32((u - Vec3::broadcast(r)).map(f32::floor));
        let mx = Int3::from_f32((u + Vec3::broadcast(r)).map(f32::floor)) + [1; 3];
        bounds = [mn, mx];
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

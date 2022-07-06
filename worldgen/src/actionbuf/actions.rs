use crate::{
    actionbuf::{Action, ActionBuf, Painter},
    prelude::*,
};

fn block(b: u8) -> BlockData {
    BlockData { data: b }
}

// VERY UNSAFE: Only use when extracting matrices from light userdata.
fn _matrix(mat: LuaLightUserData) -> Mat4 {
    unsafe { *(mat.0 as *mut Mat4 as *const Mat4) }
}

// Transform a point using a matrix, ignoring perspective division.
fn _transform(mat: &Mat4, p: Vec3) -> Vec3 {
    let mut out = p.x * mat.cols[0];
    out += p.y * mat.cols[1];
    out += p.z * mat.cols[2];
    out += mat.cols[3];
    out.xyz()
}

// Transform a vector using a matrix, ignoring perspective division.
fn _transform_vec3(mat: &Mat4, v: Vec3) -> Vec3 {
    let mut out = v.x * mat.cols[0];
    out += v.y * mat.cols[1];
    out += v.z * mat.cols[2];
    out.xyz()
}

fn invsqrt(x: f32) -> f32 {
    // Taken from Wikipedia, which was taken from Quake, with the exact original comments
    let i = x.to_bits(); // evil floating point bit level hacking
    let i = 0x5f3759df - (i >> 1); // what the fuck?
    let x2 = x * 0.5;
    let mut y = f32::from_bits(i);
    y = y * (1.5 - (x2 * y * y)); // 1st iteration
                                  //	y = y * ( 1.5 - ( x2 * y * y ) );      // 2nd iteration, this can be removed
    y
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
        let chunk = chunk.data_mut();
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
    fn entity((x, y, z, raw): (f32, f32, f32, LuaBytes)) -> bd {
        let raw = raw.bytes;
        let pos = Vec3::new(x, y, z);
        let ipos = Int3::from_f32(pos);
        bd = [ipos, ipos + [1; 3]];
    }
    fn apply(cpos, _, chunk) {
        let data = chunk.data_mut();
        if data.push_entity(pos - cpos.to_f32(), &raw[..]).is_none() {
            eprintln!("failed to add entity, entity data buffer is full");
        }
    }
}

action! {
    // place an axis-aligned cuboid.
    // `ux, uy, uz`: low vertex of the cuboid.
    // `sx, sy, sz`: size of the cuboid.
    // `b`: the block to fill with.
    fn cube((ux, uy, uz, sx, sy, sz, b): (f32, f32, f32, f32, f32, f32, u8)) -> bd {
        let u = Vec3::from([ux, uy, uz]);
        let s = Vec3::from([sx, sy, sz]);
        let b = block(b);
        let mn = Int3::from_f32(u.map(f32::round));
        let mx = Int3::from_f32((u + s).map(f32::round));
        bd = [mn, mx];
    }
    fn apply(pos, [mn, mx], chunk) {
        let chunk = chunk.data_mut();
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
    // place a sphere at a location.
    // `ux, uy, uz`: center of the sphere.
    // `r`: radius of the sphere.
    // `b`: the block to fill with.
    fn sphere((ux, uy, uz, r, b): (f32, f32, f32, f32, u8)) -> bd {
        let u = Vec3::new(ux, uy, uz);
        let b = block(b);
        let r2 = r * r;
        let mn = Int3::from_f32((u - Vec3::broadcast(r)).map(f32::floor));
        let mx = Int3::from_f32((u + Vec3::broadcast(r)).map(f32::floor)) + [1; 3];
        bd = [mn, mx];
    }
    fn apply(pos, [mn, mx], chunk) {
        let chunk = chunk.data_mut();
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
    // place an axis-aligned ellipsoid at the given location.
    // `ux, uy, uz`: center of the ellipsoid.
    // `rx, ry, rz`: radius of the ellipsoid in the different directions.
    // `b`: the block to fill with.
    fn ellipsoid((ux, uy, uz, rx, ry, rz, b): (f32, f32, f32, f32, f32, f32, u8)) -> bd {
        let u = Vec3::new(ux, uy, uz);
        let r = Vec3::new(rx, ry, rz);
        let b = block(b);
        let inv_r = r.map(f32::recip);
        let mn = Int3::from_f32((u - r).map(f32::floor));
        let mx = Int3::from_f32((u + r).map(f32::floor)) + [1; 3];
        bd = [mn, mx];
    }
    fn apply(pos, [mn, mx], chunk) {
        let chunk = chunk.data_mut();
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
    // create a cylinder using two endpoints and a radius for each endpoint.
    // the cylinder edges are rounded.
    // `x0, y0, z0`: coordinates of the first endpoint.
    // `r0`: radius of the first endpoint.
    // `x1, y1, z1`: coordinates of the second endpoint.
    // `r1`: radius of the second endpoint.
    // `b`: the block to fill with.
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
        let chunk = chunk.data_mut();
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

action! {
    // Generate a bunch of "blobs", or spheres with smooth joints between them.
    // `bs`: positions and radii of each blob. these come in groups of 4: x, y, z and r
    // `b`: the block to fill with.
    // `j`: the "join distance", or how far should the surfaces of 2 blobs be to start joining.
    //
    // This blob mode is based on the sum of distances to the power of `-1`.
    fn blobs((bs, b, j): (Vec<f32>, u8, f32)) {
        type Data = Rc<RefCell<[f32; (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize]>>;

        lua_assert!(bs.len() % 4 == 0, "blob coordinate count must be a multiple of 4");
        let b = block(b);
        let data: Data =
            Rc::new(RefCell::new([0.; (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize]));
    }
    fn place() {
        let mut mn_total = Int3::new([i32::MAX; 3]);
        let mut mx_total = Int3::new([i32::MIN; 3]);
        for bpos in bs.chunks_exact(4) {
            let r = bpos[3];
            let bpos = Vec3::new(bpos[0], bpos[1], bpos[2]);
            let x = Int3::from_f32(bpos.map(|f| (f - r - j).floor()));
            mn_total = mn_total.min(x);
            let x = Int3::from_f32(bpos.map(|f| (f + r + j).floor()));
            mx_total = mx_total.max(x);
        }
        mx_total += [1; 3];
        open([mn_total, mx_total], data.clone());
        for bpos in bs.chunks_exact(4) {
            let r = bpos[3];
            let u = Vec3::new(bpos[0], bpos[1], bpos[2]);
            let p = r + j;
            let mn = Int3::from_f32(u.map(|f| (f - p).floor()));
            let mx = Int3::from_f32(u.map(|f| (f + p).floor())) + [1; 3];
            let k = p * r / j;
            let c = -r / j;
            blob([mn, mx], data.clone(), u, k, c);
        }
        close([mn_total, mx_total], data);
    }
    fn open(pos, [mn, mx], _, data: Data) {
        let mn = mn - pos;
        let mx = mx - pos;
        let mut data = data.borrow_mut();
        for z in mn.z..mx.z {
            for y in mn.y..mx.y {
                for x in mn.x..mx.x {
                    let idx = (((z << CHUNK_BITS) + y) << CHUNK_BITS) + x;
                    data[idx as usize] = 0.;
                }
            }
        }
    }
    fn blob(cpos, [mn, mx], _, data: Data, u, k, c) {
        let mn = mn - cpos;
        let mx = mx - cpos;
        let u = u - cpos.to_f32();
        let mut data = data.borrow_mut();
        for z in mn.z..mx.z {
            for y in mn.y..mx.y {
                for x in mn.x..mx.x {
                    let pos = Vec3::new(x as f32, y as f32, z as f32);
                    let f = invsqrt((pos - u).mag_sq()).mul_add(k, c);
                    if f > 0. {
                        let idx = (((z << CHUNK_BITS) + y) << CHUNK_BITS) + x;
                        data[idx as usize] += f;
                    }
                }
            }
        }
    }
    fn close(pos, [mn, mx], chunk, data: Data) {
        let mn = mn - pos;
        let mx = mx - pos;
        let data = data.borrow_mut();
        let chunk = chunk.data_mut();
        for z in mn.z..mx.z {
            for y in mn.y..mx.y {
                for x in mn.x..mx.x {
                    let idx = (((z << CHUNK_BITS) + y) << CHUNK_BITS) + x;
                    if data[idx as usize] >= 1. {
                        chunk.set_idx(idx as usize, b);
                    }
                }
            }
        }
    }
}

action! {
    // draw a "cloud", basically a bunch of small spheres arranged in a sphere pattern.
    // `x, y, z`: center of the cloudsphere
    // `r`: radius of the cloudsphere
    // `r0, r1`: range of radii of the subspheres
    // `seed`: random seed for the subsphere sizes
    // `b`: the block to fill with
    // `n`: number of subspheres
    fn cloud((x, y, z, r, r0, r1, seed, b, n): (f32, f32, f32, f32, f32, f32, i64, u8, u32)) {
        let u = Vec3::new(x, y, z);
        let b = block(b);
        let mut rng = FastRng::seed_from_u64(seed as u64);
        let dr = r1 - r0;
    }
    fn place() {
        let sph = |u: Vec3, r: f32| {
            let mn = Int3::from_f32((u - Vec3::broadcast(r)).map(f32::floor));
            let mx = Int3::from_f32((u + Vec3::broadcast(r)).map(f32::floor)) + [1; 3];
            subsphere([mn, mx], u, r * r);
        };

        let phi = f32::PI * (3. - 5f32.sqrt());
        let dz = 2. * r / (n - 1) as f32;
        let mut theta = 0f32;
        let mut z = -r;
        sph(u, r);
        for _ in 0..n {
            let rxy = (r*r - z * z).sqrt();
            let u = u + Vec3::new(theta.cos() * rxy, theta.sin() * rxy, z);
            let r = r0 + rng.gen::<f32>() * dr;
            sph(u, r);
            theta += phi;
            z += dz;
        }
    }
    fn subsphere(pos, [mn, mx], chunk, u, r2) {
        let chunk = chunk.data_mut();
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
    fn rock((x, y, z, r, seed, n, noisy, b): (f32, f32, f32, f32, i64, u32, f32, u8)) -> bd {
        let u = Vec3::new(x, y, z);
        let b = block(b);
        let mut rng = FastRng::seed_from_u64(seed as u64);

        let mn = Int3::from_f32((u - Vec3::broadcast(r * (1. + noisy))).map(f32::floor));
        let mx = Int3::from_f32((u + Vec3::broadcast(r * (1. + noisy))).map(f32::floor)) + [1; 3];
        bd = [mn, mx];

        let phi = f32::PI * (3. - 5f32.sqrt());
        let dz = 2. * r / n as f32;
        let mut theta = rng.gen::<f32>() * (2. * f32::PI);
        let mut z = -r + 0.36 * dz;
        let noisy = noisy * r;
        // n x <= r
        // (n r / r^2) x <= 1
        let points = (0..n).map(|_i| {
            let rxy = (r * r - z * z).sqrt();
            let mut p = Vec3::new(theta.cos() * rxy, theta.sin() * rxy, z);
            p += Vec3::from(rng.gen::<[f32; 3]>()) * noisy;
            p *= 1. / p.mag_sq();
            p.x = 1. / p.x; // infinity if x = 0
            theta += phi;
            z += dz;
            p
        }).collect::<Vec<Vec3>>();
    }
    fn apply(pos, [mn, mx], chunk) {
        let chunk = chunk.data_mut();
        for az in mn.z..mx.z {
            for ay in mn.y..mx.y {
                let mut x0 = f32::NEG_INFINITY;
                let mut x1 = f32::INFINITY;
                let y = ay as f32 - u.y;
                let z = az as f32 - u.z;
                for p in &points {
                    let x = p.x * (1. - p.y * y - p.z * z);
                    if p.x >= 0. {
                        if x < x1 { x1 = x; }
                    }else{
                        if x > x0 { x0 = x; }
                    }
                }

                let x0 = (u.x + x0).max(mn.x as f32).floor() as i32;
                let x1 = ((u.x + x1).floor()+1.).min(mx.x as f32) as i32;
                for ax in x0..x1 {
                    chunk.sub_set([ax, ay, az] - pos, b);
                }
            }
        }
    }
}

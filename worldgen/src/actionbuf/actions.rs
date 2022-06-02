use crate::{
    actionbuf::{
        Action, ActionBuf, ActorArg, BufState, BuildPainter, ChunkMask, InsertAction, PainterRef,
    },
    prelude::*,
};

fn getblock(mut rgb: Vec3) -> BlockData {
    rgb *= 32.;
    let mut out = 0i16;
    for i in 0..3 {
        let r = rgb[i].max(0.).min(31.) as i16;
        out |= r << (i * 5);
    }
    BlockData { data: out }
}

// VERY UNSAFE: Only use when extracting matrices from light userdata.
fn matrix(mat: LuaLightUserData) -> Mat4 {
    unsafe { *(mat.0 as *mut Mat4 as *const Mat4) }
}

// Transform a point using a matrix, ignoring perspective division.
fn transform(mat: &Mat4, p: Vec3) -> Vec3 {
    let mut out = p.x * mat.cols[0];
    out += p.y * mat.cols[1];
    out += p.z * mat.cols[2];
    out += mat.cols[3];
    out.xyz()
}

// Transform a vector using a matrix, ignoring perspective division.
fn transform_vec3(mat: &Mat4, v: Vec3) -> Vec3 {
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

macro_rules! shape {
    (@set resetrow $state:expr, $idx:expr) => { $state.mask[$idx] = 0; };
    (@set combine $prev:expr, $x:expr, $incl:expr) => { $prev |= ($incl as ChunkSizedInt) << $x; };
    (@add resetrow $state:expr, $idx:expr) => {};
    (@add combine $prev:expr, $x:expr, $incl:expr) => { $prev |= ($incl as ChunkSizedInt) << $x; };
    (@and resetrow $state:expr, $idx:expr) => {};
    (@and combine $prev:expr, $x:expr, $incl:expr) => { $prev &= !(((!$incl) as ChunkSizedInt) << $x); };
    ([$method:tt] $arg:expr, $state:expr, $p:pat, $incl:expr) => {{
        let [mn, mx] = $arg.bbox;
        for z in mn.z..mx.z {
            for y in mn.y..mx.y {
                let idx = (Int2::new([y, z])).to_index(CHUNK_BITS);
                shape!(@$method resetrow $state, idx);
                for x in mn.x..mx.x {
                    let incl = {
                        let $p = Int3::new([x, y, z]) - $arg.origin;
                        $incl
                    };
                    shape!(@$method combine $state.mask[idx], x, incl);
                }
            }
        }
    }};
    ($arg:expr, $state:expr, $p:pat, $incl:expr) => {
        shape!([set] $arg, $state, $p, $incl);
    };
}

macro_rules! paint {
    ([getbounds] $state:expr) => {};
    ($arg:expr, $state:expr, $chunk:expr, $p:pat, $block:expr) => {
        let [mn, mx] = $arg.bbox;
        let chunk = $chunk.blocks_mut();
        for z in mn.z..mx.z {
            for y in mn.y..mx.y {
                let idx = (Int2::new([y, z])).to_index(CHUNK_BITS);
                for x in mn.x..mx.x {
                    let draw = ($state.mask[idx] >> x) & 1 != 0;
                    if draw {
                        let p = Int3::new([x, y, z]);
                        let b = {
                            let $p = p - $arg.origin;
                            $block
                        };
                        chunk.sub_set(p, b);
                    }
                }
            }
        }
    };
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
    fn apply(arg, _, chunk) {
        let chunk = chunk.blocks_mut();
        let p = u + arg.origin;
        chunk.push_portal(PortalData {
            pos: [p.x as i16, p.y as i16, p.z as i16],
            size: [s.x as i16, s.y as i16, s.z as i16],
            jump: [jx, jy, jz],
            dim: jw,
        });
    }
}

action! {
    // place an axis-aligned cuboid.
    // `ux, uy, uz`: low vertex of the cuboid.
    // `sx, sy, sz`: size of the cuboid.
    fn cube((ux, uy, uz, sx, sy, sz): (f32, f32, f32, f32, f32, f32)) -> bd {
        let u = Vec3::new(ux, uy, uz);
        let s = Vec3::new(sx, sy, sz);
        let mn = Int3::from_f32(u.map(f32::round));
        let mx = Int3::from_f32((u + s).map(f32::round));
        bd = [mn, mx];
    }
    fn apply(arg, state, _) {
        shape!(arg, state, _, true);
    }
}

action! {
    // place a sphere at a location.
    // `ux, uy, uz`: center of the sphere.
    // `r`: radius of the sphere.
    fn sphere((ux, uy, uz, r): (f32, f32, f32, f32)) -> bd {
        let u = Vec3::new(ux, uy, uz);
        let r2 = r * r;
        let mn = Int3::from_f32((u - Vec3::broadcast(r)).map(f32::floor));
        let mx = Int3::from_f32((u + Vec3::broadcast(r)).map(f32::floor)) + [1; 3];
        bd = [mn, mx];
    }
    fn apply(arg, state, _) {
        shape!(arg, state, p, (p.to_f32() - u).mag_sq() <= r2);
    }
}

action! {
    // place an ovoid using an arbitrary matrix.
    // `mat`: a 4x4 matrix that transforms a unit sphere into an ovoid.
    //        if the matrix contains shear the ovoid might be truncated.
    fn oval((mat): LuaLightUserData) -> bd {
        let mat = matrix(mat);
        let imat = mat.inversed();
        let u = transform(&mat, Vec3::zero());
        let mut s;
        s = mat.cols[0].xyz().mag_sq();
        s = s.max(mat.cols[1].xyz().mag_sq());
        s = s.max(mat.cols[2].xyz().mag_sq());
        let s = Vec3::broadcast(s.sqrt());
        let mn = Int3::from_f32((u - s).map(f32::floor));
        let mx = Int3::from_f32((u + s).map(f32::floor)) + [1; 3];
        bd = [mn, mx];
    }
    fn apply(arg, state, _) {
        shape!(arg, state, p, (transform(&imat, p.to_f32())).mag_sq() <= 1.);
    }
}

action! {
    // create a cylinder using two endpoints and a radius for each endpoint.
    // the cylinder edges are rounded.
    // `x0, y0, z0`: coordinates of the first endpoint.
    // `r0`: radius of the first endpoint.
    // `x1, y1, z1`: coordinates of the second endpoint.
    // `r1`: radius of the second endpoint.
    fn cylinder((x0, y0, z0, r0, x1, y1, z1, r1): (f32, f32, f32, f32, f32, f32, f32, f32)) -> bd {
        let u0 = Vec3::new(x0, y0, z0);
        let u1 = Vec3::new(x1, y1, z1);
        let n = u1 - u0;
        let inv = n.mag_sq().recip();
        let dr = r1 - r0;
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
    fn apply(arg, state, _) {
        shape!(arg, state, p, {
            let p = p.to_f32() - u0;
            let s = (n.dot(p) * inv).min(1.).max(0.);
            let r = r0 + s * dr;
            let d2 = (n * s - p).mag_sq();
            d2 <= r * r
        });
    }
}

action! {
    // Generate a bunch of "blobs", or spheres with smooth joints between them.
    // `bs`: positions and radii of each blob. these come in groups of 4: x, y, z and r
    // `j`: the "join distance", or how far should the surfaces of 2 blobs be to start joining.
    //
    // This blob mode is based on the sum of distances to the power of `-1`.
    fn blobs((bs, j): (Vec<f32>, f32)) {
        type Data = Rc<RefCell<[f32; (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize]>>;

        lua_assert!(bs.len() % 4 == 0, "blob coordinate count must be a multiple of 4");
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
    fn open(arg, _, _, data: Data) -> setrange {
        let [mn, mx] = arg.bbox;
        let mut data = data.borrow_mut();
        for z in mn.z..mx.z {
            for y in mn.y..mx.y {
                for x in mn.x..mx.x {
                    let ipos = Int3::new([x, y, z]);
                    data[ipos.to_index(CHUNK_BITS)] = 0.;
                }
            }
        }
    }
    fn blob(arg, _, _, data: Data, u, k, c) {
        let [mn, mx] = arg.bbox;
        let u = u + arg.origin.to_f32();
        let mut data = data.borrow_mut();
        for z in mn.z..mx.z {
            for y in mn.y..mx.y {
                for x in mn.x..mx.x {
                    let pos = Vec3::new(x as f32, y as f32, z as f32);
                    let f = invsqrt((pos - u).mag_sq()).mul_add(k, c);
                    if f > 0. {
                        let ipos = Int3::new([x, y, z]);
                        data[ipos.to_index(CHUNK_BITS)] += f;
                    }
                }
            }
        }
    }
    fn close(arg, state, _, data: Data) {
        let data = data.borrow_mut();
        shape!(arg, state, p, data[(p + arg.origin).to_index(CHUNK_BITS)] >= 1.);
    }
}

action! {
    // draw a "cloud", basically a bunch of small spheres arranged in a sphere pattern.
    // `x, y, z`: center of the cloudsphere
    // `r`: radius of the cloudsphere
    // `r0, r1`: range of radii of the subspheres
    // `seed`: random seed for the subsphere sizes
    // `n`: number of subspheres
    fn cloud((x, y, z, r, r0, r1, seed, n): (f32, f32, f32, f32, f32, f32, i64, u32)) {
        let u = Vec3::new(x, y, z);
        let mut rng = FastRng::seed_from_u64(seed as u64);
        let dr = r1 - r0;
    }
    fn place() {
        let gmn = Int3::from_f32((u - Vec3::broadcast(r + r1)).map(f32::floor));
        let gmx = Int3::from_f32((u + Vec3::broadcast(r + r1)).map(f32::floor)) + [1; 3];
        clear([gmn, gmx], ());

        let sph = |u: Vec3, r: f32| {
            let mn = Int3::from_f32((u - Vec3::broadcast(r)).map(f32::floor));
            let mx = Int3::from_f32((u + Vec3::broadcast(r)).map(f32::floor)) + [1; 3];
            subsphere([mn, mx], u, r * r);
        };

        let phi = f32::PI * (3. - 5f32.sqrt());
        let dz = 2. * r / n as f32;
        let mut theta = 0f32;
        let mut z = -r + dz * 0.36; // this offset optimizes average distance to neighbor
        sph(u, r);
        for _ in 0..n {
            let rxy = (r * r - z * z).sqrt();
            let u = u + Vec3::new(theta.cos() * rxy, theta.sin() * rxy, z);
            let r = r0 + rng.gen::<f32>() * dr;
            sph(u, r);
            theta += phi;
            z += dz;
        }
    }
    fn clear(arg, state, _, ()) -> setrange {
        shape!(arg, state, _, false);
    }
    fn subsphere(arg, state, _, u, r2) {
        // OPTIMIZE: Use sqrt and math to figure out exactly where does this row start and
        // end, and write all the row blocks in one go.
        shape!([add] arg, state, p, (p.to_f32() - u).mag_sq() <= r2);
    }
}

action! {
    // fill shape with a uniform block.
    // the block may be special.
    fn solid((b): i16) {
        let b = BlockData { data: b };
    }
    fn paint(arg, state, chunk) {
        paint!(arg, state, chunk, _p, b);
    }
}

action! {
    // fill shape with a color shifted by uniform value noise.
    // `br, bg, bb`: average RGB color.
    // `dr, dg, db`: standard deviation of RGB color.
    // `seed`: optional seed for the noise. defaults to zero.
    fn noisy((br, bg, bb, dr, dg, db, seed): (f32, f32, f32, f32, f32, f32, Option<i64>)) {
        let seed = seed.unwrap_or(0);
        let x = Vec3::new(br, bg, bb);
        let r = Vec3::new(dr, dg, db) * 3f32.sqrt();

        let x = x - r;
        let r = 2. * r;
    }
    fn paint(arg, state, chunk) {
        let mut rng = FastRng::seed_from_u64(fxhash::hash64(&(seed, arg.chunkpos)));
        paint!(arg, state, chunk, _p, {
            getblock(x + r * rng.gen::<f32>())
        });
    }
}

action! {
    // fill shape with a flat gradient from 1 point to another point.
    fn gradient((x0, y0, z0, r0, g0, b0, x1, y1, z1, r1, g1, b1): (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)) {
        let u0 = Vec3::new(x0, y0, z0);
        let c0 = Vec3::new(r0, g0, b0).map(|r| r.powf(2.2));
        let u1 = Vec3::new(x1, y1, z1);
        let c1 = Vec3::new(r1, g1, b1).map(|r| r.powf(2.2));
        let n = u1 - u0;
        let inv = n.mag_sq().recip();
        let dc = c1 - c0;
    }
    fn paint(arg, state, chunk) {
        paint!(arg, state, chunk, p, {
            let p = p.to_f32() - u0;
            let s = (n.dot(p) * inv).min(1.).max(0.);
            let c = c0 + s * dc;
            let c = c.map(|r| r.powf(1./2.2)); // linear RGB to sRGB conversion
            getblock(c)
        });
    }
}

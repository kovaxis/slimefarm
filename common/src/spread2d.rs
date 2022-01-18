use crate::prelude::*;

fn random_float(salt: i32, x: [i32; 2]) -> f32 {
    let x = fxhash::hash32(&[x[0], x[1], salt]);
    let x = (x & 0x007fffff) | 0x3f800000;
    f32::from_bits(x) - 1.
}

fn random_pos(salt: [i32; 2], x: [i32; 2]) -> Vec2 {
    Vec2::new(random_float(salt[0], x), random_float(salt[1], x))
}

fn pos_to_idx(r: i32, pos: [i32; 2]) -> usize {
    let w = 2 * r + 1;
    (pos[0] + pos[1] * w + r * (w + 1)) as usize
}

fn recenter(grid: &[Vec2; 9], center: Vec2) -> Vec2 {
    const SEGS: usize = 7;
    const OMEGA: f32 = 1.4;
    let mut avg = Vec2::zero();
    let mut n = 0.;
    for iy in 0..=SEGS {
        for ix in 0..=SEGS {
            let pos = center
                + Vec2::new(
                    -1. + 3. / SEGS as f32 * ix as f32,
                    -1. + 3. / SEGS as f32 * iy as f32,
                );
            let mut idx = 0;
            let mut mdsq = 9999.;
            let mut midx = 0;
            for _y in -1..=1 {
                for _x in -1..=1 {
                    let dsq = (grid[idx] - pos).mag_sq();
                    if dsq < mdsq {
                        mdsq = dsq;
                        midx = idx;
                    }
                    idx += 1;
                }
            }
            if midx == 4 {
                avg += pos;
                n += 1.;
            }
        }
    }
    avg /= n;
    avg = grid[4] + (avg - grid[4]) * OMEGA;
    avg.clamped(center, center + Vec2::new(1., 1.))
}

/// Generate somewhat uniform yet random points, such that no two points are less than a set
/// distance away.
///
/// More specifically, generates one point per square of an infinite grid, such that every point may
/// be located anywhere inside that square.
pub struct Spread2d {
    salt: [i32; 2],
}
impl Spread2d {
    pub fn new(seed: u64) -> Self {
        Spread2d {
            salt: [
                fxhash::hash32(&(seed, 1)) as i32,
                fxhash::hash32(&(seed, 2)) as i32,
            ],
        }
    }

    /// Generate the point at the given grid square.
    /// The generated point is in the `[0, 1]^2` unit square, such that to obtain the absolute point
    /// position it must be added to the absolute integer `pos` coordinates.
    pub fn gen(&self, pos: Int2) -> Vec2 {
        let mut grid2 = [Vec2::zero(); 5 * 5];
        let mut idx = 0;
        for y in -2..=2 {
            for x in -2..=2 {
                grid2[idx] =
                    random_pos(self.salt, [pos.x + x, pos.y + y]) + Vec2::new(x as f32, y as f32);
                idx += 1;
            }
        }

        let mut grid1 = [Vec2::zero(); 3 * 3];
        let mut idx = 0;
        for y in -1..=1 {
            for x in -1..=1 {
                let mut subgrid = [Vec2::zero(); 9];
                let mut sidx = 0;
                for sy in -1..=1 {
                    for sx in -1..=1 {
                        subgrid[sidx] = grid2[pos_to_idx(2, [x + sx, y + sy])];
                        sidx += 1;
                    }
                }
                grid1[idx] = recenter(&subgrid, Vec2::new(x as f32, y as f32));
                idx += 1;
            }
        }

        let grid0 = recenter(&grid1, Vec2::zero());

        grid0
    }
}

#[test]
fn test_spread() {
    const SCALE: i32 = 8;
    const WIDTH: i32 = 8;
    let spread = Spread2d::new(0x172385);
    let mut fpoints = [Vec2::zero(); (WIDTH * WIDTH) as usize];
    let begin = Instant::now();
    let mut idx = 0;
    for y in 0..WIDTH {
        for x in 0..WIDTH {
            fpoints[idx] = spread.gen([x, y].into()) + Vec2::new(x as f32, y as f32);
            idx += 1;
        }
    }
    let time_taken = begin.elapsed();
    let mut points = [[0; 2]; (WIDTH * WIDTH) as usize];
    for (f, i) in fpoints.iter().zip(points.iter_mut()) {
        *i = [(f.x * SCALE as f32) as i32, (f.y * SCALE as f32) as i32];
    }
    let mut buf = [false; (WIDTH * SCALE * WIDTH * SCALE) as usize];
    for &[x, y] in &points {
        if x >= 0 && x < WIDTH * SCALE && y >= 0 && y < WIDTH * SCALE {
            buf[(x + WIDTH * SCALE * y) as usize] = true;
        }
    }
    let mut idx = 0;
    for _y in 0..WIDTH * SCALE {
        for _x in 0..WIDTH * SCALE {
            let c = if buf[idx] { 'X' } else { ' ' };
            print!("{}", c);
            idx += 1;
        }
        println!();
    }
    println!(
        "generated {} points in {}ms",
        WIDTH * WIDTH,
        time_taken.as_millis()
    );
}

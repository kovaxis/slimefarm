use crate::prelude::*;

const GRADIENT_COUNT: usize = 128;
const GRADIENTS: [[f32; 2]; GRADIENT_COUNT] = [
    [1.0, 0.0],
    [0.9987954562051724, 0.049067674327418015],
    [0.9951847266721969, 0.0980171403295606],
    [0.989176509964781, 0.14673047445536175],
    [0.9807852804032304, 0.19509032201612825],
    [0.970031253194544, 0.24298017990326387],
    [0.9569403357322088, 0.29028467725446233],
    [0.9415440651830208, 0.33688985339222005],
    [0.9238795325112867, 0.3826834323650898],
    [0.9039892931234433, 0.4275550934302821],
    [0.881921264348355, 0.47139673682599764],
    [0.8577286100002721, 0.5141027441932217],
    [0.8314696123025452, 0.5555702330196022],
    [0.8032075314806449, 0.5956993044924334],
    [0.773010453362737, 0.6343932841636455],
    [0.7409511253549592, 0.6715589548470183],
    [0.7071067811865476, 0.7071067811865476],
    [0.6715589548470183, 0.7409511253549591],
    [0.6343932841636455, 0.7730104533627369],
    [0.5956993044924335, 0.8032075314806448],
    [0.5555702330196023, 0.8314696123025452],
    [0.5141027441932217, 0.8577286100002721],
    [0.4713967368259978, 0.8819212643483549],
    [0.4275550934302822, 0.9039892931234433],
    [0.38268343236508984, 0.9238795325112867],
    [0.33688985339222005, 0.9415440651830208],
    [0.29028467725446233, 0.9569403357322089],
    [0.24298017990326398, 0.970031253194544],
    [0.19509032201612833, 0.9807852804032304],
    [0.14673047445536175, 0.989176509964781],
    [0.09801714032956077, 0.9951847266721968],
    [0.049067674327418126, 0.9987954562051724],
    [6.123233995736766e-17, 1.0],
    [-0.04906767432741801, 0.9987954562051724],
    [-0.09801714032956065, 0.9951847266721969],
    [-0.14673047445536164, 0.989176509964781],
    [-0.1950903220161282, 0.9807852804032304],
    [-0.24298017990326387, 0.970031253194544],
    [-0.29028467725446216, 0.9569403357322089],
    [-0.33688985339221994, 0.9415440651830208],
    [-0.3826834323650897, 0.9238795325112867],
    [-0.42755509343028186, 0.9039892931234434],
    [-0.4713967368259977, 0.881921264348355],
    [-0.5141027441932216, 0.8577286100002721],
    [-0.555570233019602, 0.8314696123025453],
    [-0.5956993044924334, 0.8032075314806449],
    [-0.6343932841636454, 0.7730104533627371],
    [-0.6715589548470184, 0.740951125354959],
    [-0.7071067811865475, 0.7071067811865476],
    [-0.7409511253549589, 0.6715589548470186],
    [-0.773010453362737, 0.6343932841636455],
    [-0.8032075314806448, 0.5956993044924335],
    [-0.8314696123025453, 0.5555702330196022],
    [-0.857728610000272, 0.5141027441932218],
    [-0.8819212643483549, 0.4713967368259978],
    [-0.9039892931234433, 0.42755509343028203],
    [-0.9238795325112867, 0.3826834323650899],
    [-0.9415440651830207, 0.33688985339222033],
    [-0.9569403357322088, 0.2902846772544624],
    [-0.970031253194544, 0.24298017990326407],
    [-0.9807852804032304, 0.1950903220161286],
    [-0.989176509964781, 0.1467304744553618],
    [-0.9951847266721968, 0.09801714032956083],
    [-0.9987954562051724, 0.049067674327417966],
    [-1.0, 1.2246467991473532e-16],
    [-0.9987954562051724, -0.049067674327417724],
    [-0.9951847266721969, -0.09801714032956059],
    [-0.989176509964781, -0.14673047445536158],
    [-0.9807852804032304, -0.19509032201612836],
    [-0.970031253194544, -0.24298017990326382],
    [-0.9569403357322089, -0.2902846772544621],
    [-0.9415440651830208, -0.3368898533922201],
    [-0.9238795325112868, -0.38268343236508967],
    [-0.9039892931234434, -0.4275550934302818],
    [-0.881921264348355, -0.47139673682599764],
    [-0.8577286100002721, -0.5141027441932216],
    [-0.8314696123025455, -0.555570233019602],
    [-0.8032075314806449, -0.5956993044924332],
    [-0.7730104533627371, -0.6343932841636453],
    [-0.7409511253549591, -0.6715589548470184],
    [-0.7071067811865477, -0.7071067811865475],
    [-0.6715589548470187, -0.7409511253549589],
    [-0.6343932841636459, -0.7730104533627367],
    [-0.5956993044924331, -0.803207531480645],
    [-0.5555702330196022, -0.8314696123025452],
    [-0.5141027441932218, -0.857728610000272],
    [-0.47139673682599786, -0.8819212643483549],
    [-0.4275550934302825, -0.9039892931234431],
    [-0.38268343236509034, -0.9238795325112865],
    [-0.33688985339221994, -0.9415440651830208],
    [-0.29028467725446244, -0.9569403357322088],
    [-0.24298017990326412, -0.970031253194544],
    [-0.19509032201612866, -0.9807852804032303],
    [-0.1467304744553623, -0.9891765099647809],
    [-0.09801714032956045, -0.9951847266721969],
    [-0.04906767432741803, -0.9987954562051724],
    [-1.8369701987210297e-16, -1.0],
    [0.04906767432741766, -0.9987954562051724],
    [0.09801714032956009, -0.9951847266721969],
    [0.14673047445536194, -0.9891765099647809],
    [0.1950903220161283, -0.9807852804032304],
    [0.24298017990326376, -0.970031253194544],
    [0.29028467725446205, -0.9569403357322089],
    [0.3368898533922196, -0.9415440651830209],
    [0.38268343236509, -0.9238795325112866],
    [0.42755509343028214, -0.9039892931234433],
    [0.4713967368259976, -0.881921264348355],
    [0.5141027441932216, -0.8577286100002722],
    [0.5555702330196018, -0.8314696123025455],
    [0.5956993044924328, -0.8032075314806453],
    [0.6343932841636456, -0.7730104533627369],
    [0.6715589548470183, -0.7409511253549591],
    [0.7071067811865474, -0.7071067811865477],
    [0.7409511253549589, -0.6715589548470187],
    [0.7730104533627365, -0.6343932841636459],
    [0.803207531480645, -0.5956993044924332],
    [0.8314696123025452, -0.5555702330196022],
    [0.857728610000272, -0.5141027441932219],
    [0.8819212643483548, -0.4713967368259979],
    [0.9039892931234431, -0.42755509343028253],
    [0.9238795325112865, -0.3826834323650904],
    [0.9415440651830208, -0.33688985339222],
    [0.9569403357322088, -0.2902846772544625],
    [0.970031253194544, -0.24298017990326418],
    [0.9807852804032303, -0.19509032201612872],
    [0.9891765099647809, -0.1467304744553624],
    [0.9951847266721969, -0.0980171403295605],
    [0.9987954562051724, -0.04906767432741809],
];

fn gradient_for(coord: [i32; 2], salt: u32) -> Vec2 {
    GRADIENTS[fxhash::hash32(&(coord, salt)) as usize % GRADIENT_COUNT].into()
}

fn calc_dot(grad: Vec2, offset: [f32; 2], frac: [f32; 2]) -> f32 {
    grad.dot(Vec2::from(frac) - Vec2::new(offset[0] as f32, offset[1] as f32))
}

fn smooth(x: f32) -> f32 {
    (x * x) * x.mul_add(-2., 3.)
}

struct PerlinLayer {
    offset: [f64; 2],
    salt: u32,
    freq: f64,
    scale: f32,
}
impl PerlinLayer {
    /// Generates a cubic block of noise of side length `size` with its origin corner at `pos`.
    /// `out_size` points per side of the cube are sampled and their noise values placed in `out`.
    ///
    /// Note that if `generate_edges` is `true`, the first and last point correspond to the edges
    /// of the cube.
    /// Therefore the spacing between samples is `size / (out_size - 1)`, not `size / out_size`.
    pub fn noise_block(
        &self,
        pos: [f64; 2],
        size: f64,
        out_size: i32,
        out: &mut [f32],
        generate_edges: bool,
    ) {
        assert_eq!(
            (out_size * out_size) as usize,
            out.len(),
            "invalid noise output buffer size"
        );
        //Scale positions and sizes to natural perlin units, where 1 unit is one perlin block,
        //and all blocks are aligned to the origin.
        let base_pos = [
            pos[0].mul_add(self.freq, self.offset[0]),
            pos[1].mul_add(self.freq, self.offset[1]),
        ];
        let size = size * self.freq;
        //Start at the initial perlin block
        let base_block = [base_pos[0].floor() as i32, base_pos[1].floor() as i32];
        let mut cur_pos_int = [0, 0];
        let mut cur_block = base_block;
        let mut block_pos_frac = [
            (base_pos[0] - base_block[0] as f64) as f32,
            (base_pos[1] - base_block[1] as f64) as f32,
        ];
        let mut cur_block_base_pos = [0, 0];
        let calc_grads = |cur_block: [i32; 2], x_offset| {
            [
                gradient_for([cur_block[0] + x_offset, cur_block[1]], self.salt),
                gradient_for([cur_block[0] + x_offset, cur_block[1] + 1], self.salt),
            ]
        };
        let effective_out_size = if generate_edges {
            out_size - 1
        } else {
            out_size
        } as f64;
        let calc_frac = |base_pos: f64, cur_block: i32, cur_pos_int: i32| {
            // current sample pos - base block pos
            // (absolute base pos + current sample offset) - base block pos
            // (absolute base pos + (current sample integer position * sample spacing)) - base block pos
            (base_pos - cur_block as f64 + cur_pos_int as f64 * (size / effective_out_size)) as f32
        };
        let mut last_2_grads = calc_grads(cur_block, 0);
        let mut new_2_grads = calc_grads(cur_block, 1);
        loop {
            //Calculate a single noise value
            {
                let sx = smooth(block_pos_frac[0]);
                let sy = smooth(block_pos_frac[1]);
                macro_rules! calc_dot {
                    (last[$grad_n:expr], $x:expr, $y:expr) => {{
                        calc_dot(
                            last_2_grads[$grad_n],
                            [$x as f32, $y as f32],
                            block_pos_frac,
                        )
                    }};
                    (new[$grad_n:expr], $x:expr, $y:expr) => {{
                        calc_dot(new_2_grads[$grad_n], [$x as f32, $y as f32], block_pos_frac)
                    }};
                }
                let noise = Lerp::lerp(
                    &Lerp::lerp(&calc_dot!(last[0], 0, 0), calc_dot!(last[1], 0, 1), sy),
                    Lerp::lerp(&calc_dot!(new[0], 1, 0), calc_dot!(new[1], 1, 1), sy),
                    sx,
                );
                let out = &mut out[(cur_pos_int[0] + cur_pos_int[1] * out_size) as usize];
                *out = noise.mul_add(self.scale, *out);
            }
            //Advance on the X axis
            cur_pos_int[0] += 1;
            let next_x = cur_pos_int[0];
            block_pos_frac[0] = calc_frac(base_pos[0], cur_block[0], cur_pos_int[0]);
            if block_pos_frac[0] >= 1. || cur_pos_int[0] >= out_size {
                //Reached the end of this block on the X axis
                cur_pos_int[0] = cur_block_base_pos[0];
                block_pos_frac[0] = calc_frac(base_pos[0], cur_block[0], cur_pos_int[0]);
                //Advance on the Y axis
                cur_pos_int[1] += 1;
                let next_y = cur_pos_int[1];
                block_pos_frac[1] = calc_frac(base_pos[1], cur_block[1], cur_pos_int[1]);
                if block_pos_frac[1] >= 1. || cur_pos_int[1] >= out_size {
                    //Reached the end of this block on the Y axis
                    //Advance a block
                    if next_x < out_size {
                        //Advance to the next block in the X axis
                        cur_block[0] += 1;
                        cur_block_base_pos[0] = next_x;
                        last_2_grads = new_2_grads;
                    } else if next_y < out_size {
                        //Advance to the next block in the Y axis
                        cur_block[0] = base_block[0];
                        cur_block[1] += 1;
                        cur_block_base_pos[0] = 0;
                        cur_block_base_pos[1] = next_y;
                        last_2_grads = calc_grads(cur_block, 0);
                    } else {
                        //Exhausted all blocks. Done
                        break;
                    }
                    new_2_grads = calc_grads(cur_block, 1);
                    cur_pos_int = cur_block_base_pos;
                    block_pos_frac = [
                        calc_frac(base_pos[0], cur_block[0], cur_pos_int[0]),
                        calc_frac(base_pos[1], cur_block[1], cur_pos_int[1]),
                    ];
                }
            }
        }
    }
}

pub struct Noise2d {
    octaves: Vec<PerlinLayer>,
}
impl Noise2d {
    pub fn new(seed: u64, layers: &[(f64, f32)]) -> Self {
        let mut rng = FastRng::seed_from_u64(seed);
        Self {
            octaves: layers
                .iter()
                .map(|&(period, scale)| PerlinLayer {
                    offset: rng.gen(),
                    salt: rng.gen(),
                    freq: period.recip(),
                    scale,
                })
                .collect(),
        }
    }

    pub fn new_octaves(seed: u64, period: f64, octs: i32) -> Self {
        let mut rng = FastRng::seed_from_u64(seed);
        let mut freq = period.recip();
        let mut ampl = 1.;
        Self {
            octaves: (0..octs)
                .map(|_| {
                    let layer = PerlinLayer {
                        offset: rng.gen(),
                        salt: rng.gen(),
                        freq,
                        scale: ampl,
                    };
                    freq *= 2.;
                    ampl /= 2.;
                    layer
                })
                .collect(),
        }
    }

    /// Generates a cubic grid of noise sampled from a cubic grid.
    /// The cube has its origin at `pos` and side length `size`.
    /// The output grid has `out_size` samples per side.
    /// `out` must have length `out_size.pow(3)`.
    ///
    /// `generate_edges` will sample the edges too.
    /// For example, if taking 3 samples (`out_size == 3`):
    ///
    /// `generate_edges`: `false`
    /// |----------------------------------------------------|
    ///  v                 v                 v
    ///  Sample            Sample            Sample
    ///
    /// `generate_edges`: `true`
    /// |----------------------------------------------------|
    ///  v                         v                        v
    ///  Sample                    Sample                   Sample
    pub fn noise_block(
        &self,
        pos: [f64; 2],
        size: f64,
        out_size: i32,
        out: &mut [f32],
        generate_edges: bool,
    ) {
        for out in out.iter_mut() {
            *out = 0.;
        }
        for oct in self.octaves.iter() {
            oct.noise_block(pos, size, out_size, out, generate_edges);
        }
    }
}

/// Generates chunks of noise and scales it to a virtual coordinate space.
pub struct NoiseScaler2d {
    inner_size: i32,
    inner_per_outer: f32,
    buf: Vec<f32>,
}
impl NoiseScaler2d {
    pub fn new(inner: i32, outer: f32) -> Self {
        //Need to add 1 in order to account for the edges of the inner noise cube
        let inner_size = inner + 1;
        Self {
            inner_size,
            inner_per_outer: inner as f32 / outer,
            buf: vec![0.; (inner_size * inner_size * inner_size) as usize],
        }
    }

    /// Size is usually set to the same value as the `outer` arg to `Self::new`, but it doesn't
    /// _have_ to be identical.
    pub fn fill(&mut self, noise_gen: &Noise2d, pos: [f64; 2], size: f64) {
        noise_gen.noise_block(pos, size, self.inner_size, &mut self.buf, true);
    }

    fn raw_get(&self, pos: [i32; 2]) -> f32 {
        self.buf[(pos[0] + pos[1] * self.inner_size) as usize]
    }

    /// Pos should be within the outer size, otherwise panics and weird stuff may ensue.
    pub fn get(&self, pos: Vec2) -> f32 {
        let pos = pos * self.inner_per_outer;
        let base = [pos.x.floor() as i32, pos.y.floor() as i32];
        let frac = pos - Vec2::new(base[0] as f32, base[1] as f32);

        let noise = |x, y| self.raw_get([base[0] + x, base[1] + y]);
        Lerp::lerp(
            &Lerp::lerp(&noise(0, 0), noise(1, 0), frac[0]),
            Lerp::lerp(&noise(0, 1), noise(1, 1), frac[0]),
            frac[1],
        )
    }
}

use crate::prelude::*;

const GRADIENT_COUNT: usize = 128;
const GRADIENTS: [[f32; 3]; GRADIENT_COUNT] = [
    [0.16207897, -0.71325105, -0.68191154],
    [-0.064643306, 0.31292866, -0.94757422],
    [-0.46274136, 0.11220489, 0.87936369],
    [-0.99097262, 0.12996229, -0.032909928],
    [-0.4907832, -0.64637274, 0.58423808],
    [0.37202912, 0.66719118, -0.64532957],
    [-0.31736009, -0.35718845, -0.87846399],
    [0.82075001, -0.186547, 0.53997188],
    [0.60231729, -0.19461853, 0.77416891],
    [-0.57594218, -0.81314005, 0.084225146],
    [-0.76768403, 0.47611731, 0.42892137],
    [0.439462, 0.38551002, -0.81132927],
    [0.153859, 0.52874322, -0.83472032],
    [0.68689139, -0.47816242, 0.54730331],
    [0.64464113, 0.52440429, -0.55627148],
    [0.2631982, -0.94980534, -0.16910508],
    [0.086537147, -0.8980254, -0.4313487],
    [0.55431439, 0.34400081, 0.75789115],
    [-0.18078149, -0.79708482, -0.57617172],
    [0.26042393, -0.1407757, -0.9551762],
    [-0.03372666, 0.79827303, -0.60135071],
    [-0.85612054, -0.4469672, -0.25937994],
    [-0.19009012, -0.73049422, 0.65592983],
    [-0.84917317, -0.13128721, 0.51153553],
    [0.10280949, -0.98910037, 0.10540716],
    [0.31287648, -0.23085307, 0.92131166],
    [-0.62534725, 0.75802464, -0.18530909],
    [0.23296967, 0.86816389, -0.43819698],
    [0.55474604, 0.60460521, 0.57158497],
    [0.7806305, 0.39477915, 0.48452601],
    [0.64724933, 0.76164353, -0.031103794],
    [-0.72825519, 0.66338116, 0.17195878],
    [-0.53764451, 0.69311772, 0.48013145],
    [0.92876385, 0.35531707, -0.10558166],
    [0.22715766, -0.894494, 0.38507127],
    [-0.37115177, 0.9247446, -0.084224636],
    [0.51708898, -0.26892154, -0.81258857],
    [0.47161229, -0.59607765, -0.64982558],
    [-0.46420933, 0.52192668, -0.71561319],
    [-0.96088605, -0.093182523, -0.26079689],
    [0.13189617, 0.98719732, 0.089693141],
    [-0.67177957, -0.54402232, -0.5027444],
    [-0.70642073, 0.18290915, 0.68374995],
    [-0.54653497, 0.46005625, 0.69974836],
    [-0.06914212, -0.98389083, -0.1648581],
    [0.39543665, -0.81322675, -0.42695679],
    [0.7091957, -0.39058735, -0.58692672],
    [-0.31407089, -0.16610951, 0.93475511],
    [0.27037349, 0.32740826, 0.90537396],
    [-0.49008756, 0.73122408, -0.47447395],
    [-0.1587254, 0.1101962, 0.98115394],
    [0.031084605, 0.97629857, -0.21418416],
    [-0.8534182, 0.066323819, -0.51698986],
    [-0.41534981, -0.59529281, -0.68783065],
    [0.43610601, -0.71408441, 0.5476267],
    [-0.2166154, -0.96896019, 0.11913824],
    [-0.29104478, 0.38265908, 0.87684945],
    [0.65667873, 0.69908065, 0.2829475],
    [0.26168706, -0.4588044, -0.84912802],
    [-0.6426624, -0.12254471, 0.75628555],
    [-0.95883249, -0.28395145, -0.0034394621],
    [0.98693089, 0.1216346, 0.10569978],
    [0.86141681, 0.47316242, 0.18460338],
    [-0.47426522, 0.85474959, 0.21089249],
    [-0.23558595, 0.89734795, -0.37318348],
    [0.97780704, -0.19828725, -0.067642885],
    [0.86292619, -0.45202153, -0.22590911],
    [-0.82139424, -0.25238111, -0.51148341],
    [0.28495494, 0.60396874, 0.74432684],
    [0.50296924, 0.047120976, -0.86301886],
    [-0.97259661, -0.059521937, 0.22475091],
    [-0.0040964954, -0.23833624, 0.97117406],
    [0.5547666, -0.81874378, -0.14796162],
    [-0.33610363, -0.043509631, -0.94081946],
    [0.84477057, -0.46301689, 0.26828725],
    [-0.8213397, -0.56816565, 0.050881092],
    [0.52702499, 0.77225751, -0.35477176],
    [-0.6784691, -0.70686124, -0.20006713],
    [-0.28928748, 0.65097271, 0.70181713],
    [-0.90915297, 0.29219505, -0.29675399],
    [-0.26069066, 0.85199303, 0.45403552],
    [-0.71287379, -0.41636444, 0.56431517],
    [0.7687178, 0.58143664, -0.26646649],
    [0.63993198, -0.70721435, 0.30055768],
    [0.85296801, 0.31221344, -0.41829217],
    [-0.47824124, -0.39610556, 0.78382759],
    [0.6912748, 0.23415387, -0.68360158],
    [-0.47352517, 0.23122263, -0.84988823],
    [0.43265917, -0.89072565, 0.1393336],
    [0.66211957, -0.63945298, -0.39076535],
    [-0.83891258, 0.53501301, -0.099933826],
    [-0.73784807, 0.54079646, -0.40388043],
    [0.046060637, 0.91933395, 0.39077296],
    [0.0055833701, 0.5029782, 0.86428106],
    [-0.38105179, -0.91232083, -0.14990077],
    [-0.68859759, -0.64454638, 0.33225491],
    [0.76696826, -0.64153449, 0.013899355],
    [0.42755796, 0.88894184, 0.16424556],
    [-0.17859648, -0.48199137, 0.85778064],
    [0.23073275, 0.18909763, -0.95446555],
    [-0.88174248, -0.36698876, 0.29639406],
    [0.34345756, 0.92813972, -0.14350459],
    [0.73658243, -0.082750628, -0.67126646],
    [-0.16031473, 0.57997368, -0.79870503],
    [-0.092071838, -0.90827818, 0.40810969],
    [-0.0074581097, -0.31636886, -0.94860694],
    [0.012494276, 0.75038773, 0.66087983],
    [0.45677838, 0.067517782, 0.88701458],
    [-0.70164512, 0.32237695, -0.63542681],
    [-0.10491406, -0.58515679, -0.80410483],
    [0.8905555, -0.19273428, -0.41202477],
    [-0.91421201, 0.37040077, 0.16437659],
    [0.72712755, 0.085393872, 0.68117062],
    [-0.59955789, -0.32943201, -0.72938665],
    [-0.40106209, -0.84631226, 0.35057776],
    [-0.1783324, 0.96958308, 0.16764906],
    [-0.88920911, 0.17981932, 0.42068061],
    [0.44598667, -0.47384784, 0.75931819],
    [0.12983125, -0.74684601, 0.65220004],
    [-0.65528629, -0.015013823, -0.7552314],
    [-0.46383449, -0.77586121, -0.42766453],
    [-0.024149011, -0.00055302137, -0.99970822],
    [0.90360676, 0.10732291, 0.41470062],
    [0.34079459, 0.81751676, 0.46424713],
    [0.95452702, -0.16261378, 0.24986982],
    [0.14128203, -0.49908483, 0.85495831],
    [0.15377149, 0.039952746, 0.98729839],
    [0.9673277, 0.068980415, -0.24396479],
];

fn gradient_for(coord: [i32; 3], salt: u32) -> Vec3 {
    GRADIENTS[fxhash::hash32(&(coord, salt)) as usize % GRADIENT_COUNT].into()
}

fn calc_dot(grad: Vec3, offset: [f32; 3], frac: [f32; 3]) -> f32 {
    grad.dot(Vec3::from(frac) - Vec3::new(offset[0] as f32, offset[1] as f32, offset[2] as f32))
}

fn smooth(x: f32) -> f32 {
    (x * x) * x.mul_add(-2., 3.)
}

struct PerlinLayer {
    offset: [f64; 3],
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
        pos: [f64; 3],
        size: f64,
        out_size: i32,
        out: &mut [f32],
        generate_edges: bool,
    ) {
        assert_eq!(
            (out_size * out_size * out_size) as usize,
            out.len(),
            "invalid noise output buffer size"
        );
        //Scale positions and sizes to natural perlin units, where 1 unit is one perlin block,
        //and all blocks are aligned to the origin.
        let base_pos = [
            pos[0].mul_add(self.freq, self.offset[0]),
            pos[1].mul_add(self.freq, self.offset[1]),
            pos[2].mul_add(self.freq, self.offset[2]),
        ];
        let size = size * self.freq;
        //Start at the initial perlin block
        let base_block = [
            base_pos[0].floor() as i32,
            base_pos[1].floor() as i32,
            base_pos[2].floor() as i32,
        ];
        let mut cur_pos_int = [0, 0, 0];
        let mut cur_block = base_block;
        let mut block_pos_frac = [
            (base_pos[0] - base_block[0] as f64) as f32,
            (base_pos[1] - base_block[1] as f64) as f32,
            (base_pos[2] - base_block[2] as f64) as f32,
        ];
        let mut cur_block_base_pos = [0, 0, 0];
        let calc_grads = |cur_block: [i32; 3], x_offset| {
            [
                gradient_for(
                    [cur_block[0] + x_offset, cur_block[1], cur_block[2]],
                    self.salt,
                ),
                gradient_for(
                    [cur_block[0] + x_offset, cur_block[1] + 1, cur_block[2]],
                    self.salt,
                ),
                gradient_for(
                    [cur_block[0] + x_offset, cur_block[1], cur_block[2] + 1],
                    self.salt,
                ),
                gradient_for(
                    [cur_block[0] + x_offset, cur_block[1] + 1, cur_block[2] + 1],
                    self.salt,
                ),
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
        let mut last_4_grads = calc_grads(cur_block, 0);
        let mut new_4_grads = calc_grads(cur_block, 1);
        loop {
            //Calculate a single noise value
            {
                let sx = smooth(block_pos_frac[0]);
                let sy = smooth(block_pos_frac[1]);
                let sz = smooth(block_pos_frac[2]);
                macro_rules! calc_dot {
                    (last[$grad_n:expr], $x:expr, $y:expr, $z:expr) => {{
                        calc_dot(
                            last_4_grads[$grad_n],
                            [$x as f32, $y as f32, $z as f32],
                            block_pos_frac,
                        )
                    }};
                    (new[$grad_n:expr], $x:expr, $y:expr, $z:expr) => {{
                        calc_dot(
                            new_4_grads[$grad_n],
                            [$x as f32, $y as f32, $z as f32],
                            block_pos_frac,
                        )
                    }};
                }
                let noise = Lerp::lerp(
                    &Lerp::lerp(
                        &Lerp::lerp(
                            &calc_dot!(last[0], 0, 0, 0),
                            calc_dot!(last[1], 0, 1, 0),
                            sy,
                        ),
                        Lerp::lerp(
                            &calc_dot!(last[2], 0, 0, 1),
                            calc_dot!(last[3], 0, 1, 1),
                            sy,
                        ),
                        sz,
                    ),
                    Lerp::lerp(
                        &Lerp::lerp(&calc_dot!(new[0], 1, 0, 0), calc_dot!(new[1], 1, 1, 0), sy),
                        Lerp::lerp(&calc_dot!(new[2], 1, 0, 1), calc_dot!(new[3], 1, 1, 1), sy),
                        sz,
                    ),
                    sx,
                );
                let out = &mut out[(cur_pos_int[0]
                    + cur_pos_int[1] * out_size
                    + cur_pos_int[2] * (out_size * out_size))
                    as usize];
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
                    cur_pos_int[1] = cur_block_base_pos[1];
                    block_pos_frac[1] = calc_frac(base_pos[1], cur_block[1], cur_pos_int[1]);
                    //Advance the Z axis
                    cur_pos_int[2] += 1;
                    block_pos_frac[2] = calc_frac(base_pos[2], cur_block[2], cur_pos_int[2]);
                    if block_pos_frac[2] >= 1. || cur_pos_int[2] >= out_size {
                        //Reached the end of this block on the Z axis
                        //Advance a block
                        if next_x < out_size {
                            //Advance to the next block in the X axis
                            cur_block[0] += 1;
                            cur_block_base_pos[0] = next_x;
                            last_4_grads = new_4_grads;
                        } else if next_y < out_size {
                            //Advance to the next block in the Y axis
                            cur_block[0] = base_block[0];
                            cur_block[1] += 1;
                            cur_block_base_pos[0] = 0;
                            cur_block_base_pos[1] = next_y;
                            last_4_grads = calc_grads(cur_block, 0);
                        } else if cur_pos_int[2] < out_size {
                            //Advance to the next block in the Z axis
                            cur_block[0] = base_block[0];
                            cur_block[1] = base_block[1];
                            cur_block[2] += 1;
                            cur_block_base_pos[0] = 0;
                            cur_block_base_pos[1] = 0;
                            cur_block_base_pos[2] = cur_pos_int[2];
                            last_4_grads = calc_grads(cur_block, 0);
                        } else {
                            //Exhausted all blocks. Done
                            break;
                        }
                        new_4_grads = calc_grads(cur_block, 1);
                        cur_pos_int = cur_block_base_pos;
                        block_pos_frac = [
                            calc_frac(base_pos[0], cur_block[0], cur_pos_int[0]),
                            calc_frac(base_pos[1], cur_block[1], cur_pos_int[1]),
                            calc_frac(base_pos[2], cur_block[2], cur_pos_int[2]),
                        ];
                    }
                }
            }
        }
    }
}

pub struct Noise3d {
    octaves: Vec<PerlinLayer>,
}
impl Noise3d {
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
        pos: [f64; 3],
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
pub struct NoiseScaler3d {
    inner_size: i32,
    inner_per_outer: f32,
    buf: Vec<f32>,
}
impl NoiseScaler3d {
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
    pub fn fill(&mut self, noise_gen: &Noise3d, pos: [f64; 3], size: f64) {
        noise_gen.noise_block(pos, size, self.inner_size, &mut self.buf, true);
    }

    fn raw_get(&self, pos: [i32; 3]) -> f32 {
        self.buf[(pos[0] + pos[1] * self.inner_size + pos[2] * (self.inner_size * self.inner_size))
            as usize]
    }

    /// Pos should be within the outer size, otherwise panics and weird stuff may ensue.
    pub fn get(&self, pos: Vec3) -> f32 {
        let pos = pos * self.inner_per_outer;
        let base = [
            pos.x.floor() as i32,
            pos.y.floor() as i32,
            pos.z.floor() as i32,
        ];
        let frac = pos - Vec3::new(base[0] as f32, base[1] as f32, base[2] as f32);

        let noise = |x, y, z| self.raw_get([base[0] + x, base[1] + y, base[2] + z]);
        Lerp::lerp(
            &Lerp::lerp(
                &Lerp::lerp(&noise(0, 0, 0), noise(1, 0, 0), frac[0]),
                Lerp::lerp(&noise(0, 1, 0), noise(1, 1, 0), frac[0]),
                frac[1],
            ),
            Lerp::lerp(
                &Lerp::lerp(&noise(0, 0, 1), noise(1, 0, 1), frac[0]),
                Lerp::lerp(&noise(0, 1, 1), noise(1, 1, 1), frac[0]),
                frac[1],
            ),
            frac[2],
        )
    }
}

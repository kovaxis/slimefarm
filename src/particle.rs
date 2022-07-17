use common::terrain::GridKeeper4;

use crate::{prelude::*, DynVertBuf};

pub(crate) struct ParticleSystem {
    state: Rc<State>,
    particles: Vec<Particle>,
    kinds: Vec<ParticleKind>,
    particle_pos: GridKeeper4<Vec<u32>>,
    to_draw: Vec<GpuParticle>,
    vertbuf: GpuBuffer<SimpleVertex>,
    instbuf: DynVertBuf<GpuParticle>,
}
impl ParticleSystem {
    pub(crate) fn new(state: &Rc<State>) -> Self {
        Self {
            particles: default(),
            kinds: default(),
            particle_pos: default(),
            to_draw: default(),
            vertbuf: {
                let mut mesh = Mesh::default();

                let mut quad = |u: [i32; 3], axes: [usize; 2]| {
                    let mut ax0 = Vec3::zero();
                    ax0[axes[0]] = 1.;
                    let mut ax1 = Vec3::zero();
                    ax1[axes[1]] = 1.;
                    let ax2 = ax0.cross(ax1);
                    let normal = [ax2.x as i8, ax2.y as i8, ax2.z as i8, 0];
                    let color = [255; 4];
                    let u = Int3::new(u).to_f32() * 0.5;
                    let v00 = mesh.add_vertex(SimpleVertex {
                        pos: u.into(),
                        normal,
                        color,
                    });
                    let v10 = mesh.add_vertex(SimpleVertex {
                        pos: (u + ax0).into(),
                        normal,
                        color,
                    });
                    let v11 = mesh.add_vertex(SimpleVertex {
                        pos: (u + ax0 + ax1).into(),
                        normal,
                        color,
                    });
                    let v01 = mesh.add_vertex(SimpleVertex {
                        pos: (u + ax1).into(),
                        normal,
                        color,
                    });
                    mesh.add_face(v00, v10, v11);
                    mesh.add_face(v00, v11, v01);
                };

                quad([-1, -1, -1], [0, 2]);
                quad([1, -1, -1], [1, 2]);
                quad([-1, 1, -1], [2, 0]);
                quad([-1, -1, -1], [2, 1]);
                quad([-1, -1, -1], [1, 0]);
                quad([-1, -1, 1], [0, 1]);

                mesh.make_buffer(&state.display)
            },
            instbuf: DynVertBuf::new(state),
            state: state.clone(),
        }
    }

    pub fn tick(&mut self, terrain: &Terrain, dt: f32) {
        for i in (0..self.particles.len()).rev() {
            let part = &mut self.particles[i];
            let kind = &self.kinds[part.kind as usize];
            if part.time >= kind.lifetime {
                self.particles.swap_remove(i);
                continue;
            }
            part.vel *= kind.friction.powf(dt);
            part.vel += kind.acc * dt;
            let delta = part.vel * dt;
            let delta = [delta.x as f64, delta.y as f64, delta.z as f64];
            terrain.boxcast(
                &mut part.pos,
                delta,
                [kind.physical_rad as f64; 3],
                kind.sticky,
            );
            part.rot_vel = Rotor3::identity().slerp(part.rot_vel, kind.rot_friction.powf(dt));
            part.rot_vel = Rotor3::identity().slerp(kind.rot_acc, dt) * part.rot_vel;
            part.rot_pos = Rotor3::identity().slerp(part.rot_vel, dt) * part.rot_pos;
            part.time += dt;
        }

        for list in self.particle_pos.map.values_mut() {
            list.item.clear();
        }

        for (idx, part) in self.particles.iter_mut().enumerate() {
            let kind = &self.kinds[part.kind as usize];
            let size = kind.size.get(part.time);
            let particle_pos = &mut self.particle_pos;
            terrain.get_equivalent_positions(
                part.pos,
                [(size * f32::sqrt(3.)) as f64; 3],
                |_i, jmp| {
                    let mut pos = part.pos;
                    pos.coords[0] += jmp.coords.x as f64;
                    pos.coords[1] += jmp.coords.y as f64;
                    pos.coords[2] += jmp.coords.z as f64;
                    pos.dim = jmp.dim;
                    let cpos = pos.block_pos().block_to_chunk();
                    particle_pos.or_insert(cpos, Vec::new).push(idx as u32);
                },
            );
        }
    }

    pub fn queue_start(&mut self) {
        self.to_draw.clear();
    }

    pub fn queue(&mut self, cpos: ChunkPos, origin: WorldPos) {
        if let Some(list) = self.particle_pos.get(cpos) {
            for &idx in list {
                let part = &self.particles[idx as usize];
                let kind = &self.kinds[part.kind as usize];
                let color = kind.color.get(part.time);
                let size = kind.size.get(part.time);
                // Determine exact position
                let delta = [
                    part.pos.coords[0].rem_euclid(CHUNK_SIZE as f64),
                    part.pos.coords[1].rem_euclid(CHUNK_SIZE as f64),
                    part.pos.coords[2].rem_euclid(CHUNK_SIZE as f64),
                ];
                let mut pos = cpos.chunk_to_block().world_pos();
                pos.coords[0] += delta[0];
                pos.coords[1] += delta[1];
                pos.coords[2] += delta[2];
                let rpos = [
                    (pos.coords[0] - origin.coords[0]) as f32,
                    (pos.coords[1] - origin.coords[1]) as f32,
                    (pos.coords[2] - origin.coords[2]) as f32,
                ];
                // Add particle to draw list
                self.to_draw.push(GpuParticle {
                    ppos: rpos,
                    prot: part.rot_pos.into_matrix().into(),
                    pcol: [color.x as u8, color.y as u8, color.z as u8, kind.shininess],
                    psize: size,
                });
            }
        }
    }

    pub fn queue_draw(
        &mut self,
        program: &Program,
        uniforms: &crate::lua::gfx::UniformStorage,
        draw_parameters: &DrawParameters,
        frame: &mut Frame,
    ) {
        if self.to_draw.is_empty() {
            return;
        }
        self.instbuf.write(&self.state, &self.to_draw);
        frame
            .draw(
                (
                    &self.vertbuf.vertex,
                    self.instbuf.as_buf().per_instance().unwrap(),
                ),
                &self.vertbuf.index,
                program,
                uniforms,
                draw_parameters,
            )
            .unwrap();
    }

    pub fn add_kind(&mut self, kind: ParticleKind) -> u32 {
        let id = self.kinds.len() as u32;
        self.kinds.push(kind);
        id
    }

    pub fn add(&mut self, kind: u32, pos: WorldPos, vel: Vec3, rot_vel: Rotor3) {
        self.particles.push(Particle {
            pos,
            vel,
            rot_pos: Rotor3::identity(),
            rot_vel,
            time: 0.,
            kind,
        })
    }

    pub fn gc(&mut self) {
        self.particle_pos.gc();
    }
}

#[derive(Deserialize)]
pub struct ParticleKindCfg {
    pub friction: f32,
    pub acc: [f32; 3],
    pub rot_friction: f32,
    pub rot_acc: f32,
    pub rot_axis: [f32; 3],
    pub color_interval: f32,
    pub color: Vec<[f32; 3]>,
    pub size_interval: f32,
    pub size: Vec<f32>,
    pub physical_size: f32,
    pub lifetime: f32,
    pub sticky: bool,
    pub shininess: f32,
}
impl ParticleKindCfg {
    pub fn into_kind(self) -> ParticleKind {
        ParticleKind {
            friction: self.friction,
            acc: self.acc.into(),
            rot_friction: self.rot_friction,
            rot_acc: if self.rot_axis == [0.; 3] {
                Rotor3::identity()
            } else {
                Rotor3::from_angle_plane(
                    self.rot_acc,
                    Bivec3::from_normalized_axis(Vec3::from(self.rot_axis).normalized()),
                )
            },
            color: InterpList::new(
                self.color_interval,
                self.color
                    .into_iter()
                    .map(|c| Vec3::from(c) * 255.)
                    .collect(),
            ),
            size: InterpList::new(self.size_interval, self.size),
            physical_rad: self.physical_size / 2.,
            sticky: self.sticky,
            shininess: (self.shininess * 255.) as u8,
            lifetime: self.lifetime,
        }
    }
}

pub struct ParticleKind {
    pub friction: f32,
    pub acc: Vec3,
    pub rot_friction: f32,
    pub rot_acc: Rotor3,
    pub color: InterpList<Vec3>,
    pub size: InterpList<f32>,
    pub physical_rad: f32,
    pub sticky: bool,
    pub shininess: u8,
    pub lifetime: f32,
}

pub struct InterpList<T> {
    interval_inv: f32,
    list: Vec<T>,
}
impl<T> InterpList<T>
where
    T: Lerp<f32> + Copy,
{
    pub fn new(interval: f32, list: Vec<T>) -> Self {
        Self {
            interval_inv: interval.recip(),
            list,
        }
    }

    fn get(&self, t: f32) -> T {
        let t = t * self.interval_inv;
        let int = t as usize;
        let frac = t - int as f32;
        let s1 = self.list[int % self.list.len()];
        let s2 = self.list[(int + 1) % self.list.len()];
        s1.lerp(s2, frac)
    }
}

struct Particle {
    pos: WorldPos,
    vel: Vec3,
    rot_pos: Rotor3,
    rot_vel: Rotor3,
    time: f32,
    kind: u32,
}

#[derive(Copy, Clone)]
struct GpuParticle {
    ppos: [f32; 3],
    prot: [[f32; 3]; 3],
    pcol: [u8; 4],
    psize: f32,
}
implement_vertex!(GpuParticle,
    ppos normalize(false),
    prot normalize(false),
    pcol normalize(true),
    psize normalize(false)
);

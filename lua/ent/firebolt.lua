
local class = require 'class'
local Bullet = require 'ent.bullet'
local voxel = require 'voxel'
local particles = require 'particles'
local util = require 'util'

particles.register {
    name = 'firebolt.fire1',
    friction = .2,
    color_interval = 1,
    color = {{1, .20, .00}},
    size_interval = .5,
    size = {.15, 0},
    physical_size = .15,
    lifetime = .5,
}
particles.register {
    name = 'firebolt.fire2',
    friction = .2,
    color_interval = 1,
    color = {{1, .56, .00}},
    size_interval = .5,
    size = {.15, 0},
    physical_size = .15,
    lifetime = .5,
}

local Firebolt, super = class { super = Bullet }

Firebolt.model = voxel.models.firebolt
Firebolt.atk_hitbox = 6/8
Firebolt:set_bbox(2/8, 13/8)

Firebolt.particle_ids = { particles.lookup 'firebolt.fire1', particles.lookup 'firebolt.fire2' }
Firebolt.particle_per_tick = 20 / 64

Firebolt.death_particle_id = particles.lookup 'firebolt.fire1'
Firebolt.death_particle_vel = 6

function Firebolt:new()
    super.new(self)

    self.part_acc = 0
end

function Firebolt:tick(world)
    --Spawn fire particles
    self.part_acc = self.part_acc + self.particle_per_tick
    while self.part_acc >= 1 do
        self.part_acc = self.part_acc - 1

        --Spawn fire particle
        local rot_vel = 10
        local move_vel, noise = 5, .3
        local dx, dy, dz = util.normalize(self.vel_x, self.vel_y, self.vel_z, -1)
        dx = dx + world.rng:normal(-noise, noise)
        dy = dy + world.rng:normal(-noise, noise)
        dz = dz + world.rng:normal(-noise, noise)
        dx, dy, dz = util.normalize(dx, dy, dz, move_vel)
        local rx, ry, rz = world.rng:normal(-1, 1), world.rng:normal(-1, 1), world.rng:normal(-1, 1)
        local id = self.particle_ids[1 + world.rng:integer(#self.particle_ids)]
        world.terrain:add_particle(id, self.pos, dx, dy, dz, rot_vel, rx, ry, rz)
    end

    super.tick(self, world)
end

return Firebolt

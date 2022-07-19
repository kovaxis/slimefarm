
local voxel = require 'voxel'
local class = require 'class'
local util = require 'util'
local Slimy = require 'ent.slimy'
local entreg = require 'ent.reg'
local particles = require 'particles'
local Firebolt = require 'ent.firebolt'

local slimes = {}
local super = Slimy

local function wander(self, world)
    if world.rng:uniform() < 0.005 then
        local dx, dy = world.rng:normal(-1, 1), world.rng:normal(-1, 1)
        dx = dx - self.acc_x / self.wander_dist
        dy = dy - self.acc_y / self.wander_dist
        local n = dx * dx + dy * dy
        if n < 0.05 ^ 2 then
            dx, dy = 0, 0
        else
            n = n ^ -0.5
            dx, dy = dx * n, dy * n
        end
        self.wx, self.wy = dx, dy
    end
    self.wjump = world.rng:uniform() < 0.01
end

local function find_player(self, world)
    local dx, dy, dz, play = world:relative_player_pos(self.pos)
    local d = dx * dx + dy * dy + dz * dz
    if d < self.view_dist * self.view_dist then
        return d^.5, dx, dy, dz, play
    end
end

local Green = class{ super = Slimy }
slimes.Green = Green
Green:set_bbox(16/8, 11/8, 22/8)
Green.view_dist = 15
Green.max_hp = 100
Green.jump_charge = 5
Green.jump_hvel = 0.17
Green.jump_vvel = 0.5
Green.jump_keepup = 0.014
Green.jump_cooldown_land = 60
Green.death_particle_id = particles.lookup 'slimes.green.death'
Green.model = voxel.models.green_slime

particles.register(super.death_particle {
    name = 'slimes.green.death',
    color = {{0.8, 1.0, 0.459}},
})

entreg.register {
    name = 'GreenSlime',
    class = Green,
    fmt = {
        
    },
}

function Green:tick(world)
    local d, dx, dy, dz = find_player(self, world)
    if d then
        self.wx, self.wy = util.normalize2d(dx, dy, -1)
        self.wjump = true
    else
        wander(self, world)
    end

    return super.tick(self, world)
end

local Red = class{ super = Slimy }
slimes.Red = Red
Red:set_bbox(14/8, 12/8, 16/8)
Red.view_dist = 40
Red.shoot_dist = 20
Red.max_hp = 100
Red.atk_shooter = Firebolt:shooter('enemy')
Red.atk_height = 8/8
Red.atk_duration = 10
Red.atk_cooldown_duration = 20
Red.jump_hvel = 0.12
Red.jump_vvel = 0.38
Red.jump_keepup = 0.015
Red.jump_cooldown_land = 20
Red.death_particle_id = particles.lookup 'slimes.red.death'
Red.model = voxel.models.red_slime

particles.register(super.death_particle {
    name = 'slimes.red.death',
    color = {{0.918, 0.153, 0.0}},
})

entreg.register {
    name = 'RedSlime',
    class = Red,
    fmt = {
        
    },
}

function Red:tick(world)
    super.tick(self, world)

    self.watk_x, self.watk_y, self.watk_z = 0, 0, 0
    local d, dx, dy, dz, play = find_player(self, world)
    if d then
        local diff = d - self.shoot_dist
        if diff < -3 then
            self.wx, self.wy = util.normalize2d(dx, dy, -1)
            self.wjump = true
        elseif diff > 3 then
            self.wx, self.wy = util.normalize2d(dx, dy)
            self.wjump = true
        else
            self.wx, self.wy = 0, 0
            self.wjump = self.on_ground
        end
        if not self.on_ground and self.vel_z <= 0 then
            --Adjust for shooting height
            dz = dz - (self.atk_height - self.rad_z)
            --Adjust for player movement speed
            local buf = world.pos_buf
            local vx, vy, vz = play.vel_x, play.vel_y, play.vel_z
            local time = d / self.atk_vel
            vx, vy, vz = vx * time, vy * time, vz * time
            buf:copy_from(play.pos)
            vx, vy, vz = buf:move_box(world.terrain, vx, vy, vz, play.rad_x, play.rad_y, play.rad_z, true)
            dx, dy, dz = dx + vx, dy + vy, dz + vz
            self.watk_x, self.watk_y, self.watk_z = util.normalize(dx, dy, dz)
        end
    else
        wander(self, world)
    end
end

return slimes

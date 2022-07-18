
local voxel = require 'voxel'
local class = require 'class'
local input = require 'input'
local util = require 'util'
local Living = require 'ent.living'
local particles = require 'particles'

particles.register {
    name = 'slimy.attack',
    acc = {0, 0, -30},
    color_interval = 1,
    color = {{.92, 0.15, .01}},
    size_interval = 0.6,
    size = {0.3, 0},
    physical_size = 0.2,
    lifetime = 0.4,
}

local Slimy, super = class{ super = Living }

Slimy.air_maneuver = 0.011
Slimy.air_maneuver_max = 0.17

Slimy.jump_count = 1
Slimy.jump_charge = 5
Slimy.jump_hvel = 0.17
Slimy.jump_vvel = 0.5
Slimy.jump_keepup = 0.014
Slimy.jump_keepdown = 0.005
Slimy.jump_keepup_ticks = 14
Slimy.jump_cooldown_start = 10
Slimy.jump_cooldown_land = 60
Slimy.jump_cooldown_fudge = 0.4

-- The slime hitbox is a cylinder with diameter `atk_hitbox_xy` and height `atk_hitbox_z`
Slimy.atk_hitbox_xy = 3
Slimy.atk_hitbox_z = 1
Slimy.atk_damage = 20
Slimy.atk_knockback = 0.2
Slimy.atk_lift = 0.5
Slimy.atk_cooldown_duration = 30
Slimy.atk_particle = particles.lookup 'slimy.attack'
Slimy.atk_particle_n = 8

function Slimy:new()
    super.new(self)

    --Character control
    self.wx, self.wy = 0, 0
    self.watk = false
    self.wjump = false

    --Attack mechanics
    self.atk_cooldown = 0

    --Jumping mechanics
    self.jumps_left = 0
    self.jump_was_down = false
    self.jump_cooldown = 0
    self.jump_ticks = -1
    self.jump_dx = 0
    self.jump_dy = 0
    self.jump_yaw = 0
end

function Slimy:tick(world)
    local wx, wy, do_jump = self.wx, self.wy, self.wjump

    --Horizontal movement
    if not self.on_ground then
        --Maneuver in the air
        local cur_norm = (self.vel_x*self.vel_x + self.vel_y*self.vel_y)^0.5
        local max_norm = math.max(cur_norm, self.air_maneuver_max)
        self.vel_x = self.vel_x + wx * self.air_maneuver
        self.vel_y = self.vel_y + wy * self.air_maneuver
        local norm = (self.vel_x*self.vel_x + self.vel_y*self.vel_y)^0.5
        if norm > max_norm then
            --Renormalize
            local mul_by = max_norm / norm
            self.vel_x = self.vel_x * mul_by
            self.vel_y = self.vel_y * mul_by
        end
    end
    self.yaw_x, self.yaw_y = wx, wy

    --Jump
    if self.on_ground and self.vel_z < 0 and (self.jump_ticks < 0 or self.jump_ticks >= self.jump_charge) then
        --Recharge jumps
        self.jumps_left = self.jump_count
    end
    if self.jumps_left > 0 and self.jump_ticks == -1 and self.jump_cooldown <= 0 and do_jump and (self.on_ground or not self.jump_was_down) then
        self.jump_ticks = 0
        self.jump_cooldown = self.jump_cooldown_start
        self.jumps_left = self.jumps_left - 1
    end
    self.jump_was_down = do_jump
    if self.jump_cooldown > 0 then
        self.jump_cooldown = self.jump_cooldown - 1
    end
    if self.jump_ticks >= 0 then
        --Advance jump ticks
        if self.jump_ticks < self.jump_charge then
            self.jump_ticks = self.jump_ticks + 1
            if self.jump_ticks >= self.jump_charge then
                --Start jumping
                self.vel_x = wx * self.jump_hvel
                self.vel_y = wy * self.jump_hvel
                self.vel_z = self.jump_vvel
            end
        else
            if self.on_ground then
                self.jump_ticks = -1
                self.jump_cooldown = self.jump_cooldown_land
                if self.jump_cooldown_fudge > 0 then
                    self.jump_cooldown = self.jump_cooldown * world.rng:uniform(1-self.jump_cooldown_fudge, 1+self.jump_cooldown_fudge)
                end
            else
                if self.jump_ticks < self.jump_charge + self.jump_keepup_ticks then
                    if do_jump then
                        --Keep up in the air
                        self.vel_z = self.vel_z + self.jump_keepup
                    else
                        --Turn down the jump
                        self.vel_z = self.vel_z - self.jump_keepdown
                    end
                end
                self.jump_ticks = self.jump_ticks + 1
            end
        end
    end

    --Set animation from movement
    do
        local s = 0
        if self.jump_ticks < self.jump_charge then
            s = -0.1
        elseif self.on_ground then
            s = 0
        else
            s = math.abs(self.vel_z)
        end
        self.anim:event('stretch', s)
    end

    super.tick(self, world)

    --Attack
    if self.atk_cooldown <= 0 and self.on_ground and self.fall_height > 1.5 and self.watk then
        --Attack
        for i = 1, self.atk_particle_n do
            local vx, vy = util.random_circle(world.rng)
            local vz = 6
            local m, r = 8, 1
            world.pos_buf:copy_from(self.pos)
            world.pos_buf:move_box(world.terrain, r*vx, r*vy, -self.rad_z, .2, .2, .2, true)
            world.terrain:add_particle(self.atk_particle, world.pos_buf, m*vx, m*vy, vz, 0, 0, 0, 0)
        end
        local ent = world.ent_map[world.player_id]
        if ent then
            if ent ~= self and ent.hp then
                local w, h = math.max(ent.rad_x, ent.rad_y), ent.rad_z
                local buf = world.relpos_buf
                world.terrain:get_relative_positions(ent.pos, ent.rad_x, ent.rad_y, ent.rad_z, self.pos, buf)
                for i = 1, #buf, 3 do
                    local hxy, hz = self.atk_hitbox_xy, self.atk_hitbox_z
                    local dx, dy, dz = buf[i], buf[i+1], buf[i+2]
                    if dx*dx + dy*dy <= (hxy/2 + w)^2
                            and dz >= -h and dz <= hz + h then
                        -- Hit this entity
                        self.atk_cooldown = self.atk_cooldown_duration
                        self:make_damage(world, ent, self.atk_damage, self.atk_knockback, self.atk_lift, dx, dy)
                        break
                    end
                end
            end
        end
    end
    if self.atk_cooldown > 0 then
        self.atk_cooldown = self.atk_cooldown - 1
    end
end

return Slimy

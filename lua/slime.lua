
local voxel = require 'voxel'
local class = require 'class'
local input = require 'input'
local util = require 'util'

local Slime = class{}

local bbox_h = 2
local bbox_v = 1.8
local gravity = 0.025
local friction = 0.97
local walk_speed = 0.13
local yaw_anim_factor = 0.004
local yaw_anim_linear = 0.9

local jump_charge = 5
local jump_hvel = 0.17
local jump_vvel = 0.5
local jump_keepup = 0.014
local jump_keepdown = 0.005
local jump_keepup_ticks = 14
local jump_cooldown_start = 10
local jump_cooldown_land = 60

local lag_vel_z_add = 1.2
local lag_vel_z_mul = 0.0012

local air_maneuver = 0.011
local air_maneuver_max = 0.17

function Slime:new()
    assert(self.pos)
    self.mov_x = 0
    self.mov_y = 0
    self.mov_z = 0
    self.vel_x = 0
    self.vel_y = 0
    self.vel_z = 0
    self.on_ground = false
    self.visual_yaw = 0
    self.yaw = 0
    self.idle_ticks = 0
    self.visual_lag_vel_z = 0
    self.visual_fall_time = 0
    self.draw_r = 1.1 * math.sqrt(3)

    self.wx, self.wy = 0, 0

    self.anim = voxel.AnimState{
        model = voxel.models.slime,
    }

    self.jumps_left = 0
    self.jump_was_down = false

    --Jumping mechanics
    self.jump_cooldown = 0
    self.jump_ticks = -1
    self.jump_dx = 0
    self.jump_dy = 0
    self.jump_yaw = 0
end

-- Compute 3 values: walk x, walk y (2d normalized vector) and a boolean value indicating if we
-- want to jump.
local view_dist = 40
function Slime:ai(world)
    local dx, dy, dz = world.terrain:to_relative(self.pos)
    local n = dx * dx + dy * dy + dz * dz
    if n < view_dist * view_dist then
        n = (dx * dx + dy * dy) ^ -0.5
        return dx * n, dy * n, false
    else
        if world.rng:uniform() < 0.005 then
            dx, dy = world.rng:normal(-1, 1), world.rng:normal(-1, 1)
            n = dx * dx + dy * dy
            if n < 0.05 ^ 2 then
                dx, dy = 0, 0
            else
                n = n ^ -0.5
                dx, dy = dx * n, dy * n
            end
            self.wx, self.wy = dx, dy
        end
        return self.wx, self.wy, world.rng:uniform() < 0.01
    end
end

function Slime:tick(world)
    --Apply friction
    if self.on_ground then
        self.vel_x = 0
        self.vel_y = 0
        self.vel_z = 0
        if self.jump_ticks < 0 or self.jump_ticks >= jump_charge then
            self.jumps_left = 1
        end
    else
        self.vel_x = self.vel_x * friction
        self.vel_y = self.vel_y * friction
        self.vel_z = self.vel_z * friction
    end
    
    --Apply gravity
    self.vel_z = self.vel_z - gravity

    --Horizontal movement
    local wx, wy, do_jump = self:ai(world)
    if not self.on_ground then
        --Maneuver in the air
        local cur_norm = (self.vel_x*self.vel_x + self.vel_y*self.vel_y)^0.5
        local max_norm = math.max(cur_norm, air_maneuver_max)
        self.vel_x = self.vel_x + wx * air_maneuver
        self.vel_y = self.vel_y + wy * air_maneuver
        local norm = (self.vel_x*self.vel_x + self.vel_y*self.vel_y)^0.5
        if norm > max_norm then
            --Renormalize
            local mul_by = max_norm / norm
            self.vel_x = self.vel_x * mul_by
            self.vel_y = self.vel_y * mul_by
        end
    end

    --Jump
    if self.jumps_left > 0 and self.jump_ticks == -1 and self.jump_cooldown <= 0 and do_jump and (self.on_ground or not self.jump_was_down) then
        self.jump_ticks = 0
        self.jump_cooldown = jump_cooldown_start
        self.jumps_left = self.jumps_left - 1
    end
    self.jump_was_down = do_jump
    if self.jump_cooldown > 0 then
        self.jump_cooldown = self.jump_cooldown - 1
    end
    if self.jump_ticks >= 0 then
        --Advance jump ticks
        if self.jump_ticks < jump_charge then
            self.jump_ticks = self.jump_ticks + 1
            if self.jump_ticks >= jump_charge then
                --Start jumping
                self.vel_x = wx * jump_hvel
                self.vel_y = wy * jump_hvel
                self.vel_z = jump_vvel
            end
        else
            if self.on_ground then
                self.jump_ticks = -1
                self.jump_cooldown = jump_cooldown_land
            else
                if self.jump_ticks < jump_charge + jump_keepup_ticks then
                    if do_jump then
                        --Keep up in the air
                        self.vel_z = self.vel_z + jump_keepup
                    else
                        --Turn down the jump
                        self.vel_z = self.vel_z - jump_keepdown
                    end
                end
                self.jump_ticks = self.jump_ticks + 1
            end
        end
    end

    --Set yaw if moving
    if self.vel_x*self.vel_x + self.vel_y*self.vel_y > 0.02^2 then
        self.yaw = util.pos_to_yaw(self.vel_x, self.vel_y)
    end

    --Set animation from movement
    do
        local s = 0
        if self.jump_ticks < jump_charge then
            s = -0.1
        elseif self.on_ground then
            s = 0
        else
            s = math.abs(self.vel_z)
        end
        self.anim.state:motion(s)
    end

    --Apply velocity to position
    do
        local radius_h = bbox_h / 2
        local radius_v = bbox_v / 2
        local mov_x, mov_y, mov_z, cx, cy, cz = self.pos:move_box(world.terrain, self.vel_x, self.vel_y, self.vel_z, radius_h, radius_h, radius_v, true)
        self.on_ground = self.vel_z < 0 and cz
        self.mov_x, self.mov_y, self.mov_z = mov_x, mov_y, mov_z
    end
end

function Slime:draw(world)
    local frame = world.frame

    local dyaw = (self.yaw - self.visual_yaw + math.pi) % (2*math.pi) - math.pi
    self.visual_yaw = util.approach(self.yaw - dyaw, self.yaw, yaw_anim_factor, yaw_anim_linear, frame.dt) % (2*math.pi)
    frame.mvp_world:rotate_z(self.visual_yaw)

    frame.mvp_world:translate(0, 0, -bbox_v/2)
    frame.mvp_world:scale(1/8)

    self.anim:draw(frame.dt, world.shaders.terrain, frame.params_world, 'mvp', frame.mvp_world)
end

return Slime

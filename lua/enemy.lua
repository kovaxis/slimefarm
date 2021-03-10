
local Mesh = require 'mesh'
local class = require 'class'
local input = require 'input'
local util = require 'util'

local Enemy = class{}

Enemy.mesh = Mesh{}
Enemy.mesh:add_cube(0, 1, 0, 1.2, 1.2, 1.2, 0.43, 0.31, 0.13, 1)
Enemy.mesh:add_cube(0.4, 1.1,-1,   0.2,   0.6,   0.2, 0, 0, 0, 1)
Enemy.mesh:add_cube(-0.4, 1.1,-1,   0.2,   0.6,   0.2, 0, 0, 0, 1)
Enemy.mesh:add_cube(0, 1, 0,   2,   2,   2, 0.83, 0.63, 0.26, 0.4)
Enemy.mesh_buf = Enemy.mesh:as_buffer()

local bbox_h = 2
local bbox_v = 1.8
local gravity = 0.04
local friction = 0.99
local walk_speed = 0.13
local yaw_anim_factor = 0.004
local yaw_anim_linear = 0.9

local jump_charge = 5
local jump_hvel = 0.3
local jump_vvel = 0.8
local jump_keepup = 0.014
local jump_keepdown = 0.016
local jump_keepup_ticks = 14
local jump_cooldown_start = 10
local jump_cooldown_land = 0

local lag_vel_y_add = 1.2
local lag_vel_y_mul = 0.0012

local air_maneuver = 0.02
local air_maneuver_max = 0.3

local function wasd_delta(self, world)
    local play = nil
    for i, ent in ipairs(world.entities) do
        if ent.is_player then
            play = ent
            break
        end
    end
    if play then
        local dx, dz = play.x - self.x, play.z - self.z
        local magsq = dx ^ 2 + dz ^ 2
        if magsq <= 32^2 then
            local inv = jump_hvel / math.sqrt(magsq)
            return dx * inv, dz * inv
        end
    end
    return 0, 0
end

function Enemy:new()
    assert(self.x)
    assert(self.y)
    assert(self.z)
    self.vel_x = 0
    self.vel_y = 0
    self.vel_z = 0
    self.prev_x = self.x
    self.prev_y = self.y
    self.prev_z = self.z
    self.on_ground = false
    self.visual_yaw = 0
    self.yaw = 0
    self.idle_ticks = 0
    self.visual_lag_vel_y = 0
    self.visual_fall_time = 0

    self.jumps_left = 0
    self.jump_was_down = false

    self.do_jump = 0

    --Jumping mechanics
    self.jump_cooldown = 0
    self.jump_ticks = -1
    self.jump_dx = 0
    self.jump_dz = 0
    self.jump_yaw = 0
end

function Enemy:tick(world)
    --Save previous position to interpolate smoothly
    self.prev_x = self.x
    self.prev_y = self.y
    self.prev_z = self.z

    --Apply friction
    if self.on_ground then
        self.vel_x = 0
        self.vel_y = 0
        self.vel_z = 0
        if self.jump_ticks < 0 or self.jump_ticks >= jump_charge then
            self.jumps_left = 2
        end
    else
        self.vel_x = self.vel_x * friction
        self.vel_y = self.vel_y * friction
        self.vel_z = self.vel_z * friction
    end
    
    --Apply gravity
    self.vel_y = self.vel_y - gravity

    --[[
    --Check pressed keys to determine horizontal movement
    do
        local dx, dz = 0, 0
        if input.key_down.w then
            dz = dz - 1
        end
        if input.key_down.s then
            dz = dz + 1
        end
        if input.key_down.a then
            dx = dx - 1
        end
        if input.key_down.d then
            dx = dx + 1
        end
        if dx ~= 0 and dz ~= 0 then
            dx = dx * 2^-0.5
            dz = dz * 2^-0.5
        end
        if dx ~= 0 or dz ~= 0 then
            --Move horizontally
            local yaw = world.cam_yaw
            dx, dz = dx * math.cos(yaw) - dz * math.sin(yaw), dx * math.sin(yaw) + dz * math.cos(yaw)
            self.yaw = math.atan(dx, -dz)
            self.vel_x = dx * walk_speed
            self.vel_z = dz * walk_speed
        end
    end

    --Jump
    if self.on_ground and input.key_down.space then
        self.vel_y = jump_vel
    end]]

    if self.do_jump > 0 then
        self.do_jump = self.do_jump - 1
    end

    --Jump around
    if self.jumps_left > 0 and self.jump_cooldown <= 0 and self.do_jump <= 0 then
        self.jump_ticks = 0
        self.do_jump = 64 + math.random() * 64
        self.jump_cooldown = jump_cooldown_start
        self.jumps_left = self.jumps_left - 1
    end
    self.jump_was_down = input.key_down.space
    if self.jump_cooldown > 0 then
        self.jump_cooldown = self.jump_cooldown - 1
    end
    if self.jump_ticks >= 0 then
        --Advance jump ticks
        if self.jump_ticks < jump_charge then
            self.jump_ticks = self.jump_ticks + 1
            if self.jump_ticks >= jump_charge then
                --Start jumping
                local dx, dz = wasd_delta(self, world)
                self.vel_x = dx * jump_hvel
                self.vel_z = dz * jump_hvel
                self.vel_y = jump_vvel
            end
        else
            if self.on_ground then
                self.jump_ticks = -1
                self.jump_cooldown = jump_cooldown_land
            else
                if self.jump_ticks < jump_charge + jump_keepup_ticks then
                    if input.key_down.space then
                        --Keep up in the air
                        self.vel_y = self.vel_y + jump_keepup
                    else
                        --Turn down the jump
                        self.vel_y = self.vel_y - jump_keepdown
                    end
                end
                self.jump_ticks = self.jump_ticks + 1
            end
        end
    end

    --Maneuver in the air
    if not self.on_ground then
        local cur_norm = (self.vel_x*self.vel_x + self.vel_y*self.vel_y)^0.5
        local max_norm = math.max(cur_norm, air_maneuver_max)
        local dx, dz = wasd_delta(self, world)
        self.vel_x = self.vel_x + dx * air_maneuver
        self.vel_z = self.vel_z + dz * air_maneuver
        local norm = (self.vel_x*self.vel_x + self.vel_y*self.vel_y)^0.5
        if norm > max_norm then
            --Renormalize
            local mul_by = max_norm / norm
            self.vel_x = self.vel_x * mul_by
            self.vel_z = self.vel_z * mul_by
        end
    end

    --Set yaw if moving
    if self.vel_x*self.vel_x + self.vel_z*self.vel_z > 0.02^2 then
        self.yaw = util.pos_to_yaw(self.vel_x, self.vel_z)
    end

    --Count idle ticks
    if self.on_ground and self.vel_x == 0 and self.vel_z == 0 then
        self.idle_ticks = self.idle_ticks + 1
    else
        self.idle_ticks = 0
    end

    --Apply velocity to position
    do
        local radius_h = bbox_h / 2
        local radius_v = bbox_v / 2
        local fx, fy, fz = world.terrain:collide(self.x, self.y+radius_v, self.z, self.vel_x, self.vel_y, self.vel_z, radius_h, radius_v, radius_h)
        self.on_ground = self.vel_y < 0 and fy > self.y + radius_v + self.vel_y + gravity / 2
        self.x, self.y, self.z = fx, fy-radius_v, fz
    end
end

function Enemy:draw(world)
    local frame = world.frame

    local dyaw = (self.yaw - self.visual_yaw + math.pi) % (2*math.pi) - math.pi
    self.visual_yaw = util.approach(self.yaw - dyaw, self.yaw, yaw_anim_factor, yaw_anim_linear, frame.dt) % (2*math.pi)
    frame.mvp_world:rotate_y(self.visual_yaw)

    self.visual_lag_vel_y = util.approach(self.visual_lag_vel_y, self.vel_y, lag_vel_y_mul, lag_vel_y_add, frame.dt)

    local sy = 1
    if self.jump_ticks >= 0 and self.jump_ticks < jump_charge then
        --Jump charge animation
        local x = (self.jump_ticks + frame.s) / jump_charge
        sy = 0.8 + 0.2*(1+3.4*(x-1)*x)^2
    elseif self.idle_ticks > 0 then
        --On-ground animation
        local acc = self.vel_y - self.visual_lag_vel_y
        if self.vel_y <= 0 and acc > 0.0 then
            --Squash animation
            sy = 1 + 0.35 * (1/(1+2^(2.2*acc)) - 0.5)
        else
            --Idle animation
            local x = (self.idle_ticks + frame.s) + 56
            sy = 1 - 0.03 * math.sin(x*0.04) / (1+2^(30 - x*0.4))
        end
    elseif self.vel_y > 0 then
        --Rise animation
        sy = 1 - 0.8 * (1/(1+2^(8*self.vel_y)) - 0.5)
    else
        --Fall animation
        self.visual_fall_time = self.visual_fall_time + self.vel_y * frame.dt
        local mag = 0.02 * (1/(1+2^(-2*self.vel_y)) - 0.5)
        local t = self.visual_fall_time
        local x = math.sin(t * 17 + 11.758) + math.sin(t * 23 + 7.138) + math.sin(t * 38 + 2.873)
        sy = 1 + mag * x
    end
    local sxz = (1/sy)^0.5
    frame.mvp_world:scale(sxz, sy, sxz)

    world.shaders.basic:set_matrix('mvp', frame.mvp_world)
    world.shaders.basic:set_vec4('tint', 1, 1, 1, 1)
    world.shaders.basic:draw(Enemy.mesh_buf, frame.params_world)
end

return Enemy
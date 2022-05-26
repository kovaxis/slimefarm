
local voxel = require 'voxel'
local class = require 'class'
local input = require 'input'
local util = require 'util'

local Player = class{}

local bbox_h = 2
local bbox_v = 1.8
local gravity = 0.04
local friction = 0.99
local walk_speed = 0.13
local yaw_anim_factor = 0.004
local yaw_anim_linear = 0.9

local jump_charge = 5
local jump_hvel = 0.17
local jump_vvel = 0.7
local jump_keepup = 0.014
local jump_keepdown = 0.016
local jump_keepup_ticks = 14
local jump_cooldown_start = 10
local jump_cooldown_land = 0

local lag_vel_z_add = 1.2
local lag_vel_z_mul = 0.0012

local air_maneuver = 0.011
local air_maneuver_max = 0.17

local function wasd_delta(yaw)
    local dx, dy = 0, 0
    if input.key_down.a then
        dx = dx - 1
    end
    if input.key_down.d then
        dx = dx + 1
    end
    if input.key_down.w then
        dy = dy + 1
    end
    if input.key_down.s then
        dy = dy - 1
    end
    if dx ~= 0 and dy ~= 0 then
        dx = dx * 2^-0.5
        dy = dy * 2^-0.5
    end
    return util.rotate_yaw(dx, dy, yaw)
end

function Player:new()
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

    self.anim = voxel.AnimState{
        model = voxel.models.player,
    }

    self.jumps_left = 0
    self.jump_was_down = false

    --Jumping mechanics
    self.jump_cooldown = 0
    self.jump_ticks = -1
    self.jump_dx = 0
    self.jump_dy = 0
    self.jump_yaw = 0

    self.is_player = true
end

function Player:tick(world)
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
    self.vel_z = self.vel_z - gravity

    --Horizontal movement
    local wx, wy = wasd_delta(world.cam_yaw)
    if self.on_ground then
        --Run around
        self.vel_x, self.vel_y = wx * walk_speed, wy * walk_speed
    else
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
    if self.jumps_left > 0 and self.jump_cooldown <= 0 and input.key_down.space and (self.on_ground or not self.jump_was_down) then
        self.jump_ticks = 0
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
                self.vel_z = jump_vvel
            end
        else
            if self.on_ground then
                self.jump_ticks = -1
                self.jump_cooldown = jump_cooldown_land
            else
                if self.jump_ticks < jump_charge + jump_keepup_ticks then
                    if input.key_down.space then
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
    if self.on_ground then
        if self.vel_x == 0 and self.vel_y == 0 then
            --Idle
            self.anim.state:motion('idle', world.tick_count)
        else
            --Run
            self.anim.state:motion('run', world.tick_count)
        end
    else
        --Airtime
        self.anim.state:motion('air', world.tick_count)
    end

    --Apply velocity to position
    do
        local radius_h = bbox_h / 2
        local radius_v = bbox_v / 2
        local mov_x, mov_y, mov_z, cx, cy, cz = self.pos:move_box(world.terrain, self.vel_x, self.vel_y, self.vel_z, radius_h, radius_h, radius_v, true)
        self.on_ground = self.vel_z < 0 and cz
        self.mov_x, self.mov_y, self.mov_z = mov_x, mov_y, mov_z
    end

    --Move camera to point at player
    do
        local focus_height = 2
        local focus_dist = 8
        local cam_wall_dist = 0.4
        world.cam_pos:copy_from(self.pos)
        world.cam_pos:move_box(world.terrain, 0, 0, focus_height, cam_wall_dist, cam_wall_dist, cam_wall_dist)
        world.cam_mov_x, world.cam_mov_y, world.cam_mov_z = self.mov_x, self.mov_y, self.mov_z
        world.cam_rollback = focus_dist
    end
end

function Player:draw(world)
    local frame = world.frame

    local dyaw = (self.yaw - self.visual_yaw + math.pi) % (2*math.pi) - math.pi
    self.visual_yaw = util.approach(self.yaw - dyaw, self.yaw, yaw_anim_factor, yaw_anim_linear, frame.dt) % (2*math.pi)
    frame.mvp_world:rotate_z(self.visual_yaw)

    frame.mvp_world:translate(0, 0, -bbox_v/2)
    frame.mvp_world:scale(1/8)

    self.anim:draw(frame.dt, world.shaders.terrain, frame.params_world, 'mvp', frame.mvp_world)
end

return Player
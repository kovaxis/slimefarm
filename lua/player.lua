
local Mesh = require 'mesh'
local class = require 'class'
local input = require 'input'
local util = require 'util'

local Player = class{}

Player.mesh = Mesh{}
Player.mesh:add_cube(0, 1, 0, 1.2, 1.2, 1.2, 0, 0.31, 0.13, 1)
Player.mesh:add_cube(0.4, 1.1,-1,   0.2,   0.6,   0.2, 0, 0, 0)
Player.mesh:add_cube(-0.4, 1.1,-1,   0.2,   0.6,   0.2, 0, 0, 0)
Player.mesh:add_cube(0, 1, 0,   2,   2,   2, 0, 0.63, 0.26, 0.4)
Player.mesh_buf = Player.mesh:as_buffer()

local bbox_h = 2
local bbox_v = 1.8
local gravity = 0.04
local friction = 0.99
local walk_speed = 0.13
local yaw_anim_factor = 0.04
local yaw_anim_linear = 1.4

local jump_hvel = 0.4
local jump_vvel = 0.6
local jump_hnudge_ticks = 4
local jump_cooldown = 10

function Player:new()
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

    --Jumping mechanics
    self.jump_cooldown = 0
    self.jump_ticks = -1
    self.jump_dx = 0
    self.jump_dz = 0
    self.jump_yaw = 0
end

function Player:tick(world)
    --Save previous position to interpolate smoothly
    self.prev_x = self.x
    self.prev_y = self.y
    self.prev_z = self.z

    --Apply friction
    if self.on_ground then
        self.vel_x = 0
        self.vel_y = 0
        self.vel_z = 0
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

    --Jump around
    if self.jump_ticks < 0 then
        if self.on_ground and self.jump_cooldown <= 0 and input.key_down.space then
            self.jump_dx = 0
            self.jump_dz = 0
            self.jump_yaw = world.cam_yaw
            self.vel_y = jump_vvel
            self.jump_ticks = 0
        end
        if self.jump_cooldown > 0 then
            self.jump_cooldown = self.jump_cooldown - 1
        end
    else
        --Advance jump ticks
        if self.jump_ticks < jump_hnudge_ticks then
            local og_dx, og_dz = self.jump_dx, self.jump_dz
            local dx, dz = og_dx, og_dz
            if dx == 0 then
                if input.key_down.a then
                    dx = dx - 1
                end
                if input.key_down.d then
                    dx = dx + 1
                end
            end
            if dz == 0 then
                if input.key_down.w then
                    dz = dz - 1
                end
                if input.key_down.s then
                    dz = dz + 1
                end
            end
            if dx ~= og_dz or dz ~= og_dz then
                --Modify
                self.jump_dx = dx
                self.jump_dz = dz
                if og_dx ~= 0 and og_dz ~= 0 then
                    og_dx = og_dx * 2^-0.5
                    og_dz = og_dz * 2^-0.5
                end
                if dx ~= 0 and dz ~= 0 then
                    dx = dx * 2^-0.5
                    dz = dz * 2^-0.5
                end
                dx, dz = util.rotate_yaw(dx, dz, self.jump_yaw)
                og_dx, og_dz = util.rotate_yaw(og_dx, og_dz, self.jump_yaw)
                self.yaw = util.pos_to_yaw(dx, dz)
                self.vel_x = self.vel_x + (dx - og_dx) * jump_hvel
                self.vel_z = self.vel_z + (dz - og_dz) * jump_hvel
            end
        end
        if self.on_ground then
            self.jump_ticks = -1
            self.jump_cooldown = jump_cooldown
        else
            self.jump_ticks = self.jump_ticks + 1
        end
    end

    --Apply velocity to position
    do
        local radius_h = bbox_h / 2
        local radius_v = bbox_v / 2
        local fx, fy, fz = world.terrain:collide(self.x, self.y+radius_v, self.z, self.vel_x, self.vel_y, self.vel_z, radius_h, radius_v, radius_h)
        self.on_ground = self.vel_y < 0 and fy > self.y + radius_v + self.vel_y + gravity / 2
        self.x, self.y, self.z = fx, fy-radius_v, fz
    end

    --Move camera to point at player
    do
        local focus_height = 3.2
        local focus_dist = 8
        world.cam_prev_x = self.prev_x
        world.cam_prev_y = self.prev_y + focus_height
        world.cam_prev_z = self.prev_z
        world.cam_x = self.x
        world.cam_y = self.y + focus_height
        world.cam_z = self.z
        world.cam_rollback = focus_dist
    end
end

function Player:draw(world)
    local frame = world.frame

    local dyaw = (self.yaw - self.visual_yaw + math.pi) % (2*math.pi) - math.pi
    self.visual_yaw = util.approach(self.yaw - dyaw, self.yaw, yaw_anim_factor, yaw_anim_linear, frame.dt) % (2*math.pi)
    frame.mvp_world:rotate_y(self.visual_yaw)
    world.shaders.basic:set_matrix('mvp', frame.mvp_world)
    world.shaders.basic:set_vec4('tint', 1, 1, 1, 1)
    world.shaders.basic:draw(Player.mesh_buf, frame.params_world)
end

return Player
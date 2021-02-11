
local Mesh = require 'mesh'
local class = require 'class'
local input = require 'input'

local Player = class{}

Player.mesh = Mesh{}
Player.mesh:add_cube(0, 1, 0, 1.2, 1.2, 1.2, 0, 0.31, 0.13, 1)
Player.mesh:add_cube(0.4, 1.1,-1,   0.2,   0.6,   0.2, 0, 0, 0)
Player.mesh:add_cube(-0.4, 1.1,-1,   0.2,   0.6,   0.2, 0, 0, 0)
Player.mesh:add_cube(0, 1, 0,   2,   2,   2, 0, 0.63, 0.26, 0.4)
Player.mesh_buf = Player.mesh:as_buffer()

local bbox_h = 2
local bbox_v = 1.8
local gravity = 0.02
local jump_vel = 0.7
local friction = 0.97
local walk_speed = 0.13

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
    self.yaw = 0
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
    frame.mvp_world:rotate_y(self.yaw)
    world.shaders.basic:set_matrix('mvp', frame.mvp_world)
    world.shaders.basic:set_vec4('tint', 1, 1, 1, 1)
    world.shaders.basic:draw(Player.mesh_buf, frame.params_world)
end

return Player
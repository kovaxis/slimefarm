
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
local jump_vel = 0.6
local friction = 1
local walk_speed = 0.13
local yaw_anim_speed = 0.25
local yaw_anim_speed_min = 0.04

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
    self.prev_yaw = 0
    self.yaw = 0
    self.target_yaw = 0
    self.dst_x = false
    self.dst_y = false
    self.dst_z = false
    self.src_x = false
    self.src_y = false
    self.src_z = false
end

function Player:tick(world)
    --Save previous position to interpolate smoothly
    self.prev_x = self.x
    self.prev_y = self.y
    self.prev_z = self.z
    self.prev_yaw = self.yaw

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
    if self.on_ground and self.dst_x then
        self.dst_x = false
        self.dst_y = false
        self.dst_z = false
        self.src_x = false
        self.src_y = false
        self.src_z = false
    elseif self.on_ground and input.key_down.space and not self.dst_x then
        --Invert the mouse position using the last render matrix
        local max_cast_dist = 128
        local inverse = world.frame.mvp_world
        inverse:push()
        inverse:invert()
        local dx, dy, dz = inverse:transform_point(world.mouse_x, world.mouse_y, 0)
        inverse:pop()
        local factor = max_cast_dist*(dx*dx + dy*dy + dz*dz)^-0.5
        --Cast a ray to find the landing spot
        local x, y, z = world.terrain:raycast(
            world.cam_effective_x,
            world.cam_effective_y,
            world.cam_effective_z,
            dx * factor, dy * factor, dz * factor,
            0, 0, 0
        )
        --Determine jump velocity
        local dx, dy, dz = x - self.x, y - self.y, z - self.z
        local hdist = (dx*dx + dz*dz)^0.5
        local v0 = jump_vel
        local det0 = v0*v0 - 2*dy*gravity*v0 - hdist*hdist*gravity*gravity
        if det0 >= 0 then
            local det1 = v0 - dy * gravity + det0^0.5
            if det1 >= 0 then
                local dt = 2^0.5 * det1^0.5 / gravity
                local vy = dy / dt + 0.5 * dt * gravity
                local vx = hdist / dt
                self.vel_x = dx * vx / hdist
                self.vel_y = vy
                self.vel_z = dz * vx / hdist
                self.target_yaw = math.atan(dx, -dz)
                --Set source and target
                self.src_x = self.x
                self.src_y = self.y
                self.src_z = self.z
                self.dst_x = x
                self.dst_y = y
                self.dst_z = z
            end
        end
    end

    --Animate yaw
    do
        local dyaw = (self.target_yaw - self.yaw + math.pi) % (2*math.pi) - math.pi
        dyaw = util.abs_min(util.abs_max(dyaw * yaw_anim_speed, yaw_anim_speed_min), math.abs(dyaw))
        self.yaw = (self.yaw + dyaw) % (2*math.pi)
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
    frame.mvp_world:rotate_y(self.yaw + ((self.yaw - self.prev_yaw + math.pi) % (2*math.pi) - math.pi) * frame.s)
    world.shaders.basic:set_matrix('mvp', frame.mvp_world)
    world.shaders.basic:set_vec4('tint', 1, 1, 1, 1)
    world.shaders.basic:draw(Player.mesh_buf, frame.params_world)
end

return Player

local class = require 'class'
local util = require 'util'
local voxel = require 'voxel'

local Entity = class{}

Entity.rad_x = 0
Entity.rad_y = 0
Entity.rad_z = 0
Entity.draw_r = 0

Entity.gravity = 0.025
Entity.friction = 0.97

Entity.yaw_anim_factor = 0.004
Entity.yaw_anim_linear = 0.9

Entity.visual_scale = 1/8

function Entity:new()
    assert(self.draw_r ~= 0)
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

    if self.model then
        self.anim = voxel.AnimState {
            model = self.model,
        }
    end
    assert(self.anim)
end

function Entity:pretick(world)
    --Apply friction
    if self.on_ground and self.vel_z < 0 then
        self.vel_x = 0
        self.vel_y = 0
        self.vel_z = 0
    else
        local f = self.friction
        self.vel_x = self.vel_x * f
        self.vel_y = self.vel_y * f
        self.vel_z = self.vel_z * f
    end
    
    --Apply gravity
    self.vel_z = self.vel_z - self.gravity
end

function Entity:tick(world)
    --Set yaw if moving
    if self.vel_x*self.vel_x + self.vel_y*self.vel_y > 0.02^2 then
        self.yaw = util.pos_to_yaw(self.vel_x, self.vel_y)
    end

    --Apply velocity to position
    self:apply_vel(world)
end

function Entity:apply_vel(world)
    local mov_x, mov_y, mov_z, cx, cy, cz = self.pos:move_box(world.terrain, self.vel_x, self.vel_y, self.vel_z, self.rad_x, self.rad_y, self.rad_z, true)
    self.on_ground = self.vel_z < 0 and cz
    self.mov_x, self.mov_y, self.mov_z = mov_x, mov_y, mov_z
end

function Entity:draw(world)
    local frame = world.frame

    --Smooth yaw
    local dyaw = (self.yaw - self.visual_yaw + math.pi) % (2*math.pi) - math.pi
    self.visual_yaw = util.approach(self.yaw - dyaw, self.yaw, self.yaw_anim_factor, self.yaw_anim_linear, frame.dt) % (2*math.pi)
    frame.mvp_world:rotate_z(self.visual_yaw)

    --Position center of model at floor level
    frame.mvp_world:translate(0, 0, -self.rad_z)

    --Scale voxels down
    frame.mvp_world:scale(self.visual_scale)

    --Draw animated model
    self.anim:draw(frame.dt, world.shaders.terrain, frame.params_world, 'mvp', frame.mvp_world)
end

function Entity:set_bbox(x, y, z)
    x, y, z = x/2, y/2, z/2
    self.rad_x, self.rad_y, self.rad_z = x, y, z
    self.draw_r = (x*x + y*y + z*z)^0.5
end

return Entity

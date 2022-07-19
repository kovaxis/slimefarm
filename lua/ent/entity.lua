-- A world entity.

local class = require 'class'
local util = require 'util'
local voxel = require 'voxel'

local Entity = class{}

-- Radius of the physical hitbox.
Entity.rad_x = 0
Entity.rad_y = 0
Entity.rad_z = 0

-- Maximum radius of the visual model.
-- Used for view-frustum-clipping and rendering across portals.
Entity.draw_r = 0

-- Gravitational acceleration.
Entity.gravity = 0.025
-- Velocity friction.
-- A value of 1 means no friction.
Entity.friction = 0.97

-- Multiplicative yaw ease factor.
Entity.yaw_anim_factor = 0.0004
-- Linear yaw ease factor.
Entity.yaw_anim_linear = 10

-- Visual scale of the model (ie. how many blocks is a model cube).
Entity.visual_scale = 1/8

function Entity:new()
    assert(self.draw_r ~= 0)
    assert(self.pos)
    self.mov_x = 0
    self.mov_y = 0
    self.mov_z = 0
    self.vel_x = self.vel_x or 0
    self.vel_y = self.vel_y or 0
    self.vel_z = self.vel_z or 0
    self.on_ground = false
    self.yaw_x, self.yaw_y = 0, 0
    self.yaw = self.yaw or 0
    self.visual_yaw = self.visual_raw or self.yaw

    if self.model then
        self.anim = voxel.AnimState {
            model = self.model,
        }
    end
    assert(self.anim)
end

function Entity.create(cl, proto, pos)
    proto.pos = pos
    return cl(proto)
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
    --Set yaw from yaw direction
    do
        local x, y = self.yaw_x, self.yaw_y
        if x*x + y*y > 0.02^2 then
            self.yaw = util.pos_to_yaw(x, y)
        end
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

-- overloads:
-- set_bbox(size, visible_radius)
-- set_bbox(size_xy, size_z, visible_radius)
-- set_bbox(size_x, size_y, size_z, visible_radius)
function Entity:set_bbox(x, y, z, r)
    if not z then
        x, y, z = x, x, y
    end
    if not r then
        x, y, z, r = x, x, y, z
    end
    x, y, z = x/2, y/2, z/2
    self.rad_x, self.rad_y, self.rad_z = x, y, z
    self.draw_r = r * 3^.5
end

return Entity

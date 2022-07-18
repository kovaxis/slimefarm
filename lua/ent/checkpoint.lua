
local class = require 'class'
local util = require 'util'
local Entity = require 'ent.entity'
local voxel = require 'voxel'
local entreg = require 'ent.reg'

local Checkpoint, super = class { super = Entity }

Checkpoint.model = voxel.models.flag

Checkpoint.group = 'spawnpoint'

Checkpoint:set_bbox(2.5, 20/8)

entreg.register {
    name = 'Checkpoint',
    class = Checkpoint,
    fmt = {
        orient = 'u8',
        hidden = {opt = 'bool'},
    },
}

function Checkpoint:new()
    self.yaw = self.orient * math.pi/2
    super.new(self)
end

function Checkpoint:tick(world)
    self.anim:event('wave')
    return super.tick(self, world)
end

function Checkpoint:draw(...)
    if not self.hidden then
        return super.draw(self, ...)
    end
end

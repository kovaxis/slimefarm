
local class = require 'class'
local util = require 'util'
local Entity = require 'ent.entity'
local voxel = require 'voxel'

local Checkpoint, super = class { super = Entity }

Checkpoint.model = voxel.models.checkpoint

function Checkpoint:new()
    super.new(self)

    self.is_checkpoint = true
end


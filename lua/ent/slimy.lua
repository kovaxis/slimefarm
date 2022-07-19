
local voxel = require 'voxel'
local class = require 'class'
local input = require 'input'
local util = require 'util'
local Animal = require 'ent.animal'
local particles = require 'particles'

local Slimy, super = class{ super = Animal }

Slimy.walk_speed = 0

Slimy.jump_hvel = 0.17
Slimy.jump_vvel = 0.5
Slimy.jump_cooldown_fudge = 0.4

return Slimy

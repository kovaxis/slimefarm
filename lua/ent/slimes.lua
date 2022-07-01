
local voxel = require 'voxel'
local class = require 'class'
local util = require 'util'
local Slimy = require 'ent.slimy'

local slimes = {}
local super = Slimy

local function ai(self, world, view_dist)
    local dx, dy, dz = world.terrain:to_relative(self.pos)
    local n = dx * dx + dy * dy + dz * dz
    if n < view_dist * view_dist then
        local d = n^.5
        n = -(dx * dx + dy * dy) ^ -0.5
        self.wx, self.wy, self.wjump = dx * n, dy * n, true
        return d
    else
        if world.rng:uniform() < 0.005 then
            dx, dy = world.rng:normal(-1, 1), world.rng:normal(-1, 1)
            n = dx * dx + dy * dy
            if n < 0.05 ^ 2 then
                dx, dy = 0, 0
            else
                n = n ^ -0.5
                dx, dy = dx * n, dy * n
            end
            self.wx, self.wy = dx, dy
        end
        self.wjump = world.rng:uniform() < 0.01
        return 1 / 0
    end
end

local Green = class{ super = Slimy }
slimes.Green = Green
Green:set_bbox(16/8, 11/8)
Green.view_dist = 15
Green.max_hp = 600
Green.model = voxel.models.green_slime

function Green:tick(world)
    local d = ai(self, world, self.view_dist)
    if d < 1 / 0 then
        self.wx, self.wy = -self.wx, -self.wy
    end

    return super.tick(self, world)
end

local Red = class{ super = Slimy }
slimes.Red = Red
Red:set_bbox(14/8, 12/8)
Red.view_dist = 25
Red.max_hp = 650
Red.model = voxel.models.red_slime

function Red:tick(world)
    ai(self, world, self.view_dist)

    return super.tick(self, world)
end

return slimes

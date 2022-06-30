
local voxel = require 'voxel'
local class = require 'class'
local util = require 'util'
local Slimy = require 'ent.slimy'

local slimes = {}

do
    local Green, super = class{ super = Slimy }
    slimes.Green = Green

    Green:set_bbox(2, 2, 1.8)

    function Green:new()
        self.model = voxel.models.green_slime
        super.new(self)
        self.hp = 500
    end

    local view_dist = 40
    function Green:tick(world)
        local dx, dy, dz = world.terrain:to_relative(self.pos)
        local n = dx * dx + dy * dy + dz * dz
        if n < view_dist * view_dist then
            n = -(dx * dx + dy * dy) ^ -0.5
            self.wx, self.wy, self.wjump = dx * n, dy * n, true
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
        end

        return super.tick(self, world)
    end
end

return slimes

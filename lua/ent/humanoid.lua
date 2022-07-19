-- Humanoids are basically any animal that walks and should step automatically over 1-block ramps.

local voxel = require 'voxel'
local class = require 'class'
local input = require 'input'
local util = require 'util'
local Animal = require 'ent.animal'

local Humanoid, super = class{ super = Animal }

-- Step over 1-block ramps
local tmp_pos = system.world_pos()
function Humanoid:apply_vel(world)
    local rx, ry, rz = self.rad_x, self.rad_y, self.rad_z
    tmp_pos:copy_from(self.pos)
    local mx, my, mz, cx, cy, cz = self.pos:move_box(world.terrain, self.vel_x, self.vel_y, self.vel_z, rx, ry, rz, true)
    if self.vel_z < 0 and cz and (cx or cy) then
        local mx1, my1, mz1, cx1, cy1, cz1 = tmp_pos:move_box(world.terrain, 0, 0, 1.001, rx, ry, rz, true)
        if not cz1 then
            local mx2, my2, mz2, cx2, cy2, cz2 = tmp_pos:move_box(world.terrain, self.vel_x, self.vel_y, self.vel_z, rx, ry, rz, true)
            if (not cx2 or not cy2) and cz2 then
                mx, my, mz, cx, cy, cz = mx1 + mx2, my1 + my2, mz1 + mz2, cx2, cy2, cz2
                self.pos:copy_from(tmp_pos)
            end
        end
    end
    self.on_ground = self.vel_z < 0 and cz
    self.mov_x, self.mov_y, self.mov_z = mx, my, mz
end

return Humanoid

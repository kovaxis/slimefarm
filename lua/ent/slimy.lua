
local voxel = require 'voxel'
local class = require 'class'
local input = require 'input'
local util = require 'util'
local Living = require 'ent.living'

local Slimy, super = class{ super = Living }

Slimy.air_maneuver = 0.011
Slimy.air_maneuver_max = 0.17

Slimy.jump_count = 1
Slimy.jump_charge = 5
Slimy.jump_hvel = 0.17
Slimy.jump_vvel = 0.5
Slimy.jump_keepup = 0.014
Slimy.jump_keepdown = 0.005
Slimy.jump_keepup_ticks = 14
Slimy.jump_cooldown_start = 10
Slimy.jump_cooldown_land = 60

function Slimy:new()
    super.new(self)

    --Character control
    self.wx, self.wy = 0, 0
    self.wjump = false

    --Jumping mechanics
    self.jumps_left = 0
    self.jump_was_down = false
    self.jump_cooldown = 0
    self.jump_ticks = -1
    self.jump_dx = 0
    self.jump_dy = 0
    self.jump_yaw = 0
end

function Slimy:tick(world)
    local wx, wy, do_jump = self.wx, self.wy, self.wjump

    --Horizontal movement
    if not self.on_ground then
        --Maneuver in the air
        local cur_norm = (self.vel_x*self.vel_x + self.vel_y*self.vel_y)^0.5
        local max_norm = math.max(cur_norm, self.air_maneuver_max)
        self.vel_x = self.vel_x + wx * self.air_maneuver
        self.vel_y = self.vel_y + wy * self.air_maneuver
        local norm = (self.vel_x*self.vel_x + self.vel_y*self.vel_y)^0.5
        if norm > max_norm then
            --Renormalize
            local mul_by = max_norm / norm
            self.vel_x = self.vel_x * mul_by
            self.vel_y = self.vel_y * mul_by
        end
    end

    --Jump
    if self.on_ground and self.vel_z < 0 and (self.jump_ticks < 0 or self.jump_ticks >= self.jump_charge) then
        --Recharge jumps
        self.jumps_left = self.jump_count
    end
    if self.jumps_left > 0 and self.jump_ticks == -1 and self.jump_cooldown <= 0 and do_jump and (self.on_ground or not self.jump_was_down) then
        self.jump_ticks = 0
        self.jump_cooldown = self.jump_cooldown_start
        self.jumps_left = self.jumps_left - 1
    end
    self.jump_was_down = do_jump
    if self.jump_cooldown > 0 then
        self.jump_cooldown = self.jump_cooldown - 1
    end
    if self.jump_ticks >= 0 then
        --Advance jump ticks
        if self.jump_ticks < self.jump_charge then
            self.jump_ticks = self.jump_ticks + 1
            if self.jump_ticks >= self.jump_charge then
                --Start jumping
                self.vel_x = wx * self.jump_hvel
                self.vel_y = wy * self.jump_hvel
                self.vel_z = self.jump_vvel
            end
        else
            if self.on_ground then
                self.jump_ticks = -1
                self.jump_cooldown = self.jump_cooldown_land
            else
                if self.jump_ticks < self.jump_charge + self.jump_keepup_ticks then
                    if do_jump then
                        --Keep up in the air
                        self.vel_z = self.vel_z + self.jump_keepup
                    else
                        --Turn down the jump
                        self.vel_z = self.vel_z - self.jump_keepdown
                    end
                end
                self.jump_ticks = self.jump_ticks + 1
            end
        end
    end

    --Set animation from movement
    do
        local s = 0
        if self.jump_ticks < self.jump_charge then
            s = -0.1
        elseif self.on_ground then
            s = 0
        else
            s = math.abs(self.vel_z)
        end
        self.anim:event('stretch', s)
    end

    return super.tick(self, world)
end

return Slimy

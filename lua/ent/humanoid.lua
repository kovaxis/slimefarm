
local voxel = require 'voxel'
local class = require 'class'
local input = require 'input'
local util = require 'util'
local Living = require 'ent.living'
local Firebolt = require 'ent.firebolt'

local Humanoid, super = class{ super = Living }

Humanoid.walk_speed = 0.13
Humanoid.air_maneuver = 0.011
Humanoid.air_maneuver_max = 0.13

Humanoid.jump_count = 1
Humanoid.jump_charge = 5
Humanoid.jump_vvel = 0.5
Humanoid.jump_keepup = 0.014
Humanoid.jump_keepdown = 0.005
Humanoid.jump_keepup_ticks = 14
Humanoid.jump_cooldown_start = 20
Humanoid.jump_cooldown_land = 0

Humanoid.roll_pre = 7
Humanoid.roll_immune = 22
Humanoid.roll_duration = 37
Humanoid.roll_cooldown_end = 5
Humanoid.roll_speed = {0.08, 0.17, 0.08}

Humanoid.atk_duration = 18
Humanoid.atk_cooldown_end = 40
Humanoid.atk_damage = 20
Humanoid.atk_knockback = 0.2
Humanoid.atk_lift = 0.5
Humanoid.atk_vel = .4
Humanoid.atk_height = 0

function Humanoid:new()
    super.new(self)
    self.tmp_pos = self.pos:copy()

    --Character control
    self.wx, self.wy = 0, 0
    self.wjump, self.wjumpkeep = false, false
    self.wroll_x, self.wroll_y = 0, 0
    self.watk_x, self.watk_y, self.watk_z = 0, 0, 0

    --Jumping mechanics
    self.jump_ticks = -1
    self.jump_cooldown = 0
    self.jumps_left = 0

    --Roll mechanics
    self.roll_ticks = -1
    self.roll_cooldown = 0
    self.roll_dx = 0
    self.roll_dy = 0

    --Attack mechanics
    self.atk_ticks = -1
    self.atk_cooldown = 0
    self.atk_dx, self.atk_dy, self.atk_dz = 0, 0, 0
end

function Humanoid:tick(world)
    --Horizontal movement
    local wx, wy = self.wx, self.wy
    if self.roll_ticks >= 0 then
        --Rodar sin deslizar
        local speed
        if self.roll_ticks < self.roll_pre then
            speed = self.roll_speed[1]
        elseif self.roll_ticks < self.roll_immune then
            speed = self.roll_speed[2]
        elseif self.roll_ticks < self.roll_duration then
            speed = self.roll_speed[3]
        end
        self.vel_x, self.vel_y = self.roll_dx * speed, self.roll_dy * speed
        self.yaw_x, self.yaw_y = self.roll_dx, self.roll_dy
    elseif self.on_ground then
        --Run around
        local sp = self.walk_speed
        self.vel_x, self.vel_y = wx * sp, wy * sp
        self.yaw_x, self.yaw_y = wx, wy
    else
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
        self.yaw_x, self.yaw_y = wx, wy
    end
    if self.atk_ticks >= 0 then
        --Point to direction of attack
        self.yaw_x, self.yaw_y = self.atk_dx, self.atk_dy
    end

    --Roll
    if self.roll_cooldown <= 0 and self.roll_ticks < 0 and (wx ~= 0 or wy ~= 0)
            and self.jump_cooldown <= 0 and (self.wroll_x ~= 0 or self.wroll_y ~= 0) then
        --Start rolling
        self.jump_ticks = -1
        self.roll_ticks = 0
        self.roll_dx, self.roll_dy = self.wroll_x, self.wroll_y
    end
    if self.roll_cooldown > 0 then
        self.roll_cooldown = self.roll_cooldown - 1
    end
    if self.roll_ticks >= 0 then
        --Advance roll ticks
        self.roll_ticks = self.roll_ticks + 1
        if self.roll_ticks >= self.roll_duration then
            self.roll_ticks = -1
            self.roll_cooldown = self.roll_cooldown_end
        end
    end

    --Jump
    if self.on_ground and self.vel_z < 0 and (self.jump_ticks < 0 or self.jump_ticks >= self.jump_charge) then
        --Recharge jumps
        self.jumps_left = self.jump_count
    end
    if self.jumps_left > 0 and self.jump_cooldown <= 0 and self.wjump
            and self.roll_ticks < 0 and self.atk_ticks < 0 then
        self.jump_ticks = 0
        self.jump_cooldown = self.jump_cooldown_start
        self.jumps_left = self.jumps_left - 1
    end
    if self.jump_cooldown > 0 then
        self.jump_cooldown = self.jump_cooldown - 1
    end
    if self.jump_ticks >= 0 then
        --Advance jump ticks
        local charge = self.jump_charge
        if self.jump_ticks < charge then
            self.jump_ticks = self.jump_ticks + 1
            if self.jump_ticks >= charge then
                --Start jumping
                self.vel_z = self.jump_vvel
            end
        else
            if self.on_ground then
                self.jump_ticks = -1
                self.jump_cooldown = self.jump_cooldown_land
            else
                if self.jump_ticks < charge + self.jump_keepup_ticks then
                    if self.wjumpkeep then
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

    --Attack
    if self.atk_ticks < 0 and (self.watk_x ~= 0 or self.watk_y ~= 0 or self.watk_z ~= 0)
            and self.roll_ticks < 0 and self.atk_cooldown <= 0 then
        -- Start attack
        self.atk_ticks = 0
        self.jump_ticks = -1
        local ax, ay, az = self.watk_x, self.watk_y, self.watk_z
        self.atk_dx, self.atk_dy, self.atk_dz = ax, ay, az
        -- Shoot
        local pos = self.pos:copy()
        pos:move(world.terrain, 0, 0, self.atk_height - self.rad_z, .1, .1, .1)
        world:add_entity(Firebolt {
            owner = self.id,
            pos = pos,
            vel_x = self.atk_vel * ax,
            vel_y = self.atk_vel * ay,
            vel_z = self.atk_vel * az,
            group = 'ally_bullet',
            target_group = 'enemy',
            atk_damage = self.atk_damage,
            atk_knockback = self.atk_knockback,
            atk_lift = self.atk_lift,
        })

        -- Make damage
        --[[for i, ent in ipairs(world.ent_list) do
            if ent ~= self and ent.hp then
                local w, h = math.max(ent.rad_x, ent.rad_y), ent.rad_z
                local buf = world.relpos_buf
                world.terrain:get_relative_positions(ent.pos, ent.rad_x, ent.rad_y, ent.rad_z, self.pos, buf)
                for i = 1, #buf, 3 do
                    local hx, hy, hz = self.atk_hitbox_x, self.atk_hitbox_y, self.atk_hitbox_z
                    local dx, dy, dz = buf[i], buf[i+1], buf[i+2]
                    local dfw = dx * lx + dy * ly
                    local drt = dx * ly - dy * lx
                    if dfw >= -w and dfw <= hy + w
                            and drt >= -hx/2 - w and drt <= hx/2 + w
                            and dz >= -hz/2 - h and dz <= hz/2 + h then
                        -- Hit this entity
                        self:make_damage(world, ent, self.atk_damage, self.atk_knockback, self.atk_lift, lx, ly)
                        break
                    end
                end
            end
        end]]
    end
    if self.atk_cooldown > 0 then
        self.atk_cooldown = self.atk_cooldown - 1
    end
    if self.atk_ticks >= 0 then
        --Advance attack ticks
        if self.atk_ticks < self.atk_duration then
            self.atk_ticks = self.atk_ticks + 1
        else
            self.atk_ticks = -1
            self.atk_cooldown = self.atk_cooldown_end
        end
    end

    --Set animation from movement
    if self.roll_ticks >= 0 then
        self.anim:event('motion', 'roll', self.roll_ticks / self.roll_duration)
    elseif self.atk_ticks >= 0 then
        if self.atk_ticks >= self.atk_duration then
            self.anim:event('motion', 'atk', 0)
        else
            self.anim:event('motion', 'atk', 1)
        end
    elseif self.on_ground then
        if self.vel_x == 0 and self.vel_y == 0 then
            --Idle
            self.anim:event('motion', 'idle')
        else
            --Run
            self.anim:event('motion', 'run')
        end
    else
        --Airtime
        self.anim:event('motion', 'air')
    end

    return super.tick(self, world)
end

function Humanoid:apply_vel(world)
    local rx, ry, rz = self.rad_x, self.rad_y, self.rad_z
    self.tmp_pos:copy_from(self.pos)
    local mx, my, mz, cx, cy, cz = self.pos:move_box(world.terrain, self.vel_x, self.vel_y, self.vel_z, rx, ry, rz, true)
    if self.vel_z < 0 and cz and (cx or cy) then
        local mx1, my1, mz1, cx1, cy1, cz1 = self.tmp_pos:move_box(world.terrain, 0, 0, 1.001, rx, ry, rz, true)
        if not cz1 then
            local mx2, my2, mz2, cx2, cy2, cz2 = self.tmp_pos:move_box(world.terrain, self.vel_x, self.vel_y, self.vel_z, rx, ry, rz, true)
            if (not cx2 or not cy2) and cz2 then
                mx, my, mz, cx, cy, cz = mx1 + mx2, my1 + my2, mz1 + mz2, cx2, cy2, cz2
                self.pos:copy_from(self.tmp_pos)
            end
        end
    end
    self.on_ground = self.vel_z < 0 and cz
    self.mov_x, self.mov_y, self.mov_z = mx, my, mz
end

function Humanoid:damage(...)
    if self.roll_ticks >= self.roll_pre and self.roll_ticks < self.roll_immune then
        return
    end
    return super.damage(self, ...)
end

return Humanoid
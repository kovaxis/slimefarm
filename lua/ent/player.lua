
local voxel = require 'voxel'
local class = require 'class'
local input = require 'input'
local util = require 'util'
local Humanoid = require 'ent.humanoid'

local Player, super = class{ super = Humanoid }

Player:set_bbox(10/8, 15/8)

Player.atk_lounge = 0.1

local controls = {
    forward = 'w',
    left = 'a',
    back = 's',
    right = 'd',
    jump = 'space',
    roll = 'mouse_right',
    atk = 'mouse_left',
}

function Player:new()
    self.model = voxel.models.player
    super.new(self)
    self.hp = 100

    self.jump_was_down = false
    
    self.focus_height_lag = 0
    self.is_player = true
end

function Player:tick(world)
    --Check to see if colliding with entities
    do
        local buf = world.relpos_buf
        local baserad = math.max(self.rad_x, self.rad_y, self.rad_z) + 20
        for i, ent in ipairs(world.entities) do
            if ent.on_player_collision then
                local rad = baserad + math.max(ent.rad_x, ent.rad_y, ent.rad_z)
                --Get approximate relative position
                local dx, dy, dz = world.terrain:to_relative(ent.pos)
                if dx*dx + dy*dy + dz*dz <= rad*rad then
                    --Candidate to collision
                    local rx, ry, rz = self.rad_x + ent.rad_x, self.rad_y + ent.rad_y, self.rad_z + ent.rad_z
                    world.terrain:get_relative_positions(ent.pos, ent.rad_x, ent.rad_y, ent.rad_z, self.pos, buf)
                    for i = 1, #buf, 3 do
                        local dx, dy, dz = buf[i], buf[i+1], buf[i+2]
                        if dx >= -rx and dx <= rx and dy >= -ry and dy <= ry and dz >= -rz and dz <= rz then
                            --Collides with entity
                            ent:on_player_collision(world, self, dx, dy, dz)
                            break
                        end
                    end
                end
            end
        end
    end

    --Walk
    do
        local dx, dy = 0, 0
        if input.is_down[controls.left] then
            dx = dx - 1
        end
        if input.is_down[controls.right] then
            dx = dx + 1
        end
        if input.is_down[controls.forward] then
            dy = dy + 1
        end
        if input.is_down[controls.back] then
            dy = dy - 1
        end
        if dx ~= 0 and dy ~= 0 then
            dx = dx * 2^-0.5
            dy = dy * 2^-0.5
        end
        self.wx, self.wy = util.rotate_yaw(dx, dy, world.cam_yaw)
    end

    --Jump
    self.wjump = input.is_down[controls.jump] and (self.on_ground or not self.jump_was_down)
    self.wjumpkeep = input.is_down[controls.jump]
    self.jump_was_down = input.is_down[controls.jump]

    --Roll
    if input.is_down[controls.roll] then
        --Roll in the moving direction
        --Note that if not moving then we won't roll
        self.wroll_x, self.wroll_y = self.wx, self.wy
    else
        self.wroll_x, self.wroll_y = 0, 0
    end

    --Attack
    if input.is_down[controls.atk] then
        --Attack in the direction of the camera
        self.watk_x, self.watk_y = util.rotate_yaw(0, 1, world.cam_yaw)
    else
        self.watk_x, self.watk_y = 0, 0
    end

    --Smooth camera vertical jumps
    self.focus_height_lag = util.approach(self.focus_height_lag, 0, 0.9, 0.05)

    super.tick(self, world)

    --Move camera to point at player
    do
        local focus_height = 2 + self.focus_height_lag
        local focus_dist = 8
        local cam_wall_dist = 0.4
        world.cam_pos:copy_from(self.pos)
        world.cam_pos:move_box(world.terrain, 0, 0, focus_height, cam_wall_dist, cam_wall_dist, cam_wall_dist)
        world.cam_mov_x = world.cam_mov_x + self.mov_x
        world.cam_mov_y = world.cam_mov_y + self.mov_y
        world.cam_mov_z = world.cam_mov_z + self.mov_z
        world.cam_rollback = focus_dist
    end
end

function Player:apply_vel(world)
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
                self.focus_height_lag = self.focus_height_lag - 1
                world.cam_mov_z = world.cam_mov_z - 1
            end
        end
    end
    self.on_ground = self.vel_z < 0 and cz
    self.mov_x, self.mov_y, self.mov_z = mx, my, mz
end

return Player
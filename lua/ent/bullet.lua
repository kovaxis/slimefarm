
local class = require 'class'
local Entity = require 'ent.entity'
local util = require 'util'
local particles = require 'particles'

local Bullet, super = class { super = Entity }

Bullet.atk_hitbox = 0
Bullet.atk_damage = 0
Bullet.atk_knockback = 0
Bullet.atk_lift = 0

Bullet.group = ''
Bullet.target_group = ''

Bullet.timeout = 10*64
Bullet.range = 0

Bullet.death_particle_n = 6
Bullet.death_particle_id = particles.lookup 'living.death'
Bullet.death_particle_vel = 20

function Bullet:new()
    super.new(self)

    self.timeout = self.timeout
    self.acc_x = 0
    self.acc_y = 0
    self.acc_z = 0
end

function Bullet:pretick(world)
    --Purposely do not call super.pretick to avoid gravity and friction
end

function Bullet:tick(world)
    local death_anim = false

    --Time out bullet
    self.timeout = self.timeout - 1
    if self.timeout <= 0 then
        death_anim = true
        self.removing = true
    end

    --Check that owner exists
    local owner = world.ent_map[self.owner]
    if not owner then
        death_anim = true
        self.removing = true
    end

    --Check for collisions with entities in the target group
    if owner then
        local buf = world.relpos_buf
        local hitbox = self.atk_hitbox * .5
        local x1, y1, z1 = hitbox, hitbox, hitbox
        for id, ent in pairs(world.ent_groups[self.target_group]) do
            local rx, ry, rz = x1 + ent.rad_x, y1 + ent.rad_y, z1 + ent.rad_z
            world.terrain:get_relative_positions(ent.pos, rx, ry, rz, self.pos, buf)
            for i = 1, #buf, 3 do
                local dx, dy, dz = buf[i], buf[i+1], buf[i+2]
                if dx >= -rx and dx <= rx and dy >= -ry and dy <= ry and dz >= -rz and dz <= rz then
                    --Collision: hurt entity
                    if owner:make_damage(world, ent, self.atk_damage, self.atk_knockback, self.atk_lift, self.vel_x, self.vel_y) then
                        self.removing = true
                    end
                    break
                end
            end
        end
    end

    super.tick(self, world)

    --Accumulate distance and destroy on rangeout
    do
        local dx, dy, dz = self.acc_x, self.acc_y, self.acc_z
        dx = dx + self.mov_x
        dy = dy + self.mov_y
        dz = dz + self.mov_z
        if dx * dx + dy * dy + dz * dz >= self.range * self.range then
            death_anim = true
            self.removing = true
        end
        self.acc_x, self.acc_y, self.acc_z = dx, dy, dz
    end    

    --Destroy bullet on collision
    local dx, dy, dz = self.vel_x - self.mov_x, self.vel_y - self.mov_y, self.vel_z - self.mov_z
    if dx*dx + dy*dy + dz*dz > 0.001^2 then
        death_anim = true
        self.removing = true
    end

    --Spawn death particles if dying
    if death_anim then
        local id = self.death_particle_id
        local n = self.death_particle_n - 1
        for i = 0, n do
            local dx, dy, dz = util.fib_rand(i, n, world.rng, .2)
            local m = self.death_particle_vel
            world.terrain:add_particle(id, self.pos, m*dx, m*dy, m*dz, 0, 0, 0, 0)
        end
    end
end

function Bullet:draw(world)
    local frame = world.frame

    --Orient model
    local vx, vy, vz = self.vel_x, self.vel_y, self.vel_z
    local yaw, pitch = util.pos_to_yaw_pitch(vx, vy, vz)
    frame.mvp_world:rotate_z(yaw)
    frame.mvp_world:rotate_x(pitch)

    --Scale voxels down
    frame.mvp_world:scale(self.visual_scale)

    --Draw model
    self.anim:draw(frame.dt, world.shaders.terrain, frame.params_world, 'mvp', frame.mvp_world)
end

return Bullet
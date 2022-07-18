
local class = require 'class'
local util = require 'util'
local Entity = require 'ent.entity'
local Sprite = require 'sprite'
local particles = require 'particles'

local Living, super = class{ super = Entity }

Living.max_hp = 1
Living.knockback_resistance = 1
Living.armor = 1

Living.falldmg_minh = 10
Living.falldmg_maxh = 400
Living.falldmg_multiplier = 3

Living.wander_dist = 50

Living.dmg_anim_factor = 0.01
Living.dmg_anim_linear = 1
Living.healthbar_dist = 30
Living.healthbar_z = 1

Living.death_particle_n = 30
Living.death_particle_id = particles.lookup 'living.death'
Living.death_particle_vel_min = 10
Living.death_particle_vel_max = 25
Living.death_particle_rotvel = 10

Living.group = 'enemy'

function Living.death_particle(kind)
    kind.friction = kind.friction or .02
    kind.acc = kind.acc or {0, 0, -5}
    kind.rot_friction = kind.rot_friction or .15
    kind.color_interval = kind.color_interval or 1
    kind.color = kind.color or {{1, 1, 1}}
    kind.size_interval = kind.size_interval or .6
    kind.size = kind.size or {.3, 0}
    kind.physical_size = kind.physical_size or .25
    kind.lifetime = kind.lifetime or .6
    return kind
end

particles.register(Living.death_particle {
    name = 'living.death',
})

function Living:new()
    super.new(self)

    self.acc_x = 0
    self.acc_y = 0
    self.acc_z = 0

    self.hp = self.hp or self.max_hp
    self.fall_height = 0

    self.visual_dmg_r = 0
    self.visual_dmg_g = 0
    self.visual_dmg_b = 0
end

function Living:make_damage(world, target, dmg, knockback, lift, kx, ky)
    --Normalize xy knockback
    local n = (kx*kx + ky*ky)^-.5
    kx, ky = n*kx, n*ky
    --Apply lift and normalize again
    local kz
    kx, ky, kz = kx, ky, lift
    n = knockback * (kx*kx + ky*ky + kz*kz)^-.5
    kx, ky, kz = n*kx, n*ky, n*kz
    --Deal damage with the calculated knockback
    return target:damage(world, dmg, kx, ky, kz)
end

function Living:damage(world, dmg, kx, ky, kz)
    --Remove health
    self.hp = self.hp - dmg * self.armor
    --Kill if no hp left
    if self.hp <= 0 then
        --Spawn death particles
        local id = self.death_particle_id
        local n = self.death_particle_n - 1
        for i = 0, n do
            local dx, dy, dz = util.fib_rand(i, n, world.rng, .2)
            local rx, ry, rz = world.rng:normal(-1, 1), world.rng:normal(-1, 1), world.rng:normal(-1, 1)
            local m = world.rng:normal(self.death_particle_vel_min, self.death_particle_vel_max)
            world.terrain:add_particle(id, self.pos, m*dx, m*dy, m*dz, self.death_particle_rotvel, rx, ry, rz)
        end
        self.removing = true
    end
    --Apply knockback
    kx = kx or 0
    ky = ky or 0
    kz = kz or 0
    local kres = self.knockback_resistance
    kx, ky, kz = kx * kres, ky * kres, kz * kres
    self.vel_x = self.vel_x + kx
    self.vel_y = self.vel_y + ky
    self.vel_z = self.vel_z + kz
    --Start damage animation
    self.visual_dmg_r = 1
    self.visual_dmg_g = 1
    self.visual_dmg_b = 1
    
    return dmg * self.armor
end

function Living:pretick(world)
    --Accumulate total traveled distance from spawn
    self.acc_x = self.acc_x + self.mov_x
    self.acc_y = self.acc_y + self.mov_y
    self.acc_z = self.acc_z + self.mov_z

    --Accumulate fall height
    if self.mov_z >= 0 or self.on_ground then
        self.fall_height = 0
    else
        self.fall_height = self.fall_height - self.mov_z
    end

    return super.pretick(self, world)
end

function Living:tick(world)
    super.tick(self, world)

    --Deal fall damage
    if self.on_ground and self.fall_height > 0 then
        local dmg = math.floor((math.min(self.fall_height, self.falldmg_maxh) - self.falldmg_minh) * self.falldmg_multiplier)
        if dmg > 0 then
            self:damage(world, dmg)
        end
    end
end

function Living:draw(world, dx, dy, dz)
    local frame = world.frame

    --Draw healthbar
    if dx*dx + dy*dy + dz*dz < self.healthbar_dist * self.healthbar_dist then
        local x, y, z, w = frame.mvp_world:transform_vec4(0, 0, self.rad_z + self.healthbar_z, 1)
        if z > -w and z < w then
            local size = 0.07
            local icon, i = Sprite.sprites.overbars, 1
            x, y, z = x / w, y / w, z / w
            frame.mvp_hud:push()
            frame.mvp_hud:identity()
            frame.mvp_hud:translate(x, y, z)
            frame.mvp_hud:scale(size * icon.w / w, size * icon.h * frame.aspect / w, 1)
            icon:draw(i, frame.mvp_hud, frame.params_world)
            frame.mvp_hud:translate(-.5, 0, 0)
            frame.mvp_hud:scale(self.hp / self.max_hp, 1, 1)
            frame.mvp_hud:translate(.5, 0, 0)
            frame.params_world:set_polygon_offset(true, 1, 1)
            icon:draw(i+1, frame.mvp_hud, frame.params_world)
            frame.params_world:set_polygon_offset(false, 0, 0)
            frame.mvp_hud:pop()
        end
    end

    --Draw model with damage animation tint
    do
        local r, g, b = self.visual_dmg_r, self.visual_dmg_g, self.visual_dmg_b
        r = util.approach(r, 0, self.dmg_anim_factor, self.dmg_anim_linear, frame.dt)
        g = util.approach(g, 0, self.dmg_anim_factor, self.dmg_anim_linear, frame.dt)
        b = util.approach(b, 0, self.dmg_anim_factor, self.dmg_anim_linear, frame.dt)
        self.visual_dmg_r, self.visual_dmg_g, self.visual_dmg_b = r, g, b

        local upd = r ~= 0 or g ~= 0 or b ~= 0
        if upd then
            world.shaders.terrain:set_vec3('tint', r, g, b)
        end
        super.draw(self, world)
        if upd then
            world.shaders.terrain:set_vec3('tint', 0, 0, 0)
        end
    end
end

return Living

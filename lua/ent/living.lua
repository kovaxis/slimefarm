
local class = require 'class'
local util = require 'util'
local Entity = require 'ent.entity'
local Sprite = require 'sprite'

local Living, super = class{ super = Entity }

Living.max_hp = 1
Living.knockback_resistance = 1
Living.armor = 1

Living.falldmg_minh = 10
Living.falldmg_maxh = 100
Living.falldmg_multiplier = 5

Living.dmg_anim_factor = 0.01
Living.dmg_anim_linear = 1
Living.healthbar_dist = 30
Living.healthbar_z = 1

function Living:new()
    super.new(self)

    self.hp = self.hp or self.max_hp
    self.fall_height = 0

    self.visual_dmg_r = 0
    self.visual_dmg_g = 0
    self.visual_dmg_b = 0
end

function Living:damage(dmg, kx, ky, kz)
    self.hp = self.hp - dmg * self.armor
    if self.hp <= 0 then
        self.remove = true
    end
    self.visual_dmg_r = 1
    self.visual_dmg_g = 1
    self.visual_dmg_b = 1
    kx = kx or 0
    ky = ky or 0
    kz = kz or 0
    local kres = self.knockback_resistance
    kx, ky, kz = kx * kres, ky * kres, kz * kres
    self.vel_x = self.vel_x + kx
    self.vel_y = self.vel_y + ky
    self.vel_z = self.vel_z + kz
end

function Living:tick(world)
    super.tick(self, world)

    if self.on_ground and self.fall_height > 0 then
        local dmg = math.floor((math.min(self.fall_height, self.falldmg_maxh) - self.falldmg_minh) * self.falldmg_multiplier)
        if dmg > 0 then
            self:damage(dmg)
        end
    end

    if self.mov_z >= 0 or self.on_ground then
        self.fall_height = 0
    else
        self.fall_height = self.fall_height - self.mov_z
    end
end

function Living:draw(world, dx, dy, dz)
    local frame = world.frame

    --Draw healthbar
    if dx*dx + dy*dy + dz*dy < self.healthbar_dist * self.healthbar_dist then
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

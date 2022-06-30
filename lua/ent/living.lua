
local class = require 'class'
local util = require 'util'
local Entity = require 'ent.entity'

local Living, super = class{ super = Entity }

Living.knockback_resistance = 1
Living.armor = 1

Living.dmg_anim_factor = 0.01
Living.dmg_anim_linear = 1

Living.falldmg_minh = 10
Living.falldmg_maxh = 100
Living.falldmg_multiplier = 5

function Living:new()
    super.new(self)

    self.hp = 1
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

function Living:draw(world)
    local frame = world.frame

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

return Living


local class = require 'class'
local util = require 'util'

local Sprite = class{}

Sprite.shader = util.Shader{
    vertex = 'textured.vert',
    fragment = 'textured.frag',
    uniforms = {'mvp', 'tex', 'tint'},
}

function Sprite:new()
    assert(self.w)
    assert(self.h)
    assert(self.path)
    self.texture = gfx.texture(system.image('../image/'..self.path))
    self.texture:set_min('linear')
    self.texture:set_mag('nearest')
    self.full_w, self.full_h = self.texture:dimensions()
    self.pad_x = self.pad_x or 0
    self.pad_y = self.pad_y or 0
    assert(self.full_w % self.w == self.pad_x)
    assert(self.full_h % self.h == self.pad_y)
    self.cols, self.rows = (self.full_w - self.pad_x) / self.w, (self.full_h - self.pad_y) / self.h
    local tw, th = (self.full_w - self.pad_x) / self.full_w, (self.full_h - self.pad_y) / self.full_h
    self.buffers = {}
    local pos = {
        -0.5, -0.5,
         0.5, -0.5,
         0.5,  0.5,
        -0.5,  0.5,
    }
    local idx = {0, 1, 2, 2, 3, 0}
    for y = 1, self.rows do
        for x = 1, self.cols do
            local u0 = (x - 1) * tw / self.cols
            local u1 = x * tw / self.cols
            local v0 = y * th / self.rows
            local v1 = (y - 1) * th / self.rows
            local tex = {
                u0, v0,
                u1, v0,
                u1, v1,
                u0, v1,
            }
            local buf = gfx.buffer_2d(pos, tex, idx)
            table.insert(self.buffers, buf)
        end
    end
end

function Sprite:draw(i, mvp, draw_params)
    local buf = self.buffers[i]
    Sprite.shader:set_matrix('mvp', mvp)
    Sprite.shader:set_texture_2d('tex', self.texture)
    Sprite.shader:set_vec4('tint', 1, 1, 1, 1)
    Sprite.shader:draw(buf, draw_params)
end

Sprite.sprites = {}

local function load(data)
    if not data.path then
        data.path = data.name..'.png'
    end
    Sprite.sprites[data.name] = Sprite(data)
end

load {
    name = 'crosshair',
    w = 16,
    h = 16,
}

load {
    name = 'overbars',
    w = 56,
    h = 8,
    pad_x = 8,
}

return Sprite
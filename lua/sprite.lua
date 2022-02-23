
local class = require 'class'
local util = require 'util'

local Sprite = class{}

local shader = util.Shader{
    vertex = 'textured.vert',
    fragment = 'textured.frag',
    uniforms = {'mvp', 'tex'},
}

function Sprite:new()
    assert(self.w)
    assert(self.h)
    assert(self.path)
    self.texture = gfx.texture(system.image('../img/'..self.path))
    self.full_w, self.full_h = self.texture:dimensions()
    assert(self.full_w % self.w == 0)
    assert(self.full_h % self.h == 0)
    self.cols, self.rows = self.full_w / self.w, self.full_h / self.h
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
            local u0 = (x - 1) / self.cols
            local u1 = x / self.cols
            local v0 = y / self.rows
            local v1 = (y - 1) / self.rows
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
    shader:set_matrix('mvp', mvp)
    shader:set_texture('tex', self.texture)
    shader:draw(buf, draw_params)
end

return Sprite
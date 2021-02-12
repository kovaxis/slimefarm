
local class = require 'class'

local util = {}

function util.read_file(path)
    local file = assert(io.open('../'..path, 'rb'))
    local str = file:read('a')
    file:close()
    return str
end

function util.abs_min(x, abs)
    if x < 0 then
        abs = -abs
        if abs > x then
            return abs
        else
            return x
        end
    else
        if abs < x then
            return abs
        else
            return x
        end
    end
end
function util.abs_max(x, abs)
    if x < 0 then
        abs = -abs
        if abs < x then
            return abs
        else
            return x
        end
    else
        if abs > x then
            return abs
        else
            return x
        end
    end
end

util.Shader = class{}

function util.Shader:new()
    assert(self.vertex)
    assert(self.fragment)
    assert(self.uniforms)
    self.program = gfx.shader(
        util.read_file('shader/'..self.vertex),
        util.read_file('shader/'..self.fragment)
    )
    self.raw_uniforms = gfx.uniforms()
    for i, name in ipairs(self.uniforms) do
        self.raw_uniforms:add(name)
        self.uniforms[name] = i - 1
    end
end

function util.Shader:set_float(name, x)
    self.raw_uniforms:set_float(self.uniforms[name], x)
end
function util.Shader:set_vec2(name, x, y)
    self.raw_uniforms:set_vec2(self.uniforms[name], x, y)
end
function util.Shader:set_vec3(name, x, y, z)
    self.raw_uniforms:set_vec3(self.uniforms[name], x, y, z)
end
function util.Shader:set_vec4(name, x, y, z, w)
    self.raw_uniforms:set_vec4(self.uniforms[name], x, y, z, w)
end
function util.Shader:set_matrix(name, mat4)
    self.raw_uniforms:set_matrix(self.uniforms[name], mat4)
end
function util.Shader:set_texture(name, tex)
    self.raw_uniforms:set_texture(self.uniforms[name], tex)
end

function util.Shader:draw(buf, draw_params)
    gfx.draw(buf, self.program, self.raw_uniforms, draw_params)
end

function util.Shader:draw_terrain(terr, offset_uniform, draw_params, x, y, z)
    offset_uniform = self.uniforms[offset_uniform]
    terr:draw(self.program, self.raw_uniforms, offset_uniform, draw_params, x, y, z)
end

return util
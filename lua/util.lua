
local class = require 'class'

local util = {}

function util.read_file(path)
    local file = assert(io.open('../'..path, 'rb'))
    local str = file:read('a')
    file:close()
    return str
end

function util.format_time(secs)
    if secs < 0.0001 then
        return (math.ceil(secs * 10000000) / 10) .. "us"
    elseif secs < 0.25 then
        return (math.ceil(secs * 10000) / 10) .. "ms"
    elseif secs < 60 then
        return (math.ceil(secs * 10) / 10) .. "s"
    elseif secs < 3600 then
        return (secs // 60) .. "m " .. math.ceil(secs % 60) .. "s"
    else
        return (secs // 3600) .."h " .. math.ceil(secs % 3600) .. "m"
    end
end

-- rotates the given (x, y) vector around the z+ axis according to the given yaw.
function util.rotate_yaw(x, y, yaw)
    return x * math.cos(yaw) - y * math.sin(yaw), x * math.sin(yaw) + y * math.cos(yaw)
end

-- computes the yaw that the given (x, y) vector points towards.
-- yaw 0 is defined to be towards (0, 1) (into the screen by default), and angles grow around the Z
-- axis in a right-handed coordinate system, that is, angles grow from the X axis to the Y axis:
--  (0, 1) -> 0
--  (-1, 0) -> 90
--  (0, -1)  -> 180
--  (1, 0)  -> 270
function util.pos_to_yaw(dx, dy)
    return math.atan(-dx, dy)
end

function util.dot(x0, y0, z0, x1, y1, z1)
    return x0 * x1 + y0 * y1 + z0 * z1
end

function util.cross(x0, y0, z0, x1, y1, z1)
    return y0 * z1 - y1 * z0, z0 * x1 - x0 * z1, x0 * y1 - x1 * y0
end

function util.normalize(x, y, z)
    local inv = (x * x + y * y + z * z) ^ -0.5
    return inv * x, inv * y, inv * z
end

function util.approach(cur, target, factor, linear, dt)
    dt = dt or 1
    local og_delta = cur - target
    if target < cur then
        linear = -linear
    end
    local delta = og_delta * factor ^ dt + linear * dt
    if og_delta * delta <= 0 then
        return target
    else
        return target + delta
    end
end

function util.smoothstep(x)
    if x < 0 then
        return 0
    elseif x > 1 then
        return 1
    else
        local sq = x * x
        return 3 * sq - 2 * sq * x
    end
end

util.Curve = class{}

function util.Curve:new()
    assert(#self % 2 == 0, "curve values must come in pairs")
    assert(#self >= 2, "curves must have at least one point")
    local last = -math.huge
    local overall_ty = nil
    for i = 1, #self, 2 do
        if self[i] < last then
            error("curve points must be ordered", 2)
        end
        local y = self[i + 1]
        local ty
        if type(y) == 'number' then
            ty = 'single'
        elseif type(y) == 'table' then
            if #y == 2 then
                ty = 'vec2'
            elseif #y == 3 then
                ty = 'vec3'
            elseif #y == 4 then
                ty = 'vec4'
            end
        end
        if not ty then
            print("unknown value type for '"..tostring(y).."'", 2)
        end
        if overall_ty then
            assert(overall_ty == ty, "values with different types")
        else
            overall_ty = ty
        end
        last = self[i]
    end
    self.ty = overall_ty
    self.smooth = self.smooth or util.smoothstep
end

function util.Curve:at(x)
    local prev, next = #self - 1, #self - 1
    for i = 1, #self, 2 do
        if self[i] > x then
            prev = i - 2
            if prev < 1 then
                prev = 1
            end
            next = i
            break
        end
    end
    local x0 = self[prev]
    local y0 = self[prev + 1]
    local x1 = self[next]
    local y1 = self[next + 1]
    local s = self.smooth((x - x0) / (x1 - x0))
    if self.ty == 'single' then
        return y0 + s * (y1 - y0)
    elseif self.ty == 'vec2' then
        return y0[1] + s * (y1[1] - y0[1]), y0[2] + s * (y1[2] - y0[2])
    elseif self.ty == 'vec3' then
        return y0[1] + s * (y1[1] - y0[1]), y0[2] + s * (y1[2] - y0[2]), y0[3] + s * (y1[3] - y0[3])
    elseif self.ty == 'vec4' then
        return y0[1] + s * (y1[1] - y0[1]), y0[2] + s * (y1[2] - y0[2]), y0[3] + s * (y1[3] - y0[3]), y0[4] + s * (y1[4] - y0[4])
    end
end

util.DebugTimer = class{}

function util.DebugTimer:new()
    self.base = os.clock()
    self.phases = {}
    self.last_phases = {}
    self.rope = {}
end

function util.DebugTimer:finish()
    self.phases, self.last_phases = self.last_phases, self.phases
    for i = #self.phases, 1, -1 do
        self.phases[i] = nil
    end
end

function util.DebugTimer:mark(name, now)
    now = now or os.clock()
    local time = now - self.base
    self.base = now
    local phases = self.phases
    local len = #phases
    phases[len + 1] = name
    phases[len + 2] = time
end

function util.DebugTimer:to_str()
    local phases = self.last_phases
    local rope = self.rope
    for i = #rope, 1, -1 do
        rope[i] = nil
    end
    local j = 1
    for i = 1, #phases, 2 do
        rope[j + 0] = "| "
        rope[j + 1] = phases[i]
        rope[j + 2] = " "
        rope[j + 3] = math.ceil(phases[i + 1] * 100000) / 100
        rope[j + 4] = "ms "
        j = j + 5
    end
    rope[j] = "|"
    return table.concat(rope)
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
function util.Shader:set_texture_2d(name, tex)
    self.raw_uniforms:set_texture_2d(self.uniforms[name], tex)
end

function util.Shader:draw(buf, draw_params)
    gfx.draw(buf, self.program, self.raw_uniforms, draw_params)
end

function util.Shader:draw_terrain(terr, offset, atlas, draw_params, mvp, locate, subdraw)
    terr:draw(self.program, self.raw_uniforms, offset, atlas, draw_params, mvp, locate, subdraw)
end

return util

local class = require 'class'
local util = require 'util'

local Mesh = class{}

function Mesh:new()
    self.vertices = {}
    self.normals = {}
    self.colors = {}
    self.indices = {}
end

function Mesh:add_vertex(x, y, z, nx, ny, nz, r, g, b, a)
    local v = self.vertices
    local n = self.normals
    local c = self.colors
    local len = #v
    v[len + 1] = x
    v[len + 2] = y
    v[len + 3] = z
    n[len + 1] = nx
    n[len + 2] = ny
    n[len + 3] = nz
    len = #c
    c[len + 1] = r
    c[len + 2] = g
    c[len + 3] = b
    c[len + 4] = a
    return self
end

function Mesh:add_quad(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3, r, g, b, a)
    r, g, b, a = r or 1, g or 1, b or 1, a or 1
    local i = self.indices
    local base = #self.vertices / 3
    local len = #i

    local nx, ny, nz = util.cross(x1 - x0, y1 - y0, z1 - z0,   x3 - x0, y3 - y0, z3 - z0)
    nx, ny, nz = util.normalize(nx, ny, nz)

    self:add_vertex(x0, y0, z0, nx, ny, nz, r, g, b, a)
    self:add_vertex(x1, y1, z1, nx, ny, nz, r, g, b, a)
    self:add_vertex(x2, y2, z2, nx, ny, nz, r, g, b, a)
    self:add_vertex(x3, y3, z3, nx, ny, nz, r, g, b, a)

    i[len + 1] = base + 0
    i[len + 2] = base + 1
    i[len + 3] = base + 2

    i[len + 4] = base + 0
    i[len + 5] = base + 2
    i[len + 6] = base + 3

    return self
end

function Mesh:add_cube(x, y, z, w, d, h, r, g, b, a)
    r = r or 1
    g = g or 1
    b = b or 1
    a = a or 1

    w, d, h = w / 2, d / 2, h / 2
    local x0, y0, z0 = x - w, y - d, z - h
    local x1, y1, z1 = x + w, y + d, z + h

    self:add_quad(x0, y0, z0,  x1, y0, z0,  x1, y0, z1,  x0, y0, z1, r, g, b, a)
    self:add_quad(x1, y0, z0,  x1, y1, z0,  x1, y1, z1,  x1, y0, z1, r, g, b, a)
    self:add_quad(x1, y1, z0,  x0, y1, z0,  x0, y1, z1,  x1, y1, z1, r, g, b, a)
    self:add_quad(x0, y1, z0,  x0, y0, z0,  x0, y0, z1,  x0, y1, z1, r, g, b, a)
    self:add_quad(x0, y0, z0,  x0, y1, z0,  x1, y1, z0,  x1, y0, z0, r, g, b, a)
    self:add_quad(x0, y0, z1,  x1, y0, z1,  x1, y1, z1,  x0, y1, z1, r, g, b, a)

    --[[
    local v = self.vertices
    local i = self.indices
    local base = #v / 3
    local l = #i

    self:add_vertex(x0, y0, z0, r, g, b, a)
    self:add_vertex(x1, y0, z0, r, g, b, a)
    self:add_vertex(x0, y1, z0, r, g, b, a)
    self:add_vertex(x1, y1, z0, r, g, b, a)
    self:add_vertex(x0, y0, z1, r, g, b, a)
    self:add_vertex(x1, y0, z1, r, g, b, a)
    self:add_vertex(x0, y1, z1, r, g, b, a)
    self:add_vertex(x1, y1, z1, r, g, b, a)

    i[l + 1] = base + 0
    i[l + 2] = base + 2
    i[l + 3] = base + 1
    
    i[l + 4] = base + 1
    i[l + 5] = base + 2
    i[l + 6] = base + 3

    i[l + 7] = base + 0
    i[l + 8] = base + 4
    i[l + 9] = base + 2

    i[l + 10] = base + 2
    i[l + 11] = base + 4
    i[l + 12] = base + 6

    i[l + 13] = base + 0
    i[l + 14] = base + 1
    i[l + 15] = base + 5

    i[l + 16] = base + 0
    i[l + 17] = base + 5
    i[l + 18] = base + 4

    i[l + 19] = base + 1
    i[l + 20] = base + 3
    i[l + 21] = base + 5

    i[l + 22] = base + 3
    i[l + 23] = base + 7
    i[l + 24] = base + 5

    i[l + 25] = base + 4
    i[l + 26] = base + 5
    i[l + 27] = base + 7

    i[l + 28] = base + 4
    i[l + 29] = base + 7
    i[l + 30] = base + 6

    i[l + 31] = base + 2
    i[l + 32] = base + 7
    i[l + 33] = base + 3

    i[l + 34] = base + 2
    i[l + 35] = base + 6
    i[l + 36] = base + 7
    ]]

    return self
end

function Mesh:as_buffer()
    return gfx.buffer_3d(self.vertices, self.normals, self.colors, self.indices)
end

return Mesh
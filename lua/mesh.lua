
local mesh = {}
mesh.__index = mesh

function mesh:new()
    self.vertices = {}
    self.colors = {}
    self.indices = {}
    setmetatable(self, mesh)
    return self
end

function mesh:add_vertex(x, y, z, r, g, b, a)
    local v = self.vertices
    local len = #v
    v[len + 1] = x
    v[len + 2] = y
    v[len + 3] = z
    local c = self.colors
    len = #c
    c[len + 1] = r
    c[len + 2] = g
    c[len + 3] = b
    c[len + 4] = a
    return self
end

function mesh:add_cube(x, y, z, w, h, d, r, g, b, a)
    r = r or 1
    g = g or 1
    b = b or 1
    a = a or 1

    local v = self.vertices
    local i = self.indices
    local base = #v / 3
    local l = #i

    w, h, d = w / 2, h / 2, d / 2
    local x0, y0, z0 = x - w, y - h, z - d
    local x1, y1, z1 = x + w, y + h, z + d

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
end

function mesh:as_buffer()
    return gfx.buffer(self.vertices, self.colors, self.indices)
end

return mesh
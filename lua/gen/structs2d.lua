
local class = require 'class'
local native = require 'gen.native'

local Structs2d = class{}

function Structs2d:new()
    assert(self.salt ~= nil, "salt must be a value")
    assert(type(self.spread) == 'number', "spread must be a number")
    assert(type(self.margin) == 'number', "margin must be a number")
    assert(type(self.generate) == 'function', "gen must be a blockbuf-filler function")
    assert(type(self.get_z) == 'function', "get_z must be a function")
    self.generate_bound = function(wx, wy, cx, cy)
        -- world x, y
        self.wx, self.wy = wx, wy
        -- cell x, y
        self.cx, self.cy = cx, cy
        -- block x, y
        self.bx, self.by = math.floor(wx), math.floor(wy)
        -- fractional x, y
        self.fx, self.fy = wx - self.bx, wy - self.by
        -- world z
        self.wz = self:get_z()
        -- block z
        self.bz = math.floor(self.wz)
        -- fractional z
        self.fz = self.wz - self.bz
        self.bbuf:reset(self.bx, self.by, self.bz)
        self.rng:reseed(math.hash(gen.seed, self.salt, cx, cy))
        self:generate()
        return self.bbuf
    end

    self.gridbuf = native.gridbuf_2d {
        seed = math.hash(gen.seed, self.salt),
        cell_size = math.floor(self.spread + .5),
        margin = math.floor(self.margin + .5),
    }
    self.bbuf = native.action_buf()
    self.rng = math.rng(0)
end

-- Fill the chunk at X Y Z, generating structure lazily using `self.generate`.
function Structs2d:fill(x, y, z, chunk)
    self.gridbuf:fill_chunk(x, y, z, chunk, self.generate_bound)
end

return Structs2d

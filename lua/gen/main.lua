
local native = require 'gen.native'
local blocks = require 'gen.blocks'
local lightmodes = require 'gen.lightmodes'
local reclaim = require 'gen.reclaim'

gen.k = ...
gen.seed = assert(gen.k.seed, "no seed!")

local filler = require(gen.k.kind)

blocks.register {
    name = 'base.air',
    style = 'Clear',
}

blocks.register {
    name = 'base.empty',
    style = 'Clear',
}

blocks.register {
    name = 'base.portal',
    style = 'Portal',
}

lightmodes.register {
    name = 'base.std',
    light = {
        -- 1/4 per block
        base = 0,
        mul = 1,
        shr = 2,
    },
    _decay = {
        base = 8,
        mul = 0,
        shr = 0,
    },
    decay = {
        -- 8 + [0, 0]
        base = 8 * 2^16,
        mul = math.floor(0 * 2^16 / 255),
        shr = 16,
    },
}

function gen.textures()
    blocks.seal()
    return blocks.textures
end

function gen.lightmodes()
    lightmodes.seal()
    return lightmodes.modes
end

function gen.chunk(x, y, z, w)
    return filler.generate(x, y, z, w)
end

function gen.gc()
    reclaim.reclaim()
end

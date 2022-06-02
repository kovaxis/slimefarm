
local native = require 'gen.native'
local blocks = require 'gen.blocks'
local lightmodes = require 'gen.lightmodes'
local reclaim = require 'gen.reclaim'

gen.k = ...
gen.seed = assert(gen.k.seed, "no seed!")

local filler = require(gen.k.kind)

lightmodes.register {
    name = 'base.std',
    -- 1/4 per block
    light = {
        base = 0,
        mul = 1,
        shr = 2,
    },
    -- 8 per block
    decay = {
        base = 8,
        mul = 0,
        shr = 0,
    },
}

function gen.textures()
    blocks.seal()
    return blocks.textures
end

function gen.lightmodes()
    lightmodes.seal()
    print("lightmodes:")
    for k, v in pairs(lightmodes.modes) do
        print("  "..tostring(k)..": "..tostring(v))
    end
    return lightmodes.modes
end

function gen.chunk(x, y, z, w)
    return filler.generate(x, y, z, w)
end

function gen.gc()
    reclaim.reclaim()
end

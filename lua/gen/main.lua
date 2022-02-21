
local native = require 'gen.native'
local blocks = require 'gen.blocks'

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

function gen.textures()
    blocks.seal()
    return blocks.textures
end

function gen.chunk(x, y, z, w)
    return filler.generate(x, y, z, w)
end

function gen.gc()

end

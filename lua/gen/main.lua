
local native = require 'gen.native'
local blockreg = require 'gen.blockreg'

gen.k = ...
gen.seed = assert(gen.k.seed, "no seed!")

local filler = require(gen.k.kind)

blockreg.register {
    name = 'base.air',
    style = 'Clear',
}

blockreg.register {
    name = 'base.empty',
    style = 'Clear',
}

blockreg.register {
    name = 'base.portal',
    style = 'Portal',
}

function gen.textures()
    blockreg.seal()
    return blockreg.textures
end

function gen.chunk(x, y, z, w)
    local ok, res = pcall(filler.generate, x, y, z, w)
    if ok then
        return res
    else
        print("gen.chunk() errored with: "..tostring(res))
        os.exit()
    end
    --return filler.generate(x, y, z, w)
end

function gen.gc()

end

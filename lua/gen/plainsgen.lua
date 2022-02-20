
local native = require 'gen.native'
local blockreg = require 'gen.blockreg'

local plainsgen = {}

local function perlin(period, scale, octaves, lacunarity, persistence)
    lacunarity = lacunarity or 0.5
    persistence = persistence or 0.5
    local layers = {}
    for i = 1, octaves do
        layers[i] = {period, scale}
        period = period * lacunarity
        scale = scale * persistence
    end
    return layers
end

local function texture(block)
    local base = block.base
    local noise = block.noise
    local noisescales = block.octs
    if #noise == 3 then
        noise[4] = 0
    end
    local noiselvls = 6
    while #noisescales < noiselvls do
        table.insert(noisescales, 0)
    end
    local noises = {}
    for i = 1, noiselvls do
        noises[i] = {
            noise[1] * noisescales[i],
            noise[2] * noisescales[i],
            noise[3] * noisescales[i],
            noise[4] * noisescales[i],
        }
    end
    return {
        name = block.name,
        style = 'Solid',
        smooth = not block.rough,
        base = base,
        noise = noises,
    }
end

local heightmap = native.heightmap {
    seed = math.hash(gen.seed, "plains_heightmap"),
    noise = perlin(256, 0.8, 3),
    offset = 0,
    scale = 32,
    ground = blockreg.register(texture {
        name = 'base.grass',
        base = {0.43, 0.61, 0.10, 0.05},
        noise = {0.116, 0.131, 0.053},
        octs = {0, 0, 0, 0, 0, 0.8},
        rough = false,
    }),
    air = blockreg.lookup 'base.air',
}

function plainsgen.generate(x, y, z, w)
    local chunk = native.chunk(blockreg.blocks['base.air'])
    heightmap:fill_chunk(x, y, z, chunk)
    return chunk:into_raw()
end

return plainsgen
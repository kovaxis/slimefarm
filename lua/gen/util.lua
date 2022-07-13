--Worldgen utilities.

local genutil = {}

-- Prepare a perlin layer configuration using the given period, scale, number of octaves,
-- lacunarity (decrease in period per octave, defaults to 1/2) and persistence (decrease in scale
-- per octave, defaults to 1/2).
function genutil.perlin(period, scale, octaves, lacunarity, persistence)
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

-- Prepare a block texture.
function genutil.texture(block)
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

return genutil
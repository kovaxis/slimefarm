
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

blockreg.register(texture {
    name = 'base.wood',
    base = {0.31, 0.19, 0.13, 0.01},
    noise = {0.136, 0.089, 0.065},
    octs = {0.2, 0.4},
    rough = true,
})

local structs = native.structure_grid_2d {
    seed = math.hash(gen.seed, "plains_structs"),
    cell_size = 19,
    margin = 8,
}
local genstruct
do
    local bbuf = native.action_buf()
    local rng = math.rng(0)

    local max_lentotal = 100
    local min_area = 2
    local initial_area = {30, 35}
    local branch_dist = {4, 7}
    local half_dist = 30
    local area_per_len = 0.5
    local off_angle = math.rad(40)
    local off_area = {0.3, 0.4}
    local main_rotate = math.rad(133)
    local function branch(pos, up, norm, area, lentotal, depth)
        if lentotal > max_lentotal or area < min_area then
            return
        end

        local len = rng:normal(branch_dist[1], branch_dist[2]) * 0.5 ^ (lentotal / half_dist)

        local top = math.vec3(up)
        top:mul(len)
        top:add(pos)

        local r0 = math.sqrt(area)
        area = area - len * area_per_len
        
        local off_a = rng:normal(off_area[1], off_area[2])

        
        if depth < 3 then
            local subup = math.vec3(up)
            subup:rotate(off_angle, up, norm)
            
            local subnorm = math.vec3(norm)
            subup:rotate(off_angle, up, norm)

            branch(top, subup, subnorm, area * off_a, lentotal + len, depth + 1)
        end
        
        norm:rotate(main_rotate, up)
        branch(top, up, norm, area * (1 - off_a), lentotal + len, depth)

        bbuf:cylinder(
            pos:x(), pos:y(), pos:z(), r0,
            top:x(), top:y(), top:z(), math.sqrt(area * math.min(off_a, 1 - off_a)),
            blockreg.blocks['base.wood']
        )
    end
    local function tree(pos)
        local area = rng:normal(initial_area[1], initial_area[2])
        branch(pos, math.vec3(0, 0, 1), math.vec3(1, 0, 0), area, 0, 0)
    end
    function genstruct(rx, ry, sx, sy)
        local bx, by = math.floor(rx), math.floor(ry)
        local bz = heightmap:height_at(bx, by)
        local fx, fy = rx - bx, ry - by
        bbuf:reset(bx, by, bz)
        rng:reseed(math.hash(gen.seed, "plains_tree", sx, sy))
        tree(math.vec3(fx, fy, 0))
        return bbuf
    end
end

function plainsgen.generate(x, y, z, w)
    local chunk = native.chunk(blockreg.blocks['base.air'])
    heightmap:fill_chunk(x, y, z, chunk)
    structs:fill_chunk(x, y, z, chunk, genstruct)
    return chunk:into_raw()
end

return plainsgen
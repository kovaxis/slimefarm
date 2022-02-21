
local native = require 'gen.native'
local blocks = require 'gen.blocks'

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
    ground = blocks.register(texture {
        name = 'base.grass',
        base = {0.43, 0.61, 0.10, 0.05},
        noise = {0.116, 0.131, 0.053},
        octs = {0, 0, 0, 0, 0, 0.8},
        rough = false,
    }),
    air = blocks.lookup 'base.air',
}

blocks.register(texture {
    name = 'base.wood',
    base = {0.31, 0.19, 0.13, 0.01},
    noise = {0.136, 0.089, 0.065},
    octs = {0.2, 0.4},
    rough = true,
})
blocks.register(texture {
    name = 'base.leaf',
    base = {0.03, 0.26, 0.13, 0.025},
    noise = {0.045, 0.121, 0.116},
    octs = {0.1, 0, 0, 0.2},
    rough = true,
})

local structs, genstruct
do
    local bbuf = native.action_buf()
    local rng = math.rng(0)

    local s = 2
    local spread = 80 * s
    local margin = 40 * s
    local max_lentotal = 100 * s
    local min_area = 1 * s * s
    local initial_area = {35 * s * s, 40 * s * s}
    local initial_pitch = {0, math.rad(10)}
    local branch_dist = {8 * s, 10 * s}
    local half_dist = 45 * s
    local squiggle_dist = 4 * s
    local squiggle_angle = math.rad(40) / 10 * squiggle_dist / s
    --local attract_angle = math.rad(20) / 10 * squiggle_dist / s
    local attract_p = 0.4
    local area_per_len = -0.1 / s / s
    local off_angle = math.rad(60)
    local off_attractor_angle = math.rad(35)
    local off_area = {0.25, 0.3}
    local main_rotate = math.rad(133)
    local leaf_r = {5 * s, 7 * s}
    local leaf_aspect = {1, 1.4}
    local function branch(pos, up, norm, attractor, area, lentotal, depth)
        if lentotal > max_lentotal or area < min_area then
            --Paint a leaf
            local r = rng:normal(leaf_r[1], leaf_r[2])
            local aspect = rng:normal(leaf_aspect[1], leaf_aspect[2])
            local rz = r / aspect
            bbuf:oval(
                pos:x(), pos:y(), pos:z(),
                r, r, rz,
                blocks['base.leaf']
            )
            return
        end

        local len = rng:normal(branch_dist[1], branch_dist[2]) * 0.5 ^ (lentotal / half_dist)
        local off_a = rng:normal(off_area[1], off_area[2])
        local virt_area = area + area_per_len * len
        local final_area = virt_area * math.max(off_a, 1 - off_a)
        local aperlen_adj = (final_area - area) / len

        local top = math.vec3(pos)
        do
            local rot = math.vec3()
            local acc = 0
            while acc < len do
                --Advance top by a squiggle dist
                local nxt_acc = math.min(acc + squiggle_dist, len)
                local adv = nxt_acc - acc
                local nxt_area = area + adv * aperlen_adj
                up:mul(adv / up:mag()) top:add(up)
                --Paint this squiggle segment
                bbuf:cylinder(
                    pos:x(), pos:y(), pos:z(), math.sqrt(area),
                    top:x(), top:y(), top:z(), math.sqrt(nxt_area),
                    blocks['base.wood']
                )
                --Set up for next squiggle
                pos:set(top)
                acc = nxt_acc
                area = nxt_area
                --Determine a random direction to squiggle
                rot:set(norm) rot:rotate(rng:uniform(2*math.pi), up)
                --Squiggle the up and norm vectors
                up:rotate(squiggle_angle, rot) norm:rotate(squiggle_angle, rot)
                --Skew towards the attractor
                local skew = attract_p * up:angle(attractor)
                norm:rotate(skew, up, attractor) up:rotate(skew, up, attractor)
            end
        end

        
        norm:rotate(main_rotate, up)

        if depth < 3 then
            local subup = math.vec3(up)
            subup:rotate(off_angle, up, norm)
            
            local subnorm = math.vec3(norm)
            subnorm:rotate(off_angle, up, norm)

            local subattractor = math.vec3(attractor)
            subattractor:rotate(off_attractor_angle, up, norm)

            branch(top, subup, subnorm, subattractor, virt_area * off_a, lentotal + len, depth + 1)
        end
        
        branch(pos, up, norm, attractor, virt_area * (1 - off_a), lentotal + len, depth)
    end
    local function tree(pos)
        local area = rng:normal(initial_area[1], initial_area[2])
        local up = math.vec3(0, 0, 1)
        local norm = math.vec3(0, 1, 0)
        local attractor = math.vec3(0, 0, 1)
        local yaw = rng:uniform(2*math.pi)
        local pitch = rng:uniform(initial_pitch[1], initial_pitch[2])
        up:rotate_x(pitch) up:rotate_z(yaw)
        norm:rotate_x(pitch) norm:rotate_z(yaw)
        branch(pos, up, norm, attractor, area, 0, 0)
    end
    function genstruct(rx, ry, sx, sy)
        local bx, by = math.floor(rx), math.floor(ry)
        local bz = heightmap:height_at(bx, by)
        local fx, fy = rx - bx, ry - by
        bbuf:reset(bx, by, bz)
        rng:reseed(math.hash(gen.seed, "plains_tree", sx, sy))
        tree(math.vec3(fx, fy, 0))
        print("generated tree at "..rx..", "..ry)
        return bbuf
    end
    structs = native.structure_grid_2d {
        seed = math.hash(gen.seed, "plains_structs"),
        cell_size = spread,
        margin = margin,
    }
end

function plainsgen.generate(x, y, z, w)
    local chunk = native.chunk(blocks['base.air'])
    heightmap:fill_chunk(x, y, z, chunk)
    structs:fill_chunk(x, y, z, chunk, genstruct)
    return chunk:into_raw()
end

return plainsgen
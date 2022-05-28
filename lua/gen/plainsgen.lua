
local native = require 'gen.native'
local blocks = require 'gen.blocks'
local lightmodes = require 'gen.lightmodes'

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
        base = {0.53, 0.71, 0.14, 0.224},
        noise = {0.116, 0.131, 0.053},
        octs = {0, 0, 0, 0, 0, 0.8},
        rough = false,
    }),
    air = blocks.lookup 'base.air',
}

blocks.register(texture {
    name = 'base.wood',
    base = {0.31, 0.19, 0.13, 0.1},
    noise = {0.136, 0.089, 0.065},
    octs = {0.2, 0.4},
    rough = true,
})
blocks.register(texture {
    name = 'base.leaf',
    base = {0.03, 0.26, 0.13, 0.158},
    noise = {0.045, 0.121, 0.116},
    octs = {0.1, 0, 0, 0.2},
    rough = true,
})

local structs, genstruct
do
    local bbuf = native.action_buf()
    local rng = math.rng(0)
    local leaves

    local base_s = 1.5
    local s = base_s
    local spread = 40 * s
    local margin = 40 * s
    local max_lentotal = 100 * s
    local min_area = 1 * s * s
    local max_depth = 100
    local initial_area = {35 * s * s, 40 * s * s}
    local initial_pitch = {0, math.rad(10)}
    local branch_dist = {{14 * s, 16 * s}}
    local half_dist = 13 * s
    local squiggle_dist = 4 * s
    local squiggle_angle = math.rad(40) / 10 * squiggle_dist / s
    local attract_p = 0.5
    local attract_f = math.rad(10) / 10 * squiggle_dist / s
    local area_per_len = -0.1 / s / s
    local off_angle = {math.rad(90), math.rad(60)}
    local off_attractor_angle = {math.rad(50), math.rad(35)}
    local off_area = {0.45, 0.55}
    local main_bias = {0.5, 0.5}
    local off_depth_incr = 1
    local main_depth_incr = 1
    local main_rotate = math.rad(133)
    local leaf_r = {4 * s, 7 * s}
    local subleaf_r = {1.5 * s, 2.5 * s}
    local subleaf_n = 30
    local leaf_join = 1.5 * s
    local function rand(param)
        return rng:normal(param[1], param[2])
    end
    local function branch(pos, up, norm, attractor, area, lentotal, depth)
        if lentotal > max_lentotal or area < min_area or depth >= max_depth then
            --Paint a leaf
            local r = rand(leaf_r)
            leaves[#leaves + 1] = pos:x()
            leaves[#leaves + 1] = pos:y()
            leaves[#leaves + 1] = pos:z()
            leaves[#leaves + 1] = r
            do return end
            
            local r = rand(leaf_r)
            bbuf:cloud(
                pos:x(), pos:y(), pos:z(),
                r,
                subleaf_r[1], subleaf_r[2],
                rng:integer(1000000),
                blocks['base.leaf'],
                subleaf_n
            )
            return
        end

        local trunk_idx = 1
        local len = rand(branch_dist[trunk_idx]) * 0.5 ^ (lentotal / half_dist)
        local off_a = rand(off_area)
        local virt_area = area + area_per_len * len
        local final_area = virt_area * math.max(off_a, 1 - off_a)
        local aperlen_adj = (final_area - area) / len

        local top = math.vec3(pos)
        local rot = math.vec3()
        do
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
                local skew = up:angle(attractor)
                skew = math.min(attract_p * skew + attract_f, skew)
                norm:rotate(skew, up, attractor) up:rotate(skew, up, attractor)
            end
        end

        
        norm:rotate(main_rotate, up)

        local off_idx = depth == 0 and 2 or 2
        local bias = rand(main_bias)

        local o_angle = bias * off_angle[off_idx]
        local m_angle = (bias - 1) * off_angle[off_idx]
        local o_att = bias * off_attractor_angle[off_idx]
        local m_att = (bias - 1) * off_attractor_angle[off_idx]

        rot:set(up) rot:cross(norm)

        local subup = math.vec3(up)
        subup:rotate(o_angle, rot)
        up:rotate(m_angle, rot)
        
        local subnorm = math.vec3(norm)
        subnorm:rotate(o_angle, rot)
        norm:rotate(m_angle, rot)

        local subattractor = math.vec3(attractor)
        subattractor:rotate(o_att, rot)
        attractor:rotate(m_att, rot)

        branch(top, subup, subnorm, subattractor, virt_area * off_a, lentotal + len, depth + off_depth_incr)
        branch(pos, up, norm, attractor, virt_area * (1 - off_a), lentotal + len, depth + main_depth_incr)
    end
    local function tree(pos)
        leaves = {}

        s = rng:normal(base_s*0.5, base_s*1.5)
        max_lentotal = 100 * s
        min_area = 1 * s * s
        max_depth = 100
        initial_area = {35 * s * s, 40 * s * s}
        initial_pitch = {0, math.rad(10)}
        branch_dist = {{14 * s, 16 * s}}
        half_dist = 13 * s
        squiggle_dist = 4 * s
        squiggle_angle = math.rad(40) / 10 * squiggle_dist / s
        attract_p = 0.5
        attract_f = math.rad(10) / 10 * squiggle_dist / s
        area_per_len = -0.1 / s / s
        off_angle = {math.rad(90), math.rad(60)}
        off_attractor_angle = {math.rad(50), math.rad(35)}
        off_area = {0.45, 0.55}
        main_bias = {0.5, 0.5}
        off_depth_incr = 1
        main_depth_incr = 1
        main_rotate = math.rad(133)
        leaf_r = {4 * s, 7 * s}
        subleaf_r = {1.5 * s, 2.5 * s}
        subleaf_n = 30
        leaf_join = 1.5 * s

        local area = rng:normal(initial_area[1], initial_area[2])
        local up = math.vec3(0, 0, 1)
        local norm = math.vec3(0, 1, 0)
        local attractor = math.vec3(0, 0, 1)
        local yaw = rng:uniform(2*math.pi)
        local pitch = rng:uniform(initial_pitch[1], initial_pitch[2])
        up:rotate_x(pitch) up:rotate_z(yaw)
        norm:rotate_x(pitch) norm:rotate_z(yaw)
        branch(pos, up, norm, attractor, area, 0, 0)

        bbuf:blobs(leaves, blocks['base.leaf'], leaf_join)
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
    structs = native.gridbuf_2d {
        seed = math.hash(gen.seed, "plains_structs"),
        cell_size = math.floor(spread + .5),
        margin = math.floor(margin + .5),
    }
end

local lightconf = native.action_buf()
lightconf:reset(0, 0, 0)
--lightconf:cube(0, 0, 0, 1, 1, 1, 0) -- Comment to enable chunkmesh lighting
--lightconf:cube(1, 0, 0, 1, 1, 1, 0) -- Comment to enable chunkgen lighting
--

local sky_light = 1
function plainsgen.generate(x, y, z, w)
    if z >= 6 then
        return native.chunk(blocks['base.air'], lightmodes['base.std'], sky_light):into_raw()
    end
    local chunk = native.chunk(blocks['base.air'], lightmodes['base.std'])
    heightmap:fill_chunk(x, y, z, chunk)
    structs:fill_chunk(x, y, z, chunk, genstruct)
    lightconf:transfer(0, 0, 0, chunk)
    return chunk:into_raw()
end

return plainsgen
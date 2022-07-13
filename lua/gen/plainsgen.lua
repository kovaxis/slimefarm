
local native = require 'gen.native'
local blocks = require 'gen.blocks'
local lightmodes = require 'gen.lightmodes'
local spawn = require 'gen.spawn'
local genutil = require 'gen.util'
local Structs2d = require 'gen.structs2d'
local Tree = require 'gen.tree'

local plainsgen = {}

blocks.register(genutil.texture {
    name = 'base.wood',
    base = {0.31, 0.19, 0.13, 0.1},
    noise = {0.136, 0.089, 0.065},
    octs = {0.2, 0.4},
    rough = true,
})
blocks.register(genutil.texture {
    name = 'base.leaf',
    base = {0.03, 0.26, 0.13, 0.158},
    noise = {0.045, 0.121, 0.116},
    octs = {0.1, 0, 0, 0.2},
    rough = true,
})
blocks.register(genutil.texture {
    name = 'base.stone',
    base = {0.475, 0.475, 0.475, 0.158},
    noise = {0.169, 0.169, 0.169},
    octs = {0.1, 0.2, 0.3},
    rough = true,
})
blocks.register(genutil.texture {
    name = 'base.grass',
    base = {0.53, 0.71, 0.14, 0.224},
    noise = {0.116, 0.131, 0.053},
    octs = {0, 0, 0, 0, 0, 0.8},
    rough = false,
})

local heightmap = native.heightmap {
    seed = math.hash(gen.seed, "plains_heightmap"),
    noise = genutil.perlin(256, 0.8, 3),
    offset = 0,
    scale = 32,
    ground = blocks.lookup 'base.grass',
    air = blocks.lookup 'base.air',
}

local biome = native.noise2d {
    seed = math.hash(gen.seed, "plains_biome"),
    noise = genutil.perlin(256, 10, 3),
}

local rocks
do
    local chance = .1
    local spawnchance = 1
    local s = {.7, 1}
    local size = 16
    local lift = 4
    local n = 16
    local noisiness = 0.4
    local block = blocks.lookup 'base.stone'
    rocks = Structs2d {
        salt = 'plains_rocks',
        spread = 34,
        margin = 32,
        get_z = function(m)
            return heightmap:height_at(m.bx, m.by)
        end,
        generate = function(m)
            local b = biome:noise_at(m.bx, m.by)
            print("rock biome noise = "..b.." "..m.bx..", "..m.by)
            if b < -1 and m.rng:uniform() < chance then
                local s = m.rng:uniform(s[1], s[2])
                local size = s * size
                local lift = s * lift
                m.bbuf:rock(m.fx, m.fy, m.fz + lift, size, m.rng:integer(1000000000), n, noisiness, block)
                local ent = {'Checkpoint', {
                    orient = m.rng:integer(4),
                }}
                if m.rng:uniform() < spawnchance then
                    m.bbuf:entity(m.fx, m.fy, m.fz + lift + size * (2 + noisiness), spawn.serialize(ent))
                end
            end
        end,
    }
end

local trees
do
    local chance = 1
    local tree = Tree {
        k = {
            size = {1, 1.6},
            max_lentotal = 15,
            min_area = 1,
            max_depth = 10,
            initial_area = {2.5, 4.5},
            initial_pitch = {0, math.rad(40)},
            half_dist = 1000,
            area_per_len = .1,
            branch_len = {3, 4},
            squiggle_len = 1,
            squiggle_angle = math.rad(40) / 10,
            attract_linear = math.rad(10) / 10,
            attract_factor = (0.1)^(1/5),
            off_angle = {math.rad(70), math.rad(85)},
            off_attractor_angle = {math.rad(40), math.rad(50)},
            off_area = {.45, .55},
            off_bias = .5,
            main_rotate = math.rad(133),
            off_rotate = 0,
            leaf_r = {1, 1.6},
            subleaf_r = {1.8, 2.8},
            subleaf_n = 6,
            off_depth_incr = 1,
            main_depth_incr = 1,
            leaf = 'base.leaf',
            wood = 'base.wood',
        },
    }
    trees = Structs2d {
        salt = 'plains_trees',
        spread = 16,
        margin = 16,
        get_z = function(m)
            return heightmap:height_at(m.bx, m.by)
        end,
        generate = function(m)
            if biome:noise_at(m.bx, m.by) > 1 and m.rng:uniform() < chance then
                tree:make(math.vec3(m.fx, m.fy, m.fz), m.bbuf, m.rng)
            end
        end,
    }
end

local slimepack
do
    local chance = .3
    local nred = {2, 3}
    local ngreen = {4, 8}
    local r = 10
    local extra_z = 4
    slimepack = Structs2d {
        salt = 'plains_sliems',
        spread = 64,
        margin = r + 10,
        get_z = function(m)
            return 0
        end,
        generate = function(m)
            local b = biome:noise_at(m.bx, m.by)
            if b > -0.8 and b < 0.8 and m.rng:uniform() < chance then
                local ent, n
                if m.rng:uniform() < .5 then
                    ent = {'GreenSlime', {

                    }}
                    n = ngreen
                else
                    ent = {'RedSlime', {

                    }}
                    n = nred
                end
                n = m.rng:integer(n[1], n[2])
                for i = 1, n do
                    local x = m.rng:normal(-r, r)
                    local y = m.rng:normal(-r, r)
                    m.bbuf:entity(x, y, heightmap:height_at(math.floor(m.bx + x), math.floor(m.by + y)) + extra_z, spawn.serialize(ent))
                end
            end
        end,
    }
end

--[[
local structs, genstruct
do
    local bbuf = native.action_buf()
    local rng = math.rng(0)
    local leaves

    local function rock(pos, sx, sy)
        bbuf:rock(pos:x(), pos:y(), pos:z() + s * 4, s * 16, rng:integer(1000000), 16, 0.4, blocks['base.stone'])
        
        local r, ent = rng:uniform()
        if r < 0.1 then
            ent = {'Checkpoint', {
                orient = rng:integer(4),
            }}
        elseif r < 0.6 then
            ent = {'GreenSlime', {
                hp = rng:uniform(100, 1000),
            }}
        else
            ent = {'RedSlime', {
                hp = rng:uniform(100, 1000),
            }}
        end
        bbuf:entity(pos:x(), pos:y(), pos:z() + s * 4 + s * 16 * (1.4), spawn.serialize(ent))
    end
    function genstruct(rx, ry, sx, sy)
        local bx, by = math.floor(rx), math.floor(ry)
        local bz = heightmap:height_at(bx, by)
        local fx, fy = rx - bx, ry - by
        bbuf:reset(bx, by, bz)
        rng:reseed(math.hash(gen.seed, "plains_tree", sx, sy))
        --tree(math.vec3(fx, fy, 0))
        rock(math.vec3(fx, fy, 0), sx, sy)
        return bbuf
    end
    structs = native.gridbuf_2d {
        seed = math.hash(gen.seed, "plains_structs"),
        cell_size = math.floor(spread + .5),
        margin = math.floor(margin + .5),
    }
end]]

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
    rocks:fill(x, y, z, chunk)
    trees:fill(x, y, z, chunk)
    slimepack:fill(x, y, z, chunk)
    lightconf:transfer(0, 0, 0, chunk)
    return chunk:into_raw()
end

return plainsgen
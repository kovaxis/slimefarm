
local scale = 0.4
local spacing = 50

local initial_incl = 10 * math.pi / 180
local maxdepth = 4
local seg = {scale * 10, scale * 15}
local halfseg = scale * 80
local maxtotal = scale * 100
local branchoffchance = 1
local areatake = {0.3, 0.4}
local turn = 2*math.pi * 2 / 5
local turnjitter = .05
local split = {1.0, 1.4}
local branchweight = {0.2, 0.5}
local initial_area = {scale * 45, scale * 55}
local areaperlen = scale * 0.14
local minbrancharea = 1
local function makebranch(rng, total, depth, area, yaw, pitch)
    if area < minbrancharea then
        return nil
    end

    local b = {
        yaw = yaw,
        pitch = pitch,
        len = rng:normal(seg[1], seg[2]) * 0.5 ^ (total / halfseg),
        r0 = math.sqrt(area),
        r1 = 0,
        children = {},
    }

    area = area - areaperlen * b.len
    
    local areatake = rng:normal(areatake[1], areatake[2])
    if areatake * area < minbrancharea then
        areatake = 0
    end

    local submax = math.max(area * areatake, area * (1 - areatake))
    b.r1 = math.sqrt(submax + (area - submax) * 0.5)

    local mainpitch = 0
    if depth < maxdepth and rng:uniform() < branchoffchance then
        -- Add branchoff
        local splitpitch = rng:normal(split[1], split[2])
        local offlean = rng:normal(branchweight[1], branchweight[2])
        mainpitch = mainpitch - splitpitch * offlean
        table.insert(b.children, makebranch(rng, total, depth + 1, area * areatake, 0, mainpitch + splitpitch))
    end

    total = total + b.len
    if total < maxtotal then
        -- Add branch continuation
        table.insert(b.children, makebranch(rng, total, depth, area * (1 - areatake), rng:normal(turn - turnjitter, turn + turnjitter), mainpitch))
    end

    return b
end

function config()
    return {
        kind = {Plains = {
            xy_scale = 256,
            detail = 3,
            z_scale = 40,
            grass_tex = {
                solid = true,
                smooth = true,
                base = {0.43, 0.61, 0.10},
                noise = {
                    {0,0,0},
                    {0,0,0},
                    {0,0,0},
                    {0,0,0},
                    {0,0,0},
                    {0.116, 0.131, 0.053},
                },
            },

            tree = {
                spacing = spacing,
                extragen = 1,
                wood_tex = {
                    solid = true,
                    smooth = false,
                    base = {0.31, 0.19, 0.13},
                    noise = {
                        {0.136, 0.089, 0.065},
                        {0.136 * 0.2, 0.089 * 0.2, 0.065 * 0.2},
                        {0,0,0},
                        {0,0,0},
                        {0,0,0},
                        {0,0,0},
                    },
                },
                make = function(rng)
                    return makebranch(rng, 0, 1, rng:normal(initial_area[1], initial_area[2]), rng:uniform(2*math.pi), rng:normal(-initial_incl, initial_incl))
                end,
            },
        }},

        air_tex = {
            solid = false,
        },
        void_tex = {
            solid = false,
        },

        gen_radius = 512,
        seed = 123443,
    }
end

--[[{
    "_kind": {"Parkour": {
        "z_offset": 0.008,
        "delta": 0.4,
        "color": [0.43, 0.43, 0.43]
    }},
    "kind": {"Plains": {
        "xy_scale": 256,
        "detail": 3,
        "z_scale": 40,

        "tree": {
            "spacing": 128,
            "extragen": 1,
            "initial_incl": [0, 0.2],
            "initial_area": [12, 60],
            "area_per_len": [0.02, 0.03],
            "trunk_len": [26, 28],
            "halflen": [74, 75],
            "rot_angle": [2.313, 2.713],
            "offshoot_area": [0, 0.35],
            "offshoot_perturb": [0.4, 0.7],
            "main_perturb": [-0.4, -0.2],
            "prune_area": 2,
            "prune_depth": 200
        },

        "tree_width": [6, 9],
        "tree_height": [200, 240, 10],
        "tree_taperpow": 0.6,
        "tree_undergen": 5,

        "color": [0.01, 0.92, 0.20],
        "log_color": [0.53, 0.12, 0.01]
    }},
    "gen_radius": 512,
    "seed": 123443
}]]
--[[
    {Parkour = {
        z_offset = 0.008,
        delta = 0.4,
        color = {0.43, 0.43, 0.43},
    }},
]]
--[[
    "kind": {"Plains" = {
        xy_scale = 256,
        detail = 3,
        z_scale = 40,
        color = [0.01, 0.12, 0.78],
    }},
]]

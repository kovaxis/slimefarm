-- Utilities to produce trees.

local class = require 'class'
local util = require 'util'
local blocks = require 'gen.blocks'

local Tree = class{}

function Tree.inherit(k, super)
    for n, v in pairs(super) do
        if k[n] == nil then
            k[n] = v
        end
    end
end

local numpairs = {
    'size', -- general scale parameter for tree size
    'max_lentotal', -- truncate branches at this distance total to the root
    'min_area', -- truncate branches with less than this area
    'max_depth', -- truncate branches at more than this depth
    'initial_area', -- initial trunk area
    'initial_pitch', -- initial trunk inclination, with 0 being upright and pi/2 horizontal
    'half_dist', -- every this amount of total distance the branch separation distance is halved
    'area_per_len', -- area change per unit of length. usually negative to shrink branches as they go up
}
local numpairpairs = {
    'branch_len', -- length of a branch. actual length might be reduced by `half_dist`
    'squiggle_len', -- length of "squiggles", the segments that compose a branch
    'squiggle_angle', -- angle perturbation of each squiggle, in units of radians per block (taking size = 1).
    'attract_linear', -- attraction towards the attractor (ie. forward) in radians per block (taking size = 1).
    'attract_factor', -- attraction towards the attractor in proportion of the skew left after a single block (taking size = 1).
    'off_angle', -- angle between off branch and main branch at every bifurcation
    'off_attractor_angle', -- angle between off branch attractor and main branch attractor at every bifurcation
    'off_area', -- area fraction that is "given" to the off branch
    'off_bias', -- fraction of the angular deviation that is applied to the off branch instead of the main branch.
    'main_rotate', -- roll rotation of the main branch after bifurcation.
    'off_rotate', -- roll rotation of the off branch after bifurcation.
    'leaf_r', -- radius of leaf blobs at the tip of every branch.
    'subleaf_r', -- radius of the leaf spheres that compose each leaf blob
    'subleaf_n', -- amount of leaf spheres per leaf blob
    'off_depth_incr', -- increase of depth per off branch.
    'main_depth_incr', -- increate of depth per main branch.
}

function Tree:new()
    local k = self.k
    assert(type(k) == 'table', "tree.k must be a table")

    for i, name in ipairs(numpairs) do
        local p = k[name]
        if type(p) == 'number' then
            p = {p, p}
            k[name] = p
        end
        if type(p) ~= 'table' or type(p[1]) ~= 'number' or type(p[2]) ~= 'number' then
            error("k."..name.." must be a number pair")
        end
    end

    for i, name in ipairs(numpairpairs) do
        local pp = k[name]
        if type(pp) == 'number' then
            pp = {pp, pp}
            k[name] = pp
        end
        if #pp == 2 then
            pp[3], pp[4] = pp[1], pp[2]
        end
        if type(pp) ~= 'table' or type(pp[1]) ~= 'number' or type(pp[2]) ~= 'number' or type(pp[3]) ~= 'number' or type(pp[4]) ~= 'number' then
            error("k."..name.." must be a pair of number pairs")
        end
    end

    self.ktmp = {}
    for i, name in ipairs(numpairpairs) do
        self.ktmp[name] = {0, 0}
    end
    self.ktmp.wood = blocks.lookup(k.wood)
    self.ktmp.leaf = blocks.lookup(k.leaf)
end

function Tree:init(bbuf, rng)
    local k = self.ktmp

    -- Randomize tree parameters
    for i, name in ipairs(numpairs) do
        local p = self.k[name]
        k[name] = rng:normal(p[1], p[2])
    end
    for i, name in ipairs(numpairpairs) do
        local pp = self.k[name]
        local center = rng:normal(pp[1] + pp[2], pp[3] + pp[4]) / 2
        local radius = rng:normal(pp[2] - pp[1], pp[4] - pp[3]) / 2
        k[name][1] = center - radius
        k[name][2] = center + radius
    end

    -- Adjust parameters based on size parameter
    local s = k.size
    for i = 1, 2 do
        k.branch_len[i] = k.branch_len[i] * s
        k.squiggle_len[i] = k.squiggle_len[i] * s
        k.leaf_r[i] = k.leaf_r[i] * s
        k.subleaf_r[i] = k.subleaf_r[i] * s
    end
    k.max_lentotal = k.max_lentotal * s
    k.min_area = k.min_area * s * s
    k.initial_area = k.initial_area * s * s
    k.half_dist = k.half_dist * s
    k.area_per_len = k.area_per_len * s

    --Set output and rng
    self.bbuf = bbuf
    self.rng = rng
end

function Tree:make(pos, bbuf, rng)
    self:init(bbuf, rng)

    local k = self.ktmp
    local up = math.vec3(0, 0, 1)
    local norm = math.vec3(0, 1, 0)
    local attractor = math.vec3(0, 0, 1)
    local yaw = self.rng:uniform(2*math.pi)
    local pitch = k.initial_pitch
    up:rotate_x(pitch) up:rotate_z(yaw)
    norm:rotate_x(pitch) norm:rotate_z(yaw)
    self:branch(pos, up, norm, attractor, k.initial_area, 0, 0)

    self:finalize()
end

function Tree:finalize()
    local k = self.ktmp
    --self.bbuf:blobs(self.leaves, k.leaf, k.leaf_join)
end

function Tree:rand(pair)
    return self.rng:normal(pair[1], pair[2])
end

function Tree:leaf(pos, up, norm, attractor, area, lentotal, depth)
    --Paint a leaf
    local k = self.ktmp
    local r = self:rand(k.leaf_r)
    if r > 0 then
        local n = self:rand(k.subleaf_n)
        self.bbuf:cloud(
            pos:x(), pos:y(), pos:z(),
            r,
            k.subleaf_r[1], k.subleaf_r[2],
            self.rng:integer(1000000000),
            k.leaf,
            n
        )
    end
end

--[[function Tree:leaf(pos, up, norm, attractor, area, lentotal, depth)
    local leaves = self.leaves
    local r = self:rand(k.leaf_r)
    leaves[#leaves + 1] = pos:x()
    leaves[#leaves + 1] = pos:y()
    leaves[#leaves + 1] = pos:z()
    leaves[#leaves + 1] = r
end]]

function Tree:branch(pos, up, norm, attractor, area, lentotal, depth)
    local k = self.ktmp
    if lentotal > k.max_lentotal or area < k.min_area or depth >= k.max_depth then
        return self:leaf(pos, up, norm, attractor, area, lentotal, depth)
    end

    --Length of the branch
    local len = self:rand(k.branch_len) * 0.5 ^ (lentotal / k.half_dist)
    --Fraction of the area that goes to the off branch
    local off_a = self:rand(k.off_area)
    --`area` contains the area at the beggining of this branch
    --`new_area` contains the area at the end of this branch, which is to be distributed between
    --the 2 sub-branches
    local new_area = area + k.area_per_len * len
    --The max area of the sub-areas are used as the graphical `new_area`
    local max_sub_area = new_area * math.max(off_a, 1 - off_a)
    --Area delta per unit of length
    local aperlen_adj = (new_area - area) / len

    local top = math.vec3(pos)
    local rot = math.vec3()
    do
        local acc = 0
        while acc < len do
            --Advance top by a squiggle dist
            local squiggle_len = self:rand(k.squiggle_len)
            local nxt_acc = math.min(acc + squiggle_len, len)
            local adv = nxt_acc - acc
            local nxt_area = area + adv * aperlen_adj
            up:mul(adv / up:mag()) top:add(up)
            --Paint this squiggle segment
            self.bbuf:cylinder(
                pos:x(), pos:y(), pos:z(), math.sqrt(area),
                top:x(), top:y(), top:z(), math.sqrt(nxt_area),
                k.wood
            )
            --Set up for next squiggle
            pos:set(top)
            acc = nxt_acc
            area = nxt_area
            --Determine a random direction to squiggle
            rot:set(norm) rot:rotate(self.rng:uniform(2*math.pi), up)
            --Squiggle the up and norm vectors
            local rot_angle = self:rand(k.squiggle_angle) * squiggle_len
            up:rotate(rot_angle, rot) norm:rotate(rot_angle, rot)
            --Skew towards the attractor
            local skew = up:angle(attractor)
            local attract_f = self:rand(k.attract_factor) ^ squiggle_len
            local attract_l = self:rand(k.attract_linear) * squiggle_len
            skew = math.min((1 - attract_f) * skew + attract_l, skew)
            norm:rotate(skew, up, attractor) up:rotate(skew, up, attractor)
        end
    end

    
    norm:rotate(self:rand(k.main_rotate), up)

    local bias = self:rand(k.off_bias)

    local off_angle = self:rand(k.off_angle)
    local o_angle = bias * off_angle
    local m_angle = (bias - 1) * off_angle
    local off_attractor_angle = self:rand(k.off_attractor_angle)
    local o_att = bias * off_attractor_angle
    local m_att = (bias - 1) * off_attractor_angle

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

    self:branch(top, subup, subnorm, subattractor, new_area * off_a, lentotal + len, depth + self:rand(k.off_depth_incr))
    self:branch(pos, up, norm, attractor, new_area * (1 - off_a), lentotal + len, depth + self:rand(k.main_depth_incr))
end

return Tree
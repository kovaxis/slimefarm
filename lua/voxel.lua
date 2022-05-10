
local class = require 'class'
local util = require 'util'

local voxel = {}

voxel.mesher_cfg = {
    atlas_size = {64, 1024},
    -- clear, normal, blocked, blocked, base
    exposure_table = {64, 60, 40, 40, 0},
    light_uv_offset = 0.5,
}
voxel.model_mesher_cfg = {
    cfg = voxel.mesher_cfg,
    transparency = 0,
    light_value = 240,
    lighting = {
        light = {
            base = 0,
            mul = 1,
            shr = 2,
        },
        decay = {
            base = 8,
            mul = 0,
            shr = 0,
        },
    },
}
voxel.mesher = gfx.mesher(voxel.model_mesher_cfg)

-- Performs the entire process from a path to a `.vox` file to a list of voxel mesh buffers.
function voxel.dot_vox(path)
    local raw = util.read_file(path)
    local models = system.dot_vox(raw)
    for i = 1, #models do
        models[i] = voxel.mesher:mesh(models[i])
    end
    return models
end

-- Load a model complete with animations.
--
-- A model is composed by a hierarchical set of `pieces`.
-- Each piece is conceptually attached to a `bone`, or a segment in 3D space.
-- Note that a piece is always attached to a bone, but a bone is not necessarily attached to a
-- piece.
-- Normally, a root bone starts at a certain coordinate, relative to the world position of its
-- entity.
-- The bone can rotate or stretch, which will rotate or stretch its attached piece, if any.
-- A bone can have sub-bones attached to it. The root position of these bones are then affected by
-- rotation and stretching of the parent bone.
--
-- An animation is a collection of functions, each attached to a bone, that maps from time to a
-- linear transform on the delta of that bone.
-- For example, a walking animation maps from time to a cyclical rotation of the leg bones and hand
-- bones, as well as a constant tilt in the body and head superposed with some bobbing.
--
-- These transforms must be linear, that is, able to be added and multiplied. Therefore, a bone
-- transform is represented as a 3-value gyro rotation, and a 3-value scale.
-- This means that bones have "orientations".
--
-- Usually, the rest state of a model is not very meaningful. Therefore, a model is almost always
-- being animated. When transitioning between animations, it becomes useful to provide some sort of
-- smooth interpolation between them. Because animations are linear, a linear interpolation can be
-- carried out between the last animation (maybe a frozen or a slowed-down version of it) and the
-- next animation, with the interpolation weight following any visually pleasing curve (possibly an
-- S-curve).
voxel.Model = class{}

function voxel.Model:new()
    local rawvox = util.read_file(self.path)
    local rawmodels = system.dot_vox(rawvox)
    local models = {}
    self.bones = {}

    local halfvox = math.vec3(0.5)

    local refmodels = {}
    local function getrawmodel(i)
        return assert(rawmodels[i], "model "..i.." out of range, only "..#rawmodels.." models are available")
    end
    local function getmodel(i)
        if not models[i] then
            local raw = getrawmodel(i)
            models[i] = voxel.mesher:mesh(raw)
        end
        return models[i]
    end
    local strmap = {x = 1, X = 1, y = 2, Y = 2, z = 3, Z = 3}
    local function strtovec(axis)
        assert(type(axis) == 'string', "axis string must be a string")
        assert(#axis == 2, "axis strings must be 2 letters long")
        local ax = axis:sub(1, 1)
        local dr = axis:sub(2, 2)
        assert(strmap[ax], "unknown axis '"..tostring(ax).."'")
        assert(dr == '+' or dr == '-', "unknown axis direction '"..dr.."'")
        dr = dr == '+' and 1 or -1
        local v = {0, 0, 0}
        v[strmap[ax]] = dr
        return math.vec3(v)
    end
    local function findref(ref, j, base)
        if base and type(j) == 'string' then
            return math.vec3(base), strtovec(j)
        end
        local offx, offy, offz = 0, 0, 0
        if type(j) == 'table' then
            assert(type(j[1]) == 'number', "expected voxel reference as first element, got "..type(j[1]))
            assert(type(j[2]) == 'number', "expected x offset as second element, got "..type(j[2]))
            assert(type(j[3]) == 'number', "expected y offset as third element, got "..type(j[3]))
            assert(type(j[4]) == 'number', "expected z offset as fourth element, got "..type(j[4]))
            offx, offy, offz = j[2], j[3], j[4]
            j = j[1]
        end
        assert(type(j) == 'number', "expected voxel reference, got "..type(j))
        local rvox = ref[j]
        assert(rvox, "invalid reference voxel color "..tostring(j))
        assert(#rvox > 0, "found no voxel with color "..j.." in reference model")
        local u, v = math.vec3(), math.vec3()
        for i = 1, #rvox, 3 do
            v:set(rvox[i+0], rvox[i+1], rvox[i+2])
            v:add(halfvox)
            u:add(v)
        end
        u:div(#rvox / 3)
        v:set(offx, offy, offz)
        u:add(v)
        local delta = math.vec3(u)
        if base then
            delta:sub(base)
        end
        return u, delta
    end
    local function loadbone(bone, ref, parent)
        assert(type(bone) == 'table', "bone must be a table")
        assert(type(bone.name) == 'string', "bone.name must be a string")
        if type(bone.ref) == 'number' then
            ref = refmodels[bone.ref]
            if not ref then
                ref = getrawmodel(bone.ref)
                ref = ref:find()
                assert(#ref == 255)
                refmodels[bone.ref] = ref
            end
        end
        assert(ref, "bone with no reference model")

        assert(not self.bones[bone.name], "duplicate bones with name '"..bone.name.."'")
        self.bones[bone.name] = bone

        if bone.piece then
            assert(type(bone.piece) == 'number', "bone.piece must be a model index")
            bone.piece = getmodel(bone.piece)
        end

        assert(type(bone.pos) == 'table' and #bone.pos == 3, "bone.pos must be a 3-element table")
        local start, finish, front = bone.pos[1], bone.pos[2], bone.pos[3]
        start = findref(ref, start)
        local finish, delta = findref(ref, finish, start)
        local front, up = findref(ref, front, start)
        bone.abs_pos = start
        
        if bone.children then
            for i, sub in ipairs(bone.children) do
                loadbone(sub, ref, finish)
            end
        end
        
        bone.delta = finish
        bone.delta:sub(start)

        local move = math.vec3(start)
        if parent then
            move:sub(parent)
        else
            move:set(0, 0, 0)
        end
        bone.move = move

        -- Compute matrices to bring bone to normalized manipulation space
        local norm = system.matrix()
        norm:look_at(0, 0, 0, delta:x(), delta:y(), delta:z(), up:x(), up:y(), up:z())
        local denorm = system.matrix()
        denorm:reset_from(norm)
        denorm:invert()
        bone.to_std = norm
        bone.from_std = denorm
    end
    loadbone(self.rootbone, nil, nil)

    assert(type(self.animations) == 'table', "animations must be a table")
    for i = 1, #self.animations do
        local anim = self.animations[i]
        assert(type(anim) == 'table', "each anim must be a table")
        assert(type(anim.name) == 'string', "anim.name must be a string")
        assert(type(anim.speed) == 'nil' or type(anim.speed) == 'number', "anim.speed must be number or nil")
        anim.speed = anim.speed or 1
        assert(type(anim.bones) == 'table', "anim.bones must be a table")
        assert(type(anim.manage) == 'table', "anim.manage must be a table")

        self.animations[anim.name] = anim
        self.animations[i] = nil

        for bname, func in pairs(anim.bones) do
            assert(self.bones[bname], "anim bone keys must be bone names")
            assert(type(func) == 'function', "anim bone values must be animation functions returning 6 values")
        end

        for mname, func in pairs(anim.manage) do
            assert(type(mname) == 'string', "anim manager keys must be name strings")
            assert(type(func) == 'function', "anim manager values must be ease functions returning 2 values")
        end
    end
end



voxel.AnimState = class{}

--Pool of temporary animation objects
--Prevents GC
local animstate_pool = util.Pool{}

function voxel.AnimState:new()
    -- Map of active animations, indexed by animation name
    self.active = {}
end

--Start an animation with the given time/weight manager, using `time` as a reference time.
--Optionally, a finisher time/weight manager can be given, which will run when the animation is
--stopped.
--IMPORTANT: If no finisher is supplied, the animation will be assumed to be short/finite, and it
--will not be stopped when stopping all animations.
function voxel.AnimState:start(name, time, manage)
    local state = self.active[name] or animstate_pool:dirty() or {}

    print("starting animation "..name.." with manager "..(manage or 'go'))
    state.ref1 = time
    state.ref2 = time
    state.manage = manage or 'go'
    self.active[name] = state
end

--Start the given animation only if an animation with the same manager is not currently running.
function voxel.AnimState:start_lazy(name, time, manage)
    manage = manage or 'go'
    local state = self.active[name]
    if not state or state.manage ~= manage then
        self:start(name, time, manage)
    end
end

--Stop the given animation smoothly.
--If the animation has no 'stop' manager it will cause an error!
function voxel.AnimState:stop(name, time)
    local state = self.active[name]
    if state and state.manage ~= 'stop' then
        state.manage = 'stop'
        state.ref2 = time
    end
end

do
    -- Internal temporary state
    local shader, draw_params, mvp_name, mvp

    local function drawbone(self, bone)
        mvp:push()

        local tx, ty, tz, rx, ry, rz, sx, sy, sz = 0, 0, 0, 0, 0, 0, 0, 0, 0
        for name, state in pairs(self.active) do
            local f = state.bones[bone.name]
            if f then
                local tx1, ty1, tz1, rx1, ry1, rz1, sx1, sy1, sz1 = f(state.t)
                local w = state.w
                tx, ty, tz = tx + tx1 * w, ty + ty1 * w, tz + tz1 * w
                rx, ry, rz = rx + rx1 * w, ry + ry1 * w, rz + rz1 * w
                sx, sy, sz = sx + sx1 * w, sy + sy1 * w, sz + sz1 * w
            end
        end

        mvp:translate(bone.move:xyz())
        mvp:mul_right(bone.from_std)
        mvp:rotate_z(rz)
        mvp:rotate_x(rx)
        mvp:rotate_y(ry)
        mvp:translate(tx, ty, tz)
        mvp:mul_right(bone.to_std)

        if bone.piece then
            local x, y, z = bone.abs_pos:xyz()
            mvp:push()
            mvp:scale(2^sx, 2^sy, 2^sz)
            mvp:translate(-x, -y, -z)
            shader:set_matrix(mvp_name, mvp)
            shader:draw_voxel(bone.piece, draw_params)
            mvp:pop()
        end

        mvp:translate(bone.delta:xyz())
        if bone.children then
            for i = 1, #bone.children do
                drawbone(self, bone.children[i])
            end
        end

        mvp:pop()
    end

    function voxel.AnimState:draw(model, time, shader_, draw_params_, mvp_name_, mvp_)
        for name, state in pairs(self.active) do
            local anim = model.animations[name]
            state.bones = anim.bones
            state.t, state.w = anim.manage[state.manage](time - state.ref1, time - state.ref2)
            if state.w then
                state.t = state.t * anim.speed
            else
                animstate_pool:put(state)
                self.active[name] = nil
            end
        end
        shader = shader_
        draw_params = draw_params_
        mvp_name = mvp_name_
        mvp = mvp_
        drawbone(self, model.rootbone)
    end
end


-- Handle live voxel model reloading.

voxel.models = {}
voxel.watchers = {}

local function reload(w)
    print("loading file '"..w.path.."'")
    local ms, as = dofile(w.path)
    for name, model in pairs(ms) do
        voxel.models[name] = voxel.Model(model)
    end
end

function voxel.load_models(path)
    local w = {
        path = path,
        watcher = fs.watch(path),
    }
    table.insert(voxel.watchers, w)
    reload(w)
end

function voxel.check_reload()
    for i, w in ipairs(voxel.watchers) do
        if w.watcher:changed() then
            print("reloading '"..w.path.."'")
            reload(w)
        end
    end
end


return voxel
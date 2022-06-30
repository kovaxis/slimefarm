
local class = require 'class'
local util = require 'util'

local voxel = {}

voxel.mesher_cfg = {
    atlas_size = {64, 1024},
    -- clear, normal, blocked, blocked, base
    exposure_table = {64, 60, 40, 40, 0},
    light_uv_offset = 0.5,
    bleed_offset = 1/64,
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
function voxel.dot_vox(path, shininess)
    local raw = util.read_file(path)
    local models = system.dot_vox(raw, shininess)
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
    local rawmodels = system.dot_vox(rawvox, self.shininess or 0.02)
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

        if bone.piece ~= nil then
            assert(type(bone.piece) == 'number', "bone.piece must be a model index")
            bone.piece = getmodel(bone.piece)
        end

        if bone.order == nil then
            bone.order = 'zxy'
        end
        assert(type(bone.order) == 'string', "bone.order must be a string if present")
        assert(#bone.order == 3, "bone.order must be 3 characters long")
        for i = 1, 3 do
            local axis = bone.order:byte(i, i) - ('x'):byte() + 1
            assert(axis >= 1 and axis <= 3, "bone.order["..i.."] is not x, y or z")
            bone['rot'..i] = axis
        end

        assert(type(bone.pos) == 'table' and #bone.pos == 3, "bone.pos must be a 3-element table")
        local start, finish, front = bone.pos[1], bone.pos[2], bone.pos[3]
        start = findref(ref, start)
        local finish, delta = findref(ref, finish, start)
        local front, up = findref(ref, front, start)
        bone.abs_pos = start
        
        if bone.child ~= nil then
            assert(bone.children == nil, "bone.children cannot exist if bone.children is set")
            bone.children = { bone.child }
        elseif bone.children == nil then
            bone.children = {}
        end
        assert(type(bone.children) == 'table', "bone.children must be a sequence")
        for i, sub in ipairs(bone.children) do
            loadbone(sub, ref, finish)
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
        -- Normalized bone space has the bone origin at the origin, bone forward direction in the
        -- Y+ direction and bone up direction in the Z+ direction
        local norm = system.matrix()
        norm:look_at(0, 0, 0, delta:x(), delta:y(), delta:z(), up:x(), up:y(), up:z())
        local denorm = system.matrix()
        denorm:reset_from(norm)
        denorm:invert()
        bone.to_std = norm
        bone.from_std = denorm
    end
    loadbone(self.rootbone, nil, nil)

    assert(type(self.animations) == 'table', "`animations` must be a table")
    local animations = {}
    local triggers = {}
    for i, animproto in ipairs(self.animations) do
        self.animations[i] = nil
        assert(type(animproto) == 'table', "animations must be tables")
        assert(type(animproto.name) == 'string', "animation.name must be a string")
        assert(not animations[animproto.name], "duplicate animations with name '"..animproto.name.."'")
        animations[animproto.name] = animproto
        assert(type(animproto.new) == 'function', "animation.new must be a function with at least 1 argument")
        assert(type(animproto.draw) == 'function', "animation.draw must be a function with 3 arguments")
        if animproto.triggers ~= nil then
            assert(type(animproto.triggers) == 'table', "animation.triggers must be a table if present")
            for ev, listen in pairs(animproto.triggers) do
                assert(type(ev) == 'string', "animation.triggers keys must be event name strings")
                assert(type(listen) == 'function', "animation.triggers elements must be event listener functions")
                triggers[ev] = triggers[ev] or {}
                table.insert(triggers[ev], listen)
            end
        end
        animproto.__index = animproto
    end
    assert(next(self.animations) == nil, "animations is not a sequence")

    self.animations = animations
    self.triggers = triggers
end



voxel.AnimState = class{}

local animpool = util.Pool{}

function voxel.AnimState:new()
    assert(self.model, "expected a voxel model")

    self.anims = {}

    self.bones = {}
    for name, bone in pairs(self.model.bones) do
        self.bones[name] = {0, 0, 0,  0, 0, 0,  0, 0, 0}
    end
end

do
    -- Internal temporary state
    local shader, draw_params, mvp_name, mvp
    local rotfuncs = {'rotate_x', 'rotate_y', 'rotate_z'}

    local function drawbone(self, bone)
        mvp:push()

        local b = self.bones[bone.name]

        mvp:translate(bone.move:xyz())
        mvp:mul_right(bone.from_std)
        mvp[rotfuncs[bone.rot1]](mvp, b[3 + bone.rot1])
        mvp[rotfuncs[bone.rot2]](mvp, b[3 + bone.rot2])
        mvp[rotfuncs[bone.rot3]](mvp, b[3 + bone.rot3])
        mvp:translate(b[1], b[2], b[3])
        mvp:mul_right(bone.to_std)

        if bone.piece then
            local x, y, z = bone.abs_pos:xyz()
            mvp:push()
            mvp:scale(2^b[7], 2^b[8], 2^b[9])
            mvp:translate(-x, -y, -z)
            shader:set_matrix(mvp_name, mvp)
            shader:draw_voxel(bone.piece, draw_params)
            mvp:pop()
        end

        mvp:translate(bone.delta:xyz())
        for i = 1, #bone.children do
            drawbone(self, bone.children[i])
        end

        mvp:pop()
    end

    function voxel.AnimState:event(name, ...)
        local trig = true
        for i, anim in ipairs(self.anims) do
            if anim[name] then
                local go = anim[name](anim, ...) + 0
                if go ~= 0 then
                    trig = false
                end
            end
        end
        if trig then
            local ts = self.model.triggers[name]
            if ts then
                for i, trigger in ipairs(ts) do
                    trigger(self, ...)
                end
            end
        end
    end

    function voxel.AnimState:add(name, ...)
        local animproto = self.model.animations[name]
        local anim = animpool:clean()
        setmetatable(anim, animproto)
        anim:new(...)
        self.anims[#self.anims + 1] = anim
        return anim
    end

    function voxel.AnimState:draw(dt, shader_, draw_params_, mvp_name_, mvp_)
        -- Reset bone positions
        for name, bone in pairs(self.bones) do
            bone[1], bone[2], bone[3] = 0, 0, 0
            bone[4], bone[5], bone[6] = 0, 0, 0
            bone[7], bone[8], bone[9] = 0, 0, 0
        end

        -- Move bones
        local anims, i = self.anims, 1
        while anims[i] do
            local w = anims[i]:draw(self.bones, dt) + 0
            if w == 0 then
                anims[i], anims[#anims] = anims[#anims], anims[i]
                animpool:put(anims[#anims])
                anims[#anims] = nil
            else
                i = i + 1
            end
        end
        
        -- Draw moved bones
        shader = shader_
        draw_params = draw_params_
        mvp_name = mvp_name_
        mvp = mvp_
        drawbone(self, self.model.rootbone)
    end
end


-- Handle live voxel model reloading.

voxel.models = {}
voxel.watchers = {}

local function reload(w)
    local ms, as = dofile(w.path)
    for name, model in pairs(ms) do
        ms[name] = voxel.Model(model)
    end
    voxel.models = util.merge(ms, voxel.models)
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
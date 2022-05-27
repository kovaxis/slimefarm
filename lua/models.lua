
local class = require 'class'

local units_per_sec = 1
local pi, cos, sin = math.pi, math.cos, math.sin
local sqrt, floor, ceil = math.sqrt, math.floor, math.ceil

--Move a bone by a certain amount, applying a weight multiplier
local function move(bone, w, tx, ty, tz, rx, ry, rz, sx, sy, sz)
    --Translation
    bone[1] = bone[1] + w * tx
    bone[2] = bone[2] + w * ty
    bone[3] = bone[3] + w * tz
    --Rotation
    bone[4] = bone[4] + w * rx
    bone[5] = bone[5] + w * ry
    bone[6] = bone[6] + w * rz
    --Scaling
    bone[7] = bone[7] + w * sx
    bone[8] = bone[8] + w * sy
    bone[9] = bone[9] + w * sz
end

local function lerp(s, x1, x2)
    return x1 + s * (x2 - x1)
end

local function approach(cur, target, factor, linear, dt)
    local og_delta = cur - target
    if target < cur then
        linear = -linear
    end
    local delta = og_delta * factor ^ dt + linear * dt
    if og_delta * delta <= 0 then
        return target
    else
        return target + delta
    end
end

local function addsnap(x, delta, snap, phase)
    phase = phase or 0
    snap = snap or 1
    local y = x + delta
    if delta >= 0 then
        local lt, rt = ceil(x / snap - phase - 0.0001), ceil(y / snap - phase + 0.0001)
        if lt ~= rt then
            y = (lt + phase) * snap
        end
    else
        local rt, lt = floor(x / snap - phase + 0.0001), floor(y / snap - phase - 0.0001)
        if lt ~= rt then
            y = (rt + phase) * snap
        end
    end
    return y
end

local function bumpcos(t)
    local y = .5 + .5 * cos(t)
    return y * y
end

----- Voxel models -----

local models = {}

local humanoid_animation = {
    init = function(self)
        self.state = 'idle'
        self.moving = 0
        self.air = 0
        self.t = 0
        self.idle_t = 0
        self.air_t = 0
    end,
    motion = function(self, name)
        self.state = name
    end,
    draw = function(self, b, dt)
        self.moving = approach(
            self.moving,
            self.state == 'idle' and 0 or 1,
            0.001, 0.2,
            dt
        )
        self.air = approach(
            self.air,
            self.state == 'air' and 1 or 0,
            0.004, 0.2,
            dt
        )
        local idle_speed = 2
        if self.moving == 1 then
            self.idle_t = 0
        else
            self.idle_t = self.idle_t + dt * idle_speed
        end
        local move_speed = 14
        if self.moving == 0 then
            self.t = 0
        elseif self.state == 'run' then
            self.t = self.t + dt * move_speed
        else
            self.t = addsnap(self.t, dt * move_speed, pi, 0.5)
        end
        local air_speed = 3
        if self.air == 0 then
            self.air_t = 0
        else
            self.air_t = self.air_t + dt * air_speed
        end

        --Idle
        do
            local t, w = self.idle_t, 1 - self.moving
            move(b.body, w,  0, sin(t) * 0.2, 0,  0, 0, 0,  0, 0, 0)
            move(b.head, w,  0, sin(t) * 0.05, 0,  0, 0, 0,  0, 0, 0)
            move(b.lhand, w,  0, 0, 0,  sin(t*.87) * 0.04, 0, 0,  0, 0, 0)
            move(b.rhand, w,  0, 0, 0,  sin(t*-.87) * 0.04, 0, 0,  0, 0, 0)
        end
        
        --Run/Air
        do
            local t, w = self.t, self.moving
            local s = bumpcos(2*t) * .4
            local nc = lerp(self.air, 0.40, 0.05 + 0.01 * sin(self.air_t))
            local sp = lerp(self.air, 1.20, 1.45)
            move(b.body, w,  0, sin(2*t) * 0.20, 0,  nc, 0, 0,  0, 0, 0)
            move(b.head, w,  0, sin(2*t) * -0.00, 0,  0, 0, 0,  0, 0, 0)
            move(b.lhand, w,  0, 0, 0,  sin(t) * sp, 0, 0,  0, 0, 0)
            move(b.rhand, w,  0, 0, 0,  sin(-t) * sp, 0, 0,  0, 0, 0)
            move(b.lfoot, w,  0, 0, 0,  sin(t) * sp, 0, 0,  -s*.5, s, -s*.5)
            move(b.rfoot, w,  0, 0, 0,  sin(-t) * sp, 0, 0,  -s*.5, s, -s*.5)
        end
    end,
}

models.player = {
    path = 'voxel/player.vox',
    rootbone = {
        ref = 1,
        name = 'base',
        pos = {{1, 0, 0, -.5}, 'z+', 'y+'},

        children = {
            {
                name = 'body',
                pos = {2, 5, 'y+'},
                piece = 2,

                children = {
                    {
                        name = 'head',
                        pos = {6, 'z+', 'y+'},
                        piece = 3,
                    },
                    {
                        name = 'lhand',
                        pos = {8, 'z-', 'y+'},
                        piece = 4,
                    },
                    {
                        name = 'rhand',
                        pos = {7, 'z-', 'y+'},
                        piece = 5,
                    },
                },
            },
            {
                name = 'lfoot',
                pos = {4, 'z+', 'y+'},
                piece = 6,
            },
            {
                name = 'rfoot',
                pos = {3, 'z+', 'y+'},
                piece = 7,
            },
        },
    },
    animation = humanoid_animation,
}

models.slime = {
    path = 'voxel/slime.vox',
    rootbone = {
        ref = 1,
        name = 'body',
        pos = {{1, 0, 0, -.5}, 'z+', 'y+'},
        piece = 2,
    },
    animation = {
        init = function(self)
            self.stretch = 0
            self.idle_t = 0
        end,
        motion = function(self, stretch)
            self.stretch = stretch
        end,
        draw = function(self, b, dt)
            local idle_speed = 2
            self.idle_t = self.idle_t + dt * idle_speed
            sz = self.stretch
            sz = sz + sin(self.idle_t) * 0.017
            sxy = -sz / 2
            move(b.body, 1,  0, 0, 0,  0, 0, 0,  sxy, sxy, sz)
        end,
    },
}

return models

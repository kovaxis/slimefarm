
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

local function easer(speed, dur)
    dur = units_per_sec * dur
    speed = 2 * pi * units_per_sec * speed
    local invdur = 1 / dur
    local Easer = class{}

    function Easer:new()
        self.up = not not self.up
        self.t = 0
        self.w = self.up and 1 or 0
    end

    function Easer:get(dt)
        local t, w = self.t, self.w

        if self.up then
            w = 1 - w
            -- w = f^-1(w)
            w = sqrt(w)

            w = w - dt * invdur
            if w < 0 then
                w = 0
            end
            -- w = f(w)
            w = w * w

            w = 1 - w
        else
            -- w = f^-1(w)
            w = 1 - 2*w
            if w < -1 then
                w = -1
            elseif w > 1 then
                w = 1
            end
            w = .5 - sin(math.asin(w) * (1/3))
            
            w = w - dt * invdur
            if w < 0 then
                w = 0
            end
            -- w = f(w)
            w = w*w*(3 - 2*w)
        end

        if w == 0 then
            t = 0
        elseif self.up then
            t = t + dt * speed
        else
            local nt = t + dt * speed
            local lt, rt = floor(t / pi + .4999), floor(nt / pi + .5001)
            if lt ~= rt then
                t = (rt - .5) * pi
            else
                t = nt
            end
        end
        
        self.t, self.w = t, w
        return t, w
    end

    function Easer:set(up)
        self.up = up
    end

    return Easer
end

-- Smoothly drive a variable in the [0, 1] range using direction set events.
-- In particular, all state is contained in two variables, one a reference time and one a boolean
-- indicator.
-- Direction can be set (up, dn) at a certain time, and the weight can be get at any time.
local function ease_quad(speed, dur)
    dur = dur * units_per_sec
    local invdur = 1 / dur
    local function init(dir)
        return -10000, dir
    end
    local function get(reft, refd, t)
        t = t - reft
        local w = t * invdur
        t = t * speed
        if refd then
            w = 1 - w
            if w < 0 then
                w = 0
            end
            -- w = f_up(w)
            w = w * w
        else

            if w > 1 then
                w = 1
            end
            -- w = f_dn(w)
            w = w*w*(3 - 2*w)
        end
        return t, 1 - w
    end
    local function set(reft, refd, time, d)
        if refd == d then
            return reft, refd
        end
        local t = (time - reft) * invdur
        if t >= 1 then
            return time, d
        end
        if d then
            -- t = f_dn(t)
            t = t*t*(3 - 2*t)
            -- t = f_up^-1(t)
            t = math.sqrt(t)
            t = 1 - t
        else
            t = 1 - t
            -- t = f_up(t)
            t = t*t
            -- t = f_dn^-1(t)
            t = 1 - 2*t
            if t < -1 then
                t = -1
            elseif t > 1 then
                t = 1
            end
            t = .5 - math.sin(math.asin(t) * (1/3))
        end
        return time - t * dur, d
    end
    return { init = init, get = get, set = set }
end

-- Make a pair of in/out quadratic easing functions (animation managers) with the given ease
-- duration
local function ease_quad_old(speed, dur)
    dur = dur * units_per_sec
    local invdur = 1 / dur
    return {
        go = function(t)
            if t >= dur then
                return speed * t, 1
            end
            local w = (dur - t) * invdur
            w = w*w
            return speed * t, 1 - w
        end,
        stop = function(t1, t2)
            local w1 = (dur - t1) * invdur
            if w1 > 0 then
                t2 = t2 + w1
            end
            if t2 >= dur then
                return
            end
            local w = 1 - t2 * invdur
            w = w*w*(3-2*w)
            w = w*w

            t1, t2 = speed * t1, speed * t2
            local t = t1 - t2
            local mx = (-t + .5 * math.pi) % math.pi
            if t2 < mx then
                t = t + t2
            else
                t = t + mx
            end
            return t, w
        end,
    }
end

local function bumpcos(t)
    local y = .5 + .5 * cos(t)
    return y * y
end

----- Voxel models -----

local models = {}

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
    animation = {
        init = function(self)
            self.state = 'idle'
            self.moving = 0
            self.air = 0
            self.t = 0
            self.idle_t = 0
            self.air_t = 0
        end,
        motion = function(self, name, t)
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
            local move_speed = 10
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
                local t, w = self.t, self.moving * lerp(self.air, 1, 0.62 + 0.04 * sin(self.air_t))
                local s = bumpcos(2*t) * .4
                move(b.body, w,  0, sin(2*t) * 0.20, 0,  0.40 + sin(2*t) * 0.0, 0, 0,  0, 0, 0)
                move(b.head, w,  0, sin(2*t) * -0.00, 0,  0, 0, 0,  0, 0, 0)
                move(b.lhand, w,  0, 0, 0,  sin(t) * 1.2, 0, 0,  0, 0, 0)
                move(b.rhand, w,  0, 0, 0,  sin(-t) * 1.2, 0, 0,  0, 0, 0)
                move(b.lfoot, w,  0, 0, 0,  sin(t) * 1.2, 0, 0,  -s*.5, s, -s*.5)
                move(b.rfoot, w,  0, 0, 0,  sin(-t) * 1.2, 0, 0,  -s*.5, s, -s*.5)
            end
        end,
    },
    animations = {
        {
            name = 'idle',
            bones = {
                body = function(t) return 0, sin(t) * 0.2, 0, 0, 0, 0, 0, 0, 0 end,
                head = function(t) return 0, sin(t) * 0.05, 0, 0, 0, 0, 0, 0, 0 end,
                lhand = function(t) return 0, 0, 0, sin(t*.87) * 0.04, 0, 0, 0, 0, 0 end,
                rhand = function(t) return 0, 0, 0, sin(-t*.87) * 0.04, 0, 0, 0, 0, 0 end,
            },
            manage = ease_quad(.033, .3),
        },
        {
            name = 'run',
            bones = {
                body = function(t) return 0, sin(2*t) * 0.6, 0, 0.40 + sin(2*t) * 0.0, 0, 0, 0, 0, 0 end,
                head = function(t) return 0, sin(2*t) * -0.15, 0, 0, 0, 0, 0, 0, 0 end,
                lhand = function(t) return 0, 0, 0, sin(t) * 1.2, 0, 0, 0, 0, 0 end,
                rhand = function(t) return 0, 0, 0, sin(-t) * 1.2, 0, 0, 0, 0, 0 end,
                lfoot = function(t)
                    local s = bumpcos(2*t) * .4
                    return 0, 0, 0, sin(t) * 1.2, 0, 0, -s*.5, s, -s*.5
                end,
                rfoot = function(t)
                    local s = bumpcos(2*t) * .4
                    return 0, 0, 0, sin(-t) * 1.2, 0, 0, -s*.5, s, -s*.5
                end,
            },
            manage = ease_quad(.20, .3),
        },
        {
            name = 'air',
            bones = {
                
            },
            manage = ease_quad(.08, .3),
        },
    },
}

return models

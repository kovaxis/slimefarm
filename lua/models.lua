
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

----- Animations -----

local humanoid = {
    runjump = {
        name = 'runjump',
        new = function(self, air)
            self.go = 1
            self.w = 0
            self.go_air = air
            self.air = air
            self.air_t = 0
            self.run_t = 0
        end,
        triggers = {
            motion = function(ctx, state)
                if state == 'run' then
                    return ctx:add('runjump', 0)
                elseif state == 'air' then
                    return ctx:add('runjump', 1)
                end
            end,
        },
        motion = function(self, state)
            if state == 'run' or state == 'air' then
                self.go = 1
                self.go_air = state == 'air' and 1 or 0
            else
                self.go = 0
            end
            return self.go
        end,
        draw = function(self, b, dt)
            self.w = approach(self.w, self.go, 0.001, 0.2, dt)
            self.air = approach(self.air, self.go_air, 0.004, 0.2, dt)
            local move_speed = 14
            if self.go_air == 1 or self.go == 0 then
                self.run_t = addsnap(self.run_t, dt * move_speed, pi, 0.5)
            else
                self.run_t = self.run_t + dt * move_speed
            end
            local air_speed = 3
            if self.air == 0 then
                self.air_t = 0
            else
                self.air_t = self.air_t + dt * air_speed
            end

            local t, w = self.run_t, self.w
            local air_t, air_w = self.air_t, self.air
            self.t, self.w = t, w
            local s = bumpcos(2*t) * .4
            local nc = lerp(air_w, 0.35, 0.05 + 0.01 * sin(air_t))
            local sp = lerp(air_w, 1.20, 1.45)
            move(b.body, w,  0, sin(2*t) * 0.20, 0,  nc, 0, 0,  0, 0, 0)
            move(b.head, w,  0, sin(2*t) * -0.00, 0,  0, 0, 0,  0, 0, 0)
            move(b.larm, w,  0, 0, 0,  sin(t) * sp, 0, 0,  0, 0, 0)
            move(b.rarm, w,  0, 0, 0,  sin(-t) * sp, 0, 0,  0, 0, 0)
            move(b.lfoot, w,  0, 0, 0,  sin(-t) * sp, 0, 0,  -s*.5, s, -s*.5)
            move(b.rfoot, w,  0, 0, 0,  sin(t) * sp, 0, 0,  -s*.5, s, -s*.5)
            return self.w
        end,
    },
    idle = {
        name = 'idle',
        new = function(self)
            self.go = 1
            self.t = 0
            self.w = 0
        end,
        triggers = {
            motion = function(ctx, state)
                if state == 'idle' then
                    return ctx:add('idle')
                end
            end,
        },
        motion = function(self, state)
            self.go = state == 'idle' and 1 or 0
            return self.go
        end,
        draw = function(self, b, dt)
            self.w = approach(self.w, self.go, 0.001, 0.2, dt)
            local idle_speed = 2
            self.t = self.t + dt * idle_speed
            local t, w = self.t, self.w
            move(b.body, w,  0, sin(t) * 0.2, 0,  0, 0, 0,  0, 0, 0)
            move(b.head, w,  0, sin(t) * 0.05, 0,  0, 0, 0,  0, 0, 0)
            move(b.larm, w,  0, 0, 0,  sin(t*.87) * 0.04, 0, 0,  0, 0, 0)
            move(b.rarm, w,  0, 0, 0,  sin(t*-.87) * 0.04, 0, 0,  0, 0, 0)
            return self.w
        end,
    },
    roll = {
        name = 'roll',
        new = function(self)
            self.go = 1
            self.x = 0
            self.w = 0
        end,
        triggers = {
            motion = function(ctx, state)
                if state == 'roll' then
                    return ctx:add('roll')
                end
            end,
        },
        motion = function(self, state, x)
            self.go = state == 'roll' and 1 or 0
            if state == 'roll' then
                self.x = x
            end
            return self.go
        end,
        draw = function(self, b, dt)
            local go = self.go
            if self.x >= 1 then
                go = 0
            end
            self.w = approach(self.w, go, 0.001, 0.2, dt)
            local t, w = 2*pi*((self.x+.5)%1-.5), self.w
            move(b.center, w,  0, 0, 0,  t, 0, 0,  0, 0, 0)
            move(b.larm, w,  0, 0, 0,  .75*pi, 0, 0,  0, 0, 0)
            move(b.rarm, w,  0, 0, 0,  .75*pi, 0, 0,  0, 0, 0)
            move(b.head, w,  0, -1, 0,  .42, 0, 0,  0, 0, 0)
            move(b.lfoot, w,  0, -1, 0,  .53, 0, 0,  0, 0, 0)
            move(b.rfoot, w,  0, -1, 0,  .53, 0, 0,  0, 0, 0)
            return self.go == 1 and 1 or self.w
        end,
    },
    atk = {
        name = 'atk',
        new = function(self)
            self.go = 1
            self.w = 0
            self.idx = 1
            self.t = 0

            self.xbody = 0
            self.xhead = 0
            self.xlfoot = 0
            self.xrfoot = 0
            self.xlarm = 0
            self.dyarm = 0
            self.xarm = 0
            self.yarm = 0
            self.zarm = 0
            self.xwrist = 0
            self.ywrist = 0
            self.zwrist = 0
        end,
        triggers = {
            motion = function(ctx, state)
                if state == 'atk' then
                    return ctx:add('atk')
                end
            end,
        },
        motion = function(self, state, idx)
            self.go = state == 'atk' and 1 or 0
            if self.go == 1 then
                if idx ~= self.idx then
                    self.t = 0
                end
                self.idx = idx
            end
            return self.go
        end,
        draw = function(self, b, dt)
            self.t = self.t + dt
            -- xbody+: lean forward
            -- xhead+: lean head forward
            -- xlfoot+: bring left foot up
            -- xrfoot+: bring right foot up
            -- xlarm+: bring left hand up
            -- dyarm+: bring right hand out
            -- xarm+: bring right arm up
            -- yarm+: turn swing plane clockwise
            -- zarm+: swing left
            -- xwrist+: raise wrist toward self
            -- ywrist+: swing weapon to the right
            -- zwrist+: rotate weapon in its axis clockwise
            local xbody, xhead, xlfoot, xrfoot, xlarm, dyarm, xarm, yarm, zarm, xwrist, ywrist, zwrist
            local bmp = 1 + 2^(self.t * -5) * 2
            if self.idx == 1 then
                xbody, xhead, xlfoot, xrfoot, xlarm = .04*bmp, -.02*bmp, -.35, .35, -.3
                dyarm, xarm, yarm, zarm = 7, .7, .10, .4
                xwrist, ywrist, zwrist = -.5, -.3, .5
            elseif self.idx == 2 then
                xbody, xhead, xlfoot, xrfoot, xlarm = .04*bmp, -.02*bmp, .35, -.35, .05
                dyarm, xarm, yarm, zarm = 0, .65, -.3, -.3
                xwrist, ywrist, zwrist = -.5, .2, -.5
            elseif self.idx == 3 then
                xbody, xhead, xlfoot, xrfoot, xlarm = .04*bmp, -.02*bmp, -.40, .40, -.4
                dyarm, xarm, yarm, zarm = 7, .7, -.1, .4
                xwrist, ywrist, zwrist = -.5, -.4, .5
            else
                xbody, xhead, xlfoot, xrfoot, xlarm = self.xbody, self.xhead, 0, 0, self.xlarm
                dyarm, xarm, yarm, zarm = 0, .0, .0, .0
                xwrist, ywrist, zwrist = .0, .0, .0
            end
            local sp = 0.0005
            self.xbody = approach(self.xbody, xbody, sp, 0.6, dt)
            self.xhead = approach(self.xhead, xhead, sp, 0.6, dt)
            self.xlfoot = approach(self.xlfoot, xlfoot, sp, 0.6, dt)
            self.xrfoot = approach(self.xrfoot, xrfoot, sp, 0.6, dt)
            self.xlarm = approach(self.xlarm, xlarm, sp, 0.6, dt)
            self.dyarm = approach(self.dyarm, dyarm, sp, 0.6, dt)
            self.xarm = approach(self.xarm, xarm, sp, 0.6, dt)
            self.yarm = approach(self.yarm, yarm, sp, 0.6, dt)
            self.zarm = approach(self.zarm, zarm, sp, 0.6, dt)
            self.xwrist = approach(self.xwrist, xwrist, sp, 0.6, dt)
            self.ywrist = approach(self.ywrist, ywrist, sp, 0.6, dt)
            --self.zwrist = approach(self.zwrist, zwrist, sp, 0.6, dt)
            self.zwrist = zwrist

            self.w = approach(self.w, self.go, 0.0001, 0.8, dt)
            local w = self.w
            move(b.body, w,  0, 0, 0,  self.xbody*pi, 0, 0,  0, 0, 0)
            move(b.head, w,  0, 0, 0,  self.xhead*pi, 0, 0,  0, 0, 0)
            move(b.lfoot, w,  0, 0, 0,  self.xlfoot*pi, 0, 0,  0, 0, 0)
            move(b.rfoot, w,  0, 0, 0,  self.xrfoot*pi, 0, 0,  0, 0, 0)
            move(b.larm, w,  0, 0, 0,  self.xlarm*pi, 0, 0,  0, 0, 0)
            move(b.rarm, w,  0, self.dyarm, 0,  self.xarm*pi, self.yarm*pi, self.zarm*pi,  0, 0, 0)
            move(b.rwrist, w,  0, 0, 0,  self.xwrist*pi, self.ywrist*pi, self.zwrist*pi,  0, 0, 0)
            return self.w
        end,
    },
}

local slimy = {
    {
        name = 'stretch',
        new = function(self, stretch)
            self.stretch = stretch
            self.idle_t = 0
        end,
        triggers = {
            stretch = function(ctx, stretch) return ctx:add('stretch', stretch) end,
        },
        stretch = function(self, stretch)
            self.stretch = stretch
            return 1
        end,
        draw = function(self, b, dt)
            local idle_speed = 2
            self.idle_t = self.idle_t + dt * idle_speed
            local sz = self.stretch
            sz = sz + sin(self.idle_t) * 0.017
            local sxy = -sz / 2
            move(b.body, 1,  0, 0, 0,  0, 0, 0,  sxy, sxy, sz)
            return 1
        end,
    },
}

----- Voxel models -----

local models = {}

models.player = {
    path = 'voxel/player.vox',
    rootbone = {
        ref = 1,
        name = 'base',
        pos = {{1, 0, 0, -.5}, 'z+', 'y+'},

        child = {
            name = 'center',
            pos = {6, 'z+', 'y+'},

            children = {
                {
                    name = 'body',
                    pos = {2, 5, 'y+'},
                    piece = 2,

                    children = {
                        {
                            name = 'head',
                            pos = {6, 'z+', 'y+'},
                            order = 'zxy',
                            piece = 3,
                        },
                        {
                            name = 'larm',
                            pos = {8, 'z-', 'y+'},
                            order = 'xyz',
                            child = {
                                name = 'lwrist',
                                pos = {10, 'z-', 'y+'},
                                order = 'xyz',
                                piece = 4,
                            },
                        },
                        {
                            name = 'rarm',
                            pos = {7, 'z-', 'y+'},
                            order = 'xyz',
                            child = {
                                name = 'rwrist',
                                pos = {9, 'z-', 'y+'},
                                order = 'xyz',
                                piece = 5,
                            },
                        },
                    },
                },
                {
                    name = 'lfoot',
                    pos = {4, 'z-', 'y+'},
                    order = 'zxy',
                    piece = 6,
                },
                {
                    name = 'rfoot',
                    pos = {3, 'z-', 'y+'},
                    order = 'zxy',
                    piece = 7,
                },
            },
        },
    },
    animations = { humanoid.runjump, humanoid.idle, humanoid.roll, humanoid.atk },
}

models.slime = {
    path = 'voxel/slime.vox',
    rootbone = {
        ref = 1,
        name = 'body',
        pos = {{1, 0, 0, -.5}, 'z+', 'y+'},
        piece = 2,
    },
    animations = { slimy.stretch },
}

return models

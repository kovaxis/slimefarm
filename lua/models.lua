
local cos, sin = math.cos, math.sin

local units_per_sec = 64
local manage = {}

-- Make a pair of in/out quadratic easing functions (animation managers) with the given ease
-- duration
function manage.quad(speed, dur)
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
    animations = {
        {
            name = 'idle',
            bones = {
                body = function(t) return 0, sin(t) * 0.2, 0, 0, 0, 0, 0, 0, 0 end,
                head = function(t) return 0, sin(t) * 0.05, 0, 0, 0, 0, 0, 0, 0 end,
                lhand = function(t) return 0, 0, 0, sin(t*.87) * 0.04, 0, 0, 0, 0, 0 end,
                rhand = function(t) return 0, 0, 0, sin(-t*.87) * 0.04, 0, 0, 0, 0, 0 end,
            },
            manage = manage.quad(.033, .3),
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
            manage = manage.quad(.20, .3),
        },
        {
            name = 'air',
            bones = {
                
            },
            manage = manage.quad(.08, .3),
        },
    },
}

return models

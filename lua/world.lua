
local class = require 'class'
local util = require 'util'
local Mesh = require 'mesh'
local input = require 'input'
local Sprite = require 'sprite'

local World = class{}

function World:new()
    self.tick_count = 0
    self.terrain = system.terrain [[{
        "_kind": {"Parkour": {
            "z_offset": 0.008,
            "delta": 0.4,
            "color": [0.43, 0.43, 0.43]
        }},
        "kind": {"Plains": {
            "xy_scale": 256,
            "detail": 3,
            "z_scale": 40,
            "color": [0.01, 0.92, 0.20],
            "log_color": [0.53, 0.12, 0.01]
        }},
        "gen_radius": 16,
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
    self.terrain:set_view_distance(32*12)
    self.entities = {}

    self.shaders = {
        terrain = util.Shader{
            vertex = 'terrain.vert',
            fragment = 'terrain.frag',
            uniforms = {'mvp', 'mv', 'offset', 'l_dir', 'ambience', 'diffuse', 'specular', 'fog'},
        },
        basic = util.Shader{
            vertex = 'basic.vert',
            fragment = 'basic.frag',
            uniforms = {'mvp', 'tint'},
        },
        skybox = util.Shader{
            vertex = 'skybox.vert',
            fragment = 'skybox.frag',
            uniforms = {'view', 'base', 'lowest', 'highest', 'sunrise', 'sunrise_dir'},
        },
    }

    self.font_size = 6
    self.font = gfx.font(util.read_file("font/dogicapixel.ttf"), self.font_size)

    self.frame = {
        target_res = 240,
        physical_w = 1,
        physical_h = 1,
        w = 1,
        h = 1,
        params_world = gfx.draw_params(),
        params_hud = gfx.draw_params(),
        mvp_world = system.matrix(),
        mv_world = system.matrix(),
        mvp_hud = system.matrix(),
        s = 0,
        dt = 0,
    }
    self.frame.params_world:set_cull('cw')
    self.frame.params_world:set_depth('if_less', true)
    self.frame.params_world:set_color_blend('add', 'src_alpha', 'one_minus_src_alpha')
    self.frame.params_hud:set_depth('overwrite', false);
    self.frame.params_hud:set_color_blend('add', 'src_alpha', 'one_minus_src_alpha')

    self.cam_prev_x = 0
    self.cam_prev_y = 0
    self.cam_prev_z = 0
    self.cam_x = 0
    self.cam_y = 0
    self.cam_z = 0
    self.cam_rollback = 0
    self.cam_yaw = 0
    self.cam_pitch = 0
    self.cam_effective_x = 0
    self.cam_effective_y = 0
    self.cam_effective_z = 0

    self.day_cycle = 0.20

    self.fog_poll_next = os.clock()
    self.fog_poll_interval = 1 / 16
    self.fog_last_minimums = {}
    self.fog_min_idx = 0
    self.fog_current = 0
    self.fog_target = 0
    for i = 1, 16 do
        self.fog_last_minimums[i] = 0
    end

    self.tick_period = 1/64
    self.next_tick = os.clock()
    self.fps_counter = 0
    self.fps_next_reset = os.clock()
    self.fps = 0
    self.last_frame = os.clock()

    self.mouse_icon = Sprite{
        path = 'crosshair.png',
        w = 16,
        h = 16,
    }
    self.mouse_x = 0
    self.mouse_y = 0
end

function World:tick()
    --Tick entities
    for _, ent in ipairs(self.entities) do
        ent:tick(self)
    end

    --Advance day cycle
    self.day_cycle = (self.day_cycle + 1 / (64*120)) % 1
    --DEBUG: Advance day cycle quicker when in nighttime
    if self.day_cycle < 0.3 or self.day_cycle > 0.6 then
        self.day_cycle = self.day_cycle + 1 / (64*5)
    end

    --Bookkeep terrain
    local time_limit = math.max(self.next_tick - os.clock() - 0.004, 0)
    self.terrain:hint_center(self.cam_x, self.cam_y, self.cam_z)

    --Advance tick count
    self.tick_count = self.tick_count + 1
end

local last_tick = os.clock()
local timer = util.DebugTimer{}
function World:update()
    timer:mark("inter")
    --Tick 0 or more times
    while true do
        local now = os.clock()
        if now < self.next_tick then
            break
        end
        self.next_tick = self.next_tick + self.tick_period
        self:tick()
    end
    timer:mark("tick")
    --Draw the updated world
    self:draw()
    timer:mark("draw")
    gfx.finish()
    timer:mark("finish")
    
    --print(timer:to_str())
    timer:start()
end

local sky = {}
do
    local dawn, day, eve, night
    local lo, hi
    sky.screen = Mesh{}:add_quad(-1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1):as_buffer()

    dawn = {.14, .42, .77}
    day = {.08, .52, .90}
    eve = {.14, .42, .77}
    night = {.00, .01, .02}
    sky.base = util.Curve{ 0, night, 0.20, night, 0.25, dawn, 0.50, day, 0.73, eve, 0.77, night, 1, night }

    dawn = {.14, .72, .88}
    day = {.08, .78, .94}
    eve = {.14, .72, .88}
    night = {.00, .00, .01}
    sky.highest = util.Curve{ 0, night, 0.20, night, 0.25, dawn, 0.50, day, 0.73, eve, 0.77, night, 1, night }

    dawn = {.77, .62, .60}
    day = {.70, .75, .77}
    eve = {.77, .62, .60}
    night = {.01, .02, .03}
    sky.lowest = util.Curve{ 0, night, 0.20, night, 0.25, dawn, 0.50, day, 0.73, eve, 0.77, night, 1, night }

    dawn = {.68, .02, -.05}
    day = {.00, .00, .00}
    eve = {.45, .02, -.05}
    night = {.00, .00, .00}
    sky.sunrise = util.Curve{ 0, night, 0.20, night, 0.25, dawn, 0.30, day, 0.70, day, 0.75, eve, 0.77, night, 1, night }

    lo, hi = 0.00, 0.75
    sky.ambience = util.Curve{ 0, lo, 0.23, lo, 0.40, hi, 0.60, hi, 0.77, lo }

    lo, hi = 0, 0.20
    sky.diffuse  = util.Curve{ 0, lo, 0.23, lo, 0.27, hi, 0.73, hi, 0.77, lo }

    lo, hi = 0, 0.01
    sky.specular = util.Curve{ 0, lo, 0.23, lo, 0.27, hi, 0.73, hi, 0.77, lo }
end

function World:draw()
    local frame = self.frame
    local now = os.clock()
    gfx.clear(0, 0, 0, 0, 1)

    --Count FPS
    while now >= self.fps_next_reset do
        self.fps_next_reset = self.fps_next_reset + 1
        self.fps = self.fps_counter
        self.fps_counter = 0
    end
    self.fps_counter = self.fps_counter + 1

    --Figure out screen dimensions and pixelated scaling
    do
        local pw, ph = gfx.dimensions()
        frame.physical_w, frame.physical_h = pw, ph
        local scale = math.max(math.floor(math.min(pw, ph) / frame.target_res), 1)
        local w, h = math.floor(pw / scale / 2), math.floor(ph / scale / 2)
        frame.w, frame.h = w, h
        frame.mvp_hud:reset()
        frame.mvp_hud:scale(2*scale/pw, 2*scale/ph, 1)
    end

    --Get interpolation factor `s`, a weight between the previous tick and the current one
    frame.dt = now - self.last_frame
    frame.s = (now - self.next_tick) / self.tick_period + 1
    self.last_frame = now

    --Find out real camera location
    local cam_x, cam_y, cam_z
    local cam_yaw, cam_pitch
    do
        local og_cam_x = self.cam_prev_x + (self.cam_x - self.cam_prev_x) * frame.s
        local og_cam_y = self.cam_prev_y + (self.cam_y - self.cam_prev_y) * frame.s
        local og_cam_z = self.cam_prev_z + (self.cam_z - self.cam_prev_z) * frame.s
        cam_yaw = self.cam_yaw
        cam_pitch = self.cam_pitch

        local cam_wall_dist = 0.4
        local rollback = self.cam_rollback
        local dx = -math.sin(self.cam_yaw) * math.cos(self.cam_pitch) * rollback
        local dy = math.cos(self.cam_yaw) * math.cos(self.cam_pitch) * rollback
        local dz = math.sin(self.cam_pitch) * rollback
        cam_x, cam_y, cam_z = self.terrain:raycast(og_cam_x, og_cam_y, og_cam_z, -dx, -dy, -dz, cam_wall_dist, cam_wall_dist, cam_wall_dist)
        self.cam_effective_x = cam_x
        self.cam_effective_y = cam_y
        self.cam_effective_z = cam_z
        --[[
        cam_yaw = math.atan(og_cam_x - cam_x, cam_z - og_cam_z)
        local len = ((og_cam_x - cam_x)^2 + (og_cam_z - cam_z)^2)^0.5
        cam_pitch = math.atan(og_cam_y - cam_y, len)

        --Set camera back a bit
        dx, dy, dz = cam_x - og_cam_x, cam_y - og_cam_y, cam_z - og_cam_z
        len = (dx*dx + dy*dy + dz*dz)^0.5
        local factor = (len - cam_wall_dist) / len
        cam_x = og_cam_x + dx * factor
        cam_y = og_cam_y + dy * factor
        cam_z = og_cam_z + dz * factor
        ]]
    end

    --Setup model-view-projection matrix for world drawing
    local vfov = 1.1
    local hfov = vfov * frame.physical_w / frame.physical_h
    frame.mvp_world:reset()
    frame.mvp_world:perspective(vfov, frame.physical_w / frame.physical_h, 0.1, 1000)
    frame.mv_world:reset()
    frame.mv_world:rotate_x(-cam_pitch)
    frame.mv_world:rotate_z(-cam_yaw)
    frame.mvp_world:mul_right(frame.mv_world)

    --Draw skybox
    do
        local cycle = self.day_cycle
        frame.mv_world:push()
        frame.mv_world:identity()
        frame.mv_world:rotate_z(cam_yaw)
        frame.mv_world:rotate_x(cam_pitch)
        frame.mv_world:scale(math.tan(hfov / 2), math.tan(vfov / 2), 1)
        self.shaders.skybox:set_matrix('view', frame.mv_world)
        self.shaders.skybox:set_vec3('base', sky.base:at(cycle))
        self.shaders.skybox:set_vec3('highest', sky.highest:at(cycle))
        self.shaders.skybox:set_vec3('lowest', sky.lowest:at(cycle))
        self.shaders.skybox:set_vec3('sunrise', sky.sunrise:at(cycle))
        self.shaders.skybox:set_vec3('sunrise_dir', math.sin(2*math.pi*cycle), 0, -math.cos(2*math.pi*cycle))
        self.shaders.skybox:draw(sky.screen, frame.params_hud)
        frame.mv_world:pop()
    end

    --Update fog distance
    do
        local now = os.clock()
        if now > self.fog_poll_next then
            self.fog_poll_next = now + (self.fog_poll_interval - now % self.fog_poll_interval)
            --Poll fog
            local fog_now = self.terrain:visible_radius(cam_x, cam_y, cam_z)
            self.fog_min_idx = self.fog_min_idx + 1
            self.fog_last_minimums[self.fog_min_idx] = fog_now
            self.fog_min_idx = self.fog_min_idx % #self.fog_last_minimums
            local abs_min = fog_now
            for i = 1, #self.fog_last_minimums do
                local fog = self.fog_last_minimums[i]
                if fog < abs_min then
                    abs_min = fog
                end
            end
            if abs_min < self.fog_target then
                self.fog_target = abs_min
            elseif abs_min > self.fog_target + 10 then
                self.fog_target = abs_min
            end
        end
        if self.fog_target > self.fog_current then
            --Move fog out slowly
            self.fog_current = util.approach(self.fog_current, self.fog_target, 0.02, 2, frame.dt)
        else
            --Move fog in quickly
            self.fog_current = util.approach(self.fog_current, self.fog_target, 0.00001, 8, frame.dt)
        end
    end

    --Draw terrain
    do
        local cycle = self.day_cycle
        local ambience = sky.ambience:at(cycle)
        local diffuse  = sky.diffuse:at(cycle)
        local specular = sky.specular:at(cycle)
        local dx, dy, dz = -math.cos((cycle - 0.25) * 2 * math.pi), 0, -math.sin((cycle - 0.25) * 2 * math.pi)
        --ambience = 0
        --diffuse = 0.2
        --specular = 0.03
        --dx, dy, dz = 2^-0.5, 0, -2^-0.5
        dx, dy, dz = frame.mv_world:transform_vec(dx, dy, dz)
        self.shaders.terrain:set_float('fog', self.fog_current)
        self.shaders.terrain:set_matrix('mvp', frame.mvp_world)
        self.shaders.terrain:set_matrix('mv', frame.mv_world)
        self.shaders.terrain:set_vec3('ambience', ambience, ambience, ambience)
        self.shaders.terrain:set_vec3('diffuse', diffuse, diffuse, diffuse)
        self.shaders.terrain:set_vec3('specular', specular, specular, specular)
        self.shaders.terrain:set_vec3('l_dir', dx, dy, dz)
        self.shaders.terrain:draw_terrain(self.terrain, 'offset', frame.params_world, cam_x, cam_y, cam_z)
    end
    
    --Draw entities
    for _, ent in ipairs(self.entities) do
        local prevx, prevy, prevz = ent.prev_x - cam_x, ent.prev_y - cam_y, ent.prev_z - cam_z
        local x, y, z = ent.x - cam_x, ent.y - cam_y, ent.z - cam_z
        x, y, z = prevx + (x - prevx) * frame.s, prevy + (y - prevy) * frame.s, prevz + (z - prevz) * frame.s
        frame.mvp_world:push()
        frame.mvp_world:translate(x, y, z)
        ent:draw(self)
        frame.mvp_world:pop()
    end

    --Draw HUD
    frame.mvp_hud:push()
    frame.mvp_hud:translate(-frame.w + 4, frame.h - 16, 0)
    frame.mvp_hud:scale(self.font_size)
    self.font:draw("FPS: "..self.fps, frame.mvp_hud, frame.params_hud, 1, 1, 1)
    frame.mvp_hud:translate(0, -1.25, 0)
    self.font:draw("gen: "..util.format_time(self.terrain:chunk_gen_time()), frame.mvp_hud, frame.params_hud, 1, 1, 1)
    frame.mvp_hud:translate(0, -1.25, 0)
    self.font:draw("mesh: "..util.format_time(self.terrain:chunk_mesh_time()), frame.mvp_hud, frame.params_hud, 1, 1, 1)
    frame.mvp_hud:translate(0, -1.25, 0)
    self.font:draw("upload: "..util.format_time(self.terrain:chunk_mesh_upload_time()), frame.mvp_hud, frame.params_hud, 1, 1, 1)
    frame.mvp_hud:translate(0, -1.25, 0)
    self.font:draw("pos: "..math.floor(self.cam_x)..", "..math.floor(self.cam_y)..", "..math.floor(self.cam_z), frame.mvp_hud, frame.params_hud, 1, 1, 1)
    frame.mvp_hud:pop()

    --Draw crosshair
    frame.mvp_hud:push()
    frame.mvp_hud:translate(0.5, 0.5, 0)
    frame.mvp_hud:scale(self.mouse_icon.w, self.mouse_icon.h, 1)
    self.mouse_icon:draw(1, frame.mvp_hud, frame.params_hud)
    frame.mvp_hud:pop()
end

function World:mousemove(dx, dy)
    -- under a right-handed coordinate system:
    -- rotation is counterclockwise around the Y axis, so looking to the right is negative rotation
    self.cam_yaw = (self.cam_yaw - dx * 0.01) % (2*math.pi)
    -- rotation is counterclockwise around the X axis, so looking down is negative rotation
    self.cam_pitch = self.cam_pitch - dy * 0.01
    if self.cam_pitch < math.pi / -2 then
        self.cam_pitch = math.pi / -2
    elseif self.cam_pitch > math.pi / 2 then
        self.cam_pitch = math.pi / 2
    end
end

return World
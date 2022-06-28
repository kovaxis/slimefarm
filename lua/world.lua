
local class = require 'class'
local util = require 'util'
local Mesh = require 'mesh'
local input = require 'input'
local Sprite = require 'sprite'
local voxel = require 'voxel'

local World = class{}

function World:new()
    self.tick_count = 0
    self.entities = {}

    self.shaders = {
        terrain = util.Shader{
            vertex = 'terrain.vert',
            fragment = 'terrain.frag',
            -- first 3 must be (in order) 'offset', 'color', 'light'
            uniforms = {'offset', 'color', 'light', 'mvp', 'invp', 'nclip', 'clip', 'l_dir', 'ambience', 'diffuse', 'specular', 'fog', 'base', 'lowest', 'highest', 'sunrise', 'sun_dir', 'cycle'},
        },
        portal = util.Shader{
            vertex = 'portal.vert',
            fragment = 'portal.frag',
            uniforms = {'mvp', 'offset', 'nclip', 'clip'},
        },
        basic = util.Shader{
            vertex = 'basic.vert',
            fragment = 'basic.frag',
            uniforms = {'mvp', 'tint'},
        },
        skybox = util.Shader{
            vertex = 'skybox.vert',
            fragment = 'skybox.frag',
            uniforms = {'mvp', 'offset', 'view', 'base', 'lowest', 'highest', 'sunrise', 'sun_dir', 'cycle'},
        },
        dbgchunk = util.Shader{
            vertex = 'dbgchunk.vert',
            fragment = 'dbgchunk.frag',
            uniforms = {'mvp', 'offset'},
        },
    }

    self.worldgen_path = "gen"
    self.worldgen_main = "gen/main.lua"
    self.worldgen_watcher = fs.watch(self.worldgen_path)
    self.showchunkgrid = false
    self:load_terrain()

    self.font_size = 6
    self.font = gfx.font(util.read_file("font/dogicapixel.ttf"), self.font_size)

    self.frame = {
        target_res = 240,
        physical_w = 1,
        physical_h = 1,
        w = 1,
        h = 1,
        params_portalopen = gfx.draw_params(),
        params_portalclose = gfx.draw_params(),
        params_sky = gfx.draw_params(),
        params_world = gfx.draw_params(),
        params_hud = gfx.draw_params(),
        invp_world = system.matrix(),
        mvp_world = system.matrix(),
        mvp_hud = system.matrix(),
        hfov = 1,
        vfov = 1,
        entcopy_buf = {},
        cam_stack = gfx.camera_stack(),
        cam_yaw = 0,
        cam_pitch = 0,
        s = 0,
        dt = 0,
        portal_budget = 0,
    }
    self.frame.params_portalopen:set_depth('if_less', false, 'both')
    self.frame.params_portalopen:set_stencil('ccw', 'if_equal', 0, 'increment_wrap')
    self.frame.params_portalopen:set_cull('cw')
    self.frame.params_portalopen:set_clip_planes(31)
    self.frame.params_portalopen:set_color_mask(false)
    self.frame.params_portalopen:set_polygon_offset(true, -1, -2)
    self.frame.params_portalclose:set_depth('always_pass', true, 'both')
    self.frame.params_portalclose:set_stencil('ccw', 'if_equal', 0, 'decrement_wrap')
    self.frame.params_portalclose:set_cull('cw')
    self.frame.params_portalclose:set_clip_planes(31)
    self.frame.params_portalclose:set_color_mask(false)
    self.frame.params_sky:set_depth('always_pass', true)
    self.frame.params_sky:set_stencil('ccw', 'if_equal', 0)
    self.frame.params_world:set_cull('cw')
    self.frame.params_world:set_depth('if_less', true)
    self.frame.params_world:set_stencil('ccw', 'if_equal', 0)
    self.frame.params_world:set_clip_planes(31)
    self.frame.params_world:set_color_blend('add', 'src_alpha', 'one_minus_src_alpha') -- TODO: Goodbye slimes
    self.frame.params_hud:set_depth('always_pass', false);
    self.frame.params_hud:set_color_blend('add', 'src_alpha', 'one_minus_src_alpha')
    self.subdraw_bound = function()
        self:subdraw()
    end

    self.cam_pos = system.world_pos()
    self.cam_mov_x = 0
    self.cam_mov_y = 0
    self.cam_mov_z = 0
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

    self.rng = math.rng(math.hash(os.clock()))

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
    --Check whether terrain file changed
    if self.worldgen_watcher:changed() then
        print("reloading terrain")
        self:load_terrain()
    end

    --Check whether voxel models changed
    voxel.check_reload()

    --Tick entities
    for _, ent in ipairs(self.entities) do
        ent:tick(self)
    end

    --Advance day cycle
    self.day_cycle = (self.day_cycle + 1 / (64*120)) % 1
    --self.day_cycle = (self.day_cycle + 1 / (64*30)) % 1
    --DEBUG: Advance day cycle quicker when in nighttime
    if self.day_cycle < 0.3 or self.day_cycle > 0.7 then -- Skip night
    --if self.day_cycle < 0.2 or self.day_cycle > 0.8 then -- Skip night but not sunset and sunrise
    --if self.day_cycle > 0.2 and self.day_cycle < 0.8 then -- Skip day
    --if self.day_cycle > 0.3 and self.day_cycle < 0.7 then -- Skip day but not sunset and sunrise
        self.day_cycle = self.day_cycle + 1 / (64*5)
    end

    --Bookkeep terrain
    local time_limit = math.max(self.next_tick - os.clock() - 0.004, 0)
    self.terrain:bookkeep(self.cam_pos)

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
    
    timer:finish()
    --print(timer:to_str())
end

function World:load_terrain()
    local file = io.open(self.worldgen_main, 'r')
    local gen_main = file:read('a')
    file:close()
    self.terrain = system.terrain {
        gen = {
            src = gen_main,
            args = {{
                seed = 6813264,
                kind = 'gen.plainsgen',
            }},
        },
        mesher = voxel.mesher_cfg,
    }
    self.terrain:set_interpolation(false, true)
    --self.terrain:set_view_distance(32*12, 32*14)
    self.terrain:set_view_distance(32*6, 32*8)

    --Set chunkframe
    self:update_showchunkgrid(self.showchunkgrid)
end

function World:update_showchunkgrid(show)
    if show then
        local j, w = 0.1, 32
        local r, g, b, a = 1, 0, 0, 1
        local mesh = Mesh{}
        local function addquad(bx, by, bz, dx, dy, dz, d1x, d1y, d1z, d2x, d2y, d2z)
            mesh:add_quad(
                bx + d1x, by + d1y, bz + d1z,
                bx + d2x, by + d2y, bz + d2z,
                bx + dx + d2x, by + dy + d2y, bz + dz + d2z,
                bx + dx + d1x, by + dy + d1y, bz + dz + d1z,
                r, g, b, a
            )
        end
        addquad(0, 0, 0,   w, 0, 0,   0, j, 0,   0, 0, j)
        addquad(w, 0, 0,   0, w, 0,  -j, 0, 0,   0, 0, j)
        addquad(w, w, 0,  -w, 0, 0,  0, -j, 0,   0, 0, j)
        addquad(0, w, 0,  0, -w, 0,   j, 0, 0,   0, 0, j)

        addquad(0, 0, 0,   0, 0, w,   j, 0, 0,   0, j, 0)
        addquad(w, 0, 0,   0, 0, w,   0, j, 0,  -j, 0, 0)
        addquad(w, w, 0,   0, 0, w,  -j, 0, 0,  0, -j, 0)
        addquad(w, 0, 0,   0, 0, w,  0, -j, 0,   j, 0, 0)

        addquad(0, 0, w,   w, 0, 0,  0, 0, -j,   0, j, 0)
        addquad(w, 0, w,   0, w, 0,  0, 0, -j,  -j, 0, 0)
        addquad(w, w, w,  -w, 0, 0,  0, 0, -j,  0, -j, 0)
        addquad(0, w, w,  0, -w, 0,  0, 0, -j,   j, 0, 0)

        self.terrain:set_dbg_chunkframe(self.shaders.dbgchunk.program, mesh:as_buffer())
    else
        self.terrain:set_dbg_chunkframe(nil, nil)
    end
    self.showchunkgrid = show
end

local sky = {}
do
    local dawn, day, eve, night
    local lo, hi
    sky.screen = Mesh{}:add_quad(-1, -1, 1,   1, -1, 1,   1, 1, 1,   -1, 1, 1):as_buffer()
    sky.portalscreen = gfx.buffer_empty()

    dawn = {.14, .72, .88}
    day = {.08, .78, .94}
    eve = {.14, .72, .88}
    night = {.002, .008, .018}
    sky.base = util.Curve{ 0, night, 0.20, night, 0.25, dawn, 0.50, day, 0.73, eve, 0.77, night, 1, night }

    dawn = {.07, .38, .77}
    day = {.01, .48, .90}
    eve = {.07, .38, .77}
    night = {.001, .005, .008}
    sky.highest = util.Curve{ 0, night, 0.20, night, 0.25, dawn, 0.50, day, 0.73, eve, 0.77, night, 1, night }

    dawn = {.77, .62, .60}
    day = {.70, .75, .77}
    eve = {.77, .62, .60}
    night = {.001, .02, .03}
    sky.lowest = util.Curve{ 0, night, 0.20, night, 0.25, dawn, 0.50, day, 0.73, eve, 0.77, night, 1, night }

    dawn = {.68, .02, -.05}
    day = {.00, .00, .00}
    eve = {.45, .02, -.05}
    night = {.00, .00, .00}
    sky.sunrise = util.Curve{ 0, night, 0.20, night, 0.25, dawn, 0.30, day, 0.70, day, 0.75, eve, 0.77, night, 1, night }

    lo, hi = 0.00, 0.75
    sky.ambience = util.Curve{ 0, lo, 0.22, lo, 0.40, hi, 0.60, hi, 0.78, lo }

    lo, hi = 0, 0.20
    sky.diffuse  = util.Curve{ 0, lo, 0.22, lo, 0.27, hi, 0.73, hi, 0.78, lo }

    lo, hi = 0, 1
    sky.specular = util.Curve{ 0, lo, 0.22, lo, 0.27, hi, 0.73, hi, 0.78, lo }
    
    local colors = {'base', 'highest', 'lowest', 'sunrise'}
    local lighting = {'ambience', 'diffuse', 'specular'}
    function sky.colors(shader, cycle)
        for i = 1, #colors do
            local r, g, b = sky[colors[i]]:at(cycle)
            shader:set_vec3(colors[i], r, g, b)
        end
        shader:set_float('cycle', cycle)
        shader:set_vec3('sun_dir', math.sin(2*math.pi*cycle), 0, -math.cos(2*math.pi*cycle))
    end
    function sky.lighting(shader, cycle)
        for i = 1, #lighting do
            local v = sky[lighting[i]]:at(cycle)
            shader:set_vec3(lighting[i], v, v, v)
        end
    end
end

-- Set the `nclip` and `clip` shader uniforms to the clipping planes of the given camera
local set_clip_planes
do
    local c, m = {}, system.matrix()
    function set_clip_planes(shader, cam)
        cam:clip_planes(c)
        m:set_col(0, c[5], c[6], c[7], c[8])
        m:set_col(1, c[9], c[10], c[11], c[12])
        m:set_col(2, c[13], c[14], c[15], c[16])
        m:set_col(3, c[17], c[18], c[19], c[20])
        shader:set_vec4('nclip', c[1], c[2], c[3], c[4])
        shader:set_matrix('clip', m)
    end
end

-- Draw terrain within a frame or portal (the camera being a frame)
local portalbuf = gfx.buffer_empty()
local entpos_buf = system.world_pos()
local campos_buf = system.world_pos()
function World:subdraw()
    local frame = self.frame
    local cam = frame.cam_stack
    local depth = cam:depth()
    
    local x0, y0, z0 = cam:framequad(0)
    local x1, y1, z1 = cam:framequad(1)
    local x2, y2, z2 = cam:framequad(2)
    local x3, y3, z3 = cam:framequad(3)

    -- Whether the portal geometry intersects the near clipping plane
    local proper = cam:proper()

    --Draw portal itself into the stencil buffer
    if depth > 0 then
        cam:geometry(portalbuf)
        self.shaders.portal:set_matrix('mvp', frame.mvp_world)
        self.shaders.portal:set_vec3('offset', cam:geometry_offset())
        set_clip_planes(self.shaders.portal, cam)
        frame.params_portalopen:set_depth('if_less', false, proper and 'none' or 'both')
        frame.params_portalopen:set_stencil_ref(depth - 1)
        self.shaders.portal:draw(portalbuf, frame.params_portalopen)
    end

    --Draw skybox
    do
        local cycle = self.day_cycle
        local whole_screen = depth == 0 or not proper

        local skybuf, dx, dy, dz
        if whole_screen then
            skybuf = sky.screen
            dx, dy, dz = 0, 0, 0
        else
            skybuf = sky.portalscreen
            cam:geometry(skybuf)
            dx, dy, dz = cam:geometry_offset()
        end

        if whole_screen then
            frame.mvp_world:push()
            frame.mvp_world:identity()
            self.shaders.skybox:set_matrix('mvp', frame.mvp_world)
            frame.mvp_world:pop()
        else
            self.shaders.skybox:set_matrix('mvp', frame.mvp_world)
        end
        self.shaders.skybox:set_vec3('offset', dx, dy, dz)

        frame.mvp_world:push()
        frame.mvp_world:identity()
        frame.mvp_world:rotate_z(frame.cam_yaw)
        frame.mvp_world:rotate_x(frame.cam_pitch)
        frame.mvp_world:scale(math.tan(frame.hfov / 2), 1, math.tan(frame.vfov / 2))
        self.shaders.skybox:set_matrix('view', frame.mvp_world)
        frame.mvp_world:pop()

        frame.params_sky:set_stencil_ref(depth)
        sky.colors(self.shaders.skybox, cycle)
        self.shaders.skybox:draw(skybuf, frame.params_sky)
    end
    
    --If this portal cannot be rendered, this is a good point to stop drawing
    if depth >= 4 then
        return
    end
    if frame.portal_budget <= 0 then
        return
    end
    frame.portal_budget = frame.portal_budget - 1

    --Draw terrain
    do
        local cycle = self.day_cycle
        --ambience = 0
        --diffuse = 0.2
        --specular = 0.03
        --dx, dy, dz = 2^-0.5, 0, -2^-0.5
        set_clip_planes(self.shaders.terrain, cam)
        frame.params_world:set_stencil_ref(depth)
        self.shaders.terrain:set_float('fog', self.fog_current)
        self.shaders.terrain:set_matrix('invp', frame.invp_world)
        self.shaders.terrain:set_matrix('mvp', frame.mvp_world)
        sky.colors(self.shaders.terrain, cycle)
        sky.lighting(self.shaders.terrain, cycle)
        self.shaders.terrain:draw_terrain(self.terrain, frame.params_world, frame.mvp_world, cam, self.subdraw_bound)
    end
    
    --Draw entities
    local entcopies = frame.entcopy_buf
    frame.params_world:set_stencil_ref(depth)
    for _, ent in ipairs(self.entities) do
        local movx, movy, movz = ent.mov_x, ent.mov_y, ent.mov_z
        entpos_buf:copy_from(ent.pos)
        entpos_buf:move_box(self.terrain, movx * frame.s, movy * frame.s, movz * frame.s, 0.1, 0.1, 0.1) -- TODO: Replace with a raycast
        self.terrain:get_draw_positions(entpos_buf, ent.draw_r, ent.draw_r, ent.draw_r, cam, entcopies)
        for i = 1, #entcopies, 3 do
            local dx, dy, dz = entcopies[i], entcopies[i+1], entcopies[i+2]
            if cam:can_view(dx, dy, dz, ent.draw_r) then
                frame.mvp_world:push()
                frame.mvp_world:translate(dx, dy, dz)
                ent:draw(self)
                frame.mvp_world:pop()
            end
        end
    end

    if false and depth ~= 0 then
        local buf = Mesh{}
        buf:add_quad(
            x0, y0, z0,
            x1, y1, z1,
            x2, y2, z2,
            x3, y3, z3,
            0, 0, 0, 1
        )
        self.shaders.basic:set_matrix('mvp', frame.mvp_world)
        frame.params_hud:set_color_blend('add', 'src_alpha', 'one_minus_src_alpha')
        self.shaders.basic:set_vec4('tint', 1, 1, 1, 0.5)
        self.shaders.basic:draw(buf:as_buffer(), frame.params_hud)
    end

    --Draw portal itself out of the stencil buffer and into the depth buffer
    if depth > 0 then
        cam:geometry(portalbuf)
        self.shaders.portal:set_matrix('mvp', frame.mvp_world)
        self.shaders.portal:set_vec3('offset', cam:geometry_offset())
        set_clip_planes(self.shaders.portal, cam)
        frame.params_portalclose:set_stencil_ref(depth)
        self.shaders.portal:draw(portalbuf, frame.params_portalclose)
    end
end

local raw_origin = system.world_pos()
local last_atlas
function World:draw()
    local frame = self.frame
    local now = os.clock()
    gfx.clear(0, 0, 0, 0, 1, 0)

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
    frame.s = (now - self.next_tick) / self.tick_period
    frame.t = self.tick_count + (1 + frame.s)
    self.last_frame = now

    --Find out real camera location
    --TODO: Maybe center fog around the player instead of the camera
    local cam_yaw, cam_pitch
    do
        local movx, movy, movz = self.cam_mov_x * frame.s, self.cam_mov_y * frame.s, self.cam_mov_z * frame.s
        campos_buf:copy_from(self.cam_pos)
        campos_buf:move_box(self.terrain, movx, movy, movz, 0.1, 0.1, 0.1) --TODO: Replace with a raycast
        cam_yaw = self.cam_yaw
        cam_pitch = self.cam_pitch

        local cam_wall_dist = 0.4
        local rollback = self.cam_rollback
        local dx = -math.sin(self.cam_yaw) * math.cos(self.cam_pitch) * rollback
        local dy = math.cos(self.cam_yaw) * math.cos(self.cam_pitch) * rollback
        local dz = math.sin(self.cam_pitch) * rollback
        campos_buf:move_box(self.terrain, -dx, -dy, -dz, cam_wall_dist, cam_wall_dist, cam_wall_dist)
        frame.cam_yaw = cam_yaw
        frame.cam_pitch = cam_pitch
    end

    --Setup model-view-projection matrix for world drawing
    do
        local vfov = 1.1
        local hfov = vfov * frame.physical_w / frame.physical_h
        frame.hfov = hfov
        frame.vfov = vfov
        frame.mvp_world:reset()
        frame.mvp_world:perspective(vfov, frame.physical_w / frame.physical_h, 0.2, 800)
        frame.mvp_world:rotate_x(-cam_pitch)
        frame.mvp_world:rotate_z(-cam_yaw)
        frame.invp_world:reset_from(frame.mvp_world)
        frame.invp_world:invert()

        -- Update initial drawing conditions
        frame.cam_stack:reset(campos_buf, frame.mvp_world, nil, 0, 0, 0)
    end

    --Update fog distance
    do
        local now = os.clock()
        if now > self.fog_poll_next then
            self.fog_poll_next = now + (self.fog_poll_interval - now % self.fog_poll_interval)
            --Poll fog
            local fog_now = self.terrain:visible_radius()
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
    frame.portal_budget = 16
    self.terrain:reset_draw_stats()
    self:subdraw()

    --Draw HUD
    frame.mvp_hud:push()
    frame.mvp_hud:translate(-frame.w + 4, frame.h - 16, 0)
    frame.mvp_hud:scale(self.font_size)
    self.font:draw("FPS: "..self.fps, frame.mvp_hud, frame.params_hud, 1, 1, 1)
    if true then
        -- chunk gen/mesh/upload average times
        frame.mvp_hud:translate(0, -1.25, 0)
        self.font:draw("gen: "..util.format_time(self.terrain:get_stat("gentime")), frame.mvp_hud, frame.params_hud, 1, 1, 1)
        frame.mvp_hud:translate(0, -1.25, 0)
        self.font:draw("light: "..util.format_time(self.terrain:get_stat("lighttime")), frame.mvp_hud, frame.params_hud, 1, 1, 1)
        frame.mvp_hud:translate(0, -1.25, 0)
        self.font:draw("mesh: "..util.format_time(self.terrain:get_stat("meshtime")), frame.mvp_hud, frame.params_hud, 1, 1, 1)
        frame.mvp_hud:translate(0, -1.25, 0)
        self.font:draw("upload: "..util.format_time(self.terrain:get_stat("uploadtime")), frame.mvp_hud, frame.params_hud, 1, 1, 1)
    end
    if true then
        -- chunk draw stats
        frame.mvp_hud:translate(0, -1.25, 0)
        self.font:draw("drawnchunks: "..self.terrain:get_stat('drawnchunks'), frame.mvp_hud, frame.params_hud, 1, 1, 1)
        frame.mvp_hud:translate(0, -1.25, 0)
        self.font:draw("vertices: "..math.ceil(self.terrain:get_stat('vertbytes')/1024).."KB", frame.mvp_hud, frame.params_hud, 1, 1, 1)
        frame.mvp_hud:translate(0, -1.25, 0)
        self.font:draw("indices: "..math.ceil(self.terrain:get_stat('idxbytes')/1024).."KB", frame.mvp_hud, frame.params_hud, 1, 1, 1)
        frame.mvp_hud:translate(0, -1.25, 0)
        self.font:draw("color: "..math.ceil(self.terrain:get_stat('colorbytes')/1024).."KB", frame.mvp_hud, frame.params_hud, 1, 1, 1)
    end
    if true then
        -- absolute world coordinates
        frame.mvp_hud:translate(0, -1.25, 0)
        local raw_x, raw_y, raw_z, raw_w = self.cam_pos:raw_difference(raw_origin)
        self.font:draw("pos: "..math.floor(raw_x)..", "..math.floor(raw_y)..", "..math.floor(raw_z).." : "..math.floor(raw_w), frame.mvp_hud, frame.params_hud, 1, 1, 1)
        frame.mvp_hud:pop()
    end

    --Draw crosshair
    frame.mvp_hud:push()
    frame.mvp_hud:translate(0.5, 0.5, 0)
    frame.mvp_hud:scale(self.mouse_icon.w, self.mouse_icon.h, 1)
    self.mouse_icon:draw(1, frame.mvp_hud, frame.params_hud)
    frame.mvp_hud:pop()

    --DEBUG: Draw texture atlas of the current chunk
    if true then
        local atlas
        --atlas = self.terrain:atlas_at(self.cam_pos)
        if atlas then
            atlas:set_mag('nearest')
            last_atlas = atlas
        end
        if last_atlas then
            local buf = gfx.buffer_2d({
                0, 0,
                1, 0,
                1, 1,
                0, 1,
            }, {
                0, 0,
                1, 0,
                1, 1,
                0, 1,
            }, {
                0, 1, 2, 2, 3, 0
            })
            frame.mvp_hud:push()
            frame.mvp_hud:translate(-frame.w * 0.75, -frame.h * 0.75, 0)
            local tw, th = last_atlas:dimensions()
            local pixelsize = frame.h / 100
            frame.mvp_hud:scale(pixelsize * tw, pixelsize * th, 1)
            Sprite.shader:set_matrix('mvp', frame.mvp_hud)
            Sprite.shader:set_texture_2d('tex', last_atlas)
            Sprite.shader:set_vec4('tint', 1, 1, 1, 100)
            Sprite.shader:draw(buf, frame.params_hud)
            frame.mvp_hud:pop()
        end
    end
end

function World:keydown(key)
    if key == 'f7' then
        self:update_showchunkgrid(not self.showchunkgrid)
    end
end

function World:keyup(key)

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
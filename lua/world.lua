
local class = require 'class'
local util = require 'util'
local Mesh = require 'mesh'
local input = require 'input'

local World = class{}

function World:new()
    self.terrain = terrain.new{
        seed = 123443,
    }
    self.entities = {}

    self.shaders = {
        terrain = util.Shader{
            vertex = 'terrain.vert',
            fragment = 'terrain.frag',
            uniforms = {'mvp', 'offset'},
        },
        basic = util.Shader{
            vertex = 'basic.vert',
            fragment = 'basic.frag',
            uniforms = {'mvp', 'tint'},
        },
    }

    do
        local m = Mesh{}
        m:add_cube(0, 1, 0, 0.8, 2, 0.8, 0, 0.9, 0.4)
        self.cube_buf = m:as_buffer()
    end

    self.frame = {
        target_res = 240,
        physical_w = 1,
        physical_h = 1,
        w = 1,
        h = 1,
        params_world = gfx.draw_params(),
        params_hud = gfx.draw_params(),
        mvp_world = algebra.matrix(),
        mvp_hud = algebra.matrix(),
        s = 0,
    }
    self.frame.params_world:set_cull('cw')
    self.frame.params_world:set_depth('if_less', true)
    self.frame.params_world:set_color_blend('add', 'src_alpha', 'one_minus_src_alpha')
    self.frame.params_hud:set_cull('cw')
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

    self.tick_period = 1/100
    self.next_tick = os.clock()
end

function World:tick()
    --Tick entities
    for _, ent in ipairs(self.entities) do
        ent:tick(self)
    end

    --Bookkeep terrain
    self.terrain:book_keep(self.cam_x, self.cam_y, self.cam_z, 0.010)
end

function World:update()
    --Tick 0 or more times
    while true do
        local now = os.clock()
        if now < self.next_tick then
            break
        end
        self:tick()
        self.next_tick = self.next_tick + self.tick_period
    end
end

function World:draw()
    local frame = self.frame
    local now = os.clock()
    gfx.clear()

    --Figure out screen dimensions and pixelated scaling
    do
        local pw, ph = gfx.dimensions()
        frame.physical_w, frame.physical_h = pw, ph
        local scale = math.max(math.floor(math.min(pw, ph) / frame.target_res), 1)
        local w, h = math.floor(pw / scale / 2), math.floor(ph / scale / 2)
        frame.w, frame.h = w, h
        frame.mvp_hud:reset()
        frame.mvp_hud:scale(2*scale/pw, 2*scale/ph)
    end

    --Get interpolation factor `s`, a weight between the previous tick and the current one
    frame.s = (now - self.next_tick) / self.tick_period + 1

    --Find out real camera location
    local cam_x, cam_y, cam_z
    local cam_yaw, cam_pitch
    do
        local og_cam_x = self.cam_prev_x + (self.cam_x - self.cam_prev_x) * frame.s
        local og_cam_y = self.cam_prev_y + (self.cam_y - self.cam_prev_y) * frame.s
        local og_cam_z = self.cam_prev_z + (self.cam_z - self.cam_prev_z) * frame.s

        local cam_wall_dist = 1.2
        local rollback = self.cam_rollback
        local dx = math.sin(self.cam_yaw) * math.cos(self.cam_pitch) * rollback
        local dy = math.sin(self.cam_pitch) * rollback
        local dz = -math.cos(self.cam_yaw) * math.cos(self.cam_pitch) * rollback
        cam_x, cam_y, cam_z = self.terrain:collide(og_cam_x, og_cam_y, og_cam_z, -dx, -dy, -dz, 0, 0, 0)
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
    end

    --Setup model-view-projection matrix for world drawing
    frame.mvp_world:reset()
    frame.mvp_world:perspective(1.1, frame.physical_w / frame.physical_h, 0.1, 1000)
    frame.mvp_world:rotate_x(-cam_pitch)
    frame.mvp_world:rotate_y(-cam_yaw)

    --Draw terrain
    self.shaders.terrain:set_matrix('mvp', frame.mvp_world)
    self.shaders.terrain:draw_terrain(self.terrain, 'offset', frame.params_world, cam_x, cam_y, cam_z)
    
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
    
end

function World:mousemove(dx, dy)
    if input.mouse_down.right then
        self.cam_yaw = (self.cam_yaw + dx * 0.01) % (2*math.pi)
        self.cam_pitch = self.cam_pitch - dy * 0.01
        if self.cam_pitch < math.pi / -2 then
            self.cam_pitch = math.pi / -2
        elseif self.cam_pitch > math.pi / 2 then
            self.cam_pitch = math.pi / 2
        end
    end
end

return World
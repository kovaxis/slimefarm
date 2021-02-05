
local keys = require 'keys'
local mesh = require 'mesh'

local function read_file(path)
    local file = io.open('../'..path, 'rb')
    local str = file:read('a')
    file:close()
    return str
end

local shader = gfx.shader(read_file('shader/basic.vert'), read_file('shader/basic.frag'))

local m = mesh.new{}
m:add_cube(0, 0, 0, 1, 1, 1, 0.2, 0.2, 0.9)
for i in ipairs(m.colors) do
    m.colors[i] = math.random()
end
local buf = m:as_buffer()

local uni = gfx.uniforms()
uni:add('mvp')
uni:add('tint')

local mvp = algebra.matrix()

local keys_down = {}
local cam_x, cam_y, cam_z = 0, 0, 0
local cam_yaw, cam_pitch = 0, 0
local last_tick = os.clock()

local function tick(dt)
    local now = os.clock()
    local dt = now - last_tick
    last_tick = now

    --Move
    local dx, dy, dz = 0, 0, 0
    if keys_down.w then
        dz = dz - 1
    end
    if keys_down.s then
        dz = dz + 1
    end
    if keys_down.a then
        dx = dx - 1
    end
    if keys_down.d then
        dx = dx + 1
    end
    if keys_down.lshift then
        dy = dy - 1
    end
    if keys_down.space then
        dy = dy + 1
    end
    if dx ~= 0 and dz ~= 0 then
        dx = dx * 2^0.5
        dz = dz * 2^0.5
    end
    local speed = 2 * dt
    if dx ~= 0 or dz ~= 0 then
        --Move horizontally
        dx, dz = dx * math.cos(cam_yaw) - dz * math.sin(cam_yaw), dx * math.sin(cam_yaw) + dz * math.cos(cam_yaw)
        cam_x, cam_z = cam_x + dx * speed, cam_z + dz * speed
    end
    cam_y = cam_y + dy * speed
end

local function draw()
    local w, h = gfx.dimensions()
    gfx.clear()

    mvp:reset()
    mvp:perspective(1.1, w / h, 0.1, 1000)
    mvp:rotate_x(-cam_pitch)
    mvp:rotate_y(-cam_yaw)
    mvp:translate(-cam_x, -cam_y, -cam_z)
    
    mvp:push()
    mvp:translate(0, 0, -2)
    uni:set_matrix(0, mvp)
    uni:set_vec4(1, 0.1, 0.1, 1, 1)
    gfx.draw(buf, shader, uni)
    mvp:pop()

    gfx.finish()
end

while true do
    local ev, a, b, c, d, e, f = coroutine.yield()
    if ev == 'quit' then
        coroutine.yield(true)
    elseif ev == 'key' then
        local scancode, state = keys[a], b
        if scancode then
            if state then
                if scancode == 'escape' then
                    coroutine.yield(true)
                end
            end
            keys_down[scancode] = state
        end
    elseif ev == 'mousemove' then
        local dx, dy = a, b
        cam_yaw = (cam_yaw + dx * 0.01) % (2*math.pi)
        cam_pitch = cam_pitch - dy * 0.01
        if cam_pitch < math.pi / -2 then
            cam_pitch = math.pi / -2
        elseif cam_pitch > math.pi / 2 then
            cam_pitch = math.pi / 2
        end
    elseif ev == 'update' then
        --Tick
        tick()
        --Draw
        draw()
    end
end
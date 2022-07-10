
setmetatable(_G, {
    __index = function(g, name)
        error("attempt to access undefined global '"..tostring(name).."'", 2)
    end,
    __newindex = function(g, name, value)
        error("attempt to set undefined global '"..tostring(name).."'", 2)
    end,
})

local input = require 'input'
local World = require 'world'
local voxel = require 'voxel'

local world = World{}

local has_focus = true
while true do
    local ev, a, b, c, d, e, f = coroutine.yield()
    if ev == 'quit' then
        coroutine.yield(true)
    elseif ev == 'key' then
        if has_focus then
            local scancode, state = input.scancodes[a], b
            if scancode then
                if state ~= input.is_down[scancode] then
                    if state then
                        world:keydown(scancode)
                    else
                        world:keyup(scancode)
                    end
                end
                input.is_down[scancode] = state
                if state then
                    if scancode == 'escape' then
                        coroutine.yield(true)
                    elseif scancode == 'f11' then
                        --`true` to use exclusive mode
                        gfx.toggle_fullscreen(false)
                    end
                end
            end
        end
    elseif ev == 'mousemove' then
        if has_focus then
            local dx, dy = a, b
            world:mousemove(dx, dy)
        end
    elseif ev == 'click' then
        if has_focus then
            local button, state = input.mouse_buttons[a], b
            if button then
                input.is_down[button] = state
                if state then
                    world:keydown(button)
                else
                    world:keyup(button)
                end
            end
        end
    elseif ev == 'update' then
        world:update()
    elseif ev == 'focus' then
        has_focus = a
    end
end
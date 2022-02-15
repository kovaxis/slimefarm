
local input = require 'input'
local World = require 'world'
local Player = require 'player'
local Enemy = require 'enemy'

local world = World{}
table.insert(world.entities, Player{
    x = 0,
    y = 0,
    z = 100,
})
--[[for x = -10, 10 do
    table.insert(world.entities, Enemy{
        x = x * 10,
        y = 20,
        z = 100,
    })
end]]

local has_focus = true
while true do
    local ev, a, b, c, d, e, f = coroutine.yield()
    if ev == 'quit' then
        coroutine.yield(true)
    elseif ev == 'key' then
        if has_focus then
            local scancode, state = input.scancodes[a], b
            if scancode then
                input.key_down[scancode] = state
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
            local button, state = a, b
            local name = input.mouse_buttons[button]
            if name then
                input.mouse_down[name] = state
            end
            input.mouse_down[button] = state
        end
    elseif ev == 'update' then
        world:update()
    elseif ev == 'focus' then
        has_focus = a
    end
end
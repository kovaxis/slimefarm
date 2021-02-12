
local input = require 'input'
local World = require 'world'
local Player = require 'player'

local world = World{}
table.insert(world.entities, Player{
    x = 0,
    y = 0,
    z = 0,
})

while true do
    local ev, a, b, c, d, e, f = coroutine.yield()
    if ev == 'quit' then
        coroutine.yield(true)
    elseif ev == 'key' then
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
    elseif ev == 'mousemove' then
        local dx, dy = a, b
        world:mousemove(dx, dy)
    elseif ev == 'click' then
        local button, state = a, b
        local name = input.mouse_buttons[button]
        if name then
            input.mouse_down[name] = state
        end
        input.mouse_down[button] = state
    elseif ev == 'update' then
        --Update (0 or more ticks)
        world:update()
        --Draw
        world:draw()
        --Finish
        gfx.finish()
    end
end
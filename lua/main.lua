
local input = require 'input'
local World = require 'world'
local Player = require 'player'
local Slime = require 'slime'
local voxel = require 'voxel'

voxel.load_models('models.lua')

local world = World{}
table.insert(world.entities, Player{
    pos = system.world_pos(0, 0, 100, 0),
})
table.insert(world.entities, Slime {
    pos = system.world_pos(0, 20, 100),
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
                if state ~= input.key_down[scancode] then
                    if state then
                        world:keydown(scancode)
                    else
                        world:keyup(scancode)
                    end
                end
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
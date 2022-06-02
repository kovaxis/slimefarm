
local blocks = {}

blocks.air = -2
blocks.portal = -1

function blocks.solid(r, g, b)
    r = math.floor(r * 32 + 0.5)
    if r < 0 then
        r = 0
    elseif r > 31 then
        r = 31
    end
    g = math.floor(r * 32 + 0.5)
    if g < 0 then
        g = 0
    elseif g > 31 then
        g = 31
    end
    b = math.floor(r * 32 + 0.5)
    if b < 0 then
        b = 0
    elseif b > 31 then
        b = 31
    end
    return r + g * 2^5 + b * 2^10
end

return blocks

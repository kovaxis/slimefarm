
local native = require 'native'
local blocks = require 'blocks'

local voidgen = {}

function voidgen.generate(x, y, z, w)
    return native.chunk(blocks['base.air'])
end

return voidgen
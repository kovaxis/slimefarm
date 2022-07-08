-- Utilities for transforming key/value entity specs into raw binary strings according to specs
-- supplied by the main thread.

local spawn = {}

local unpacker = string.binpack(gen.k.entspecs)

function spawn.serialize(ent)
    return unpacker:pack(ent)
end

return spawn

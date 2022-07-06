-- Utilities for transforming key/value entity specs into raw binary strings according to specs
-- supplied by the main thread.

local spawn = {}

local name2id = gen.k.entspecs.name2id
local id2spec = gen.k.entspecs.id2spec

function spawn.serialize(ent)
    local typeid = name2id[ent.ty]
    if not typeid then
        error("unknown entity type '"..tostring(ent.ty).."'")
    end
    local fmt = id2spec[typeid]
    return string.binpack(fmt, {
        ty = typeid,
        ent = ent,
    })
end

return spawn

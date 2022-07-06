-- A registry for entities, to allow the worldgen thread to refer to entities by name instead of
-- by entspec id.
-- Ids are also automatically assigned.

local entreg = {}

local gen_id
do
    local next_id, id_stride
    local max_ids = 65536
    local rng = math.rng()
    next_id = rng:integer(max_ids)
    id_stride = rng:integer(max_ids)
    if id_stride % 2 == 0 then
        id_stride = (id_stride + 1) % max_ids
    end
    function gen_id()
        local id = next_id
        next_id = (next_id + id_stride) % max_ids
        return id
    end
end

entreg.by_id = {}
entreg.by_name = {}

function entreg.register(spec)
    assert(type(spec.name) == 'string', "spec.name must be a string")
    assert(entreg.by_name[spec.name] == nil, "duplicate entities with name '"..spec.name.."'")
    spec.id = gen_id()
    assert(type(spec.class) == 'table', "spec.class must be an entity class")
    assert(type(spec.fmt) == 'string', "spec.fmt must be a format string")

    entreg.by_id[spec.id] = spec
    entreg.by_name[spec.name] = spec
end

function entreg.get_specs()
    local specs = {
        name2id = {},
        id2spec = {},
    }
    for id, spec in pairs(entreg.by_id) do
        specs.name2id[spec.name] = id
        local fmt = '{ty=u2 ent='..spec.fmt..'}'
        specs.id2spec[id] = fmt
    end
    return specs
end

function entreg.instantiate(pos, raw)
    local lo, hi = raw:byte(1, 2)
    local id = lo + hi * 256
    local spec = entreg.by_id[id]
    assert(spec, "unknown entspec id "..id)

    local proto = string.binunpack(spec.fmt, raw:sub(3))
    local ent = spec.class:create(proto, pos)
    return ent
end

return entreg

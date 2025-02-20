-- A registry for entities, which assigns names to entities and allows the worldgen thread to refer
-- to entity kinds by name and automatically convert these names to a numerical id which is
-- converted back to a name and a class on the main thread side.

local entreg = {}

entreg.by_name = {}

local spec_blueprint = {enum = {}}
local spec_packer = nil

function entreg.register(spec)
    assert(spec_packer == nil, "cannot register entity specs after registry has been sealed")
    assert(type(spec.name) == 'string', "spec.name must be a string")
    assert(entreg.by_name[spec.name] == nil, "duplicate entities with name '"..spec.name.."'")
    assert(type(spec.class) == 'table', "spec.class must be an entity class")
    assert(type(spec.fmt) == 'table', "spec.fmt must be a format table")

    entreg.by_name[spec.name] = spec
    spec_blueprint.enum[spec.name] = {struct = spec.fmt}
end

function entreg.seal()
    spec_packer = string.binpack(spec_blueprint)
    return spec_blueprint
end

function entreg.instantiate(pos, data)
    local proto = spec_packer:unpack(data)
    local name, proto = proto[1], proto[2]
    local spec = entreg.by_name[name]
    local ent = spec.class:create(proto, pos)
    return ent
end

return entreg

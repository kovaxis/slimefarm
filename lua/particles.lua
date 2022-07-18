
local particles = {}

local next_id = 0
local list = {}
local by_name = {}

local default = {
    friction = 1,
    acc = {0, 0, 0},
    rot_friction = 1,
    rot_acc = 0,
    rot_axis = {0, 0, 0},
    sticky = true,
    shininess = .20,
}

local function get_id()
    local id = next_id
    next_id = next_id + 1
    return id
end

function particles.register(kind)
    assert(type(kind) == 'table', "kind must be a table")
    assert(type(kind.name) == 'string', "kind.name must be a name string")

    local id = by_name[kind.name]
    if not id then
        id = get_id()
    end
    assert(type(list[id + 1]) ~= 'table', "duplicate particle kinds with name '"..kind.name.."'")

    for k, v in pairs(default) do
        if kind[k] == nil then
            kind[k] = v
        end
    end

    kind.id = id
    by_name[kind.name] = id
    list[id + 1] = kind

    return id
end

function particles.lookup(name)
    local id = by_name[name]
    if id then
        return id
    end
    id = get_id()
    by_name[name] = id
    list[id + 1] = name
    return id
end

function particles.get_particles()
    for id = 0, next_id - 1 do
        if type(list[id + 1]) == 'string' then
            error("particle '"..list[id + 1].."' is never registered")
        end
    end

    return list
end

return particles

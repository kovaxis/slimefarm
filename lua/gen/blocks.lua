
local blockreg = {}

local max_ids = 256

local rng = math.rng()
local next_id = rng:integer(0, max_ids)
local total = 0
local advance = rng:integer(0, max_ids)
if advance % 2 == 0 then
    advance = advance + 1
end

local name2id = {}
local sealed = false

blockreg.textures = {}

function blockreg.lookup(name)
    if sealed then
        error("cannot register or lookup blocks at this moment, registry has already been sealed")
    end

    if name2id[name] then
        return name2id[name]
    end

    if not name:find('%.') then
        error("block names must be of the form <namespace>.<block>")
    end
    if total >= max_ids then
        error("ran out of all "..max_ids.." block ids!")
    end
    local id = next_id
    next_id = (next_id + advance) % max_ids
    name2id[name] = id
    return id
end

function blockreg.register(block)
    local id = blockreg.lookup(block.name)
    blockreg.textures[id] = block
    return id
end

function blockreg.seal()
    sealed = true
    for name, id in pairs(name2id) do
        blockreg[name] = id
        blockreg[id] = name
    end
end

return blockreg
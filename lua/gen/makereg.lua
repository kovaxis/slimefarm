
local makereg = {}

function makereg.make_registry(reg_name, assoc_name, max_ids)
    local reg = {}

    local rng = math.rng()
    local next_id = rng:integer(0, max_ids)
    local total = 0
    local advance = rng:integer(0, max_ids)
    if advance % 2 == 0 then
        advance = advance + 1
    end
    
    local name2id = {}
    local sealed = false
    
    reg[assoc_name] = {}
    
    function reg.lookup(name)
        if sealed then
            error("cannot register or lookup "..reg_name.." at this moment, registry has already been sealed")
        end
    
        if name2id[name] then
            return name2id[name]
        end
    
        if not name:find('%.') then
            error(reg_name.." names must be of the form <namespace>.<"..reg_name..">")
        end
        if total >= max_ids then
            error("ran out of all "..max_ids.." "..reg_name.." ids!")
        end
        local id = next_id
        next_id = (next_id + advance) % max_ids
        name2id[name] = id
        return id
    end
    
    function reg.register(data)
        local id = reg.lookup(data.name)
        reg[assoc_name][id] = data
        return id
    end
    
    function reg.seal()
        sealed = true
        for name, id in pairs(name2id) do
            reg[name] = id
            reg[id] = name
        end
    end

    return reg
end

return makereg


local reclaim = {}

reclaim.reclaimable = setmetatable({}, {
    __mode = 'kv',
})

function reclaim.track(data)
    reclaim.reclaimable[data] = true
end

function reclaim.wrap(a, b)
    if b == nil then
        assert(type(a) == 'function')
        return function(...)
            local x = a(...)
            reclaim.track(x)
            return x
        end
    else
        local c = a[b]
        assert(type(c) == 'function')
        a[b] = function(...)
            local x = c(...)
            reclaim.track(x)
            return x
        end
    end
end

function reclaim.reclaim()
    for data in pairs(reclaim.reclaimable) do
        data:gc()
    end
end

return reclaim

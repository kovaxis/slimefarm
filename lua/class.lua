
local class = {}

setmetatable(class, {
    __call = function(_, cl)
        return class.make(cl)
    end,
})

function class.make(cl)
    local instance_metatable = {
        __index = cl,
    }
    local class_metatable = {
        __call = function(cl, self, ...)
            setmetatable(self, instance_metatable)
            self:new(...)
            return self
        end,
    }
    setmetatable(cl, class_metatable)
    return cl
end

return class
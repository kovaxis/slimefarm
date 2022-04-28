
local util = require 'util'

local voxel = {}

voxel.mesher_cfg = {
    atlas_size = {64, 1024},
    -- clear, normal, blocked, blocked, base
    exposure_table = {65, 60, 45, 45, 0},
    light_uv_offset = 0.5,
}
voxel.model_mesher_cfg = {
    cfg = voxel.mesher_cfg,
    transparency = {0, 0, 0, 0},
    light_value = 240,
    lighting = {
        light = {
            base = 0,
            mul = 1,
            shr = 2,
        },
        decay = {
            base = 8,
            mul = 0,
            shr = 0,
        },
    },
}
voxel.mesher = gfx.mesher(voxel.model_mesher_cfg)

-- Performs the entire process from a path to a `.vox` file to a voxel buffer.
function voxel.dot_vox(path, idx)
    local raw = util.read_file(path)
    local data, sx, sy, sz = system.dot_vox(raw, idx)
    return voxel.mesher:mesh(data, sx, sy, sz)
end

return voxel
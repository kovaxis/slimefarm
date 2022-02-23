
local build_rust = true
local use_built = true
local require_build = true
local use_debug = true

local reclaim = require 'gen.reclaim'

if build_rust then
    local release
    if use_debug then
        release = ""
    else
        release = "--release"
    end
    local cmd = "cargo build -p worldgen "..release
    print("building native worldgen lib using command \""..cmd.."\"")
    local ok = os.execute(cmd)
    if not ok and require_build then
        error("rust failed to build correctly!")
    end
end

local path
if use_built then
    if use_debug then
        path = "../target/debug/worldgen"
    else
        path = "../target/release/worldgen"
    end
else
    path = "gen/worldgen"
end
local native = fs.open_lib(path)

reclaim.wrap(native, 'heightmap')
reclaim.wrap(native, 'gridbuf_2d')

return native

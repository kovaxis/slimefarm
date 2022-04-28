
TODO:

- Skylight occlusion.
- Object-centered lighting.
- Poll for events in a separate process and timestamp them for smoother input.
- Add an option to upload meshes in single thread, for stability.
- Fix `rlua 0.18` to print error messages from Rust callbacks.
    Check if `0.19` fixes this.
- Merge adjacent chunk meshes into supermeshes for an extra ~20 FPS.
    More important now that portals are a thing.
- Decorations using GPU instancing.
- Queries for the shortest path to the player or at least shortest distance with portals.
    A must for combat.


Maybes:
- Prevent discontinuities in squash animations by making it stateful.
- Special-case 2x and 4x noise scalers for big noise performance boost.
- SIMD for noise.
- Optimize gen and mesh priorities based on player velocity.
- Do something about ambient occlusion around portals.
- Do something about noise texturing around portals?
- Stitch atlas textures together.


Threads:

- Main thread
- Worldgen
- Mesher


Potential future threads:
- Worldkeeper (to allow worldgen to use 100% of its time generating while the worldkeep thread can
    block on the ChunkStorage lock).
- Multiple mesher threads (quite easy target).
- Decouple rendering/input from logic, creating a separate logic thread that sends render commands.
    Position interpolation with portals is a hard topic here.

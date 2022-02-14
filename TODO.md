
TODO:

- Skylight occlusion.
- Object-centered lighting.
- Set a chunkload and a view radius, unloading chunks and meshes beyond it.
- Prevent discontinuities in squash animations by making it stateful.
- Special-case 2x and 4x noise scalers for big noise performance boost.
- SIMD for noise.
- Optimize render priorities based on player velocity.
- Poll for events in a separate process and timestamp them for smoother input.
- Add an option to upload meshes in single thread, for stability.
- Fix `rlua 0.18` to print error messages from Rust callbacks.
- Merge adjacent chunk meshes into supermeshes for an extra ~20 FPS.
- Chunk recentering hysteresis.
- Decorations using GPU instancing.

Threads:

- Main thread
- Worldgen (x cores/2)
- Mesher
- Bookkeep

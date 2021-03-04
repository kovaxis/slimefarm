
- Mesh in a separate thread.
- Discard faraway chunks.
- Prevent discontinuities in squash animations by making it stateful.
- Special-case 2x and 4x noise scalers for big worldgen performance boost.
- Merge adjacent chunk meshes into supermeshes.
- SIMD for noise.
- Optimize render priorities based on player velocity.
- Poll for events in a separate process and timestamp them for smoother input.

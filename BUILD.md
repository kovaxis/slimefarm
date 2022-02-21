
# Notes on building, v1

When building for release, it is best to export the `AmdPowerXpressRequestHighPerformance` and
`NvOptimusEnablement` symbols from _the executable_.
Rust does not support this very well.
The current workaround, at least on Windows, is to enable the following `RUSTFLAGS` and then
_specify the target_.
The reason behind this is that when the target is specified Cargo passes the RUSTFLAGS only to
the final linking stage for some reason?
If the target is not specified the build scripts are also run with these linker arguments, and
since they do not provide the necessary symbols linking will fail with "unresolved symbol".

Note: "-C link-args=-Wl,-export-dynamic", "-C link-args=-rdynamic" and
"-C link-args=-Wl,--dynamic-list=dynsyms.txt" DID NOT WORK, no matter how
hard I tried to use the static variables so the linker would not strip them out.
They need to be specified explicitly, at least on Windows.

The best command to execute on Windows and get the symbols exported is:

```
CARGO_ENCODED_RUSTFLAGS=$'-C\037link-args=/EXPORT:AmdPowerXpressRequestHighPerformance /EXPORT:NvOptimusEnablement' cargo build --release --target x86_64-pc-windows-msvc
```

# Notes on building, v2

Currently, setting RUSTFLAGS is such a pain (because it causes miscompilations), that everything is
being done to avoid dynamic linking at binary load time.
This means that normal `cargo run` and `cargo build` can be used `:D`.

Note that although Lua may try to build `worldgen` automatically, the `commonmem` shared library
still needs to be built manually before building the main game.

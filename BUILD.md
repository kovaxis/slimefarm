
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

Currently, setting RUSTFLAGS is such a pain (because it causes recompilations), that everything is
being done to avoid dynamic linking at binary load time.
This means that normal `cargo run` and `cargo build` can be used `:D`.

For archiving reasons, I learned that:
- Dynamic linking with `extern` blocks is a pain, but it is not _too_ bad.
- The rust compiler needs to be able to find a `<libname>.lib` file that promises it that the
    symbols will actually be there at runtime. Doing this requires a `-l <libname>` rustflag.
- This means that the `.lib` file must be somewhere that the linker searches for.
- Even if the `.lib` file is placed somewhere visible for the main artifact, build scripts might
    not see it. This leads to errors if the target is not specified (again, why are these two
    orthogonal options linked???)
- Every little change to RUSTFLAGS requires an entire project rebuild. At this point the game
    takes at least a minute to full-recompile, so it's not something to take lightly.
- Conclusion: just do true dynamic linking and avoid these issues.

Therefore, the solution is to create one extra dynamic library, which holds all statics that need
to be shared between the different modules.

Note that although Lua may try to build `worldgen` automatically, the `commonmem` shared library
still needs to be built manually before building the main game.

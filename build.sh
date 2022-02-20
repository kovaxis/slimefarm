#!/bin/bash

# When building for release, it is best to export the `AmdPowerXpressRequestHighPerformance` and
# `NvOptimusEnablement` symbols from _the executable_.
# Rust does not support this very well.
# The current workaround, at least on Windows, is to enable the following `RUSTFLAGS` and then
# _specify the target_.
# The reason behind this is that when the target is specified Cargo passes the RUSTFLAGS only to
# the final linking stage for some reason?
# If the target is not specified the build scripts are also run with these linker arguments, and
# since they do not provide the necessary symbols linking will fail with "unresolved symbol".

# Note: "-C link-args=-Wl,-export-dynamic", "-C link-args=-rdynamic" and
# "-C link-args=-Wl,--dynamic-list=dynsyms.txt" DID NOT WORK, no matter how
# hard I tried to use the static variables so the linker would not strip them out.
# They need to be specified explicitly, at least on Windows.

CARGO_ENCODED_RUSTFLAGS=$'-C\037link-args=/EXPORT:AmdPowerXpressRequestHighPerformance /EXPORT:NvOptimusEnablement' cargo build --release --target x86_64-pc-windows-msvc

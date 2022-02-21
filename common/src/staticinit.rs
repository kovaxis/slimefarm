//! Initialize shared static functions and variables from the `commonmem` shared library.

use crate::prelude::*;
use libloading::{Library, Symbol};

macro_rules! load_statics {
    ($lib:expr => $($path:path = $name:literal;)*) => {{
        $({
            let name = $name;
            assert_eq!(name.last().unwrap(), &0u8);
            $path = match $lib.get($name) {
                Ok(sym) => *sym,
                Err(e) => panic!("failed to get shared symbol \"{}\": {}", String::from_utf8_lossy(&name[..name.len() - 1]), e),
            };
        })*
    }};
}

pub unsafe fn static_init() {
    let path = libloading::library_filename("commonmem");
    let lib = match Library::new(&path) {
        Ok(lib) => lib,
        Err(e) => {
            eprintln!("failed to load dynamic library `commonmem`: {}", e);
            eprintln!(
                "place `{}` next to the executable and try again",
                path.to_string_lossy()
            );
            std::process::exit(1);
        }
    };
    load_statics! {lib =>
        crate::arena::ARENA_ALLOC = b"game_arena_alloc\0";
        crate::arena::ARENA_DEALLOC = b"game_arena_dealloc\0";
    }
    mem::forget(lib);
}

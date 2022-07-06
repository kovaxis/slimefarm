use crate::prelude::*;
use common::{lua::LuaValueStatic, lua_assert, lua_bail, lua_func, lua_lib, lua_type};

/// Skip over whitespace.
/// (Also ',' and '=')
fn trim_space(s: &mut &[u8]) {
    loop {
        match s.first() {
            Some(&c) => match c {
                b' ' | b',' | b'=' => *s = &s[1..],
                _ => break,
            },
            None => break,
        }
    }
}

/// Get the next non-whitespace byte.
fn peek_byte(s: &mut &[u8]) -> Option<u8> {
    trim_space(s);
    s.first().copied()
}

/// Read a single ident, erroring if it can't be found.
fn read_ident<'a>(s: &mut &'a [u8]) -> LuaResult<&'a [u8]> {
    trim_space(s);
    lua_assert!(
        s.first()
            .map(|&b| b == b'_' || b.is_ascii_alphabetic())
            .unwrap_or(false),
        "expected ident"
    );
    let end = s
        .iter()
        .position(|&b| b != b'_' && !b.is_ascii_alphanumeric())
        .unwrap_or(s.len());
    let ident = &s[..end];
    *s = &s[end..];
    Ok(ident)
}

/// Read a single byte, erroring if it can't be found.
fn expect_byte(s: &mut &[u8]) -> LuaResult<u8> {
    trim_space(s);
    lua_assert!(!s.is_empty(), "unexpected end of format string");
    let b = s[0];
    *s = &s[1..];
    Ok(b)
}

/// Read a compile-time amount of bytes, erroring if they can't be found.
fn expect_bytes<T>(s: &mut &[u8]) -> LuaResult<T>
where
    T: Default + AsMut<[u8]>,
{
    let mut buf = T::default();
    let len = buf.as_mut().len();
    lua_assert!(s.len() >= len, "unexpected end of string");
    buf.as_mut().copy_from_slice(&s[..len]);
    *s = &s[len..];
    Ok(buf)
}

/// Skip over a single formatting element, skipping over any nested elements recursively.
fn skip_fmt(fmt: &mut &[u8]) -> LuaResult<()> {
    let elem = expect_byte(fmt)?;
    match elem {
        b'b' => {}
        b'u' | b'i' | b'f' | b's' => {
            expect_byte(fmt)?;
        }
        b'{' | b'(' => loop {
            let close = match elem {
                b'{' => b'}',
                b'(' => b')',
                _ => unreachable!(),
            };
            if peek_byte(fmt) == Some(close) {
                *fmt = &fmt[1..];
                break;
            }
            read_ident(fmt)?;
            skip_fmt(fmt)?;
        },
        b => lua_bail!("unknown element '{}'", b as char),
    }
    Ok(())
}

fn inner_pack<'lua>(
    lua: LuaContext<'lua>,
    fmt: &mut &[u8],
    val: LuaValue<'lua>,
    out: &mut Vec<u8>,
) -> LuaResult<()> {
    match expect_byte(fmt)? {
        b'b' => {
            let b = bool::from_lua(val, lua)?;
            out.push(b as u8);
        }
        b'u' => match expect_byte(fmt)? {
            b'1' => {
                out.push(u8::from_lua(val, lua)? as u8);
            }
            b'2' => {
                out.extend_from_slice(&u16::from_lua(val, lua)?.to_le_bytes());
            }
            b'4' => {
                out.extend_from_slice(&u32::from_lua(val, lua)?.to_le_bytes());
            }
            b'8' => {
                out.extend_from_slice(&u64::from_lua(val, lua)?.to_le_bytes());
            }
            l => lua_bail!("unexpected unsigned integer length '{}'", l as char),
        },
        b'i' => match expect_byte(fmt)? {
            b'1' => {
                out.push(i8::from_lua(val, lua)? as u8);
            }
            b'2' => {
                out.extend_from_slice(&i16::from_lua(val, lua)?.to_le_bytes());
            }
            b'4' => {
                out.extend_from_slice(&i32::from_lua(val, lua)?.to_le_bytes());
            }
            b'8' => {
                out.extend_from_slice(&i64::from_lua(val, lua)?.to_le_bytes());
            }
            l => lua_bail!("unexpected signed integer length '{}'", l as char),
        },
        b'f' => match expect_byte(fmt)? {
            b'4' => {
                out.extend_from_slice(&f32::from_lua(val, lua)?.to_bits().to_le_bytes());
            }
            b'8' => {
                out.extend_from_slice(&f64::from_lua(val, lua)?.to_bits().to_le_bytes());
            }
            l => lua_bail!("unexpected float length '{}'", l as char),
        },
        b's' => {
            let s = LuaString::from_lua(val, lua)?;
            let s = s.as_bytes();
            match expect_byte(fmt)? {
                b'1' => {
                    out.push(
                        u8::try_from(s.len())
                            .map_err(|_| "cannot fit length of string in u8")
                            .to_lua_err()?,
                    );
                }
                b'2' => {
                    out.extend_from_slice(
                        &u16::try_from(s.len())
                            .map_err(|_| "cannot fit length of string in u16")
                            .to_lua_err()?
                            .to_le_bytes(),
                    );
                }
                b'4' => {
                    out.extend_from_slice(
                        &u32::try_from(s.len())
                            .map_err(|_| "cannot fit length of string in u32")
                            .to_lua_err()?
                            .to_le_bytes(),
                    );
                }
                b'8' => {
                    out.extend_from_slice(
                        &u64::try_from(s.len())
                            .map_err(|_| "cannot fit length of string in u64")
                            .to_lua_err()?
                            .to_le_bytes(),
                    );
                }
                l => lua_bail!("unexpected string length encoding length '{}'", l as char),
            }
            out.extend_from_slice(s);
        }
        b'{' => {
            let tab = LuaTable::from_lua(val, lua)?;
            loop {
                if let Some(b'}') = peek_byte(fmt) {
                    // Close table
                    *fmt = &fmt[1..];
                    break;
                }
                let ident = read_ident(fmt)?;
                let ident = lua.create_string(ident)?;
                inner_pack(lua, fmt, tab.raw_get(ident)?, out)?;
            }
        }
        b'(' => {
            let tab = LuaTable::from_lua(val, lua)?;
            let mut found = false;
            let mut idx = 0;
            loop {
                if let Some(b')') = peek_byte(fmt) {
                    // Close table
                    *fmt = &fmt[1..];
                    break;
                }
                let ident = read_ident(fmt)?;
                if found {
                    skip_fmt(fmt)?;
                } else {
                    let ident = lua.create_string(ident)?;
                    let inner = tab.raw_get(ident)?;
                    match inner {
                        LuaValue::Nil => skip_fmt(fmt)?,
                        inner => {
                            out.push(idx);
                            inner_pack(lua, fmt, inner, out)?;
                            found = true;
                        }
                    }
                }
                idx = idx
                    .checked_add(1)
                    .ok_or("enum with over 255 variants")
                    .to_lua_err()?;
            }
            lua_assert!(found, "unknown enum variant name");
        }
        b => lua_bail!("unknown element '{}'", b as char),
    }
    Ok(())
}

pub(crate) fn pack<'lua>(
    lua: LuaContext<'lua>,
    mut fmt: &[u8],
    val: LuaValue<'lua>,
    out: &mut Vec<u8>,
) -> LuaResult<()> {
    inner_pack(lua, &mut fmt, val, out)?;
    trim_space(&mut fmt);
    lua_assert!(fmt.is_empty(), "trailing characters in format string");
    Ok(())
}

fn inner_unpack<'lua>(
    lua: LuaContext<'lua>,
    fmt: &mut &[u8],
    bin: &mut &[u8],
) -> LuaResult<LuaValue<'lua>> {
    Ok(match expect_byte(fmt)? {
        b'b' => {
            let b = expect_byte(bin)?;
            lua_assert!(b < 2, "invalid boolean byte");
            LuaValue::Boolean(b != 0)
        }
        b'u' => LuaValue::Integer(match expect_byte(fmt)? {
            b'1' => expect_byte(bin)? as i64,
            b'2' => u16::from_le_bytes(expect_bytes(bin)?) as i64,
            b'4' => u32::from_le_bytes(expect_bytes(bin)?) as i64,
            b'8' => u64::from_le_bytes(expect_bytes(bin)?) as i64,
            l => lua_bail!("unexpected unsigned integer length '{}'", l as char),
        }),
        b'i' => LuaValue::Integer(match expect_byte(fmt)? {
            b'1' => expect_byte(bin)? as i8 as i64,
            b'2' => i16::from_le_bytes(expect_bytes(bin)?) as i64,
            b'4' => i32::from_le_bytes(expect_bytes(bin)?) as i64,
            b'8' => i64::from_le_bytes(expect_bytes(bin)?) as i64,
            l => lua_bail!("unexpected unsigned integer length '{}'", l as char),
        }),
        b'f' => LuaValue::Number(match expect_byte(fmt)? {
            b'4' => f32::from_bits(u32::from_le_bytes(expect_bytes(bin)?)) as f64,
            b'8' => f64::from_bits(u64::from_le_bytes(expect_bytes(bin)?)) as f64,
            l => lua_bail!("unexpected float length '{}'", l as char),
        }),
        b's' => {
            let len = match expect_byte(fmt)? {
                b'1' => expect_byte(bin)? as usize,
                b'2' => u16::from_le_bytes(expect_bytes(bin)?) as usize,
                b'4' => u32::from_le_bytes(expect_bytes(bin)?) as usize,
                b'8' => u64::from_le_bytes(expect_bytes(bin)?) as usize,
                l => lua_bail!("unexpected unsigned integer length '{}'", l as char),
            };
            lua_assert!(bin.len() >= len, "truncated string");
            let s = lua.create_string(&bin[..len])?;
            *bin = &bin[len..];
            LuaValue::String(s)
        }
        b'{' => {
            let tab = lua.create_table()?;
            loop {
                if let Some(b'}') = peek_byte(fmt) {
                    // Close table
                    *fmt = &fmt[1..];
                    break;
                }
                let ident = read_ident(fmt)?;
                let ident = lua.create_string(ident)?;
                let val = inner_unpack(lua, fmt, bin)?;
                tab.raw_set(ident, val)?;
            }
            LuaValue::Table(tab)
        }
        b'(' => {
            let enc_idx = expect_byte(bin)?;
            let mut idx = 0;
            let mut out = None;
            loop {
                if let Some(b')') = peek_byte(fmt) {
                    // Close table
                    *fmt = &fmt[1..];
                    break;
                }
                if enc_idx == idx {
                    let ident = read_ident(fmt)?;
                    let ident = lua.create_string(ident)?;
                    let inner = inner_unpack(lua, fmt, bin)?;
                    let tab = lua.create_table()?;
                    tab.raw_set(ident, inner)?;
                    out = Some(tab);
                } else {
                    skip_fmt(fmt)?;
                }
                idx = idx
                    .checked_add(1)
                    .ok_or("enum with over 255 variants")
                    .to_lua_err()?;
            }
            let out = out.ok_or("invalid enum discriminant").to_lua_err()?;
            LuaValue::Table(out)
        }
        b => lua_bail!("unknown element '{}'", b as char),
    })
}

pub(crate) fn unpack<'lua>(
    lua: LuaContext<'lua>,
    mut fmt: &[u8],
    mut bin: &[u8],
) -> LuaResult<LuaValue<'lua>> {
    let val = inner_unpack(lua, &mut fmt, &mut bin)?;
    trim_space(&mut fmt);
    lua_assert!(fmt.is_empty(), "trailing characters in format string");
    lua_assert!(
        bin.is_empty(),
        "trailing characters in binary string: '{:?}'",
        bin
    );
    Ok(val)
}

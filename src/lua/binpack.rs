use crate::prelude::*;
use common::{lua::LuaValueStatic, lua_assert, lua_bail, lua_func, lua_lib, lua_type};

#[derive(Serialize, Deserialize)]
#[allow(non_camel_case_types)]
enum BinpackFmt {
    bool,
    u8,
    u16,
    u32,
    u64,
    i8,
    i16,
    i32,
    i64,
    s8,
    s16,
    s32,
    s64,
    f32,
    f64,
    opt(Box<BinpackFmt>),
    r#struct(HashMap<String, BinpackFmt>),
    r#enum(HashMap<String, BinpackFmt>),
}
impl BinpackFmt {
    fn transform(self) -> Binpack {
        use self::{Binpack::*, BinpackFmt::*, Intsize::*};
        match self {
            bool => Bool,
            u8 => Uint(I8),
            u16 => Uint(I16),
            u32 => Uint(I32),
            u64 => Uint(I64),
            i8 => Int(I8),
            i16 => Int(I16),
            i32 => Int(I32),
            i64 => Int(I64),
            s8 => Str(I8),
            s16 => Str(I16),
            s32 => Str(I32),
            s64 => Str(I64),
            f32 => F32,
            f64 => F64,
            opt(v) => Option(Box::new(v.transform())),
            r#struct(fs) => {
                let mut must = Vec::with_capacity(fs.len());
                let mut may = Vec::with_capacity(fs.len());
                for (name, f) in fs {
                    match f {
                        opt(f) => may.push((name.into_bytes(), f.transform())),
                        f => must.push((name.into_bytes(), f.transform())),
                    }
                }
                Struct { must, may }
            }
            r#enum(vs) => {
                let dense = vs
                    .into_iter()
                    .map(|(name, v)| (name.into_bytes(), v.transform()))
                    .collect::<Vec<_>>();
                let sparse = dense
                    .iter()
                    .enumerate()
                    .map(|(i, (n, _v))| (n.clone(), i))
                    .collect();
                Enum { dense, sparse }
            }
        }
    }
}

#[derive(Copy, Clone)]
pub enum Intsize {
    I8,
    I16,
    I32,
    I64,
}
impl Intsize {
    fn to_fit(x: usize) -> Self {
        use self::Intsize::*;
        if x <= u8::MAX as usize {
            I8
        } else if x <= u16::MAX as usize {
            I16
        } else if x <= u32::MAX as usize {
            I32
        } else {
            I64
        }
    }

    fn read(self, bin: &mut &[u8]) -> LuaResult<usize> {
        use self::Intsize::*;
        Ok(match self {
            I8 => expect_byte(bin)? as usize,
            I16 => u16::from_le_bytes(expect_bytes(bin)?) as usize,
            I32 => u32::from_le_bytes(expect_bytes(bin)?) as usize,
            I64 => u64::from_le_bytes(expect_bytes(bin)?) as usize,
        })
    }

    fn write(self, x: usize, out: &mut Vec<u8>) {
        use self::Intsize::*;
        match self {
            I8 => out.push(x as u8),
            I16 => out.extend_from_slice(&(x as u16).to_le_bytes()),
            I32 => out.extend_from_slice(&(x as u32).to_le_bytes()),
            I64 => out.extend_from_slice(&(x as u64).to_le_bytes()),
        }
    }
}

/// Read a single byte, erroring if it can't be found.
fn expect_byte(s: &mut &[u8]) -> LuaResult<u8> {
    lua_assert!(!s.is_empty(), "unexpected eof");
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
    lua_assert!(s.len() >= len, "unexpected eof");
    buf.as_mut().copy_from_slice(&s[..len]);
    *s = &s[len..];
    Ok(buf)
}

pub enum Binpack {
    Bool,
    Uint(Intsize),
    Int(Intsize),
    Str(Intsize),
    F32,
    F64,
    Option(Box<Binpack>),
    Struct {
        must: Vec<(Vec<u8>, Binpack)>,
        may: Vec<(Vec<u8>, Binpack)>,
    },
    Enum {
        dense: Vec<(Vec<u8>, Binpack)>,
        sparse: HashMap<Vec<u8>, usize>,
    },
}

lua_type! {Binpack, lua, this,
    fn pack(val: LuaValue) {
        thread_local! {
            static BUF: Cell<Vec<u8>> = Cell::new(Vec::new());
        }
        BUF.with(|buf| -> LuaResult<LuaString> {
            let mut out = buf.take();
            out.clear();
            this.pack(lua, val, &mut out)?;
            let bin = lua.create_string(&out[..])?;
            buf.replace(out);
            Ok(bin)
        })?
    }

    fn unpack(bin: LuaString) {
        this.unpack(lua, &mut bin.as_bytes())?
    }
}

impl Binpack {
    pub fn create(fmt: LuaValue) -> LuaResult<Self> {
        let proto: BinpackFmt = rlua_serde::from_value(fmt)?;
        Ok(proto.transform())
    }

    pub fn pack<'lua>(
        &self,
        lua: LuaContext<'lua>,
        val: LuaValue<'lua>,
        out: &mut Vec<u8>,
    ) -> LuaResult<()> {
        use self::{Binpack::*, Intsize::*};
        match self {
            Bool => out.push(bool::from_lua(val, lua)? as u8),
            Uint(sz) => match sz {
                I8 => out.push(u8::from_lua(val, lua)?),
                I16 => out.extend_from_slice(&u16::from_lua(val, lua)?.to_le_bytes()),
                I32 => out.extend_from_slice(&u32::from_lua(val, lua)?.to_le_bytes()),
                I64 => out.extend_from_slice(&u64::from_lua(val, lua)?.to_le_bytes()),
            },
            Int(sz) => match sz {
                I8 => out.push(i8::from_lua(val, lua)? as u8),
                I16 => out.extend_from_slice(&i16::from_lua(val, lua)?.to_le_bytes()),
                I32 => out.extend_from_slice(&i32::from_lua(val, lua)?.to_le_bytes()),
                I64 => out.extend_from_slice(&i64::from_lua(val, lua)?.to_le_bytes()),
            },
            Str(sz) => {
                let s = LuaString::from_lua(val, lua)?;
                let s = s.as_bytes();
                macro_rules! e {
                    ($t:tt) => {
                        out.extend_from_slice(
                            &$t::try_from(s.len())
                                .map_err(|_| "string is too long")
                                .to_lua_err()?
                                .to_le_bytes(),
                        )
                    };
                }
                match sz {
                    I8 => e!(u8),
                    I16 => e!(u16),
                    I32 => e!(u32),
                    I64 => e!(u64),
                }
            }
            F32 => out.extend_from_slice(&f32::from_lua(val, lua)?.to_bits().to_le_bytes()),
            F64 => out.extend_from_slice(&f64::from_lua(val, lua)?.to_bits().to_le_bytes()),
            Option(fmt) => match val {
                LuaValue::Nil => out.push(0u8),
                val => {
                    out.push(1u8);
                    fmt.pack(lua, val, out)?;
                }
            },
            Struct { must, may } => {
                let tab = LuaTable::from_lua(val, lua)?;
                for (name, fmt) in must {
                    let subval = tab.raw_get(lua.create_string(name)?)?;
                    fmt.pack(lua, subval, out)?;
                }
                if !may.is_empty() {
                    let int = Intsize::to_fit(may.len());
                    for (idx, (name, fmt)) in may.iter().enumerate() {
                        let subval = tab.raw_get(lua.create_string(name)?)?;
                        match subval {
                            LuaValue::Nil => {}
                            subval => {
                                int.write(idx, out);
                                fmt.pack(lua, subval, out)?;
                            }
                        }
                    }
                    int.write(may.len(), out);
                }
            }
            Enum { dense, sparse } => {
                let tab = LuaTable::from_lua(val, lua)?;
                let name = tab.raw_get::<_, LuaString>(1)?;
                let subval = tab.raw_get(2)?;
                let id = *sparse
                    .get(name.as_bytes())
                    .ok_or("unknown variant name")
                    .to_lua_err()?;
                let (_, fmt) = &dense[id];
                let int = Intsize::to_fit(dense.len() - 1);
                int.write(id, out);
                fmt.pack(lua, subval, out)?;
            }
        }
        Ok(())
    }

    pub fn unpack<'lua>(
        &self,
        lua: LuaContext<'lua>,
        bin: &mut &[u8],
    ) -> LuaResult<LuaValue<'lua>> {
        use self::{Binpack::*, Intsize::*};
        Ok(match self {
            Bool => LuaValue::Boolean(match expect_byte(bin)? {
                0 => false,
                1 => true,
                _ => lua_bail!("invalid boolean value"),
            }),
            Uint(sz) => LuaValue::Integer(match sz {
                I8 => u8::from_le_bytes(expect_bytes(bin)?) as i64,
                I16 => u16::from_le_bytes(expect_bytes(bin)?) as i64,
                I32 => u32::from_le_bytes(expect_bytes(bin)?) as i64,
                I64 => u64::from_le_bytes(expect_bytes(bin)?) as i64,
            }),
            Int(sz) => LuaValue::Integer(match sz {
                I8 => i8::from_le_bytes(expect_bytes(bin)?) as i64,
                I16 => i16::from_le_bytes(expect_bytes(bin)?) as i64,
                I32 => i32::from_le_bytes(expect_bytes(bin)?) as i64,
                I64 => i64::from_le_bytes(expect_bytes(bin)?) as i64,
            }),
            Str(sz) => {
                let len = match sz {
                    I8 => u8::from_le_bytes(expect_bytes(bin)?) as usize,
                    I16 => u16::from_le_bytes(expect_bytes(bin)?) as usize,
                    I32 => u32::from_le_bytes(expect_bytes(bin)?) as usize,
                    I64 => u64::from_le_bytes(expect_bytes(bin)?) as usize,
                };
                lua_assert!(bin.len() >= len, "truncated string");
                let s = &bin[..len];
                *bin = &bin[len..];
                LuaValue::String(lua.create_string(s)?)
            }
            F32 => LuaValue::Number(f32::from_bits(u32::from_le_bytes(expect_bytes(bin)?)) as f64),
            F64 => LuaValue::Number(f64::from_bits(u64::from_le_bytes(expect_bytes(bin)?))),
            Option(fmt) => match expect_byte(bin)? {
                0 => LuaValue::Nil,
                1 => fmt.unpack(lua, bin)?,
                _ => lua_bail!("invalid option tag"),
            },
            Struct { must, may } => {
                let tab = lua.create_table()?;
                for (name, fmt) in must {
                    let name = lua.create_string(name)?;
                    let subval = fmt.unpack(lua, bin)?;
                    tab.raw_set(name, subval)?;
                }
                if !may.is_empty() {
                    let int = Intsize::to_fit(may.len());
                    loop {
                        let tag = int.read(bin)?;
                        lua_assert!(tag <= may.len(), "invalid struct field tag");
                        if tag == may.len() {
                            break;
                        }
                        let (name, fmt) = &may[tag];
                        let name = lua.create_string(name)?;
                        let subval = fmt.unpack(lua, bin)?;
                        tab.raw_set(name, subval)?;
                    }
                }
                LuaValue::Table(tab)
            }
            Enum { dense, sparse: _ } => {
                let int = Intsize::to_fit(dense.len() - 1);
                let tag = int.read(bin)?;
                lua_assert!(tag < dense.len(), "invalid enum variant tag");
                let (name, fmt) = &dense[tag];
                let name = lua.create_string(name)?;
                let subval = fmt.unpack(lua, bin)?;
                [LuaValue::String(name), subval].to_lua(lua)?
            }
        })
    }
}

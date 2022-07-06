use crate::prelude::*;

#[macro_export]
macro_rules! lua_bail {
    ($err:expr, $($rest:tt)*) => {{
        return Err(format!($err, $($rest)*)).to_lua_err();
    }};
    ($err:expr) => {{
        return Err($err).to_lua_err();
    }};
}

#[macro_export]
macro_rules! lua_assert {
    ($cond:expr, $($err:tt)*) => {{
        if !$cond {
            lua_bail!($($err)*);
        }
    }};
}

#[macro_export]
macro_rules! lua_type {
    (@$m:ident $lua:ident $this:ident fn $fn_name:ident () $fn_code:block $($rest:tt)*) => {{
        $m.add_method(stringify!($fn_name), |lua, this, ()| {
            #[allow(unused_variables)]
            let $lua = lua;
            #[allow(unused_variables)]
            let $this = this;
            Ok($fn_code)
        });
        $crate::lua_type!(@$m $lua $this $($rest)*);
    }};
    (@$m:ident $lua:ident $this:ident fn $fn_name:ident ($($args:tt)*) $fn_code:block $($rest:tt)*) => {{
        $m.add_method(stringify!($fn_name), |lua, this, $($args)*| {
            #[allow(unused_variables)]
            let $lua = lua;
            #[allow(unused_variables)]
            let $this = this;
            Ok($fn_code)
        });
        $crate::lua_type!(@$m $lua $this $($rest)*);
    }};
    (@$m:ident $lua:ident $this:ident mut fn $fn_name:ident () $fn_code:block $($rest:tt)*) => {{
        $m.add_method_mut(stringify!($fn_name), |lua, this, ()| {
            #[allow(unused_variables)]
            let $lua = lua;
            #[allow(unused_variables)]
            let $this = this;
            Ok($fn_code)
        });
        $crate::lua_type!(@$m $lua $this $($rest)*);
    }};
    (@$m:ident $lua:ident $this:ident mut fn $fn_name:ident ($($args:tt)*) $fn_code:block $($rest:tt)*) => {{
        $m.add_method_mut(stringify!($fn_name), |lua, this, $($args)*| {
            #[allow(unused_variables)]
            let $lua = lua;
            #[allow(unused_variables)]
            let $this = this;
            Ok($fn_code)
        });
        $crate::lua_type!(@$m $lua $this $($rest)*);
    }};
    (@$m:ident $lua:ident $this:ident) => {};
    ($ty:ty, $lua:ident, $this:ident, $($rest:tt)*) => {
        impl LuaUserData for $ty {
            fn add_methods<'lua, M: LuaUserDataMethods<'lua, Self>>(m: &mut M) {
                $crate::lua_type!(@m $lua $this $($rest)*);
            }
        }
    };
}

#[macro_export]
macro_rules! lua_func {
    ($lua:ident, $state:ident, fn($($fn_args:tt)*) $fn_code:block) => {{
        let state = $state.clone();
        $lua.create_function(move |ctx, $($fn_args)*| {
            #[allow(unused_variables)]
            let $lua = ctx;
            #[allow(unused_variables)]
            let $state = &state;
            Ok($fn_code)
        }).unwrap()
    }};
}

#[macro_export]
macro_rules! lua_lib {
    ($lua:ident, $state:ident, $( fn $fn_name:ident($($fn_args:tt)*) $fn_code:block )*) => {{
        let lib = $lua.create_table().unwrap();
        $(
            lib.set(stringify!($fn_name), $crate::lua_func!($lua, $state, fn($($fn_args)*) $fn_code)).unwrap();
        )*
        lib
    }};
}

#[derive(Serialize, Deserialize)]
pub enum LuaValueStatic {
    Nil,
    Bool(bool),
    LightUserData(usize),
    Int(i64),
    Float(f64),
    String(Vec<u8>),
    Table(Vec<(LuaValueStatic, LuaValueStatic)>),
}
impl<'lua> FromLua<'lua> for LuaValueStatic {
    fn from_lua(v: LuaValue<'lua>, lua: LuaContext<'lua>) -> LuaResult<Self> {
        Ok(match v {
            LuaValue::Nil => Self::Nil,
            LuaValue::Boolean(b) => Self::Bool(b),
            LuaValue::LightUserData(u) => Self::LightUserData(u.0 as usize),
            LuaValue::Integer(i) => Self::Int(i),
            LuaValue::Number(f) => Self::Float(f),
            LuaValue::String(s) => Self::String(s.as_bytes().to_vec()),
            LuaValue::Table(t) => Self::Table(
                t.pairs()
                    .map(|res| {
                        let (k, v) = res?;
                        Ok((Self::from_lua(k, lua)?, Self::from_lua(v, lua)?))
                    })
                    .collect::<LuaResult<_>>()?,
            ),
            _ => lua_bail!("cannot convert {:?} to a static lua value", v),
        })
    }
}
impl<'lua> ToLua<'lua> for LuaValueStatic {
    fn to_lua(self, lua: LuaContext<'lua>) -> LuaResult<LuaValue<'lua>> {
        Ok(match self {
            Self::Nil => LuaValue::Nil,
            Self::Bool(b) => LuaValue::Boolean(b),
            Self::LightUserData(u) => LuaValue::LightUserData(LuaLightUserData(u as *mut _)),
            Self::Int(i) => LuaValue::Integer(i),
            Self::Float(f) => LuaValue::Number(f),
            Self::String(s) => LuaValue::String(lua.create_string(&s)?),
            Self::Table(pairs) => LuaValue::Table(lua.create_table_from(pairs.into_iter())?),
        })
    }
}

pub struct LuaBytes {
    pub bytes: Vec<u8>,
}
impl<'lua> FromLua<'lua> for LuaBytes {
    fn from_lua(v: LuaValue<'lua>, lua: LuaContext<'lua>) -> LuaResult<Self> {
        let s = LuaString::from_lua(v, lua)?;
        Ok(Self{ bytes: s.as_bytes().to_vec() })
    }
}
impl<'lua> ToLua<'lua> for LuaBytes {
    fn to_lua(self, lua: LuaContext<'lua>) -> LuaResult<LuaValue<'lua>> {
        Ok(LuaValue::String(
            lua.create_string(&self.bytes[..])?
        ))
    }
}

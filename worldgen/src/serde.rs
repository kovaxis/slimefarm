use std::ptr::read_unaligned;

use crate::prelude::*;
use serde::{
    de::{
        self, DeserializeSeed, Deserializer as _, EnumAccess, Error as _, IntoDeserializer,
        MapAccess, SeqAccess, Unexpected, VariantAccess, Visitor,
    },
    Deserialize,
};

type Result<T> = std::result::Result<T, Error>;

pub(crate) fn deserialize<'de, T: Deserialize<'de>>(
    lua: LuaContext<'de>,
    val: LuaValue<'de>,
) -> Result<T> {
    let mut de = Deserializer::<'de, 'static>::from_value(lua, val);
    T::deserialize(&mut de)
}

#[derive(Debug)]
pub struct Error {
    src: String,
    msg: String,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if !self.src.is_empty() {
            write!(f, "{}: ", self.src)?;
        }
        write!(f, "{}", self.msg)
    }
}
impl StdError for Error {}
impl de::Error for Error {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Error {
            src: String::new(),
            msg: format!("{}", msg),
        }
    }
}
impl Error {
    fn locate(mut self, node: &Node) -> Self {
        fn navigate(node: &Node, buf: &mut String) {
            if let Some(p) = node.parent {
                navigate(p, buf);
            }
            let _ = write!(buf, "{}", node.name);
        }
        let mut buf = String::new();
        navigate(node, &mut buf);
        self.src = buf;
        self
    }
}

trait ResultExt<T> {
    fn locate(self, node: &Node) -> Result<T>;
}
impl<T> ResultExt<T> for Result<T> {
    fn locate(self, node: &Node) -> Result<T> {
        self.map_err(|e| e.locate(node))
    }
}

struct Node<'a> {
    name: String,
    parent: Option<&'a Node<'a>>,
}
impl<'a> Node<'a> {
    fn navigate(&self, buf: &mut String) {
        if let Some(p) = self.parent {
            p.navigate(buf);
        }
        let _ = write!(buf, "{}", self.name);
    }

    fn error<T: fmt::Display>(&self, msg: T) -> Error {
        let mut buf = String::new();
        self.navigate(&mut buf);
        let _ = write!(buf, ": {}", msg);
        Error {
            src: String::new(),
            msg: buf,
        }
    }
}

struct Deserializer<'de, 'a> {
    lua: LuaContext<'de>,
    val: LuaValue<'de>,
    node: Node<'a>,
}

impl<'de, 'a> Deserializer<'de, 'a> {
    fn from_value(lua: LuaContext<'de>, val: LuaValue<'de>) -> Self {
        Self {
            lua,
            node: Node {
                name: "root".into(),
                parent: None,
            },
            val,
        }
    }

    fn with_name(
        lua: LuaContext<'de>,
        val: LuaValue<'de>,
        name: String,
        parent: &'a Node<'a>,
    ) -> Self {
        Self {
            lua,
            node: Node {
                name,
                parent: Some(parent),
            },
            val,
        }
    }

    fn get_unexpected(val: &LuaValue) -> Unexpected<'static> {
        match val {
            LuaValue::Nil => Unexpected::Unit,
            LuaValue::Boolean(b) => Unexpected::Bool(*b),
            LuaValue::Integer(num) => Unexpected::Signed(*num),
            LuaValue::Number(num) => Unexpected::Float(*num),
            LuaValue::String(_) => Unexpected::Bytes(b""),
            LuaValue::Table(_) => Unexpected::Map,
            LuaValue::Function(_) => Unexpected::Other("function"),
            LuaValue::LightUserData(_) => Unexpected::Other("lightuserdata"),
            LuaValue::Thread(_) => Unexpected::Other("thread"),
            LuaValue::UserData(_) => Unexpected::Other("userdata"),
            LuaValue::Error(_) => Unexpected::Other("error"),
        }
    }

    fn unexpected<T>(&self, val: &LuaValue, ex: &'static str) -> Result<T> {
        Err(Error::invalid_type(Self::get_unexpected(val), &ex).locate(&self.node))
    }

    fn parse_bool(&self) -> Result<bool> {
        if let LuaValue::Boolean(b) = self.val {
            Ok(b)
        } else {
            self.unexpected(&self.val, "boolean")
        }
    }

    fn parse_int(&self) -> Result<i64> {
        match &self.val {
            &LuaValue::Integer(num) => Ok(num),
            &LuaValue::Number(num) => {
                if num as i64 as f64 == num {
                    Ok(num as i64)
                } else {
                    self.unexpected(&self.val, "integer")
                }
            }
            _ => self.unexpected(&self.val, "integer"),
        }
    }

    fn parse_float(&self) -> Result<f64> {
        match &self.val {
            &LuaValue::Number(num) => Ok(num),
            &LuaValue::Integer(num) => Ok(num as f64),
            _ => self.unexpected(&self.val, "float"),
        }
    }

    fn parse_bytes(&self) -> Result<&[u8]> {
        if let LuaValue::String(b) = &self.val {
            Ok(b.as_bytes())
        } else {
            self.unexpected(&self.val, "string")
        }
    }

    fn parse_str(&self) -> Result<&str> {
        let s = self.parse_bytes()?;
        let s = std::str::from_utf8(s).map_err(|_| self.node.error("invalid utf-8"))?;
        Ok(s)
    }

    fn parse_sequence<'b>(&'b mut self) -> Result<Sequence<'de, 'b>> {
        if let LuaValue::Table(_) = &self.val {
            let tab = if let LuaValue::Table(tab) = mem::replace(&mut self.val, LuaValue::Nil) {
                tab
            } else {
                unreachable!()
            };
            Ok(Sequence {
                lua: self.lua,
                seq: tab.sequence_values(),
                nxt: 0,
                parent: &self.node,
            })
        } else {
            self.unexpected(&self.val, "list")
        }
    }

    fn parse_map<'b>(&'b mut self) -> Result<Map<'de, 'b>> {
        if let LuaValue::Table(_) = &self.val {
            let tab = if let LuaValue::Table(tab) = mem::replace(&mut self.val, LuaValue::Nil) {
                tab
            } else {
                unreachable!()
            };
            Ok(Map {
                lua: self.lua,
                pairs: tab.pairs(),
                next_val: None,
                kname: String::new(),
                parent: &self.node,
            })
        } else {
            self.unexpected(&self.val, "list")
        }
    }

    fn parse_function(&mut self) -> Result<[u8; mem::size_of::<LuaRegistryKey>()]> {
        if let LuaValue::Function(_) = &self.val {
            let f = if let LuaValue::Function(f) = mem::replace(&mut self.val, LuaValue::Nil) {
                f
            } else {
                unreachable!()
            };
            let regkey = self
                .lua
                .create_registry_value(f)
                .map_err(|e| self.node.error(e))?;
            unsafe { Ok(mem::transmute(regkey)) }
        } else {
            self.unexpected(&self.val, "function")
        }
    }
}

fn stringify(val: &LuaValue) -> String {
    match val {
        LuaValue::Nil => "nil".into(),
        LuaValue::Boolean(b) => if *b { "true" } else { "false" }.into(),
        LuaValue::String(s) => String::from_utf8_lossy(s.as_bytes()).into_owned(),
        LuaValue::Integer(num) => format!("{}", *num),
        LuaValue::Number(num) => format!("{}", *num),
        LuaValue::Table(_) => "<table>".into(),
        LuaValue::Function(_) => "<function>".into(),
        LuaValue::Thread(_) => "<thread>".into(),
        LuaValue::UserData(_) => "<userdata>".into(),
        LuaValue::LightUserData(_) => "<lightuserdata>".into(),
        LuaValue::Error(_) => "<error>".into(),
    }
}

struct Sequence<'de, 'a> {
    lua: LuaContext<'de>,
    seq: LuaTableSequence<'de, LuaValue<'de>>,
    nxt: usize,
    parent: &'a Node<'a>,
}
impl<'de, 'a> SeqAccess<'de> for Sequence<'de, 'a> {
    type Error = Error;

    fn next_element_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<Option<T::Value>> {
        match self.seq.next() {
            Some(t) => {
                let t = t.map_err(|e| self.parent.error(e))?;
                self.nxt += 1;
                seed.deserialize(&mut Deserializer::with_name(
                    self.lua,
                    t,
                    format!("[{}]", self.nxt),
                    self.parent,
                ))
                .map(Some)
            }
            None => Ok(None),
        }
    }
}

struct Map<'de, 'a> {
    lua: LuaContext<'de>,
    pairs: LuaTablePairs<'de, LuaValue<'de>, LuaValue<'de>>,
    next_val: Option<LuaValue<'de>>,
    kname: String,
    parent: &'a Node<'a>,
}
impl<'de, 'a> MapAccess<'de> for Map<'de, 'a> {
    type Error = Error;

    fn next_key_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<Option<T::Value>> {
        match self.pairs.next() {
            Some(t) => {
                let (k, v) = t.map_err(|e| self.parent.error(e))?;
                let kname = stringify(&k);
                let selfkname = format!("{{{}}}", kname);
                if let LuaValue::String(_) = &k {
                    self.kname = format!(".{}", kname);
                } else {
                    self.kname = format!("[{}]", kname);
                }
                self.next_val = Some(v);
                seed.deserialize(&mut Deserializer::with_name(
                    self.lua,
                    k,
                    selfkname,
                    self.parent,
                ))
                .map(Some)
            }
            None => Ok(None),
        }
    }

    fn next_value_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<T::Value> {
        let v = self
            .next_val
            .take()
            .ok_or_else(|| self.parent.error("expected map value"))?;
        seed.deserialize(&mut Deserializer::with_name(
            self.lua,
            v,
            mem::take(&mut self.kname),
            self.parent,
        ))
    }
}

struct Enum<'de, 'a> {
    lua: LuaContext<'de>,
    key: LuaValue<'de>,
    val: LuaValue<'de>,
    kname: String,
    parent: &'a Node<'a>,
}
impl<'de, 'a> EnumAccess<'de> for Enum<'de, 'a> {
    type Error = Error;
    type Variant = Self;

    fn variant_seed<V: DeserializeSeed<'de>>(
        mut self,
        seed: V,
    ) -> Result<(V::Value, Self::Variant)> {
        self.kname = stringify(&self.key);
        let key = seed
            .deserialize(&mut Deserializer::with_name(
                self.lua,
                mem::replace(&mut self.key, LuaValue::Nil),
                format!("{{{}}}", self.kname),
                self.parent,
            ))
            .locate(self.parent)?;
        Ok((key, self))
    }
}
impl<'de, 'a> VariantAccess<'de> for Enum<'de, 'a> {
    type Error = Error;

    fn unit_variant(self) -> Result<()> {
        Err(Error::custom("expected string"))
    }

    fn newtype_variant_seed<V: DeserializeSeed<'de>>(mut self, seed: V) -> Result<V::Value> {
        seed.deserialize(&mut Deserializer::with_name(
            self.lua,
            mem::replace(&mut self.val, LuaValue::Nil),
            format!("({})", self.kname),
            self.parent,
        ))
    }

    fn tuple_variant<V: Visitor<'de>>(mut self, _len: usize, v: V) -> Result<V::Value> {
        (&mut Deserializer::with_name(
            self.lua,
            mem::replace(&mut self.val, LuaValue::Nil),
            format!("({})", self.kname),
            self.parent,
        ))
            .deserialize_seq(v)
    }

    fn struct_variant<V: Visitor<'de>>(
        mut self,
        _fields: &'static [&'static str],
        v: V,
    ) -> Result<V::Value> {
        (&mut Deserializer::with_name(
            self.lua,
            mem::replace(&mut self.val, LuaValue::Nil),
            format!("({})", self.kname),
            self.parent,
        ))
            .deserialize_map(v)
    }
}

fn try_fit<T, U>(from: T) -> Result<U>
where
    T: TryInto<U>,
{
    match from.try_into() {
        Ok(u) => Ok(u),
        Err(_) => Err(Error::custom("number out of range")),
    }
}

impl<'de, 'a, 'b> de::Deserializer<'de> for &'a mut Deserializer<'de, 'b> {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        match &self.val {
            LuaValue::Nil => self.deserialize_unit(v),
            LuaValue::Boolean(_) => self.deserialize_bool(v),
            LuaValue::Integer(_) => self.deserialize_i64(v),
            LuaValue::Number(_) => self.deserialize_f64(v),
            LuaValue::String(_) => self.deserialize_bytes(v),
            LuaValue::Table(_) => self.deserialize_map(v),
            LuaValue::Function(_) => self.deserialize_u32(v),
            LuaValue::LightUserData(_) => Err(Error::invalid_value(
                Unexpected::Other("lightuserdata"),
                &"anything",
            )),
            LuaValue::Thread(_) => Err(Error::invalid_value(
                Unexpected::Other("thread"),
                &"anything",
            )),
            LuaValue::UserData(_) => Err(Error::invalid_value(
                Unexpected::Other("userdata"),
                &"anything",
            )),
            LuaValue::Error(_) => Err(Error::invalid_value(
                Unexpected::Other("error"),
                &"anything",
            )),
        }
    }

    fn deserialize_bool<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        v.visit_bool(self.parse_bool()?)
    }

    fn deserialize_i8<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        v.visit_i8(try_fit(self.parse_int()?)?)
    }

    fn deserialize_i16<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        v.visit_i16(try_fit(self.parse_int()?)?)
    }

    fn deserialize_i32<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        v.visit_i32(try_fit(self.parse_int()?)?)
    }

    fn deserialize_i64<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        v.visit_i64(self.parse_int()?)
    }

    fn deserialize_u8<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        v.visit_u8(try_fit(self.parse_int()?)?)
    }

    fn deserialize_u16<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        v.visit_u16(try_fit(self.parse_int()?)?)
    }

    fn deserialize_u32<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        v.visit_u32(try_fit(self.parse_int()?)?)
    }

    fn deserialize_u64<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        v.visit_u64(try_fit(self.parse_int()?)?)
    }

    fn deserialize_f32<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        v.visit_f32(self.parse_float()? as f32)
    }

    fn deserialize_f64<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        v.visit_f64(self.parse_float()?)
    }

    fn deserialize_char<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        let s = self.parse_str()?;
        let mut cs = s.chars();
        let c = cs
            .next()
            .ok_or_else(|| Error::invalid_value(Unexpected::Str(s), &"character"))
            .locate(&self.node)?;
        if cs.next().is_some() {
            return Err(Error::invalid_value(Unexpected::Str(s), &"character").locate(&self.node));
        }
        v.visit_char(c)
    }

    fn deserialize_str<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        let s = self.parse_str()?;
        v.visit_string(s.to_string())
    }

    fn deserialize_string<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        self.deserialize_str(v)
    }

    fn deserialize_bytes<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        let s = self.parse_bytes()?;
        v.visit_byte_buf(s.to_vec())
    }

    fn deserialize_byte_buf<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        self.deserialize_bytes(v)
    }

    fn deserialize_option<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        if let LuaValue::Nil = self.val {
            v.visit_none()
        } else {
            v.visit_some(self)
        }
    }

    fn deserialize_unit<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        if let LuaValue::Nil = self.val {
            v.visit_unit()
        } else {
            self.unexpected(&self.val, "nil")
        }
    }

    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        v: V,
    ) -> Result<V::Value> {
        self.deserialize_unit(v)
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        v: V,
    ) -> Result<V::Value> {
        if name == "luafunction" {
            v.visit_bytes(&self.parse_function()?[..])
        } else {
            v.visit_newtype_struct(self)
        }
    }

    fn deserialize_seq<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        v.visit_seq(self.parse_sequence()?)
    }

    fn deserialize_tuple<V: Visitor<'de>>(self, _len: usize, v: V) -> Result<V::Value> {
        self.deserialize_seq(v)
    }

    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _len: usize,
        v: V,
    ) -> Result<V::Value> {
        self.deserialize_seq(v)
    }

    fn deserialize_map<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        v.visit_map(self.parse_map()?)
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        v: V,
    ) -> Result<V::Value> {
        self.deserialize_map(v)
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        v: V,
    ) -> Result<V::Value> {
        match &self.val {
            LuaValue::Table(_) => {
                let tab = if let LuaValue::Table(tab) = mem::replace(&mut self.val, LuaValue::Nil) {
                    tab
                } else {
                    unreachable!()
                };
                let mut it = tab.pairs::<LuaValue, LuaValue>();
                let (key, val) = it
                    .next()
                    .ok_or_else(|| self.node.error("expected enum variant and data"))?
                    .map_err(|e| self.node.error(e))?;
                if it.next().is_some() {
                    Err(self
                        .node
                        .error("expected enum container to have only 1 field"))?;
                }
                v.visit_enum(Enum {
                    lua: self.lua,
                    key,
                    val,
                    kname: String::new(),
                    parent: &self.node,
                })
            }
            LuaValue::String(s) => {
                let s = s.as_bytes();
                let s = std::str::from_utf8(s).map_err(|_| self.node.error("invalid utf-8"))?;
                v.visit_enum(s.into_deserializer())
            }
            _ => self.unexpected(&self.val, "list"),
        }
    }

    fn deserialize_identifier<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        self.deserialize_str(v)
    }

    fn deserialize_ignored_any<V: Visitor<'de>>(self, v: V) -> Result<V::Value> {
        v.visit_unit()
    }
}

#[repr(transparent)]
pub(crate) struct LuaFuncRef {
    pub key: LuaRegistryKey,
}
impl LuaFuncRef {
    pub fn get<'lua>(&self, lua: LuaContext<'lua>) -> LuaResult<LuaFunction<'lua>> {
        lua.registry_value::<LuaFunction>(&self.key)
    }
}
impl<'de> Deserialize<'de> for LuaFuncRef {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        d.deserialize_newtype_struct("luafunction", LuaFuncVisitor)
    }
}

struct LuaFuncVisitor;
impl<'de> Visitor<'de> for LuaFuncVisitor {
    type Value = LuaFuncRef;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "a lua function")
    }

    fn visit_bytes<E: de::Error>(self, b: &[u8]) -> std::result::Result<LuaFuncRef, E> {
        // SUPER-UNSAFE!!!
        unsafe {
            if b.len() != mem::size_of::<LuaFuncRef>() {
                return Err(E::custom("invalid LuaFuncRef deserializer representation"));
            }
            let funcref = (b.as_ptr() as *const LuaFuncRef).read_unaligned();
            Ok(funcref)
        }
    }
}

use crate::prelude::*;
use rand_distr::StandardNormal;

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
    (@$m:ident fn $fn_name:ident ($lua:ident, $this:ident, $($args:tt)*) $fn_code:block $($rest:tt)*) => {{
        $m.add_method(stringify!($fn_name), |lua, this, $($args)*| {
            #[allow(unused_variables)]
            let $lua = lua;
            #[allow(unused_variables)]
            let $this = this;
            Ok($fn_code)
        });
        lua_type!(@$m $($rest)*);
    }};
    (@$m:ident mut fn $fn_name:ident ($lua:ident, $this:ident, $($args:tt)*) $fn_code:block $($rest:tt)*) => {{
        $m.add_method_mut(stringify!($fn_name), |lua, this, $($args)*| {
            #[allow(unused_variables)]
            let $lua = lua;
            #[allow(unused_variables)]
            let $this = this;
            Ok($fn_code)
        });
        lua_type!(@$m $($rest)*);
    }};
    (@$m:ident) => {};
    ($ty:ty, $($rest:tt)*) => {
        impl LuaUserData for $ty {
            fn add_methods<'lua, M: LuaUserDataMethods<'lua, Self>>(m: &mut M) {
                lua_type!(@m $($rest)*);
            }
        }
    };
}

#[macro_export]
macro_rules! lua_func {
    ($lua:ident, $state:ident, fn($($fn_args:tt)*) $fn_code:block) => {{
        let state = AssertSync($state.clone());
        $lua.create_function(move |ctx, $($fn_args)*| {
            #[allow(unused_variables)]
            let $lua = ctx;
            #[allow(unused_variables)]
            let $state = &*state;
            Ok($fn_code)
        }).unwrap()
    }};
}

#[macro_export]
macro_rules! lua_lib {
    ($lua:ident, $state:ident, $( fn $fn_name:ident($($fn_args:tt)*) $fn_code:block )*) => {{
        let lib = $lua.create_table().unwrap();
        $(
            lib.set(stringify!($fn_name), lua_func!($lua, $state, fn($($fn_args)*) $fn_code)).unwrap();
        )*
        lib
    }};
}

pub struct LuaRng {
    pub rng: Cell<FastRng>,
}
impl LuaRng {
    pub fn seed(seed: u64) -> Self {
        Self {
            rng: FastRng::seed_from_u64(seed).into(),
        }
    }

    pub fn new(rng: FastRng) -> Self {
        Self {
            rng: Cell::new(rng),
        }
    }

    fn get(&self) -> FastRng {
        self.rng.replace(unsafe { mem::zeroed() })
    }
    fn set(&self, rng: FastRng) {
        self.rng.set(rng);
    }
}
lua_type! {LuaRng,
    // uniform() -> uniform(0, 1)
    // uniform(r) -> uniform(0, r)
    // uniform(l, r) -> uniform(l, r)
    fn uniform(lua, this, (a, b): (Option<f64>, Option<f64>)) {
        let mut rng = this.get();
        let (l, r) = match (a, b) {
            (Some(l), Some(r)) => (l, r),
            (Some(r), _) => (0., r),
            _ => (0., 1.),
        };
        let v = rng.gen_range(l..= r);
        this.set(rng);
        v
    }

    // normal() -> normal(1/2, 1/6) clamped to [0, 1]
    // normal(x) -> normal(x/2, x/6) clamped to [0, x]
    // normal(l, r) -> normal((l+r)/2, (r-l)/6) clamped to [l, r]
    fn normal(lua, this, (a, b): (Option<f64>, Option<f64>)) {
        let mut rng = this.get();
        let (mu, sd) = match (a, b) {
            (Some(l), Some(r)) => (0.5 * (l + r), 1./6. * (r - l)),
            (Some(x), _) => (0.5, 1./6.*x),
            (_, _) => (0.5, 1./6.),
        };
        let z = rng.sample::<f64, _>(StandardNormal).clamp(-3., 3.);
        this.set(rng);
        mu + sd * z
    }
}

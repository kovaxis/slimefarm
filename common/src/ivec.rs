use crate::prelude::*;

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Int2 {
    pub x: i32,
    pub y: i32,
}
impl fmt::Debug for Int2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}, {}]", self.x, self.y)
    }
}
impl Hash for Int2 {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        h.write_u64(self.x as u32 as u64 | ((self.y as u32 as u64) << 32));
    }
}
impl ops::Deref for Int2 {
    type Target = [i32; 2];
    #[inline]
    fn deref(&self) -> &[i32; 2] {
        unsafe { mem::transmute(self) }
    }
}
impl ops::DerefMut for Int2 {
    #[inline]
    fn deref_mut(&mut self) -> &mut [i32; 2] {
        unsafe { mem::transmute(self) }
    }
}
impl ops::AddAssign for Int2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}
impl ops::Add for Int2 {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: Self) -> Self {
        self += rhs;
        self
    }
}
impl ops::AddAssign<[i32; 2]> for Int2 {
    #[inline]
    fn add_assign(&mut self, rhs: [i32; 2]) {
        *self += Self::new(rhs);
    }
}
impl ops::Add<[i32; 2]> for Int2 {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: [i32; 2]) -> Self {
        self += rhs;
        self
    }
}
impl ops::Add<Int2> for [i32; 2] {
    type Output = Int2;
    #[inline]
    fn add(self, mut rhs: Int2) -> Int2 {
        rhs += self;
        rhs
    }
}
impl ops::SubAssign for Int2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}
impl ops::Sub for Int2 {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: Self) -> Self {
        self -= rhs;
        self
    }
}
impl ops::SubAssign<[i32; 2]> for Int2 {
    #[inline]
    fn sub_assign(&mut self, rhs: [i32; 2]) {
        *self -= Self::new(rhs);
    }
}
impl ops::Sub<[i32; 2]> for Int2 {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: [i32; 2]) -> Self {
        self -= rhs;
        self
    }
}
impl ops::Sub<Int2> for [i32; 2] {
    type Output = Int2;
    #[inline]
    fn sub(self, rhs: Int2) -> Int2 {
        Int2::new(self) - rhs
    }
}
impl ops::Neg for Int2 {
    type Output = Self;
    #[inline]
    fn neg(mut self) -> Self {
        self.x = -self.x;
        self.y = -self.y;
        self
    }
}
impl ops::MulAssign<i32> for Int2 {
    #[inline]
    fn mul_assign(&mut self, rhs: i32) {
        self.x *= rhs;
        self.y *= rhs;
    }
}
impl ops::Mul<i32> for Int2 {
    type Output = Self;
    #[inline]
    fn mul(mut self, rhs: i32) -> Self {
        self *= rhs;
        self
    }
}
impl ops::Mul<Int2> for i32 {
    type Output = Int2;
    #[inline]
    fn mul(self, mut rhs: Int2) -> Int2 {
        rhs *= self;
        rhs
    }
}
impl ops::DivAssign<i32> for Int2 {
    #[inline]
    fn div_assign(&mut self, rhs: i32) {
        self.x = self.x.div_euclid(rhs);
        self.y = self.y.div_euclid(rhs);
    }
}
impl ops::Div<i32> for Int2 {
    type Output = Self;
    #[inline]
    fn div(mut self, rhs: i32) -> Self {
        self /= rhs;
        self
    }
}
impl ops::RemAssign<i32> for Int2 {
    #[inline]
    fn rem_assign(&mut self, rhs: i32) {
        self.x = self.x.rem_euclid(rhs);
        self.y = self.y.rem_euclid(rhs);
    }
}
impl ops::Rem<i32> for Int2 {
    type Output = Self;
    #[inline]
    fn rem(mut self, rhs: i32) -> Self {
        self %= rhs;
        self
    }
}
impl ops::ShrAssign<i32> for Int2 {
    #[inline]
    fn shr_assign(&mut self, rhs: i32) {
        self.x >>= rhs;
        self.y >>= rhs;
    }
}
impl ops::Shr<i32> for Int2 {
    type Output = Self;
    #[inline]
    fn shr(mut self, rhs: i32) -> Self {
        self >>= rhs;
        self
    }
}
impl ops::ShlAssign<i32> for Int2 {
    #[inline]
    fn shl_assign(&mut self, rhs: i32) {
        self.x <<= rhs;
        self.y <<= rhs;
    }
}
impl ops::Shl<i32> for Int2 {
    type Output = Self;
    #[inline]
    fn shl(mut self, rhs: i32) -> Self {
        self <<= rhs;
        self
    }
}
impl From<[i32; 2]> for Int2 {
    #[inline]
    fn from(x: [i32; 2]) -> Self {
        Self::new(x)
    }
}
impl From<Int2> for [i32; 2] {
    #[inline]
    fn from(x: Int2) -> Self {
        *x
    }
}
impl Int2 {
    #[inline]
    pub fn new(u: [i32; 2]) -> Self {
        Self { x: u[0], y: u[1] }
    }

    #[inline]
    pub fn splat(u: i32) -> Self {
        Self { x: u, y: u }
    }

    #[inline]
    pub fn zero() -> Self {
        Self::splat(0)
    }

    #[inline]
    pub fn lowbits_mut(&mut self, bits: i32) {
        let mask = (1 << bits) - 1;
        self.x &= mask;
        self.y &= mask;
    }

    #[inline]
    pub fn lowbits(mut self, bits: i32) -> Self {
        self.lowbits_mut(bits);
        self
    }

    #[inline]
    pub fn min_mut(&mut self, rhs: Int2) {
        self.x = self.x.min(rhs.x);
        self.y = self.y.min(rhs.y);
    }

    #[inline]
    pub fn min(mut self, rhs: Int2) -> Self {
        self.min_mut(rhs);
        self
    }

    #[inline]
    pub fn max_mut(&mut self, rhs: Int2) {
        self.x = self.x.max(rhs.x);
        self.y = self.y.max(rhs.y);
    }

    #[inline]
    pub fn max(mut self, rhs: Int2) -> Self {
        self.max_mut(rhs);
        self
    }

    #[inline]
    pub fn from_f32(float: Vec2) -> Self {
        Self::new([float.x.floor() as i32, float.y.floor() as i32])
    }

    #[inline]
    pub fn from_f64(float: [f64; 2]) -> Self {
        Self::new([float[0].floor() as i32, float[1].floor() as i32])
    }

    #[inline]
    pub fn to_f32(self) -> Vec2 {
        Vec2::new(self.x as f32, self.y as f32)
    }

    #[inline]
    pub fn to_f32_center(self) -> Vec2 {
        Vec2::new(self.x as f32 + 0.5, self.y as f32 + 0.5)
    }

    #[inline]
    pub fn to_f64(self) -> [f64; 2] {
        [self.x as f64, self.y as f64]
    }

    #[inline]
    pub fn to_f64_center(self) -> [f64; 2] {
        [self.x as f64 + 0.5, self.y as f64 + 0.5]
    }

    #[inline]
    pub fn to_index(self, bits: Int2) -> usize {
        (self.x | (self.y << bits.x)) as usize
    }

    #[inline]
    pub fn from_index(bits: Int2, idx: usize) -> Self {
        let idx = idx as i32;
        Self::new([
            idx & ((1 << bits.x) - 1),
            (idx >> bits.x) & ((1 << bits.y) - 1),
        ])
    }

    #[inline]
    pub fn is_within(self, size: Int2) -> bool {
        (self.x as u32) < (size.x as u32) && (self.y as u32) < (size.y as u32)
    }

    #[inline]
    pub fn with_z(self, z: i32) -> Int3 {
        Int3::new([self.x, self.y, z])
    }

    #[inline]
    pub fn mag_sq(self) -> i32 {
        self.x * self.x + self.y * self.y
    }
}

#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Int3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}
impl fmt::Debug for Int3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}, {}, {}]", self.x, self.y, self.z)
    }
}
impl Hash for Int3 {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        h.write_u64(self.x as u32 as u64 | ((self.y as u32 as u64) << 32));
        h.write_u32(self.z as u32);
    }
}
impl ops::Deref for Int3 {
    type Target = [i32; 3];
    #[inline]
    fn deref(&self) -> &[i32; 3] {
        unsafe { mem::transmute(self) }
    }
}
impl ops::DerefMut for Int3 {
    #[inline]
    fn deref_mut(&mut self) -> &mut [i32; 3] {
        unsafe { mem::transmute(self) }
    }
}
impl ops::AddAssign for Int3 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}
impl ops::Add for Int3 {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: Self) -> Self {
        self += rhs;
        self
    }
}
impl ops::AddAssign<[i32; 3]> for Int3 {
    #[inline]
    fn add_assign(&mut self, rhs: [i32; 3]) {
        *self += Int3::new(rhs);
    }
}
impl ops::Add<[i32; 3]> for Int3 {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: [i32; 3]) -> Self {
        self += rhs;
        self
    }
}
impl ops::Add<Int3> for [i32; 3] {
    type Output = Int3;
    #[inline]
    fn add(self, mut rhs: Int3) -> Int3 {
        rhs += self;
        rhs
    }
}
impl ops::SubAssign for Int3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}
impl ops::Sub for Int3 {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: Self) -> Self {
        self -= rhs;
        self
    }
}
impl ops::SubAssign<[i32; 3]> for Int3 {
    #[inline]
    fn sub_assign(&mut self, rhs: [i32; 3]) {
        *self -= Int3::new(rhs);
    }
}
impl ops::Sub<[i32; 3]> for Int3 {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: [i32; 3]) -> Self {
        self -= rhs;
        self
    }
}
impl ops::Sub<Int3> for [i32; 3] {
    type Output = Int3;
    #[inline]
    fn sub(self, rhs: Int3) -> Int3 {
        Int3::new(self) - rhs
    }
}
impl ops::Neg for Int3 {
    type Output = Self;
    #[inline]
    fn neg(mut self) -> Self {
        self.x = -self.x;
        self.y = -self.y;
        self.z = -self.z;
        self
    }
}
impl ops::MulAssign<i32> for Int3 {
    #[inline]
    fn mul_assign(&mut self, rhs: i32) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}
impl ops::Mul<i32> for Int3 {
    type Output = Self;
    #[inline]
    fn mul(mut self, rhs: i32) -> Self {
        self *= rhs;
        self
    }
}
impl ops::Mul<Int3> for i32 {
    type Output = Int3;
    #[inline]
    fn mul(self, mut rhs: Int3) -> Int3 {
        rhs *= self;
        rhs
    }
}
impl ops::DivAssign<i32> for Int3 {
    #[inline]
    fn div_assign(&mut self, rhs: i32) {
        self.x = self.x.div_euclid(rhs);
        self.y = self.y.div_euclid(rhs);
        self.z = self.z.div_euclid(rhs);
    }
}
impl ops::Div<i32> for Int3 {
    type Output = Self;
    #[inline]
    fn div(mut self, rhs: i32) -> Self {
        self /= rhs;
        self
    }
}
impl ops::RemAssign<i32> for Int3 {
    #[inline]
    fn rem_assign(&mut self, rhs: i32) {
        self.x = self.x.rem_euclid(rhs);
        self.y = self.y.rem_euclid(rhs);
        self.z = self.z.rem_euclid(rhs);
    }
}
impl ops::Rem<i32> for Int3 {
    type Output = Self;
    #[inline]
    fn rem(mut self, rhs: i32) -> Self {
        self %= rhs;
        self
    }
}
impl ops::ShrAssign<i32> for Int3 {
    #[inline]
    fn shr_assign(&mut self, rhs: i32) {
        self.x >>= rhs;
        self.y >>= rhs;
        self.z >>= rhs;
    }
}
impl ops::Shr<i32> for Int3 {
    type Output = Self;
    #[inline]
    fn shr(mut self, rhs: i32) -> Self {
        self >>= rhs;
        self
    }
}
impl ops::ShlAssign<i32> for Int3 {
    #[inline]
    fn shl_assign(&mut self, rhs: i32) {
        self.x <<= rhs;
        self.y <<= rhs;
        self.z <<= rhs;
    }
}
impl ops::Shl<i32> for Int3 {
    type Output = Self;
    #[inline]
    fn shl(mut self, rhs: i32) -> Self {
        self <<= rhs;
        self
    }
}
impl From<[i32; 3]> for Int3 {
    #[inline]
    fn from(x: [i32; 3]) -> Self {
        Self::new(x)
    }
}
impl From<Int3> for [i32; 3] {
    #[inline]
    fn from(x: Int3) -> Self {
        *x
    }
}
impl Int3 {
    #[inline]
    pub fn new(x: [i32; 3]) -> Self {
        Self {
            x: x[0],
            y: x[1],
            z: x[2],
        }
    }

    #[inline]
    pub fn splat(x: i32) -> Self {
        Self { x: x, y: x, z: x }
    }

    #[inline]
    pub fn zero() -> Self {
        Self::splat(0)
    }

    #[inline]
    pub fn lowbits_mut(&mut self, bits: i32) {
        let mask = (1 << bits) - 1;
        self.x &= mask;
        self.y &= mask;
        self.z &= mask;
    }

    #[inline]
    pub fn lowbits(mut self, bits: i32) -> Self {
        self.lowbits_mut(bits);
        self
    }

    #[inline]
    pub fn min_mut(&mut self, rhs: Int3) {
        self.x = self.x.min(rhs.x);
        self.y = self.y.min(rhs.y);
        self.z = self.z.min(rhs.z);
    }

    #[inline]
    pub fn min(mut self, rhs: Int3) -> Self {
        self.min_mut(rhs);
        self
    }

    #[inline]
    pub fn max_mut(&mut self, rhs: Int3) {
        self.x = self.x.max(rhs.x);
        self.y = self.y.max(rhs.y);
        self.z = self.z.max(rhs.z);
    }

    #[inline]
    pub fn max(mut self, rhs: Int3) -> Self {
        self.max_mut(rhs);
        self
    }

    #[inline]
    pub fn from_f32(float: Vec3) -> Self {
        Self::new([
            float.x.floor() as i32,
            float.y.floor() as i32,
            float.z.floor() as i32,
        ])
    }

    #[inline]
    pub fn from_f64(float: [f64; 3]) -> Self {
        Self::new([
            float[0].floor() as i32,
            float[1].floor() as i32,
            float[2].floor() as i32,
        ])
    }

    #[inline]
    pub fn to_f32(self) -> Vec3 {
        Vec3::new(self.x as f32, self.y as f32, self.z as f32)
    }

    #[inline]
    pub fn to_f32_center(self) -> Vec3 {
        Vec3::new(
            self.x as f32 + 0.5,
            self.y as f32 + 0.5,
            self.z as f32 + 0.5,
        )
    }

    #[inline]
    pub fn to_f64(self) -> [f64; 3] {
        [self.x as f64, self.y as f64, self.z as f64]
    }

    #[inline]
    pub fn to_f64_center(self) -> [f64; 3] {
        [
            self.x as f64 + 0.5,
            self.y as f64 + 0.5,
            self.z as f64 + 0.5,
        ]
    }

    #[inline]
    pub fn to_index(self, bits: Int3) -> usize {
        (self.x | ((self.y | ((self.z) << bits.y)) << bits.x)) as usize
    }

    #[inline]
    pub fn from_index(bits: Int3, idx: usize) -> Self {
        let idx = idx as i32;
        Self::new([
            idx & ((1 << bits.x) - 1),
            (idx >> bits.x) & ((1 << bits.y) - 1),
            (idx >> (bits.x + bits.y)) & ((1 << bits.z) - 1),
        ])
    }

    #[inline]
    pub fn is_within(self, size: Int3) -> bool {
        (self.x as u32) < (size.x as u32)
            && (self.y as u32) < (size.y as u32)
            && (self.z as u32) < (size.z as u32)
    }

    #[inline]
    pub fn xy(self) -> Int2 {
        Int2::new([self.x, self.y])
    }

    #[inline]
    pub fn mag_sq(self) -> i32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
}

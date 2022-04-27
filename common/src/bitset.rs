use crate::prelude::*;

#[macro_export]
macro_rules! bit_array {
    (@ [$($pub:tt)?] $name:ident $bits:expr) => {
        $($pub)? struct $name(pub [usize; Self::WORDS]);
        impl $name {
            #[allow(dead_code)]
            const WORD_BITS: usize = mem::size_of::<usize>() * 8;
            #[allow(dead_code)]
            const BITS: usize = $bits;
            #[allow(dead_code)]
            const WORDS: usize = (Self::BITS + Self::WORD_BITS - 1) / Self::WORD_BITS;

            #[allow(dead_code)]
            #[inline]
            pub fn new() -> Self {
                Self([0; Self::WORDS])
            }

            #[allow(dead_code)]
            #[inline]
            pub fn clear(&mut self) {
                self.0.fill(0);
            }

            #[allow(dead_code)]
            #[inline]
            pub fn get(&self, idx: usize) -> bool {
                (self.0[idx / Self::WORD_BITS] >> (idx % Self::WORD_BITS)) & 1 != 0
            }

            #[allow(dead_code)]
            #[inline]
            pub fn set(&mut self, idx: usize) {
                self.0[idx / Self::WORD_BITS] |= 1 << (idx % Self::WORD_BITS);
            }

            #[allow(dead_code)]
            #[inline]
            pub fn unset(&mut self, idx: usize) {
                self.0[idx / Self::WORD_BITS] &= !(1 << (idx % Self::WORD_BITS));
            }

            #[allow(dead_code)]
            #[inline]
            pub fn set_row(&mut self, idx: usize, len: usize) {
                let mut word = idx / Self::WORD_BITS;
                let mut subword = idx % Self::WORD_BITS;
                let mut rem = len;
                loop {
                    let fill = Self::WORD_BITS - subword;
                    if rem >= fill {
                        self.0[word] |= usize::MAX << subword;
                        word += 1;
                        subword = 0;
                        rem -= fill;
                    } else {
                        self.0[word] |= !(usize::MAX << rem) << subword;
                        break;
                    }
                }
            }
        }
    };

    (pub $name:ident($len:expr);) => {
        bit_array! {
            @ [pub] $name $len
        }
    };
    ($name:ident($len:expr);) => {
        bit_array! {
            @ [] $name $len
        }
    };
}

bit_array! {
    ExampleBits(100);
}

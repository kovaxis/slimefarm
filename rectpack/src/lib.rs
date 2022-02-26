//! Pack small rectangles into a larger one. This is useful for creating texture atlases for the efficient GPU rendering.
//!
//! Usage example:
//!
//! ```
//! use rect_packer::Packer;
//!
//! let config = rect_packer::Config {
//!     width: 1024,
//!     height: 1024,
//!
//!     border_padding: 5,
//!     rectangle_padding: 10,
//! };
//!
//! let rectangles = [(50, 70), (350, 210), (255, 410)];
//!
//! let mut packer = Packer::new(config);
//! for &(width, height) in &rectangles {
//!     if let Some(rect) = packer.pack(width, height, false) {
//!         println!("Rectangle is at position ({}, {}) within the encompassing rectangle",
//!             rect.x,
//!             rect.y);
//!     }
//! }
//!

pub use packer::DensePacker;
pub use packer::Packer;
pub use rect::Rect;

mod packer;
mod rect;

/// Describes size and padding requirements of rectangle packing.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Config {
    /// Width of the encompassing rectangle.
    pub width: i32,
    /// Height of the encompassing rectangle.
    pub height: i32,

    /// Minimum spacing between border and rectangles.
    pub border_padding: i32,
    /// Minimum spacing between rectangles.
    pub rectangle_padding: i32,
}

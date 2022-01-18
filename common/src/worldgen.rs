//! Current worldgen plans:
//!
//! The process of generating a chunk in no moment processes multiple chunks simultaneously, it
//! always processes one chunk at a time. This means that structures that span multiple chunks
//! must be able to be "resumed": generating parts of them in separate chunk. In the worst case,
//! this entails generating a structure as many times as the chunks it spans.
//!
//! However, it comes in handy sometimes to use information from neighbouring chunks in order to
//! determine the generation of the current chunk. For example, when generating trees in a chunk
//! we need to know which trees touch this chunk. This entails checking whether a tree seed occurs
//! in a neighboring chunk. However, determining where exactly do seeds go benefits from knowing the
//! structure of terrain, so ideally before checking where to place seeds in a chunk, the terrain of
//! that chunk has to be generated. Therefore, to generate trees in a chunk, terrain has to be
//! generated in the neigboring chunks.
//!
//! Additionally, imagine that we wanted to generate ponds and trees in a forest.
//! When generating a chunk, we need to know whether a pond or a tree touches this chunk. To know
//! that, tree seeds and pond seeds must be placed in the 3x3x3 matrix of chunks centered in this
//! chunk. As said above, we want seed placement in a chunk to have information about the chunk
//! terrain, so first the terrain must be generated in this 3x3x3 matrix. Then comes the problem of
//! avoiding pond-tree collisions.
//!
//! The problem is defined as so: we want to place pond and tree seeds in a chunk, but we want to
//! make sure that around a certain radius of a pond seed there are no tree seeds. This radius might
//! depend on the specific pond and tree seed (some ponds are larger, some trees are larger).
//!
//! A naive solution might be to place a restriction of pond seeding to occur before tree seeding,
//! since then the tree seeding process would have full access to pond seed information. However,
//! when determining where to place trees to satisfy the restrictions, we need to know about all
//! pond seeds in the 3x3x3 matrix around the currently-seeding chunk. Remember that when generating
//! the tree blocks in a chunk, we need to know about all tree seeds in a 3x3x3 matrix. If each of
//! these chunk seeds requires in turn 3x3x3 knowledge about pond seeds, and each pond seed requires
//! 1x1x1 knowledge about terrain, then to generate a chunk we would need all 5x5x5 terrain to be
//! generated. If more dependencies are established, this radius grows further, which is not ideal.
//!
//! Instead, pond seeding and tree seeding could happen independently, allowing ponds and trees to
//! collide. However, when generating the actual blocks of a tree, the seed is tested for rejection.
//! If there is a pond seed close enough to the tree seed, then the seed is rejected, and no tree
//! is built. Under this model, pond and tree seed placement requires 1x1x1 terrain knowledge only,
//! pond building requires 3x3x3 pond seed knowledge and tree building requires 3x3x3 pond and tree
//! seed information. This means that to generate a chunk only a 3x3x3 matrix is required, which no
//! further scaling, which is an acceptable distance.
//!
//! There is a limitation, though. When assuming that tree seed rejection only needs information
//! about the 3x3x3 pond seed matrix, we assume that a pond seed that is at least 1 chunk away from
//! a tree seed cannot affect it. This "locality" is a necessary feature to build a procedural
//! world. However, in some cases, long-range interactions are desired and necessary, for example
//! when building a large temple. When building a large temple, we would like for trees and ponds
//! to be suppressed in the range of the temple, of course. This means that, for example, temple
//! seeding cannot use information about the terrain, or otherwise enormous amounts of terrain
//! would have to be generated just to determine whether a temple touches a given chunk or not.
//!
//! To solve this problem, we use "tentative tiered seeds". To distribute objects and structures
//! around the world in a way that prevents collisions, gives a semi-constant density and allows
//! large objects, a universal "object seed mesh" is placed on the world. For example, for
//! tree-sized objects, a voronoi-style uniform distribution of locations is generated, such that
//! even if all tree-sized locations had a tree generated in it, tree collisions would be minimal
//! or none. Note that a "location" does not imply concrete coordinates. For example, in a typical
//! 2.5D world, locations would consist only of horizontal coordinates, with no vertical
//! information, because determining the vertical location of a coordinate would require terrain
//! information. This means that each location has the shape of an infinitely tall and deep column.
//! Apart from these tree-sized locations, one would like to place larger structures. Say, something
//! in between a tree and a temple, something like a house. Therefore, a new set of locations can
//! be generated, which are house-sized locations. Let's call tree-sized locations "0-locations",
//! and house-sized locations "1-locations". Even if houses were generated in all 1-locations, they
//! would not collide. However, some 1-locations might have houses, some may not, and each house
//! might have a different radius within it suppresses 0-locations. However, this information must
//! be able to be determined cheaply, _without_ any dependencies on terrain. By using this
//! information, tree placement on 0-locations can check whether it collides with a house, and
//! therefore trees can be suppressed: if a tree is placed on a 0-location, but the sum of its
//! radius and a neighboring 1-location is larger that their distance, then the tree is suppresed.
//! This means that first, 1-locations have to be filled. Afterwards, 0-locations can be filled and
//! rejected. Afterwards, when generating a chunk, the blocks corresponding to nearby 0-locations
//! are generated, and the blocks corresponding to nearby 1-locations are generated. Of course, this
//! 0-location/1-location scheme can be extended to any number of location tiers, with m-locations
//! having to check all n-locations, with n > m. Since this process can be lengthy for lots of
//! low-locations which have to check several high-locations, when a high-location is placed entire
//! chunks can be marked as having no low-locations, and entire chunks can be marked as far away
//! from this high-location (information about distance to higher-locations is similar for nearby
//! low-locations).
//!
//! However, there is a better solution that skips these performance issues and deals with irregular
//! structures (eg. irregular temples in a jungle, where we want to suppress the generation of
//! trees in contact with the temple, but we do not want to create a "clear area" around the
//! temple). Instead of using distances between locations to determine whether they overlap,
//! instead create an "occupation map". The world is divided into "cells", each with a shape
//! typically like an infinite column, various blocks wide and long (2x2 or 4x4 blocks), and which
//! can be occupied or free. Whenever a structure is placed on a location, the cells that it
//! occupies are computed, and they are marked as occupied. Afterwards, whenever a lower-rank
//! location structure is placed, it can check against the cells that it occupies to know whether
//! it should generate or not.
//!
//! Take a concrete example of temple, lagoon and tree generation, (2-locations, 1-locations,
//! 0-locations). Let's say that 90% of 2-locations contain a temple, in a uniformly random way.
//! Note that choosing whether a location contains a temple (or what kind of temple) can depend on
//! any number of factors, specifically the biome, or, if a small 1x1 terrain dependency is
//! introduced, or otherwise heightmaps are used, on altitude. Let's have a humidity noise variable
//! indicate whether a 1-locations contain lagoons, so as to cluster lagoons together. Additionally,
//! let's use humidity and a secondary noise variable to choose the percentage of 0-locations that
//! contain trees, and their size, so as to place clustered trees, and ensure that all lagoons have
//! trees around them. When building a chunk, we need to make sure that all in-range 2-locations are
//! placed. The above placement strategy takes place. Afterwards, we need to make sure that all
//! in-range 1-locations are placed. The above placement strategy takes place, and each 1-location
//! to be placed recursively makes sure that all 2-locations in range are properly placed, and
//! rejects 1-locations according to the occupation map, which was filled by 2-locations in the
//! previous step. Afterwards, we need to make sure that all 0-locations that are in range of the
//! chunk to be generated are placed. 0-locations recursively place any 2-locations in range, and
//! check for overlaps against them using the occupation map. They also recursively place any
//! 1-locations in range, and check for overlaps. Note that any 1-locations that crash against a
//! 2-location are rejected _before_ 0-locations check for overlap, so any lagoon that is suppressed
//! by a temple will have its void filled by trees. After all in-range 0, 1, and 2-locations have
//! been placed, we can finally generate the chunk. We go from smaller to larger.
//!
//! First, all 0-location trees are expanded, in a well-defined order. When expanding a seed, we
//! need to know the exact layout of the structure, at least the layout that collides with this
//! chunk. This means that when placing a tree seed the tree parameters need to be immediately
//! calculated. Using these parameters, the tree trunk and foliage can be determined. Let's say that
//! the tree trunk is based on a perturbed line that goes from ground to a tree center height, at
//! which height it branches off in different directions, going a certain direction, spawning new
//! branches in slightly perturbed directions and so on recursively for a set depth/distance. After
//! this process is done, the 2D projected shape of the tree can be known, and so is the shape of
//! the tree trunk. Since we don't care about the foliage overlapping with other structures (eg. We
//! don't care about the foliage hanging over a lagoon. Although we _do_ care that the foliage does
//! not go into a temple, this will be enforced by the temple itself, since it will overwrite its
//! insides with air. Trees might be very close to the temple, but that is normal. Trees might even
//! go through the temple, but whatever). This means that the tree only checks for the trunk cells
//! to be unused, and only marks these as used too. After these are marked, the tree position is
//! determined, and for this terrain information is needed. The surface height is generated (to do
//! this terrain generation has to be done in the given chunk column). The tree position is
//! determined, and now the tree blocks are set in stone, although they haven't been actually placed
//! yet. This means that the tree shape is determined without looking at the actual blocks in the
//! location, meaning that the tree might overlap terrain, which it will overwrite. Afterwards, when
//! generating a chunk the tree is built in-place, overwriting any terrain.
//!
//! Afterwards, lagoons are generated. In a similar manner to trees, lagoons generate a
//! 3-dimensional lagoon shape upon placement, which is set in stone as soon as it is placed,
//! without looking at the surrounding terrain. Since we want the lagoon shape to be kind of
//! irregular, we use smooth perlin noise added with a radial parameter, so as to limit the diameter
//! of the lagoon. Since there is a maximum radius for 1-locations, if the lagoon shape goes beyond
//! the maximum radius it will have to be truncated. When seeding, occupation cells are filled
//! according to the lagoon shape, and these very same cells are checked for occupation. If they
//! are already occupied, the lagoon bails out from generation, marks no cells and leaves this
//! location unfilled. When placing the seed, or otherwise when first expanding the seed (lazy
//! computation), the lagoon generates the terrain at its seed, and gets the surface height of the
//! terrain at its center. Using this information, it sets the water level to be a few blocks below
//! this point, assuming that the inclination of the terrain around it is small enough (terrain
//! generation in lagoon-filled biomes must be quite flat, which makes sense). When actually
//! generating a chunk that is touched by a lagoon, any ground blocks which are touched by the
//! lagoon are replaced by water or air (depending on whether it is above or below the water level).
//!
//! Finally, temples are generated. These are the most complex structures, since they must provide
//! a fun and challenging maze within them, with complex enough layout that makes them fun to
//! navigate. To generate a temple, rooms are placed with connections between them. A good and
//! simple temple layout might be a rectangular grid, with a center room that contains the altar,
//! boss and treasure, and surrounding rooms that contain minions to fight with to reach the center.
//! To connect the rooms, the rooms can be considered as nodes, and the walls between them as edges.
//! To create connections, a spanning tree is generated by picking a random edge between
//! disconnected rooms, such that at the end the entire dungeon is a tree. To prevent quick paths
//! between the center and the entrance, connections between rooms at the same "level" might be
//! given higher chance to be chosen before connections between rooms at different distances from
//! the center. Additionally, rooms at the center might be higher in order to give an aesthetic
//! view from the outside. Once done, the room that is farthest away from the central room and in
//! contact with the outside is picked as the entrace, a door direction is picked. Once the layout
//! is set in stone (although not built yet), the corresponding occupation cells are marked. Note
//! that because this is the highest-level location rank, 2-locations do not have to check whether
//! their cells are occupied, since they are always unoccupied (this is because 2-locations are
//! placed far apart enough so that structure with a certain radius cannot collide with each other).
//! When actually building the temple, first the height level of the temple is determined. For this,
//! the door terrain must be generated, and the entrance height is sampled. Afterwards, the rooms
//! that make contact with the currently-generating chunk are placed, and any entities are placed
//! in it.
//!
//! A few notes:
//! - Generating a chunk entails determining all locations that could touch it. Because locations
//!     have a fixed maximum radius, this can be done quickly.
//! - Note that structures have state that is shared among multiple chunk generations. There are
//!     also actions like "request a certain structure to place itself here", which means that
//!     structures are classes in and of itself, with associated state and behaviour.
//! - Structures must have an action "generate any of your blocks that go into this chunk".
//!     Determining exactly which blocks go into an arbitrary chunk may be difficult, so structures
//!     may use intermediate buffers, where they generate the structure into a buffer and then copy
//!     it into a chunk.
//! - Generating a chunk entails placing all structures within a fixed radius. Recursively, placing
//!     a structure requires placing nearby higher-rank structures, so as to carry out overlapping.
//!     Placing a 0-location requires placing all 1-locations in (1-range + 0-range) range,
//!     and placing each of those 1-locations requires placing all 2-locations in
//!     (2-range + 2 * 1-range + 0-range) range. If location ranges are chosen to be powers
//!     of two, then generating a chunk requires all k-locations in at most 3 * k-range range to
//!     be generated. Note however, that since k-locations are spread at least 2 * k-range apart,
//!     generating a point requires checking up to about 9 k-locations for each k. A chunk is of
//!     course not a point, so this amount might go up for lower-tier locations. However, the amount
//!     of locations to check is bounded. Additionally, when generating adjacent chunks most
//!     locations will already have been checked.
//! - The assignment of color to every block type is done globally, with every block type obtaining
//!     a separate color scheme. Note that a single block type does not have a uniform color, only
//!     a single _color scheme_. For example, grass blocks might have a color that depends on a
//!     perlin variable such as humidity or temperature, stone bricks might have sharp variations
//!     of color, with darker or mossy bricks sprinkled randomly around, etc...
//! - OPTIMIZE: For each chunk save a bit indicating whether the entire chunk is made out of the
//!     same block. Do this recursively in an octree.

use crate::{prelude::*, terrain::GridKeeper2d};

pub trait GenStore {
    fn register_raw(&self, name: &[u8], obj: [usize; 2], destroy: unsafe fn([usize; 2]));
    fn lookup_raw(&self, name: &[u8]) -> Option<[usize; 2]>;

    fn listen_raw(&self, name: &[u8], listener: Box<dyn FnMut(*const u8)>);
    unsafe fn trigger_raw(&self, name: &[u8], args: *const u8);
}
impl dyn GenStore {
    pub fn register<T: ?Sized>(&self, name: &str, t: Box<T>) {
        unsafe fn destroy<T: ?Sized>(obj: [usize; 2]) {
            let ptr = mem::transmute_copy::<[usize; 2], *mut T>(&obj);
            drop(Box::from_raw(ptr));
        }
        let ptr = Box::into_raw(t);
        let mut obj = [0usize; 2];
        unsafe {
            (&mut obj as *mut [usize; 2] as *mut *mut T).write(ptr);
        }
        self.register_raw(name.as_bytes(), obj, destroy::<T>)
    }

    pub unsafe fn lookup<T: ?Sized>(&self, name: &str) -> &'static T {
        let obj = match self.lookup_raw(name.as_bytes()) {
            Some(obj) => obj,
            None => panic!("gencapsule \"{}\" not found", name),
        };
        let ptr = mem::transmute_copy::<[usize; 2], *mut T>(&obj);
        &*ptr
    }

    pub unsafe fn listen<A, F>(&self, name: &str, mut listener: F)
    where
        F: FnMut(&A) + 'static,
    {
        self.listen_raw(
            name.as_bytes(),
            Box::new(move |a| listener(&*(a as *const A))),
        )
    }

    pub unsafe fn trigger<A>(&self, name: &str, args: &A) {
        self.trigger_raw(name.as_bytes(), args as *const A as *const u8)
    }
}

pub trait ChunkFiller {
    fn fill(&self, pos: ChunkPos) -> Option<ChunkBox>;
}

/// Divides a chunk into 8x8 equal pieces, assigning each piece a single bit.
/// This means that the entire chunk can be stored in a single `u64`.
#[derive(Default)]
pub struct OccupChunk {
    bits: u64,
}
impl OccupChunk {
    pub fn is_occup(&self, [x, y]: [i32; 2]) -> bool {
        (self.bits >> (x + y * 8)) & 1 != 0
    }

    pub fn set_occup(&mut self, [x, y]: [i32; 2]) {
        self.bits |= 1 << (x + y * 8);
    }
}

pub struct OccupMap {
    map: RefCell<GridKeeper2d<OccupChunk>>,
}
impl OccupMap {
    pub fn new(size: i32) -> Self {
        Self {
            map: GridKeeper2d::new(size, [0, 0]).into(),
        }
    }

    pub fn set_center(&self, center: [i32; 2]) {
        self.map.borrow_mut().set_center(center);
    }

    fn decompose(&self, pos: [i32; 2]) -> ([i32; 2], [i32; 2]) {
        let cnkpos = [pos[0].div_euclid(CHUNK_SIZE), pos[1].div_euclid(CHUNK_SIZE)];
        let subpos = [
            pos[0].rem_euclid(CHUNK_SIZE) / (CHUNK_SIZE / 8),
            pos[1].rem_euclid(CHUNK_SIZE) / (CHUNK_SIZE / 8),
        ];
        (cnkpos, subpos)
    }

    pub fn is_occup(&self, pos: [i32; 2]) -> bool {
        let (cnkpos, subpos) = self.decompose(pos);
        match self.map.borrow().get(cnkpos) {
            Some(cnk) => cnk.is_occup(subpos),
            None => false,
        }
    }

    pub fn set_occup(&self, pos: [i32; 2]) {
        let (cnkpos, subpos) = self.decompose(pos);
        match self.map.borrow_mut().get_mut(cnkpos) {
            Some(cnk) => cnk.set_occup(subpos),
            None => (),
        }
    }
}

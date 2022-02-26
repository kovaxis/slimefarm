use crate::prelude::*;

#[derive(Clone, Debug)]
pub(crate) struct Mesh<V, I = VertIdx> {
    pub vertices: Vec<V>,
    pub indices: Vec<I>,
}

impl<V, I> Default for Mesh<V, I> {
    fn default() -> Self {
        Mesh {
            vertices: default(),
            indices: default(),
        }
    }
}

fn vert(pos: Vec3, normal: [i8; 3], color: [u8; 4]) -> SimpleVertex {
    SimpleVertex {
        pos: pos.into(),
        normal: [normal[0], normal[1], normal[2], 0],
        color: color,
    }
}

pub trait MeshIndex {
    const MAX: usize;
    fn from_usize(x: usize) -> Self;
}
impl MeshIndex for u8 {
    const MAX: usize = u8::MAX as usize;
    #[inline]
    fn from_usize(x: usize) -> u8 {
        x as u8
    }
}
impl MeshIndex for u16 {
    const MAX: usize = u16::MAX as usize;
    #[inline]
    fn from_usize(x: usize) -> u16 {
        x as u16
    }
}
impl MeshIndex for u32 {
    const MAX: usize = u32::MAX as usize;
    #[inline]
    fn from_usize(x: usize) -> u32 {
        x as u32
    }
}

impl<V, I> Mesh<V, I> {
    pub fn with_capacity(verts: usize, faces: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(verts),
            indices: Vec::with_capacity(faces * 3),
        }
    }

    /// Remove all vertices and faces.
    pub fn _clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    /// Add a single triangular face.
    pub fn add_face(&mut self, i0: I, i1: I, i2: I) {
        self.indices.push(i0);
        self.indices.push(i1);
        self.indices.push(i2);
    }
}
impl<V: glium::Vertex, I: glium::index::Index> Mesh<V, I> {
    /// Upload mesh to GPU.
    pub fn make_buffer<F: glium::backend::Facade + ?Sized>(&self, display: &F) -> GpuBuffer<V, I> {
        if self.vertices.len() > VertIdx::MAX as usize {
            eprintln!(
                "over {} vertices in mesh! graphic glitches may occur",
                VertIdx::MAX
            );
        }
        GpuBuffer {
            vertex: VertexBuffer::immutable(display, &self.vertices).unwrap(),
            index: IndexBuffer::immutable(display, PrimitiveType::TrianglesList, &self.indices)
                .unwrap(),
        }
    }
}
impl<V, I: MeshIndex> Mesh<V, I> {
    /// Add a single point.
    pub fn add_vertex(&mut self, v: V) -> I {
        let idx = I::from_usize(self.vertices.len());
        self.vertices.push(v);
        idx
    }
}
impl Mesh<SimpleVertex> {
    /// Add a single vertex and return its index.
    pub fn add_vertex_simple(&mut self, v: Vec3, normal: [i8; 3], color: [u8; 4]) -> VertIdx {
        let idx = self.vertices.len() as VertIdx;
        self.vertices.push(vert(v, normal, color));
        idx
    }
}

pub(crate) struct RawBufPackage<V, I: glium::index::Index = VertIdx> {
    vert: RawVertexPackage<V>,
    idx: RawIndexPackage<I>,
}
impl<V: glium::Vertex> RawBufPackage<V> {
    pub(crate) fn pack(buf: GpuBuffer<V>) -> Self {
        Self {
            vert: RawVertexPackage::pack(buf.vertex),
            idx: RawIndexPackage::pack(buf.index),
        }
    }

    pub(crate) unsafe fn unpack(self, display: &Display) -> GpuBuffer<V> {
        GpuBuffer {
            vertex: self.vert.unpack(display),
            index: self.idx.unpack(display),
        }
    }
}

use crate::prelude::*;

#[derive(Clone, Debug)]
pub(crate) struct Mesh<V = SimpleVertex> {
    pub vertices: Vec<V>,
    pub indices: Vec<VertIdx>,
}

impl<V> Default for Mesh<V> {
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

impl<V> Mesh<V> {
    /// Remove all vertices and faces.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    /// Add a single triangular face.
    pub fn add_face(&mut self, i0: VertIdx, i1: VertIdx, i2: VertIdx) {
        self.indices.push(i0);
        self.indices.push(i1);
        self.indices.push(i2);
    }
}
impl Mesh<SimpleVertex> {
    /// Add a single vertex and return its index.
    pub fn add_vertex(&mut self, v: Vec3, normal: [i8; 3], color: [u8; 4]) -> VertIdx {
        let idx = self.vertices.len() as VertIdx;
        self.vertices.push(vert(v, normal, color));
        idx
    }

    /// Upload mesh to GPU.
    pub fn make_buffer<F: glium::backend::Facade + ?Sized>(&self, display: &F) -> Buffer3d {
        if self.vertices.len() > u16::MAX as usize {
            eprintln!(
                "over {} vertices in mesh! graphic glitches may occur",
                u16::MAX
            );
        }
        Buffer3d {
            vertex: VertexBuffer::immutable(display, &self.vertices).unwrap(),
            index: IndexBuffer::immutable(display, PrimitiveType::TrianglesList, &self.indices)
                .unwrap(),
        }
    }
}

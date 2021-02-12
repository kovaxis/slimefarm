use crate::prelude::*;

#[derive(Clone, Default, Debug)]
pub(crate) struct Mesh {
    pub vertices: Vec<SimpleVertex>,
    pub indices: Vec<u16>,
}

fn vert(pos: Vec3, color: [u8; 4]) -> SimpleVertex {
    SimpleVertex {
        pos: pos.into(),
        color,
    }
}

impl Mesh {
    /// Should receive vertices in counter-clockwise order.
    pub fn add_quad(&mut self, v0: Vec3, v1: Vec3, v2: Vec3, v3: Vec3, color: [u8; 4]) {
        let v = self.vertices.len();
        self.vertices.push(vert(v0, color));
        self.vertices.push(vert(v1, color));
        self.vertices.push(vert(v2, color));
        self.vertices.push(vert(v3, color));
        let v0 = v as u16;
        let v1 = v as u16 + 1;
        let v2 = v as u16 + 2;
        let v3 = v as u16 + 3;
        self.add_face(v0, v1, v2);
        self.add_face(v2, v3, v0);
    }

    /// Add a single triangular face.
    pub fn add_face(&mut self, i0: u16, i1: u16, i2: u16) {
        self.indices.push(i0);
        self.indices.push(i1);
        self.indices.push(i2);
    }

    /// Upload mesh to GPU.
    pub fn make_buffer(&self, display: &Display) -> Buffer3d {
        Buffer3d {
            vertex: VertexBuffer::new(display, &self.vertices).unwrap(),
            index: IndexBuffer::new(display, PrimitiveType::TrianglesList, &self.indices).unwrap(),
        }
    }
}

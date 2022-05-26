use crate::prelude::*;

pub struct VoxelModel {
    pub palette: Box<[[u8; 4]; 256]>,
    size: Int3,
    data: Vec<u8>,
}
impl VoxelModel {
    pub fn size(&self) -> Int3 {
        self.size
    }
    pub fn data(&self) -> &[u8] {
        &self.data[..]
    }
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data[..]
    }

    pub fn find(&self, find: &[bool; 256]) -> [Vec<Int3>; 256] {
        let mut idx = 0;
        let mut out = arr![Vec::new(); 256];
        for z in 0..self.size.z {
            for y in 0..self.size.y {
                for x in 0..self.size.x {
                    let v = self.data[idx];
                    if find[v as usize] {
                        out[v as usize].push(Int3::new([x, y, z]));
                    }
                    idx += 1;
                }
            }
        }
        out
    }
}

fn parse_model(
    vox: &dot_vox::DotVoxData,
    model: dot_vox::Model,
    shininess: u8,
) -> Result<VoxelModel> {
    let size = model.size;
    let size = Int3::new([size.x as i32, size.y as i32, size.z as i32]);
    let mut data = vec![0; (size.x * size.y * size.z) as usize];
    let mut palette = Box::new([[0; 4]; 256]);
    for (i, color) in vox.palette.iter().enumerate() {
        let idx = (i as u8).wrapping_add(1);
        palette[idx as usize] = color.to_le_bytes();
        palette[idx as usize][3] = shininess;
    }
    for v in model.voxels {
        ensure!(
            (v.x as i32) < size.x && (v.y as i32) < size.y && (v.z as i32) < size.z,
            "voxel out of range"
        );
        let i = v.i + 1;
        data[(v.x as i32 + size.x * (v.y as i32 + size.y * v.z as i32)) as usize] = i;
    }
    Ok(VoxelModel {
        palette,
        size,
        data,
    })
}

pub fn load_vox(bytes: &[u8], shininess: u8) -> Result<Vec<VoxelModel>> {
    let mut vox = dot_vox::load_bytes(bytes)
        .map_err(|e| anyhow!(e))
        .with_context(|| anyhow!("failed to load .vox file"))?;
    let models = mem::take(&mut vox.models);
    let models = models
        .into_iter()
        .map(|m| parse_model(&vox, m, shininess))
        .collect::<Result<_>>()?;
    Ok(models)
}

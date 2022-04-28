use crate::prelude::*;

fn parse_model(vox: &dot_vox::DotVoxData, model: dot_vox::Model) -> Result<(Vec<[u8; 4]>, Int3)> {
    let size = model.size;
    let size = Int3::new([size.x as i32, size.y as i32, size.z as i32]);
    let mut data = vec![[0; 4]; (size.x * size.y * size.z) as usize];
    for v in model.voxels {
        ensure!(
            (v.x as i32) < size.x && (v.y as i32) < size.y && (v.z as i32) < size.z,
            "voxel out of range"
        );
        ensure!(
            (v.i as usize) < vox.palette.len(),
            "voxel color out of range"
        );
        data[(v.x as i32 + size.x * (v.y as i32 + size.y * v.z as i32)) as usize] =
            vox.palette[v.i as usize].to_le_bytes();
    }
    Ok((data, size))
}

pub fn load_vox(bytes: &[u8]) -> Result<Vec<(Vec<[u8; 4]>, Int3)>> {
    let mut vox = dot_vox::load_bytes(bytes)
        .map_err(|e| anyhow!(e))
        .with_context(|| anyhow!("failed to load .vox file"))?;
    let models = mem::take(&mut vox.models);
    let models = models
        .into_iter()
        .map(|m| parse_model(&vox, m))
        .collect::<Result<_>>()?;
    Ok(models)
}

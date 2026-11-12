//! Export posed figures as static meshes (GLB with embedded textures, or OBJ + MTL).
use anyhow::{Context, Result};
use glam::{Vec2, Vec3};
use poser_core::model::{Material, TextureData};
use serde_json::{Value, json};
use std::io::Write;
use std::path::Path;

pub struct ExportMesh<'a> {
    pub name: String,
    /// World-space, already posed.
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: &'a [Vec2],
    pub indices: &'a [u32],
    pub material: &'a Material,
    pub texture: Option<&'a TextureData>,
}

fn png_bytes(t: &TextureData) -> Result<Vec<u8>> {
    let img = image::RgbaImage::from_raw(t.width, t.height, t.rgba.to_vec()).context("bad texture")?;
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png)?;
    Ok(out.into_inner())
}

fn pad4(v: &mut Vec<u8>, byte: u8) {
    while v.len() % 4 != 0 {
        v.push(byte);
    }
}

pub fn write_glb(path: &Path, meshes: &[ExportMesh]) -> Result<()> {
    let mut bin: Vec<u8> = vec![];
    let mut views: Vec<Value> = vec![];
    let mut accessors: Vec<Value> = vec![];
    let mut add_view = |bin: &mut Vec<u8>, data: &[u8], target: Option<u32>| -> usize {
        pad4(bin, 0);
        let off = bin.len();
        bin.extend_from_slice(data);
        let mut v = json!({"buffer": 0, "byteOffset": off, "byteLength": data.len()});
        if let Some(t) = target {
            v["target"] = json!(t);
        }
        views.push(v);
        views.len() - 1
    };
    let mut gl_meshes = vec![];
    let mut nodes = vec![];
    let mut materials = vec![];
    let mut textures = vec![];
    let mut images = vec![];
    let mut tex_cache: Vec<(*const TextureData, usize)> = vec![];
    for (mi, m) in meshes.iter().enumerate() {
        let n = m.positions.len();
        if n == 0 || m.indices.is_empty() {
            continue;
        }
        let (lo, hi) = m.positions.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), p| (a.min(*p), b.max(*p)));
        let pv = add_view(&mut bin, bytemuck::cast_slice(&m.positions), Some(34962));
        accessors.push(json!({"bufferView": pv, "componentType": 5126, "count": n, "type": "VEC3", "min": [lo.x, lo.y, lo.z], "max": [hi.x, hi.y, hi.z]}));
        let pa = accessors.len() - 1;
        let nv = add_view(&mut bin, bytemuck::cast_slice(&m.normals), Some(34962));
        accessors.push(json!({"bufferView": nv, "componentType": 5126, "count": n, "type": "VEC3"}));
        let na = accessors.len() - 1;
        let mut attrs = json!({"POSITION": pa, "NORMAL": na});
        if m.uvs.len() == n {
            let uv = add_view(&mut bin, bytemuck::cast_slice(m.uvs), Some(34962));
            accessors.push(json!({"bufferView": uv, "componentType": 5126, "count": n, "type": "VEC2"}));
            attrs["TEXCOORD_0"] = json!(accessors.len() - 1);
        }
        let iv = add_view(&mut bin, bytemuck::cast_slice(m.indices), Some(34963));
        accessors.push(json!({"bufferView": iv, "componentType": 5125, "count": m.indices.len(), "type": "SCALAR"}));
        let ia = accessors.len() - 1;
        let c = m.material.base_color;
        let mut mat = json!({
            "name": m.material.name,
            "pbrMetallicRoughness": {"baseColorFactor": [c.x, c.y, c.z, c.w], "metallicFactor": 0.0, "roughnessFactor": 0.8},
            "doubleSided": true,
        });
        if let Some(t) = m.texture {
            let key = t as *const TextureData;
            let ti = match tex_cache.iter().find(|(k, _)| *k == key) {
                Some((_, i)) => *i,
                None => {
                    let png = png_bytes(t)?;
                    let v = add_view(&mut bin, &png, None);
                    images.push(json!({"bufferView": v, "mimeType": "image/png"}));
                    textures.push(json!({"source": images.len() - 1, "sampler": 0}));
                    tex_cache.push((key, textures.len() - 1));
                    textures.len() - 1
                }
            };
            mat["pbrMetallicRoughness"]["baseColorTexture"] = json!({"index": ti});
        }
        materials.push(mat);
        gl_meshes.push(json!({"name": m.name, "primitives": [{"attributes": attrs, "indices": ia, "material": materials.len() - 1}]}));
        nodes.push(json!({"name": if m.name.is_empty() { format!("mesh{mi}") } else { m.name.clone() }, "mesh": gl_meshes.len() - 1}));
    }
    pad4(&mut bin, 0);
    let mut doc = json!({
        "asset": {"version": "2.0", "generator": "3D Model Poser"},
        "scene": 0,
        "scenes": [{"nodes": (0..nodes.len()).collect::<Vec<_>>()}],
        "nodes": nodes,
        "meshes": gl_meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": bin.len()}],
    });
    if !images.is_empty() {
        doc["images"] = json!(images);
        doc["textures"] = json!(textures);
        doc["samplers"] = json!([{"magFilter": 9729, "minFilter": 9729}]);
    }
    let mut js = serde_json::to_vec(&doc)?;
    pad4(&mut js, b' ');
    let total = 12 + 8 + js.len() + 8 + bin.len();
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"glTF")?;
    f.write_all(&2u32.to_le_bytes())?;
    f.write_all(&(total as u32).to_le_bytes())?;
    f.write_all(&(js.len() as u32).to_le_bytes())?;
    f.write_all(b"JSON")?;
    f.write_all(&js)?;
    f.write_all(&(bin.len() as u32).to_le_bytes())?;
    f.write_all(b"BIN\0")?;
    f.write_all(&bin)?;
    Ok(())
}

pub fn write_obj(path: &Path, meshes: &[ExportMesh]) -> Result<()> {
    let stem = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "pose".into());
    let dir = path.parent().unwrap_or(Path::new("."));
    let mtl_name = format!("{stem}.mtl");
    let mut obj = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut mtl = std::io::BufWriter::new(std::fs::File::create(dir.join(&mtl_name))?);
    writeln!(obj, "# 3D Model Poser export\nmtllib {mtl_name}")?;
    let mut base = 1usize;
    let mut vt_base = 1usize;
    let mut tex_written: Vec<(*const TextureData, String)> = vec![];
    for (mi, m) in meshes.iter().enumerate() {
        let mname = format!("m{mi}_{}", m.material.name.replace(' ', "_"));
        let c = m.material.base_color;
        writeln!(mtl, "newmtl {mname}\nKd {} {} {}\nd {}", c.x, c.y, c.z, c.w)?;
        if let Some(t) = m.texture {
            let key = t as *const TextureData;
            let file = match tex_written.iter().find(|(k, _)| *k == key) {
                Some((_, f)) => f.clone(),
                None => {
                    let f = format!("{stem}_tex{}.png", tex_written.len());
                    std::fs::write(dir.join(&f), png_bytes(t)?)?;
                    tex_written.push((key, f.clone()));
                    f
                }
            };
            writeln!(mtl, "map_Kd {file}")?;
        }
        writeln!(obj, "o {}\nusemtl {mname}", if m.name.is_empty() { format!("mesh{mi}") } else { m.name.replace(' ', "_") })?;
        for p in &m.positions {
            writeln!(obj, "v {} {} {}", p.x, p.y, p.z)?;
        }
        for n in &m.normals {
            writeln!(obj, "vn {} {} {}", n.x, n.y, n.z)?;
        }
        let has_uv = m.uvs.len() == m.positions.len();
        if has_uv {
            for uv in m.uvs {
                writeln!(obj, "vt {} {}", uv.x, 1.0 - uv.y)?;
            }
        }
        for t in m.indices.chunks_exact(3) {
            let (a, b, c) = (t[0] as usize + base, t[1] as usize + base, t[2] as usize + base);
            if has_uv {
                let (ta, tb, tc) = (t[0] as usize + vt_base, t[1] as usize + vt_base, t[2] as usize + vt_base);
                writeln!(obj, "f {a}/{ta}/{a} {b}/{tb}/{b} {c}/{tc}/{c}")?;
            } else {
                writeln!(obj, "f {a}//{a} {b}//{b} {c}//{c}")?;
            }
        }
        base += m.positions.len();
        if has_uv {
            vt_base += m.positions.len();
        }
    }
    Ok(())
}

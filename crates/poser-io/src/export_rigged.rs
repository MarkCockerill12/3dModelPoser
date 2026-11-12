//! Rigged glTF export: skeleton + skin weights + morph targets, posed as in the app.
use anyhow::{Context, Result};
use glam::{Mat4, Quat, Vec3};
use poser_core::model::Model;
use poser_core::skeleton::BoneShape;
use poser_core::skinning::{DeformedMesh, deform};
use poser_core::transform::Transform;
use serde_json::{Value, json};
use std::io::Write;
use std::path::Path;

/// Bake non-inherited bone shapes into the rest pose so the result is a plain skeleton.
/// Returns (model with new rest + inverse binds + mesh bind positions, adjusted pose locals).
pub fn bake_shapes(model: &Model, shapes: &[BoneShape], locals: &[Transform]) -> (Model, Vec<Transform>) {
    // Bone-attached rigid parts become meshes skinned 100% to their bone (portable across apps).
    let mut model = model.clone();
    let rest_w = model.skeleton.rest_world();
    for m in &mut model.meshes {
        if let Some(b) = m.rigid_bone.filter(|b| *b < rest_w.len()) {
            let t = rest_w[b] * m.rigid_transform;
            let nm = glam::Mat3::from_mat4(t).inverse().transpose();
            for p in &mut m.positions {
                *p = t.transform_point3(*p);
            }
            for n in &mut m.normals {
                *n = (nm * *n).normalize_or(Vec3::Y);
            }
            m.joints = vec![[b as u16, 0, 0, 0]; m.positions.len()];
            m.weights = vec![[1.0, 0.0, 0.0, 0.0]; m.positions.len()];
            m.rigid_bone = None;
            m.rigid_transform = Mat4::IDENTITY;
        }
    }
    let model = &model;
    let mut out = model.clone();
    let sk = &model.skeleton;
    if shapes.iter().all(|s| s.is_identity()) || shapes.len() != sk.len() {
        return (out, locals.to_vec());
    }
    let rest = sk.rest_locals();
    let pm = sk.evaluate(&rest, Some(shapes));
    // New bind geometry = rest deformation with shapes applied.
    for (mi, m) in out.meshes.iter_mut().enumerate() {
        if !m.is_skinned() {
            continue;
        }
        let mut d = DeformedMesh::default();
        deform(&model.meshes[mi], &[], &pm.skin, Mat4::IDENTITY, &mut d);
        m.positions = d.positions;
        m.normals = d.normals;
    }
    let mut new_locals = locals.to_vec();
    for (i, b) in out.skeleton.bones.iter_mut().enumerate() {
        if let Some(p) = b.parent {
            let m = shapes[p].matrix(sk.bones[p].axis);
            b.rest.translation = m * b.rest.translation;
            new_locals[i].translation = m * locals[i].translation;
        }
    }
    out.skeleton.recompute_inverse_bind();
    (out, new_locals)
}

fn pad4(v: &mut Vec<u8>, byte: u8) {
    while v.len() % 4 != 0 {
        v.push(byte);
    }
}

fn mat_arr(m: &Mat4) -> Vec<f32> {
    m.to_cols_array().to_vec()
}

/// Write a rigged, posed GLB. `transform` places the figure in the world.
pub fn write_rigged_glb(path: &Path, model: &Model, shapes: &[BoneShape], locals: &[Transform], morph_weights: &[(String, f32)], transform: &Transform) -> Result<()> {
    let (model, locals) = bake_shapes(model, shapes, locals);
    let sk = &model.skeleton;
    let mut bin: Vec<u8> = vec![];
    let mut views: Vec<Value> = vec![];
    let mut accessors: Vec<Value> = vec![];
    let mut add = |bin: &mut Vec<u8>, data: &[u8], target: Option<u32>| -> usize {
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
    // Nodes: 0 = figure root (placement + armature transform), then bones, then meshes.
    let mut nodes: Vec<Value> = vec![];
    let root_m = transform.to_mat4() * sk.root_transform;
    nodes.push(json!({"name": model.name, "matrix": mat_arr(&root_m), "children": []}));
    let bone_node = |i: usize| i + 1;
    for (i, b) in sk.bones.iter().enumerate() {
        let l = locals.get(i).copied().unwrap_or(b.rest);
        nodes.push(json!({
            "name": b.name,
            "translation": l.translation.to_array(),
            "rotation": l.rotation.normalize().to_array(),
            "scale": l.scale.to_array(),
            "children": [],
        }));
    }
    for (i, b) in sk.bones.iter().enumerate() {
        match b.parent {
            Some(p) => nodes[bone_node(p)]["children"].as_array_mut().unwrap().push(json!(bone_node(i))),
            None => nodes[0]["children"].as_array_mut().unwrap().push(json!(bone_node(i))),
        }
    }
    // Skin
    let mut skins = vec![];
    if !sk.is_empty() {
        let ibm: Vec<f32> = sk.bones.iter().flat_map(|b| {
            // IBMs are relative to the armature (root_transform), which lives on node 0.
            mat_arr(&b.inverse_bind)
        }).collect();
        let v = add(&mut bin, bytemuck::cast_slice(&ibm), None);
        accessors.push(json!({"bufferView": v, "componentType": 5126, "count": sk.len(), "type": "MAT4"}));
        skins.push(json!({"inverseBindMatrices": accessors.len() - 1, "joints": (0..sk.len()).map(bone_node).collect::<Vec<_>>(), "skeleton": 0}));
    }
    let mut meshes = vec![];
    let mut materials = vec![];
    let mut images = vec![];
    let mut textures = vec![];
    let mut tex_map: Vec<Option<usize>> = vec![None; model.textures.len()];
    for (mi, m) in model.meshes.iter().enumerate() {
        if m.positions.is_empty() || m.indices.is_empty() {
            continue;
        }
        let n = m.positions.len();
        let (lo, hi) = m.positions.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), p| (a.min(*p), b.max(*p)));
        let pv = add(&mut bin, bytemuck::cast_slice(&m.positions), Some(34962));
        accessors.push(json!({"bufferView": pv, "componentType": 5126, "count": n, "type": "VEC3", "min": lo.to_array(), "max": hi.to_array()}));
        let mut attrs = json!({"POSITION": accessors.len() - 1});
        let nv = add(&mut bin, bytemuck::cast_slice(&m.normals), Some(34962));
        accessors.push(json!({"bufferView": nv, "componentType": 5126, "count": n, "type": "VEC3"}));
        attrs["NORMAL"] = json!(accessors.len() - 1);
        if m.uvs.len() == n {
            let uv = add(&mut bin, bytemuck::cast_slice(&m.uvs), Some(34962));
            accessors.push(json!({"bufferView": uv, "componentType": 5126, "count": n, "type": "VEC2"}));
            attrs["TEXCOORD_0"] = json!(accessors.len() - 1);
        }
        if m.is_skinned() {
            let j: Vec<u16> = m.joints.iter().flatten().copied().collect();
            let jv = add(&mut bin, bytemuck::cast_slice(&j), Some(34962));
            accessors.push(json!({"bufferView": jv, "componentType": 5123, "count": n, "type": "VEC4"}));
            attrs["JOINTS_0"] = json!(accessors.len() - 1);
            let w: Vec<f32> = m.weights.iter().flatten().copied().collect();
            let wv = add(&mut bin, bytemuck::cast_slice(&w), Some(34962));
            accessors.push(json!({"bufferView": wv, "componentType": 5126, "count": n, "type": "VEC4"}));
            attrs["WEIGHTS_0"] = json!(accessors.len() - 1);
        }
        let iv = add(&mut bin, bytemuck::cast_slice(&m.indices), Some(34963));
        accessors.push(json!({"bufferView": iv, "componentType": 5125, "count": m.indices.len(), "type": "SCALAR"}));
        let ia = accessors.len() - 1;
        // Morph targets (dense position deltas).
        let mut targets = vec![];
        let mut tw = vec![];
        let mut names = vec![];
        for t in &m.morphs {
            let mut dense = vec![Vec3::ZERO; n];
            for (k, &i) in t.indices.iter().enumerate() {
                if (i as usize) < n {
                    dense[i as usize] = t.position_deltas[k];
                }
            }
            let (lo, hi) = dense.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), p| (a.min(*p), b.max(*p)));
            let dv = add(&mut bin, bytemuck::cast_slice(&dense), Some(34962));
            accessors.push(json!({"bufferView": dv, "componentType": 5126, "count": n, "type": "VEC3", "min": lo.to_array(), "max": hi.to_array()}));
            targets.push(json!({"POSITION": accessors.len() - 1}));
            tw.push(morph_weights.iter().find(|(nm, _)| *nm == t.name).map(|x| x.1).unwrap_or(0.0));
            names.push(t.name.clone());
        }
        let mat = &model.materials[m.material.min(model.materials.len() - 1)];
        let c = mat.base_color;
        let mut mj = json!({"name": mat.name, "pbrMetallicRoughness": {"baseColorFactor": c.to_array(), "metallicFactor": 0.0, "roughnessFactor": 0.8}, "doubleSided": true});
        if mat.alpha_mode == poser_core::model::AlphaMode::Mask {
            mj["alphaMode"] = json!("MASK");
            mj["alphaCutoff"] = json!(mat.alpha_cutoff);
        }
        if let Some(t) = mat.base_color_texture.filter(|t| *t < model.textures.len()) {
            let ti = match tex_map[t] {
                Some(i) => i,
                None => {
                    let td = &model.textures[t];
                    let img = image::RgbaImage::from_raw(td.width, td.height, td.rgba.to_vec()).context("texture")?;
                    let mut cur = std::io::Cursor::new(vec![]);
                    img.write_to(&mut cur, image::ImageFormat::Png)?;
                    let v = add(&mut bin, &cur.into_inner(), None);
                    images.push(json!({"bufferView": v, "mimeType": "image/png"}));
                    textures.push(json!({"source": images.len() - 1}));
                    tex_map[t] = Some(textures.len() - 1);
                    textures.len() - 1
                }
            };
            mj["pbrMetallicRoughness"]["baseColorTexture"] = json!({"index": ti});
        }
        materials.push(mj);
        let mut prim = json!({"attributes": attrs, "indices": ia, "material": materials.len() - 1});
        let mut mesh = json!({"name": m.name});
        if !targets.is_empty() {
            prim["targets"] = json!(targets);
            mesh["weights"] = json!(tw);
            mesh["extras"] = json!({"targetNames": names});
        }
        mesh["primitives"] = json!([prim]);
        meshes.push(mesh);
        let mut node = json!({"name": if m.name.is_empty() { format!("mesh{mi}") } else { m.name.clone() }, "mesh": meshes.len() - 1});
        if m.is_skinned() && !skins.is_empty() {
            node["skin"] = json!(0);
            nodes.push(node);
            let idx = nodes.len() - 1;
            nodes[0]["children"].as_array_mut().unwrap().push(json!(idx));
        } else {
            let mm = if m.rigid_bone.is_some() { m.rigid_transform } else { sk.root_transform.inverse() * m.rigid_transform };
            node["matrix"] = json!(mat_arr(&mm));
            nodes.push(node);
            let idx = nodes.len() - 1;
            match m.rigid_bone {
                Some(b) => nodes[bone_node(b)]["children"].as_array_mut().unwrap().push(json!(idx)),
                None => nodes[0]["children"].as_array_mut().unwrap().push(json!(idx)),
            }
        }
    }
    for n in nodes.iter_mut() {
        if n["children"].as_array().is_some_and(|c| c.is_empty()) {
            n.as_object_mut().unwrap().remove("children");
        }
    }
    pad4(&mut bin, 0);
    let mut doc = json!({
        "asset": {"version": "2.0", "generator": "3D Model Poser"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": bin.len()}],
    });
    if !skins.is_empty() {
        doc["skins"] = json!(skins);
    }
    if !images.is_empty() {
        doc["images"] = json!(images);
        doc["textures"] = json!(textures);
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
    let _ = Quat::IDENTITY;
    Ok(())
}

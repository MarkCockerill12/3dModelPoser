//! FBX + OBJ import through ufbx (converted to right-handed, Y-up, metres).
use crate::util::{downscale_rgba, finalize};
use anyhow::{Result, anyhow, bail};
use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use poser_core::animation::{AnimationClip, BoneChannel, MorphChannel, Track};
use poser_core::model::{AlphaMode, Material, Mesh, Model, MorphTarget, SourceFormat, TextureData};
use poser_core::skeleton::{Bone, Skeleton};
use poser_core::transform::Transform;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn v3(v: ufbx::Vec3) -> Vec3 {
    Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}

fn quat(q: ufbx::Quat) -> Quat {
    Quat::from_xyzw(q.x as f32, q.y as f32, q.z as f32, q.w as f32).normalize()
}

fn mat(m: &ufbx::Matrix) -> Mat4 {
    Mat4::from_cols(
        Vec4::new(m.m00 as f32, m.m10 as f32, m.m20 as f32, 0.0),
        Vec4::new(m.m01 as f32, m.m11 as f32, m.m21 as f32, 0.0),
        Vec4::new(m.m02 as f32, m.m12 as f32, m.m22 as f32, 0.0),
        Vec4::new(m.m03 as f32, m.m13 as f32, m.m23 as f32, 1.0),
    )
}

fn xf(t: &ufbx::Transform) -> Transform {
    Transform { translation: v3(t.translation), rotation: quat(t.rotation), scale: v3(t.scale) }
}

fn load_texture(tex: &ufbx::Texture, base: &Path) -> Option<TextureData> {
    let img = if !tex.content.is_empty() {
        image::load_from_memory(&tex.content).ok()
    } else {
        let mut cands: Vec<PathBuf> = vec![];
        for s in [tex.absolute_filename.as_ref(), tex.relative_filename.as_ref(), tex.filename.as_ref()] {
            let s: &str = s;
            if s.is_empty() {
                continue;
            }
            let p = PathBuf::from(s);
            cands.push(p.clone());
            cands.push(base.join(&p));
            if let Some(f) = p.file_name() {
                cands.push(base.join(f));
                cands.push(base.join("textures").join(f));
                cands.push(base.join("Textures").join(f));
            }
        }
        cands.into_iter().find(|p| p.is_file()).and_then(|p| image::open(p).ok())
    }?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let (w, h, data) = downscale_rgba(w, h, rgba.into_raw(), 2048);
    Some(TextureData { name: tex.element.name.to_string(), width: w, height: h, rgba: Arc::new(data) })
}

pub fn load(path: &Path) -> Result<Model> {
    let is_obj = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("obj"));
    let opts = ufbx::LoadOpts {
        target_axes: ufbx::CoordinateAxes {
            right: ufbx::CoordinateAxis::PositiveX,
            up: ufbx::CoordinateAxis::PositiveY,
            front: ufbx::CoordinateAxis::PositiveZ,
        },
        target_unit_meters: 1.0,
        space_conversion: ufbx::SpaceConversion::ModifyGeometry,
        generate_missing_normals: true,
        load_external_files: true,
        ignore_missing_external_files: true,
        clean_skin_weights: true,
        ..Default::default()
    };
    let path_str = path.to_str().ok_or_else(|| anyhow!("non-UTF8 path"))?;
    let scene = ufbx::load_file(path_str, opts).map_err(|e| anyhow!("ufbx: {}", e.description))?;
    let base = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "Model".into());
    let mut model = Model::empty(name, if is_obj { SourceFormat::Obj } else { SourceFormat::Fbx });
    model.path = Some(path.to_path_buf());

    // ---- Bones ----------------------------------------------------------------------------
    let nodes: Vec<&ufbx::Node> = scene.nodes.iter().map(|n| &**n).collect();
    let id_of = |n: &ufbx::Node| n.element.typed_id as usize;
    let parent: Vec<Option<usize>> = nodes.iter().map(|n| n.parent.as_ref().map(|p| p.element.typed_id as usize)).collect();
    let mut bone_set: HashSet<usize> = HashSet::new();
    for n in &nodes {
        if n.bone.is_some() {
            bone_set.insert(id_of(n));
        }
    }
    for c in scene.skin_clusters.iter() {
        if let Some(b) = &c.bone_node {
            bone_set.insert(b.element.typed_id as usize);
        }
    }
    // Intermediate nodes between bones.
    for b in bone_set.clone() {
        let mut chain = vec![];
        let mut p = parent[b];
        while let Some(x) = p {
            if bone_set.contains(&x) {
                bone_set.extend(chain.drain(..));
                break;
            }
            chain.push(x);
            p = parent[x];
        }
    }
    // Drop the implicit scene root.
    bone_set.remove(&id_of(&scene.root_node));
    loop {
        let roots: HashSet<Option<usize>> = bone_set
            .iter()
            .filter(|b| !parent[**b].is_some_and(|p| bone_set.contains(&p)))
            .map(|b| parent[*b].filter(|p| *p != id_of(&scene.root_node)))
            .collect();
        if roots.len() <= 1 {
            break;
        }
        let mut added = false;
        for p in roots.into_iter().flatten() {
            added |= bone_set.insert(p);
        }
        if !added {
            break;
        }
    }
    let mut order = vec![];
    let mut stack = vec![id_of(&scene.root_node)];
    while let Some(i) = stack.pop() {
        if bone_set.contains(&i) {
            order.push(i);
        }
        for c in nodes[i].children.iter().rev() {
            stack.push(c.element.typed_id as usize);
        }
    }
    let node_to_bone: HashMap<usize, usize> = order.iter().enumerate().map(|(b, n)| (*n, b)).collect();
    let root_parent = order.first().and_then(|r| parent[*r]);
    let mut skel = Skeleton {
        bones: vec![],
        root_transform: root_parent.map(|p| mat(&nodes[p].node_to_world)).unwrap_or(Mat4::IDENTITY),
    };
    for &n in &order {
        let p = parent[n].and_then(|p| node_to_bone.get(&p).copied());
        skel.bones.push(Bone::new(nodes[n].element.name.to_string(), p, xf(&nodes[n].local_transform)));
    }
    skel.recompute_inverse_bind();
    let mut ibm_set = vec![false; skel.len()];
    // Inverse bind = geometry->bone at bind time, re-expressed for model-space vertices
    // (we pre-transform skinned vertices by the cluster's geometry_to_world).
    for c in scene.skin_clusters.iter() {
        if let Some(b) = c.bone_node.as_ref().and_then(|b| node_to_bone.get(&(b.element.typed_id as usize))) {
            if !ibm_set[*b] {
                skel.bones[*b].inverse_bind = mat(&c.geometry_to_bone) * mat(&c.geometry_to_world).inverse();
                ibm_set[*b] = true;
            }
        }
    }
    model.skeleton = skel;

    // ---- Materials --------------------------------------------------------------------------
    model.materials.clear();
    model.textures.clear();
    let mut tex_cache: HashMap<u32, Option<usize>> = HashMap::new();
    let mut mat_index: HashMap<u32, usize> = HashMap::new();
    for m in scene.materials.iter() {
        let pbr = &m.pbr.base_color;
        let fbx = &m.fbx.diffuse_color;
        let color = if pbr.has_value {
            let f = if m.pbr.base_factor.has_value { m.pbr.base_factor.value_vec4.x as f32 } else { 1.0 };
            Vec4::new(pbr.value_vec4.x as f32 * f, pbr.value_vec4.y as f32 * f, pbr.value_vec4.z as f32 * f, 1.0)
        } else if fbx.has_value {
            Vec4::new(fbx.value_vec4.x as f32, fbx.value_vec4.y as f32, fbx.value_vec4.z as f32, 1.0)
        } else {
            Vec4::new(0.8, 0.8, 0.8, 1.0)
        };
        let tex_ref = pbr.texture.as_ref().or(fbx.texture.as_ref());
        let tex_idx = tex_ref.and_then(|t| {
            let id = t.element.typed_id;
            *tex_cache.entry(id).or_insert_with(|| {
                let td = load_texture(t, &base)?;
                model.textures.push(td);
                Some(model.textures.len() - 1)
            })
        });
        let opacity = m.pbr.opacity.has_value.then(|| m.pbr.opacity.value_vec4.x as f32).unwrap_or(1.0);
        mat_index.insert(m.element.typed_id, model.materials.len());
        model.materials.push(Material {
            name: m.element.name.to_string(),
            base_color: Vec4::new(color.x, color.y, color.z, opacity),
            base_color_texture: tex_idx,
            alpha_mode: if opacity < 0.99 { AlphaMode::Blend } else { AlphaMode::Opaque },
            alpha_cutoff: 0.5,
            double_sided: true,
            clay_shade: 1.0,
        });
    }
    let default_material = model.materials.len();
    model.materials.push(Material::default());

    // ---- Meshes -------------------------------------------------------------------------
    let mut node_meshes: HashMap<usize, Vec<(usize, Vec<u32>)>> = HashMap::new(); // mesh node -> (model mesh, logical vertex per out vertex)
    for n in &nodes {
        let Some(mesh) = n.mesh.as_ref() else { continue };
        let skin = mesh.skin_deformers.first();
        let cluster_bone: Vec<Option<usize>> = skin
            .map(|s| {
                s.clusters
                    .iter()
                    .map(|c| c.bone_node.as_ref().and_then(|b| node_to_bone.get(&(b.element.typed_id as usize)).copied()))
                    .collect()
            })
            .unwrap_or_default();
        let geo_to_world = match skin.and_then(|s| s.clusters.first()) {
            Some(c) => mat(&c.geometry_to_world),
            None => mat(&n.geometry_to_world),
        };
        // Rigid parent bone for unskinned meshes.
        let rigid_bone = if skin.is_none() {
            let mut p = parent[id_of(n)];
            let mut b = None;
            while let Some(x) = p {
                if let Some(&bi) = node_to_bone.get(&x) {
                    b = Some(bi);
                    break;
                }
                p = parent[x];
            }
            b
        } else {
            None
        };
        let parts: Vec<(Option<u32>, Vec<u32>)> = if mesh.material_parts.len() > 1 {
            mesh.material_parts
                .iter()
                .enumerate()
                .map(|(i, p)| (mesh.materials.get(i).map(|m| m.element.typed_id), p.face_indices.iter().copied().collect()))
                .collect()
        } else {
            vec![(mesh.materials.first().map(|m| m.element.typed_id), (0..mesh.faces.len() as u32).collect())]
        };
        let mut tri = Vec::new();
        for (mat_id, faces) in parts {
            if faces.is_empty() {
                continue;
            }
            let mut out = Mesh {
                name: if mesh.material_parts.len() > 1 {
                    format!("{}.{}", n.element.name, mat_id.and_then(|m| mat_index.get(&m)).map(|i| model.materials[*i].name.clone()).unwrap_or_default())
                } else {
                    n.element.name.to_string()
                },
                material: mat_id.and_then(|m| mat_index.get(&m).copied()).unwrap_or(default_material),
                rigid_transform: Mat4::IDENTITY,
                ..Default::default()
            };
            let mut dedup: HashMap<(u32, u32, u32), u32> = HashMap::new();
            let mut logical: Vec<u32> = vec![];
            let has_uv = mesh.vertex_uv.exists;
            let has_nrm = mesh.vertex_normal.exists;
            for &fi in &faces {
                let face = mesh.faces[fi as usize];
                let nt = ufbx::triangulate_face_vec(&mut tri, mesh, face);
                for k in 0..(nt as usize * 3) {
                    let ix = tri[k] as usize;
                    let vi = mesh.vertex_indices[ix];
                    let ni = if has_nrm { mesh.vertex_normal.indices[ix] } else { 0 };
                    let ui = if has_uv { mesh.vertex_uv.indices[ix] } else { 0 };
                    let id = *dedup.entry((vi, ni, ui)).or_insert_with(|| {
                        out.positions.push(v3(mesh.vertex_position[ix]));
                        if has_nrm {
                            out.normals.push(v3(mesh.vertex_normal[ix]));
                        }
                        if has_uv {
                            let uv = mesh.vertex_uv[ix];
                            out.uvs.push(Vec2::new(uv.x as f32, 1.0 - uv.y as f32));
                        }
                        logical.push(vi);
                        (out.positions.len() - 1) as u32
                    });
                    out.indices.push(id);
                }
            }
            // Bring geometry into model space (bind space for skinned meshes).
            let nm = Mat4::from_mat3(glam::Mat3::from_mat4(geo_to_world).inverse().transpose());
            if skin.is_some() {
                for p in &mut out.positions {
                    *p = geo_to_world.transform_point3(*p);
                }
                for nr in &mut out.normals {
                    *nr = nm.transform_vector3(*nr).normalize_or(Vec3::Y);
                }
            } else {
                let gw = mat(&n.geometry_to_world);
                match rigid_bone {
                    Some(b) => {
                        let bw = model.skeleton.rest_world()[b];
                        out.rigid_bone = Some(b);
                        out.rigid_transform = bw.inverse() * gw;
                    }
                    None => out.rigid_transform = gw,
                }
            }
            if let Some(s) = skin {
                out.joints = Vec::with_capacity(logical.len());
                out.weights = Vec::with_capacity(logical.len());
                for &lv in &logical {
                    let sv = s.vertices.get(lv as usize);
                    let mut ws: Vec<(u16, f32)> = vec![];
                    if let Some(sv) = sv {
                        for wi in sv.weight_begin..sv.weight_begin + sv.num_weights {
                            let w = s.weights[wi as usize];
                            if let Some(Some(b)) = cluster_bone.get(w.cluster_index as usize) {
                                ws.push((*b as u16, w.weight as f32));
                            }
                        }
                    }
                    ws.sort_by(|a, b| b.1.total_cmp(&a.1));
                    let mut j = [0u16; 4];
                    let mut w = [0f32; 4];
                    for (k, (bj, bw)) in ws.into_iter().take(4).enumerate() {
                        j[k] = bj;
                        w[k] = bw;
                    }
                    out.joints.push(j);
                    out.weights.push(w);
                }
            }
            // Blend shapes.
            for bd in mesh.blend_deformers.iter() {
                for ch in bd.channels.iter() {
                    let Some(shape) = ch.target_shape.as_ref() else { continue };
                    let mut by_logical: HashMap<u32, Vec3> = HashMap::new();
                    for (k, &v) in shape.offset_vertices.iter().enumerate() {
                        by_logical.insert(v, v3(shape.position_offsets[k]));
                    }
                    let lin = glam::Mat3::from_mat4(if skin.is_some() { geo_to_world } else { Mat4::IDENTITY });
                    let dense: Vec<Vec3> =
                        logical.iter().map(|l| by_logical.get(l).map(|d| lin * *d).unwrap_or(Vec3::ZERO)).collect();
                    let t = MorphTarget::from_dense(ch.element.name.to_string(), &dense, None);
                    if !t.indices.is_empty() {
                        out.morphs.push(t);
                    }
                }
            }
            out.sanitize();
            node_meshes.entry(id_of(n)).or_default().push((model.meshes.len(), logical));
            model.meshes.push(out);
        }
    }
    if model.meshes.is_empty() {
        bail!("no meshes found");
    }

    // ---- Animations (baked at 30 fps) -------------------------------------------------------
    if !model.skeleton.is_empty() {
        for stack in scene.anim_stacks.iter() {
            let (t0, t1) = (stack.time_begin, stack.time_end);
            if t1 <= t0 {
                continue;
            }
            let fps = 30.0;
            let frames = (((t1 - t0) * fps).ceil() as usize + 1).min(30 * 60 * 10);
            let times: Vec<f32> = (0..frames).map(|f| f as f32 / fps as f32).collect();
            let mut clip = AnimationClip { name: stack.element.name.to_string(), fps: fps as f32, ..Default::default() };
            for (b, &n) in order.iter().enumerate() {
                let node = nodes[n];
                let mut tr = Vec::with_capacity(frames);
                let mut rot = Vec::with_capacity(frames);
                let mut sc = Vec::with_capacity(frames);
                for f in 0..frames {
                    let t = t0 + f as f64 / fps;
                    let x = ufbx::evaluate_transform(&stack.anim, node, t);
                    tr.push(v3(x.translation));
                    let mut q = quat(x.rotation);
                    if let Some(prev) = rot.last() {
                        if q.dot(*prev) < 0.0 {
                            q = -q;
                        }
                    }
                    rot.push(q);
                    sc.push(v3(x.scale));
                }
                let constant_v = |v: &Vec<Vec3>| v.iter().all(|x| x.distance_squared(v[0]) < 1e-12);
                let constant_q = |v: &Vec<Quat>| v.iter().all(|x| x.dot(v[0]).abs() > 0.999_999);
                let rest = model.skeleton.bones[b].rest;
                let mut ch = BoneChannel { bone: b, ..Default::default() };
                if !(constant_v(&tr) && tr[0].distance_squared(rest.translation) < 1e-10) {
                    ch.translation = Some(Track { times: times.clone(), values: tr, step: false });
                }
                if !(constant_q(&rot) && rot[0].dot(rest.rotation).abs() > 0.999_999) {
                    ch.rotation = Some(Track { times: times.clone(), values: rot, step: false });
                }
                if !(constant_v(&sc) && sc[0].distance_squared(rest.scale) < 1e-10) {
                    ch.scale = Some(Track { times: times.clone(), values: sc, step: false });
                }
                if ch.translation.is_some() || ch.rotation.is_some() || ch.scale.is_some() {
                    clip.channels.push(ch);
                }
            }
            // Blend-shape weights.
            for n in &nodes {
                let Some(mesh) = n.mesh.as_ref() else { continue };
                let Some(list) = node_meshes.get(&id_of(n)) else { continue };
                let chans: Vec<&ufbx::BlendChannel> = mesh.blend_deformers.iter().flat_map(|d| d.channels.iter().map(|c| &**c)).collect();
                if chans.is_empty() {
                    continue;
                }
                let names: Vec<String> = chans.iter().map(|c| c.element.name.to_string()).collect();
                let values: Vec<Vec<f32>> = (0..frames)
                    .map(|f| {
                        let t = t0 + f as f64 / fps;
                        chans.iter().map(|c| ufbx::evaluate_blend_weight(&stack.anim, c, t) as f32).collect()
                    })
                    .collect();
                let animated = values.iter().any(|v| v.iter().zip(&values[0]).any(|(a, b)| (a - b).abs() > 1e-4));
                if animated && !list.is_empty() {
                    clip.morph_channels.push(MorphChannel { names, weights: Track { times: times.clone(), values, step: false } });
                }
            }
            if !clip.channels.is_empty() || !clip.morph_channels.is_empty() {
                clip.recompute_duration();
                model.animations.push(clip);
            }
        }
    }
    finalize(&mut model);
    Ok(model)
}

//! glTF 2.0 / GLB / VRM import.
use crate::util::{downscale_rgba, finalize};
use anyhow::{Context, Result, bail};
use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use poser_core::animation::{AnimationClip, BoneChannel, MorphChannel, Track};
use poser_core::humanoid::{HumanBone, HumanoidMap};
use poser_core::model::{AlphaMode, Material, Mesh, Model, MorphTarget, SourceFormat, TextureData};
use poser_core::skeleton::{Bone, Skeleton};
use poser_core::transform::Transform;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

fn pick<T>(cubic: bool, v: Vec<T>) -> Vec<T> {
    if cubic { v.into_iter().skip(1).step_by(3).collect() } else { v }
}

fn node_local(n: &gltf::Node) -> Transform {
    let (t, r, s) = n.transform().decomposed();
    Transform {
        translation: Vec3::from(t),
        rotation: Quat::from_array(r).normalize(),
        scale: Vec3::from(s),
    }
}

/// Raw JSON of a .gltf/.glb/.vrm (for extensions such as VRM).
fn read_json(path: &Path) -> Option<serde_json::Value> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() >= 20 && &bytes[0..4] == b"glTF" {
        let len = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
        let kind = &bytes[16..20];
        if kind != b"JSON" || 20 + len > bytes.len() {
            return None;
        }
        serde_json::from_slice(&bytes[20..20 + len]).ok()
    } else {
        serde_json::from_slice(&bytes).ok()
    }
}

fn image_to_rgba(img: &gltf::image::Data, name: String) -> TextureData {
    use gltf::image::Format as F;
    let (w, h) = (img.width, img.height);
    let px = &img.pixels;
    let n = (w * h) as usize;
    let mut out = Vec::with_capacity(n * 4);
    match img.format {
        F::R8G8B8A8 => out.extend_from_slice(px),
        F::R8G8B8 => {
            for c in px.chunks_exact(3) {
                out.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        F::R8 => {
            for &c in px {
                out.extend_from_slice(&[c, c, c, 255]);
            }
        }
        F::R8G8 => {
            for c in px.chunks_exact(2) {
                out.extend_from_slice(&[c[0], c[0], c[0], c[1]]);
            }
        }
        F::R16G16B16A16 => {
            for c in px.chunks_exact(8) {
                out.extend_from_slice(&[c[1], c[3], c[5], c[7]]);
            }
        }
        F::R16G16B16 => {
            for c in px.chunks_exact(6) {
                out.extend_from_slice(&[c[1], c[3], c[5], 255]);
            }
        }
        _ => out.resize(n * 4, 255),
    }
    out.resize(n * 4, 255);
    let (w, h, out) = downscale_rgba(w, h, out, 2048);
    TextureData { name, width: w, height: h, rgba: Arc::new(out) }
}

pub fn load(path: &Path) -> Result<Model> {
    let (doc, buffers, images) = gltf::import(path).with_context(|| format!("reading {}", path.display()))?;
    let json = read_json(path);
    let is_vrm = json
        .as_ref()
        .and_then(|j| j.get("extensions"))
        .is_some_and(|e| e.get("VRM").is_some() || e.get("VRMC_vrm").is_some());
    let name = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "Model".into());
    let mut model = Model::empty(name, if is_vrm { SourceFormat::Vrm } else { SourceFormat::Gltf });
    model.path = Some(path.to_path_buf());

    let nodes: Vec<gltf::Node> = doc.nodes().collect();
    let nn = nodes.len();
    let mut parent: Vec<Option<usize>> = vec![None; nn];
    for n in &nodes {
        for c in n.children() {
            parent[c.index()] = Some(n.index());
        }
    }
    let locals: Vec<Transform> = nodes.iter().map(node_local).collect();
    let mut world_cache: Vec<Option<Mat4>> = vec![None; nn];
    fn world_of(i: usize, parent: &[Option<usize>], locals: &[Transform], cache: &mut [Option<Mat4>]) -> Mat4 {
        if let Some(m) = cache[i] {
            return m;
        }
        let m = match parent[i] {
            Some(p) => world_of(p, parent, locals, cache) * locals[i].to_mat4(),
            None => locals[i].to_mat4(),
        };
        cache[i] = Some(m);
        m
    }

    // ---- Skeleton ------------------------------------------------------------------------
    let mut bone_set: HashSet<usize> = HashSet::new();
    for skin in doc.skins() {
        for j in skin.joints() {
            bone_set.insert(j.index());
        }
    }
    // Include nodes between joints.
    let joints: Vec<usize> = bone_set.iter().copied().collect();
    for j in joints {
        let mut chain = vec![];
        let mut p = parent[j];
        while let Some(x) = p {
            if bone_set.contains(&x) {
                for c in chain.drain(..) {
                    bone_set.insert(c);
                }
                break;
            }
            chain.push(x);
            p = parent[x];
        }
    }
    // Make all roots share one parent.
    loop {
        let roots: HashSet<Option<usize>> =
            bone_set.iter().filter(|b| !parent[**b].is_some_and(|p| bone_set.contains(&p))).map(|b| parent[*b]).collect();
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
    // Topological order via DFS from scene roots.
    let mut order: Vec<usize> = vec![];
    let mut stack: Vec<usize> = (0..nn).filter(|i| parent[*i].is_none()).rev().collect();
    while let Some(i) = stack.pop() {
        if bone_set.contains(&i) {
            order.push(i);
        }
        for c in nodes[i].children().collect::<Vec<_>>().into_iter().rev() {
            stack.push(c.index());
        }
    }
    let node_to_bone: HashMap<usize, usize> = order.iter().enumerate().map(|(b, n)| (*n, b)).collect();
    let root_parent = order.first().and_then(|r| parent[*r]);
    let mut skel = Skeleton {
        bones: vec![],
        root_transform: root_parent.map(|p| world_of(p, &parent, &locals, &mut world_cache)).unwrap_or(Mat4::IDENTITY),
    };
    for &n in &order {
        let p = parent[n].and_then(|p| node_to_bone.get(&p).copied());
        let name = nodes[n].name().map(str::to_string).unwrap_or_else(|| format!("bone_{n}"));
        skel.bones.push(Bone::new(name, p, locals[n]));
    }
    skel.recompute_inverse_bind();
    let mut ibm_set = vec![false; skel.len()];
    for skin in doc.skins() {
        let reader = skin.reader(|b| Some(&buffers[b.index()]));
        if let Some(ibms) = reader.read_inverse_bind_matrices() {
            for (j, m) in skin.joints().zip(ibms) {
                if let Some(&b) = node_to_bone.get(&j.index()) {
                    if !ibm_set[b] {
                        skel.bones[b].inverse_bind = Mat4::from_cols_array_2d(&m);
                        ibm_set[b] = true;
                    }
                }
            }
        }
    }
    let moved = skel.align_rest_to_bind(&ibm_set);
    if moved > 0 {
        log::info!("glTF: moved {moved} displaced bones back to their bind pose");
    }
    model.skeleton = skel;

    // ---- Materials / textures -----------------------------------------------------------
    for (i, img) in images.iter().enumerate() {
        model.textures.push(image_to_rgba(img, format!("image_{i}")));
    }
    model.materials.clear();
    for m in doc.materials() {
        let pbr = m.pbr_metallic_roughness();
        model.materials.push(Material {
            name: m.name().unwrap_or("material").to_string(),
            base_color: Vec4::from(pbr.base_color_factor()),
            base_color_texture: pbr.base_color_texture().map(|t| t.texture().source().index()),
            alpha_mode: match m.alpha_mode() {
                gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                gltf::material::AlphaMode::Mask => AlphaMode::Mask,
                gltf::material::AlphaMode::Blend => AlphaMode::Blend,
            },
            alpha_cutoff: m.alpha_cutoff().unwrap_or(0.5),
            double_sided: m.double_sided(),
            clay_shade: 1.0,
        });
    }
    let default_material = model.materials.len();
    model.materials.push(Material::default());

    // ---- Meshes -------------------------------------------------------------------------
    let mut mesh_node_morph: HashMap<usize, Vec<usize>> = HashMap::new(); // node -> model mesh indices
    for n in &nodes {
        let Some(mesh) = n.mesh() else { continue };
        let target_names: Vec<String> = mesh
            .extras()
            .as_ref()
            .and_then(|r| serde_json::from_str::<serde_json::Value>(r.get()).ok())
            .and_then(|v| v.get("targetNames").cloned())
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default();
        let skin_map: Option<Vec<usize>> =
            n.skin().map(|s| s.joints().map(|j| node_to_bone.get(&j.index()).copied().unwrap_or(0)).collect());
        let node_world = world_of(n.index(), &parent, &locals, &mut world_cache);
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let r = prim.reader(|b| Some(&buffers[b.index()]));
            let Some(pos) = r.read_positions() else { continue };
            let mut m = Mesh {
                name: format!("{}{}", mesh.name().unwrap_or("mesh"), if mesh.primitives().len() > 1 { format!(".{}", prim.index()) } else { String::new() }),
                positions: pos.map(Vec3::from).collect(),
                material: prim.material().index().unwrap_or(default_material),
                rigid_transform: Mat4::IDENTITY,
                ..Default::default()
            };
            if let Some(nr) = r.read_normals() {
                m.normals = nr.map(Vec3::from).collect();
            }
            if let Some(uv) = r.read_tex_coords(0) {
                m.uvs = uv.into_f32().map(Vec2::from).collect();
            }
            if let Some(c) = r.read_colors(0) {
                m.colors = c.into_rgba_f32().map(Vec4::from).collect();
            }
            m.indices = match r.read_indices() {
                Some(i) => i.into_u32().collect(),
                None => (0..m.positions.len() as u32).collect(),
            };
            if let (Some(map), Some(j), Some(w)) = (&skin_map, r.read_joints(0), r.read_weights(0)) {
                m.joints = j
                    .into_u16()
                    .map(|j| j.map(|x| map.get(x as usize).copied().unwrap_or(0) as u16))
                    .collect();
                m.weights = w.into_f32().collect();
            } else {
                // Rigid: follow the closest bone ancestor.
                let mut p = parent[n.index()];
                let mut bone = None;
                while let Some(x) = p {
                    if let Some(&b) = node_to_bone.get(&x) {
                        bone = Some(b);
                        break;
                    }
                    p = parent[x];
                }
                match bone {
                    Some(b) => {
                        let bw = world_of(order[b], &parent, &locals, &mut world_cache);
                        m.rigid_bone = Some(b);
                        m.rigid_transform = bw.inverse() * node_world;
                    }
                    None => m.rigid_transform = node_world,
                }
            }
            for (ti, (p, nrm, _)) in r.read_morph_targets().enumerate() {
                let pd: Vec<Vec3> = p.map(|it| it.map(Vec3::from).collect()).unwrap_or_else(|| vec![Vec3::ZERO; m.positions.len()]);
                let nd: Option<Vec<Vec3>> = nrm.map(|it| it.map(Vec3::from).collect());
                let name = target_names.get(ti).cloned().unwrap_or_else(|| format!("{} morph {ti}", mesh.name().unwrap_or("mesh")));
                m.morphs.push(MorphTarget::from_dense(name, &pd, nd.as_deref()));
            }
            m.sanitize();
            mesh_node_morph.entry(n.index()).or_default().push(model.meshes.len());
            model.meshes.push(m);
        }
    }
    if model.meshes.is_empty() {
        bail!("no triangle meshes found");
    }

    // ---- Animations -----------------------------------------------------------------------
    for (ai, anim) in doc.animations().enumerate() {
        let mut clip = AnimationClip {
            name: anim.name().map(str::to_string).unwrap_or_else(|| format!("Animation {}", ai + 1)),
            fps: 30.0,
            ..Default::default()
        };
        let mut chans: HashMap<usize, BoneChannel> = HashMap::new();
        for ch in anim.channels() {
            let target = ch.target().node().index();
            let interp = ch.sampler().interpolation();
            let step = interp == gltf::animation::Interpolation::Step;
            let cubic = interp == gltf::animation::Interpolation::CubicSpline;
            let r = ch.reader(|b| Some(&buffers[b.index()]));
            let Some(times) = r.read_inputs() else { continue };
            let times: Vec<f32> = times.collect();
            let Some(outputs) = r.read_outputs() else { continue };
            use gltf::animation::util::ReadOutputs as O;
            match outputs {
                O::MorphTargetWeights(w) => {
                    let flat: Vec<f32> = pick(cubic, w.into_f32().collect());
                    let k = times.len().max(1);
                    let per = flat.len() / k;
                    if let Some(meshes) = mesh_node_morph.get(&target) {
                        if let Some(&mi) = meshes.first() {
                            let names: Vec<String> = model.meshes[mi].morphs.iter().map(|t| t.name.clone()).collect();
                            let values = flat.chunks(per.max(1)).map(|c| c.to_vec()).collect();
                            clip.morph_channels.push(MorphChannel { names, weights: Track { times: times.clone(), values, step } });
                        }
                    }
                }
                other => {
                    let Some(&bone) = node_to_bone.get(&target) else { continue };
                    let c = chans.entry(bone).or_insert_with(|| BoneChannel { bone, ..Default::default() });
                    match other {
                        O::Translations(t) => {
                            c.translation = Some(Track { times, values: pick(cubic, t.map(Vec3::from).collect()), step })
                        }
                        O::Rotations(rq) => {
                            c.rotation = Some(Track {
                                times,
                                values: pick(cubic, rq.into_f32().map(|q| Quat::from_array(q).normalize()).collect()),
                                step,
                            })
                        }
                        O::Scales(s) => c.scale = Some(Track { times, values: pick(cubic, s.map(Vec3::from).collect()), step }),
                        O::MorphTargetWeights(_) => {}
                    }
                }
            }
        }
        clip.channels = chans.into_values().collect();
        clip.channels.sort_by_key(|c| c.bone);
        clip.recompute_duration();
        model.animations.push(clip);
    }

    // ---- VRM humanoid ---------------------------------------------------------------------
    let mut human = HumanoidMap::default();
    if let Some(ext) = json.as_ref().and_then(|j| j.get("extensions")) {
        if let Some(bones) = ext.pointer("/VRMC_vrm/humanoid/humanBones").and_then(|b| b.as_object()) {
            for (k, v) in bones {
                if let (Some(h), Some(node)) = (HumanBone::from_name(k), v.get("node").and_then(|n| n.as_u64())) {
                    if let Some(&b) = node_to_bone.get(&(node as usize)) {
                        human.bones.insert(h, b);
                    }
                }
            }
        } else if let Some(arr) = ext.pointer("/VRM/humanoid/humanBones").and_then(|b| b.as_array()) {
            for v in arr {
                let name = v.get("bone").and_then(|b| b.as_str()).unwrap_or("");
                // VRM 0.x thumb naming differs from 1.0.
                let mapped = match name {
                    "leftThumbProximal" => "leftThumbMetacarpal",
                    "leftThumbIntermediate" => "leftThumbProximal",
                    "rightThumbProximal" => "rightThumbMetacarpal",
                    "rightThumbIntermediate" => "rightThumbProximal",
                    n => n,
                };
                if let (Some(h), Some(node)) = (HumanBone::from_name(mapped), v.get("node").and_then(|n| n.as_i64())) {
                    if let Some(&b) = node_to_bone.get(&(node as usize)) {
                        human.bones.insert(h, b);
                    }
                }
            }
        }
    }
    if !human.bones.is_empty() {
        model.humanoid = human;
    }
    finalize(&mut model);
    Ok(model)
}

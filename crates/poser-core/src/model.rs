//! Imported asset data: meshes, materials, textures, skeleton, animations.
use crate::animation::AnimationClip;
use crate::humanoid::HumanoidMap;
use crate::skeleton::Skeleton;
use glam::{Mat4, Vec2, Vec3, Vec4};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct TextureData {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// RGBA8, sRGB.
    pub rgba: Arc<Vec<u8>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

#[derive(Clone, Debug)]
pub struct Material {
    pub name: String,
    pub base_color: Vec4,
    pub base_color_texture: Option<usize>,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            name: "Default".into(),
            base_color: Vec4::new(0.8, 0.8, 0.8, 1.0),
            base_color_texture: None,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
        }
    }
}

/// Sparse morph target (blend shape).
#[derive(Clone, Debug, Default)]
pub struct MorphTarget {
    pub name: String,
    pub indices: Vec<u32>,
    pub position_deltas: Vec<Vec3>,
    /// Same length as `position_deltas` when present.
    pub normal_deltas: Vec<Vec3>,
}

impl MorphTarget {
    /// Build a sparse target from dense deltas, dropping near-zero entries.
    pub fn from_dense(name: String, pos: &[Vec3], nrm: Option<&[Vec3]>) -> Self {
        let mut t = MorphTarget { name, ..Default::default() };
        for (i, d) in pos.iter().enumerate() {
            let n = nrm.map(|n| n[i]).unwrap_or(Vec3::ZERO);
            if d.length_squared() > 1e-14 || n.length_squared() > 1e-10 {
                t.indices.push(i as u32);
                t.position_deltas.push(*d);
                if nrm.is_some() {
                    t.normal_deltas.push(n);
                }
            }
        }
        t
    }
}

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub name: String,
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<Vec2>,
    pub colors: Vec<Vec4>,
    pub indices: Vec<u32>,
    /// Per-vertex joint indices into the model skeleton (empty for rigid meshes).
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    /// For rigid meshes: bone the mesh follows (if any).
    pub rigid_bone: Option<usize>,
    /// For rigid meshes: transform from mesh space to model space at rest
    /// (or to bone space when `rigid_bone` is set).
    pub rigid_transform: Mat4,
    pub morphs: Vec<MorphTarget>,
    pub material: usize,
    /// Optional per-vertex part id (used by procedural mannequin for click parts).
    pub part_ids: Vec<u16>,
    /// Force linear blend skinning (weights authored/approximated for LBS).
    pub linear_blend: bool,
}

impl Mesh {
    pub fn is_skinned(&self) -> bool {
        !self.joints.is_empty() && self.joints.len() == self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Recompute smooth vertex normals from triangles.
    pub fn compute_normals(&mut self) {
        let mut n = vec![Vec3::ZERO; self.positions.len()];
        for tri in self.indices.chunks_exact(3) {
            let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let fnrm = (self.positions[b] - self.positions[a]).cross(self.positions[c] - self.positions[a]);
            n[a] += fnrm;
            n[b] += fnrm;
            n[c] += fnrm;
        }
        for v in &mut n {
            *v = v.try_normalize().unwrap_or(Vec3::Y);
        }
        self.normals = n;
    }

    /// Make sure optional attribute arrays are either empty or full length.
    pub fn sanitize(&mut self) {
        let n = self.positions.len();
        if self.normals.len() != n {
            self.compute_normals();
        }
        if self.uvs.len() != n {
            self.uvs = vec![Vec2::ZERO; n];
        }
        if !self.colors.is_empty() && self.colors.len() != n {
            self.colors.clear();
        }
        if self.joints.len() != n || self.weights.len() != n {
            self.joints.clear();
            self.weights.clear();
        }
        for w in &mut self.weights {
            let s: f32 = w.iter().sum();
            if s > 1e-6 {
                for x in w.iter_mut() {
                    *x /= s;
                }
            } else {
                *w = [1.0, 0.0, 0.0, 0.0];
            }
        }
        self.indices.retain(|&i| (i as usize) < n);
        let tri_len = self.indices.len() / 3 * 3;
        self.indices.truncate(tri_len);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceFormat {
    Gltf,
    Vrm,
    Fbx,
    Obj,
    Pmx,
    Procedural,
}

/// A loaded character or prop. Shared (Arc) between scene instances.
#[derive(Clone, Debug)]
pub struct Model {
    pub name: String,
    pub source: SourceFormat,
    pub path: Option<std::path::PathBuf>,
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub textures: Vec<TextureData>,
    pub skeleton: Skeleton,
    pub humanoid: HumanoidMap,
    pub animations: Vec<AnimationClip>,
}

impl Model {
    pub fn empty(name: impl Into<String>, source: SourceFormat) -> Self {
        Self {
            name: name.into(),
            source,
            path: None,
            meshes: vec![],
            materials: vec![Material::default()],
            textures: vec![],
            skeleton: Skeleton::default(),
            humanoid: HumanoidMap::default(),
            animations: vec![],
        }
    }

    pub fn has_skeleton(&self) -> bool {
        !self.skeleton.bones.is_empty()
    }

    pub fn vertex_count(&self) -> usize {
        self.meshes.iter().map(|m| m.positions.len()).sum()
    }

    pub fn triangle_count(&self) -> usize {
        self.meshes.iter().map(|m| m.triangle_count()).sum()
    }

    /// Unique morph target names over all meshes (morph sliders are shared by name).
    pub fn morph_names(&self) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        for m in &self.meshes {
            for t in &m.morphs {
                if !names.contains(&t.name) {
                    names.push(t.name.clone());
                }
            }
        }
        names
    }

    /// Rest-pose vertex positions (model space) for every mesh.
    pub fn rest_positions(&self) -> Vec<Vec<Vec3>> {
        let pm = self.skeleton.evaluate(&self.skeleton.rest_locals(), None);
        self.meshes
            .iter()
            .map(|m| {
                let rigid = match m.rigid_bone {
                    Some(b) if b < pm.world.len() => pm.world[b] * m.rigid_transform,
                    _ => m.rigid_transform,
                };
                let mut out = crate::skinning::DeformedMesh::default();
                crate::skinning::deform(m, &[], &pm.skin, rigid, &mut out);
                out.positions
            })
            .collect()
    }

    /// Axis-aligned bounds of the rest-pose geometry in model space.
    pub fn rest_bounds(&self) -> (Vec3, Vec3) {
        let mut lo = Vec3::splat(f32::MAX);
        let mut hi = Vec3::splat(f32::MIN);
        for ps in self.rest_positions() {
            for p in ps {
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        if lo.x > hi.x { (Vec3::ZERO, Vec3::ZERO) } else { (lo, hi) }
    }

    /// (floor height, top, horizontal centre) at rest. For humanoids the floor is taken from
    /// vertices driven by the feet, so stray geometry below the soles is ignored.
    pub fn rest_extent(&self) -> (f32, f32, Vec3) {
        let pos = self.rest_positions();
        let (lo, hi) = self.rest_bounds();
        let feet: Vec<usize> = [
            crate::humanoid::HumanBone::LeftFoot,
            crate::humanoid::HumanBone::RightFoot,
            crate::humanoid::HumanBone::LeftToes,
            crate::humanoid::HumanBone::RightToes,
        ]
        .iter()
        .filter_map(|h| self.humanoid.get(*h))
        .collect();
        let mut floor = f32::MAX;
        if !feet.is_empty() {
            for (m, ps) in self.meshes.iter().zip(&pos) {
                if !m.is_skinned() {
                    continue;
                }
                for (i, p) in ps.iter().enumerate() {
                    let (j, w) = (m.joints[i], m.weights[i]);
                    let k = (0..4).max_by(|a, b| w[*a].total_cmp(&w[*b])).unwrap();
                    if feet.contains(&(j[k] as usize)) && w[k] > 0.5 {
                        floor = floor.min(p.y);
                    }
                }
            }
        }
        if floor == f32::MAX {
            floor = lo.y;
        }
        (floor, hi.y, (lo + hi) * 0.5)
    }
}

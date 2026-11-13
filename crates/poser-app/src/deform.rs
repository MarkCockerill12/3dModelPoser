//! Per-figure deformation cache: bone matrices, skinned vertices, highlight overlays.
use poser_core::glam::{Mat4, Vec3, Vec4};
use poser_core::model::Mesh;
use poser_core::scene::Scene;
use poser_core::skeleton::PoseMatrices;
use poser_core::skinning::{DeformedMesh, deform};
use std::collections::HashMap;

#[derive(Default)]
pub struct FigureCache {
    pub revision: u64,
    pub pose: PoseMatrices,
    pub world: Mat4,
    pub meshes: Vec<DeformedMesh>,
    pub overlays: Vec<Vec<Vec4>>,
    pub overlay_key: (Option<usize>, Option<usize>, bool),
    pub highlight_rev: u64,
    pub overlay_dirty: bool,
    pub deform_dirty: bool,
    pub center: Vec3,
    pub radius: f32,
}

#[derive(Default)]
pub struct DeformCache {
    pub figs: HashMap<u64, FigureCache>,
    /// Extra per-vertex highlights (edit-parts tool): figure id -> (mesh, vertices, colour).
    pub highlight: HashMap<u64, Vec<(usize, Vec<u32>, Vec4)>>,
    pub highlight_rev: u64,
}

fn rigid_matrix(mesh: &Mesh, pose: &PoseMatrices, root: Mat4) -> Mat4 {
    match mesh.rigid_bone {
        Some(b) if b < pose.world.len() => pose.world[b] * mesh.rigid_transform,
        _ => {
            let _ = root;
            mesh.rigid_transform
        }
    }
}

/// Weight of `bone` (and its descendants when `subtree`) for each vertex of `mesh`.
fn bone_weights(mesh: &Mesh, bone: usize, include: &[bool]) -> Vec<f32> {
    if !mesh.is_skinned() {
        let w = if mesh.rigid_bone.is_some_and(|b| b == bone || include.get(b).copied().unwrap_or(false)) { 1.0 } else { 0.0 };
        return vec![w; mesh.positions.len()];
    }
    mesh.joints
        .iter()
        .zip(&mesh.weights)
        .map(|(j, w)| {
            let mut s = 0.0;
            for k in 0..4 {
                let b = j[k] as usize;
                if b == bone || include.get(b).copied().unwrap_or(false) {
                    s += w[k];
                }
            }
            s
        })
        .collect()
}

impl DeformCache {
    /// Bring every figure up to date. Returns true when anything changed.
    pub fn update(&mut self, scene: &Scene, selected: Option<(usize, Option<usize>)>, hovered: Option<(usize, Option<usize>)>) -> bool {
        self.update_grouped(scene, selected, hovered, true)
    }

    pub fn update_grouped(&mut self, scene: &Scene, selected: Option<(usize, Option<usize>)>, hovered: Option<(usize, Option<usize>)>, group: bool) -> bool {
        let mut changed = false;
        let ids: Vec<u64> = scene.figures.iter().map(|f| f.id).collect();
        let before = self.figs.len();
        self.figs.retain(|k, _| ids.contains(k));
        changed |= self.figs.len() != before;

        // Pose evaluation first (attachments need parent bone matrices).
        let mut evals: Vec<Option<PoseMatrices>> = Vec::with_capacity(scene.figures.len());
        for f in &scene.figures {
            let c = self.figs.entry(f.id).or_default();
            if c.revision != f.revision {
                c.pose = f.evaluate();
                // Skin with limb twist spread onto twist helper bones (joints keep their place).
                let pairs = poser_core::skinning::twist_split_pairs(&f.model.skeleton, &f.model.humanoid);
                if !pairs.is_empty() {
                    let split = poser_core::skinning::apply_twist_split(&f.model.skeleton, &pairs, &f.pose.locals);
                    c.pose.skin = f.model.skeleton.evaluate(&split, f.shapes_opt()).skin;
                }
            }
            evals.push(Some(c.pose.clone()));
        }
        for (i, f) in scene.figures.iter().enumerate() {
            let world = scene.figure_world(i, &evals);
            let c = self.figs.get_mut(&f.id).unwrap();
            let moved = c.world != world;
            c.world = world;
            if c.revision != f.revision {
                c.revision = f.revision;
                c.meshes.resize_with(f.model.meshes.len(), DeformedMesh::default);
                let root = f.model.skeleton.root_transform;
                for (mi, mesh) in f.model.meshes.iter().enumerate() {
                    let w = f.mesh_morph_weights(mi);
                    let rigid = rigid_matrix(mesh, &c.pose, root);
                    deform(mesh, &w, &c.pose.skin, rigid, &mut c.meshes[mi]);
                }
                // Bounds in world space.
                let mut lo = Vec3::splat(f32::MAX);
                let mut hi = Vec3::splat(f32::MIN);
                for m in &c.meshes {
                    for p in m.positions.iter().step_by(7) {
                        lo = lo.min(*p);
                        hi = hi.max(*p);
                    }
                }
                if lo.x <= hi.x {
                    let center = (lo + hi) * 0.5;
                    c.center = center;
                    c.radius = (hi - lo).length() * 0.5;
                }
                c.deform_dirty = true;
                c.overlay_key.2 = false; // force overlay rebuild when vertex counts change
                changed = true;
            }
            if moved {
                changed = true;
            }
            // Overlays
            let sel_bone = selected.filter(|s| s.0 == i).and_then(|s| s.1);
            let hov_bone = hovered.filter(|h| h.0 == i).and_then(|h| h.1);
            let fig_selected = selected.is_some_and(|s| s.0 == i);
            let key = (sel_bone, hov_bone, fig_selected);
            if c.overlay_key != key || c.overlays.len() != f.model.meshes.len() || c.highlight_rev != self.highlight_rev {
                c.overlay_key = key;
                c.highlight_rev = self.highlight_rev;
                c.overlays.clear();
                // Helper bones (face, twist, toes) light up with their main joint.
                let group_of = |b: usize| -> Vec<bool> {
                    let sk = &f.model.skeleton;
                    let hm = &f.model.humanoid;
                    if !group || !hm.is_valid() {
                        return vec![];
                    }
                    (0..sk.len())
                        .map(|i| {
                            if hm.human_of(i).is_some() {
                                return false;
                            }
                            let mut x = i;
                            while let Some(p) = sk.bones[x].parent {
                                if p == b {
                                    return true;
                                }
                                if hm.human_of(p).is_some() {
                                    return false;
                                }
                                x = p;
                            }
                            false
                        })
                        .collect()
                };
                let hov_inc = hov_bone.map(group_of).unwrap_or_default();
                let sel_inc = sel_bone.map(group_of).unwrap_or_default();
                for mesh in &f.model.meshes {
                    let n = mesh.positions.len();
                    let mut ov = vec![Vec4::ZERO; n];
                    if let Some(b) = hov_bone.filter(|h| Some(*h) != sel_bone) {
                        for (o, w) in ov.iter_mut().zip(bone_weights(mesh, b, &hov_inc)) {
                            if w > 0.0 {
                                *o = Vec4::new(1.0, 1.0, 1.0, 0.18 * w);
                            }
                        }
                    }
                    if let Some(b) = sel_bone {
                        for (o, w) in ov.iter_mut().zip(bone_weights(mesh, b, &sel_inc)) {
                            if w > 0.0 {
                                *o = Vec4::new(1.0, 0.45, 0.05, 0.42 * w);
                            }
                        }
                    }
                    c.overlays.push(ov);
                }
                for (m, verts, col) in self.highlight.get(&f.id).map(|v| v.as_slice()).unwrap_or(&[]) {
                    if let Some(ov) = c.overlays.get_mut(*m) {
                        for v in verts {
                            if let Some(o) = ov.get_mut(*v as usize) {
                                *o = *col;
                            }
                        }
                    }
                }
                c.overlay_dirty = true;
                changed = true;
            }
        }
        changed
    }

    pub fn scene_bounds(&self, scene: &Scene) -> (Vec3, f32) {
        let mut lo = Vec3::splat(f32::MAX);
        let mut hi = Vec3::splat(f32::MIN);
        for f in scene.figures.iter().filter(|f| f.visible) {
            if let Some(c) = self.figs.get(&f.id) {
                let wc = c.world.transform_point3(c.center);
                let r = c.radius * c.world.x_axis.truncate().length().max(c.world.y_axis.truncate().length());
                lo = lo.min(wc - Vec3::splat(r));
                hi = hi.max(wc + Vec3::splat(r));
            }
        }
        if lo.x > hi.x {
            return (Vec3::new(0.0, 0.9, 0.0), 1.0);
        }
        ((lo + hi) * 0.5, ((hi - lo).length() * 0.5).max(0.3))
    }
}

use crate::transform::Transform;
use glam::{Mat3, Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct Bone {
    pub name: String,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    /// Rest (bind) local transform relative to the parent.
    pub rest: Transform,
    /// Model space -> bone space at bind time.
    pub inverse_bind: Mat4,
    /// Unit direction (bone local space) from the bone head toward its tail.
    pub axis: Vec3,
    /// Rest length in bone-local units.
    pub length: f32,
    /// MMD-style "append" transform: copy (a fraction of) another bone's posed rotation.
    pub inherit: Option<(usize, f32)>,
}

impl Bone {
    pub fn new(name: impl Into<String>, parent: Option<usize>, rest: Transform) -> Self {
        Self {
            name: name.into(),
            parent,
            children: vec![],
            rest,
            inverse_bind: Mat4::IDENTITY,
            axis: Vec3::Y,
            length: 0.1,
            inherit: None,
        }
    }
}

/// Bones are always stored parents-before-children.
#[derive(Clone, Debug)]
pub struct Skeleton {
    pub bones: Vec<Bone>,
    /// Transform of any non-bone ancestors (armature object) applied above root bones.
    pub root_transform: Mat4,
}

impl Default for Skeleton {
    fn default() -> Self {
        Self { bones: vec![], root_transform: Mat4::IDENTITY }
    }
}

/// Per-bone proportion scaling (non-inherited, Design-Doll style).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoneShape {
    /// Scale along the bone axis. Children are pushed out, not scaled.
    pub length: f32,
    /// Scale perpendicular to the bone axis (thickness).
    pub girth: f32,
}

impl Default for BoneShape {
    fn default() -> Self {
        Self { length: 1.0, girth: 1.0 }
    }
}

impl BoneShape {
    pub fn is_identity(&self) -> bool {
        (self.length - 1.0).abs() < 1e-6 && (self.girth - 1.0).abs() < 1e-6
    }

    /// 3x3 scale matrix aligned to `axis`: g*I + (l-g)*a*a^T.
    pub fn matrix(&self, axis: Vec3) -> Mat3 {
        let a = axis;
        let g = self.girth;
        let d = self.length - self.girth;
        Mat3::from_cols(
            Vec3::new(g + d * a.x * a.x, d * a.y * a.x, d * a.z * a.x),
            Vec3::new(d * a.x * a.y, g + d * a.y * a.y, d * a.z * a.y),
            Vec3::new(d * a.x * a.z, d * a.y * a.z, g + d * a.z * a.z),
        )
    }
}

/// Result of evaluating a pose.
#[derive(Clone, Debug, Default)]
pub struct PoseMatrices {
    /// Bone world (model space) matrices, without the bone's own shape scale.
    pub world: Vec<Mat4>,
    /// Skinning matrices: world * shape * inverse_bind.
    pub skin: Vec<Mat4>,
}

impl Skeleton {
    pub fn len(&self) -> usize {
        self.bones.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bones.is_empty()
    }

    pub fn find(&self, name: &str) -> Option<usize> {
        self.bones.iter().position(|b| b.name == name)
    }

    pub fn rest_locals(&self) -> Vec<Transform> {
        self.bones.iter().map(|b| b.rest).collect()
    }

    pub fn rest_world(&self) -> Vec<Mat4> {
        self.evaluate(&self.rest_locals(), None).world
    }

    pub fn is_ancestor(&self, ancestor: usize, mut bone: usize) -> bool {
        while let Some(p) = self.bones[bone].parent {
            if p == ancestor {
                return true;
            }
            bone = p;
        }
        false
    }

    /// Chain of bones from `from` (exclusive ancestor) down to `to` inclusive. Returns None
    /// when `from` is not an ancestor of `to`.
    pub fn chain(&self, from: usize, to: usize) -> Option<Vec<usize>> {
        let mut out = vec![to];
        let mut b = to;
        while let Some(p) = self.bones[b].parent {
            if p == from {
                out.reverse();
                return Some(out);
            }
            out.push(p);
            b = p;
        }
        None
    }

    pub fn evaluate(&self, locals: &[Transform], shapes: Option<&[BoneShape]>) -> PoseMatrices {
        let n = self.bones.len();
        let mut world = Vec::with_capacity(n);
        let mut skin = Vec::with_capacity(n);
        for (i, bone) in self.bones.iter().enumerate() {
            let mut local = locals.get(i).copied().unwrap_or(bone.rest);
            if let Some((src, w)) = bone.inherit {
                if let (Some(sl), Some(sb)) = (locals.get(src), self.bones.get(src)) {
                    let delta = sl.rotation * sb.rest.rotation.inverse();
                    let d = Quat::IDENTITY.slerp(delta, w);
                    local.rotation = (d * local.rotation).normalize();
                }
            }
            let (parent_world, parent_shape) = match bone.parent {
                Some(p) => (
                    world[p],
                    shapes.map(|s| s[p]).filter(|s| !s.is_identity()).map(|s| s.matrix(self.bones[p].axis)),
                ),
                None => (self.root_transform, None),
            };
            let t = match parent_shape {
                Some(m) => m * local.translation,
                None => local.translation,
            };
            let w: Mat4 = parent_world * Mat4::from_scale_rotation_translation(local.scale, local.rotation, t);
            world.push(w);
            let own = shapes.map(|s| s[i]).filter(|s| !s.is_identity());
            let s = match own {
                Some(sh) => w * Mat4::from_mat3(sh.matrix(bone.axis)) * bone.inverse_bind,
                None => w * bone.inverse_bind,
            };
            skin.push(s);
        }
        PoseMatrices { world, skin }
    }

    /// Rebuild children lists from parent links.
    pub fn rebuild_children(&mut self) {
        for b in &mut self.bones {
            b.children.clear();
        }
        for i in 0..self.bones.len() {
            if let Some(p) = self.bones[i].parent {
                self.bones[p].children.push(i);
            }
        }
    }

    /// Number of descendants for each bone.
    pub fn descendant_counts(&self) -> Vec<usize> {
        let mut c = vec![0usize; self.bones.len()];
        for i in (0..self.bones.len()).rev() {
            if let Some(p) = self.bones[i].parent {
                c[p] += c[i] + 1;
            }
        }
        c
    }

    /// Compute `axis` / `length` for every bone. `preferred` may return a specific child to
    /// aim at (e.g. hips -> spine for humanoids).
    pub fn compute_axes(&mut self, preferred: impl Fn(usize) -> Option<usize>) {
        self.rebuild_children();
        let counts = self.descendant_counts();
        let n = self.bones.len();
        let mut axes = vec![(Vec3::Y, 0.0f32); n];
        let mut has = vec![false; n];
        for i in 0..n {
            let b = &self.bones[i];
            let target = preferred(i).filter(|c| self.bones[*c].parent == Some(i)).or_else(|| {
                b.children
                    .iter()
                    .copied()
                    .filter(|c| self.bones[*c].rest.translation.length_squared() > 1e-12)
                    .max_by_key(|c| counts[*c])
            });
            if let Some(c) = target {
                let t = self.bones[c].rest.translation;
                let len = t.length();
                if len > 1e-6 {
                    axes[i] = (t / len, len);
                    has[i] = true;
                }
            }
        }
        // Leaves: continue the parent's direction, half its length.
        for i in 0..n {
            if !has[i] {
                if let Some(p) = self.bones[i].parent {
                    let (pa, pl) = axes[p];
                    let rot = self.bones[i].rest.rotation;
                    let a = (rot.inverse() * pa).normalize_or(Vec3::Y);
                    let s = self.bones[i].rest.scale.max_element().max(1e-6);
                    axes[i] = (a, (pl * 0.5 / s).max(1e-4));
                } else {
                    axes[i] = (Vec3::Y, 0.1);
                }
            }
        }
        for (b, (a, l)) in self.bones.iter_mut().zip(axes) {
            b.axis = a;
            b.length = l;
        }
    }

    /// Recompute inverse bind matrices from the current rest pose.
    /// Make the rest pose match the bind pose for bones an exporter left displaced (e.g. a
    /// game prop bone whose node transform puts a briefcase 60 cm from the hand). At rest every
    /// bone's skin matrix should be the same (undeformed mesh); bones that disagree with the
    /// transform most bones share are moved to where the mesh was bound. Bones must be ordered
    /// parents-first. Returns how many bones were moved.
    pub fn align_rest_to_bind(&mut self, has_bind: &[bool]) -> usize {
        let n = self.bones.len();
        let world = self.rest_world();
        let d: Vec<Option<Mat4>> = (0..n).map(|b| has_bind.get(b).copied().unwrap_or(false).then(|| world[b] * self.bones[b].inverse_bind)).collect();
        let ext = world.iter().map(|m| m.w_axis.truncate().length()).fold(0.0f32, f32::max).max(1e-3);
        let close = |a: &Mat4, b: &Mat4| {
            (0..3).all(|c| (a.col(c) - b.col(c)).truncate().length() <= 3e-3 * a.col(c).truncate().length().max(1e-6))
                && (a.w_axis - b.w_axis).truncate().length() <= 3e-3 * ext
        };
        let cands: Vec<usize> = (0..n).filter(|b| d[*b].is_some()).collect();
        if cands.len() < 2 || self.bones.iter().enumerate().any(|(i, b)| b.parent.is_some_and(|p| p >= i)) {
            return 0;
        }
        let step = (cands.len() / 64).max(1);
        let mut best = (0usize, cands[0]);
        for &r in cands.iter().step_by(step) {
            let c = cands.iter().filter(|b| close(&d[r].unwrap(), &d[**b].unwrap())).count();
            if c > best.0 {
                best = (c, r);
            }
        }
        // Only when most bones agree (otherwise the file is consistent in its own way).
        if best.0 * 2 < cands.len() || best.0 == cands.len() {
            return 0;
        }
        let c = d[best.1].unwrap();
        let mut fixed = 0;
        let mut new_world: Vec<Mat4> = Vec::with_capacity(n);
        for b in 0..n {
            let pw = match self.bones[b].parent {
                Some(p) => new_world[p],
                None => self.root_transform,
            };
            if let Some(db) = d[b] {
                if !close(&c, &db) {
                    let want = c * self.bones[b].inverse_bind.inverse();
                    self.bones[b].rest = Transform::from_mat4(&(pw.inverse() * want));
                    fixed += 1;
                }
            }
            new_world.push(pw * self.bones[b].rest.to_mat4());
        }
        fixed
    }

    pub fn recompute_inverse_bind(&mut self) {
        let w = self.rest_world();
        for (b, m) in self.bones.iter_mut().zip(w) {
            b.inverse_bind = m.inverse();
        }
    }
}

/// Convert a world-space rotation delta applied to `bone` into a new local rotation.
pub fn apply_world_rotation(local: Quat, parent_world_rot: Quat, delta_world: Quat) -> Quat {
    let world = parent_world_rot * local;
    (parent_world_rot.inverse() * delta_world * world).normalize()
}

/// Extract pure rotation from an affine matrix (handles non-uniform scale).
pub fn mat_rotation(m: &Mat4) -> Quat {
    let (_, r, _) = m.to_scale_rotation_translation();
    r.normalize()
}

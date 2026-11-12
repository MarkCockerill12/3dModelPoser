//! Pose storage, portable (canonical) poses, mirroring.
use crate::humanoid::{HumanBone, HumanoidMap};
use crate::skeleton::{BoneShape, Skeleton, mat_rotation};
use crate::transform::Transform;
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Portable pose keyed by canonical bones, expressed in the normalised T-pose frame.
/// Works on any humanoid regardless of its rest pose or bone orientation conventions.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct HumanPose {
    pub name: String,
    #[serde(default)]
    pub category: String,
    pub rotations: BTreeMap<HumanBone, Quat>,
    /// Hips displacement from the T-pose position, in units of hips height (model space).
    #[serde(default)]
    pub hips_offset: Option<Vec3>,
}

fn mirror_quat(q: Quat) -> Quat {
    Quat::from_xyzw(q.x, -q.y, -q.z, q.w)
}

impl HumanPose {
    pub fn mirrored(&self) -> Self {
        let mut out = self.clone();
        out.rotations = self.rotations.iter().map(|(b, q)| (b.mirror(), mirror_quat(*q))).collect();
        out.hips_offset = self.hips_offset.map(|v| Vec3::new(-v.x, v.y, v.z));
        out
    }

    /// Keep only bones matching the predicate.
    pub fn filtered(&self, keep: impl Fn(HumanBone) -> bool) -> Self {
        let mut out = self.clone();
        out.rotations.retain(|b, _| keep(*b));
        if !keep(HumanBone::Hips) {
            out.hips_offset = None;
        }
        out
    }

    pub fn blend(&self, other: &Self, t: f32) -> Self {
        let mut out = self.clone();
        for (b, q) in &other.rotations {
            let a = self.rotations.get(b).copied().unwrap_or(Quat::IDENTITY);
            let mut q2 = *q;
            if a.dot(q2) < 0.0 {
                q2 = -q2;
            }
            out.rotations.insert(*b, a.slerp(q2, t));
        }
        if let (Some(a), Some(b)) = (self.hips_offset, other.hips_offset) {
            out.hips_offset = Some(a.lerp(b, t));
        }
        out
    }
}

/// Normalised world rotation N_x = W_x * Wt_x^-1 for each bone.
fn normalized_world(map: &HumanoidMap, world: &[Mat4]) -> Vec<Quat> {
    world
        .iter()
        .zip(&map.tpose_world)
        .map(|(m, t)| (mat_rotation(m) * t.inverse()).normalize())
        .collect()
}

fn mapped_ancestor(skel: &Skeleton, rev: &[Option<HumanBone>], mut b: usize) -> Option<usize> {
    while let Some(p) = skel.bones[b].parent {
        if rev[p].is_some() {
            return Some(p);
        }
        b = p;
    }
    None
}

/// Capture a portable pose from skeleton locals.
pub fn extract_human_pose(skel: &Skeleton, map: &HumanoidMap, locals: &[Transform]) -> HumanPose {
    let mut out = HumanPose::default();
    if !map.is_valid() || map.tpose_world.len() != skel.len() {
        return out;
    }
    let world = skel.evaluate(locals, None).world;
    let nw = normalized_world(map, &world);
    let rev = map.reverse(skel.len());
    for (h, &b) in &map.bones {
        // Root bones are relative to the model frame (identity normalised rotation).
        let parent_n = mapped_ancestor(skel, &rev, b).map(|p| nw[p]).unwrap_or(Quat::IDENTITY);
        let d = (parent_n.inverse() * nw[b]).normalize();
        out.rotations.insert(*h, d);
    }
    if let Some(h) = map.get(HumanBone::Hips) {
        let mut tl = skel.rest_locals();
        for (i, t) in tl.iter_mut().enumerate() {
            t.rotation = map.tpose_local[i];
        }
        let tw = skel.evaluate(&tl, None).world;
        let d = world[h].w_axis.truncate() - tw[h].w_axis.truncate();
        out.hips_offset = Some(d / map.hips_height);
    }
    out
}

/// Apply a portable pose onto skeleton locals. Bones not in the pose are left untouched unless
/// `reset_missing` is set, in which case mapped bones go back to the T-pose and others to rest.
pub fn apply_human_pose(
    skel: &Skeleton,
    map: &HumanoidMap,
    pose: &HumanPose,
    locals: &mut [Transform],
    shapes: Option<&[BoneShape]>,
    reset_missing: bool,
) {
    if !map.is_valid() || map.tpose_world.len() != skel.len() {
        return;
    }
    let rev = map.reverse(skel.len());
    if reset_missing {
        for (i, l) in locals.iter_mut().enumerate() {
            l.rotation = if rev[i].is_some() { map.tpose_local[i] } else { skel.bones[i].rest.rotation };
            if i != map.get(HumanBone::Hips).unwrap_or(usize::MAX) {
                l.translation = skel.bones[i].rest.translation;
            }
        }
    }
    // Normalised frames of mapped ancestors must be computed top-down from the current pose.
    for i in 0..skel.len() {
        let Some(h) = rev[i] else { continue };
        let Some(d) = pose.rotations.get(&h) else { continue };
        // N_parentmapped from current locals.
        let world = skel.evaluate(locals, shapes).world;
        let pn = match mapped_ancestor(skel, &rev, i) {
            Some(p) => (mat_rotation(&world[p]) * map.tpose_world[p].inverse()).normalize(),
            None => Quat::IDENTITY,
        };
        let target_world = pn * *d * map.tpose_world[i];
        let parent_world = match skel.bones[i].parent {
            Some(p) => mat_rotation(&world[p]),
            None => mat_rotation(&skel.root_transform),
        };
        locals[i].rotation = (parent_world.inverse() * target_world).normalize();
    }
    if let (Some(h), Some(off)) = (map.get(HumanBone::Hips), pose.hips_offset) {
        let mut tl = skel.rest_locals();
        for (i, t) in tl.iter_mut().enumerate() {
            t.rotation = map.tpose_local[i];
        }
        let tw = skel.evaluate(&tl, shapes).world;
        let target = tw[h].w_axis.truncate() + off * map.hips_height;
        let world = skel.evaluate(locals, shapes).world;
        let parent = match skel.bones[h].parent {
            Some(p) => world[p],
            None => skel.root_transform,
        };
        let mut local_t = parent.inverse().transform_point3(target);
        if let (Some(p), Some(sh)) = (skel.bones[h].parent, shapes) {
            // Undo the parent shape scaling applied to child offsets.
            let m = sh[p].matrix(skel.bones[p].axis);
            local_t = m.inverse() * local_t;
        }
        locals[h].translation = local_t;
    }
}

/// Pose for one figure instance.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Pose {
    pub locals: Vec<Transform>,
}

impl Pose {
    pub fn rest(skel: &Skeleton) -> Self {
        Self { locals: skel.rest_locals() }
    }

    pub fn tpose(skel: &Skeleton, map: &HumanoidMap) -> Self {
        let mut p = Self::rest(skel);
        if map.tpose_local.len() == p.locals.len() {
            for (l, r) in p.locals.iter_mut().zip(&map.tpose_local) {
                l.rotation = *r;
            }
        }
        p
    }
}

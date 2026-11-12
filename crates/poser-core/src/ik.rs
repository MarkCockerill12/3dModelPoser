//! Inverse kinematics on skeleton locals.
use crate::skeleton::{BoneShape, Skeleton, mat_rotation};
use crate::transform::Transform;
use glam::{Mat4, Quat, Vec3};

fn pos(m: &Mat4) -> Vec3 {
    m.w_axis.truncate()
}

fn parent_rot(skel: &Skeleton, world: &[Mat4], b: usize) -> Quat {
    match skel.bones[b].parent {
        Some(p) => mat_rotation(&world[p]),
        None => mat_rotation(&skel.root_transform),
    }
}

/// Apply a world-space rotation delta to `bone`'s local rotation.
pub fn rotate_bone_world(skel: &Skeleton, world: &[Mat4], locals: &mut [Transform], bone: usize, delta: Quat) {
    let pr = parent_rot(skel, world, bone);
    let w = pr * locals[bone].rotation;
    locals[bone].rotation = (pr.inverse() * delta * w).normalize();
}

/// Analytic two-bone IK (root -> mid -> end), e.g. upper arm / forearm / hand.
/// Keeps the current bend plane; `bend_hint` is used when the chain is straight
/// (direction the middle joint should move towards).
#[allow(clippy::too_many_arguments)]
pub fn two_bone_ik(
    skel: &Skeleton,
    shapes: Option<&[BoneShape]>,
    locals: &mut [Transform],
    root: usize,
    mid: usize,
    end: usize,
    target: Vec3,
    bend_hint: Vec3,
) {
    let world = skel.evaluate(locals, shapes).world;
    let a = pos(&world[root]);
    let b = pos(&world[mid]);
    let c = pos(&world[end]);
    let t = target;
    let lab = (b - a).length();
    let lcb = (b - c).length();
    if lab < 1e-6 || lcb < 1e-6 {
        return;
    }
    let eps = 1e-4 * (lab + lcb);
    let lat = (t - a).length().clamp(eps, lab + lcb - eps);
    let acos = |x: f32| x.clamp(-1.0, 1.0).acos();
    let ac = (c - a).normalize_or_zero();
    let ab = (b - a).normalize_or_zero();
    let ba = -ab;
    let bc = (c - b).normalize_or_zero();
    let at = (t - a).normalize_or_zero();
    let ac_ab_0 = acos(ac.dot(ab));
    let ba_bc_0 = acos(ba.dot(bc));
    let ac_at_0 = acos(ac.dot(at));
    let ac_ab_1 = acos((lcb * lcb - lab * lab - lat * lat) / (-2.0 * lab * lat));
    let ba_bc_1 = acos((lat * lat - lab * lab - lcb * lcb) / (-2.0 * lab * lcb));

    let mut axis0 = ac.cross(ab);
    if axis0.length_squared() < 1e-8 {
        // Straight chain: bend towards the hint.
        axis0 = ac.cross(bend_hint.normalize_or_zero());
        if axis0.length_squared() < 1e-8 {
            axis0 = ac.any_orthonormal_vector();
        }
    }
    let axis0 = axis0.normalize();
    let mut axis1 = ac.cross(at);
    let r2 = if axis1.length_squared() < 1e-10 {
        Quat::IDENTITY
    } else {
        axis1 = axis1.normalize();
        Quat::from_axis_angle(axis1, ac_at_0)
    };
    let r0 = Quat::from_axis_angle(axis0, ac_ab_1 - ac_ab_0);
    let r1 = Quat::from_axis_angle(axis0, ba_bc_1 - ba_bc_0);

    // Mid first (in the pre-rotation world frame), then root.
    rotate_bone_world(skel, &world, locals, mid, r1);
    rotate_bone_world(skel, &world, locals, root, r2 * r0);
}

/// Twist a two-bone chain around the root->end axis (elbow/knee direction, the
/// "donut" ring in Design Doll). End position stays fixed.
pub fn swing_chain(
    skel: &Skeleton,
    shapes: Option<&[BoneShape]>,
    locals: &mut [Transform],
    root: usize,
    end: usize,
    end_bone_keeps_world_rotation: bool,
    angle: f32,
) {
    let world = skel.evaluate(locals, shapes).world;
    let a = pos(&world[root]);
    let c = pos(&world[end]);
    let axis = (c - a).normalize_or_zero();
    if axis == Vec3::ZERO {
        return;
    }
    let end_world = mat_rotation(&world[end]);
    rotate_bone_world(skel, &world, locals, root, Quat::from_axis_angle(axis, angle));
    if end_bone_keeps_world_rotation {
        let w2 = skel.evaluate(locals, shapes).world;
        let pr = parent_rot(skel, &w2, end);
        locals[end].rotation = (pr.inverse() * end_world).normalize();
    }
}

/// Cyclic coordinate descent over `chain` (root first) so that `effector` reaches `target`.
/// `limit` is called after each step to clamp the local rotation.
#[allow(clippy::too_many_arguments)]
pub fn ccd(
    skel: &Skeleton,
    shapes: Option<&[BoneShape]>,
    locals: &mut [Transform],
    chain: &[usize],
    effector: usize,
    target: Vec3,
    iterations: usize,
    mut limit: impl FnMut(usize, &mut Transform),
    damping: f32,
) {
    for _ in 0..iterations {
        let mut done = false;
        for &j in chain.iter().rev() {
            let world = skel.evaluate(locals, shapes).world;
            let e = pos(&world[effector]);
            if (e - target).length_squared() < 1e-8 {
                done = true;
                break;
            }
            let p = pos(&world[j]);
            let to_e = (e - p).normalize_or_zero();
            let to_t = (target - p).normalize_or_zero();
            if to_e == Vec3::ZERO || to_t == Vec3::ZERO {
                continue;
            }
            let full = Quat::from_rotation_arc(to_e, to_t);
            let delta = Quat::IDENTITY.slerp(full, damping.clamp(0.01, 1.0));
            rotate_bone_world(skel, &world, locals, j, delta);
            limit(j, &mut locals[j]);
        }
        if done {
            break;
        }
    }
}

/// Rotate `bone` so that its axis points at `target` (world space), keeping twist minimal.
/// Bend a chain (root first) so `effector` moves towards `target`, spreading the bend evenly
/// over every joint (a smooth curve, like a real spine) instead of CCD's "all in one joint".
/// `max_angle[i]` caps the extra rotation of joint i (radians); `limit` clamps each joint.
#[allow(clippy::too_many_arguments)]
pub fn distribute_aim(
    skel: &Skeleton,
    shapes: Option<&[BoneShape]>,
    locals: &mut [Transform],
    chain: &[usize],
    effector: usize,
    target: Vec3,
    max_angle: &[f32],
    mut limit: impl FnMut(usize, &mut Transform),
) {
    let n = chain.len();
    for pass in 0..2 {
        for (i, &b) in chain.iter().enumerate() {
            let world = skel.evaluate(locals, shapes).world;
            let p = pos(&world[b]);
            let e = (pos(&world[effector]) - p).normalize_or_zero();
            let t = (target - p).normalize_or_zero();
            if e == Vec3::ZERO || t == Vec3::ZERO {
                continue;
            }
            let (axis, angle) = Quat::from_rotation_arc(e, t).to_axis_angle();
            // First pass: even share of what is left; second pass: small correction.
            let share = if pass == 0 { 1.0 / (n - i) as f32 } else { 0.5 / n as f32 };
            let cap = max_angle.get(i).copied().unwrap_or(f32::MAX);
            let a = (angle * share).min(cap);
            if a.abs() < 1e-5 || !axis.is_finite() {
                continue;
            }
            rotate_bone_world(skel, &world, locals, b, Quat::from_axis_angle(axis, a));
            limit(b, &mut locals[b]);
        }
    }
}

pub fn aim_bone(skel: &Skeleton, shapes: Option<&[BoneShape]>, locals: &mut [Transform], bone: usize, target: Vec3) {
    let world = skel.evaluate(locals, shapes).world;
    let p = pos(&world[bone]);
    let axis_world = (mat_rotation(&world[bone]) * skel.bones[bone].axis).normalize_or_zero();
    let want = (target - p).normalize_or_zero();
    if axis_world == Vec3::ZERO || want == Vec3::ZERO {
        return;
    }
    rotate_bone_world(skel, &world, locals, bone, Quat::from_rotation_arc(axis_world, want));
}

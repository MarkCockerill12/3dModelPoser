//! CPU morph + linear blend skinning. Results feed both the renderer and picking.
use crate::model::Mesh;
use glam::{Mat3, Mat4, Quat, Vec3, Vec4};
use std::sync::atomic::{AtomicBool, Ordering};

/// Dual-quaternion skinning (volume preserving) on/off.
pub static DUAL_QUATERNION: AtomicBool = AtomicBool::new(true);

/// Rigid joint as a dual quaternion (real, dual). None when the matrix has scale.
fn to_dq(m: &Mat4) -> Option<(Quat, Quat)> {
    let (s, r, t) = m.to_scale_rotation_translation();
    if (s - Vec3::ONE).abs().max_element() > 1e-3 {
        return None;
    }
    let d = Quat::from_xyzw(t.x, t.y, t.z, 0.0) * r * 0.5;
    Some((r, d))
}
use rayon::prelude::*;

/// Deformed vertex data for one mesh.
#[derive(Clone, Debug, Default)]
pub struct DeformedMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
}

/// Morph weights for a mesh, matched by its morph target order.
pub fn apply_morphs(mesh: &Mesh, weights: &[f32], pos: &mut [Vec3], nrm: &mut [Vec3]) {
    for (t, w) in mesh.morphs.iter().zip(weights) {
        if w.abs() < 1e-5 {
            continue;
        }
        for (k, &i) in t.indices.iter().enumerate() {
            let i = i as usize;
            if i < pos.len() {
                pos[i] += t.position_deltas[k] * *w;
                if let Some(d) = t.normal_deltas.get(k) {
                    nrm[i] += *d * *w;
                }
            }
        }
    }
}

/// Skin `mesh` with `skin` matrices (or a rigid matrix) into `out`.
pub fn deform(mesh: &Mesh, morph_weights: &[f32], skin: &[Mat4], rigid: Mat4, out: &mut DeformedMesh) {
    let n = mesh.positions.len();
    out.positions.clear();
    out.positions.extend_from_slice(&mesh.positions);
    out.normals.clear();
    out.normals.extend_from_slice(&mesh.normals);
    if out.normals.len() != n {
        out.normals.resize(n, Vec3::Y);
    }
    if morph_weights.iter().any(|w| w.abs() > 1e-5) {
        apply_morphs(mesh, morph_weights, &mut out.positions, &mut out.normals);
    }
    if mesh.is_skinned() && !skin.is_empty() {
        let nmats: Vec<Mat3> = skin.iter().map(|m| Mat3::from_mat4(*m).inverse().transpose()).collect();
        let dqs: Vec<Option<(Quat, Quat)>> = if DUAL_QUATERNION.load(Ordering::Relaxed) && !mesh.linear_blend { skin.iter().map(to_dq).collect() } else { vec![None; skin.len()] };
        let joints = &mesh.joints;
        let weights = &mesh.weights;
        let nj = skin.len();
        out.positions
            .par_iter_mut()
            .zip(out.normals.par_iter_mut())
            .enumerate()
            .with_min_len(1024)
            .for_each(|(i, (p, nr))| {
                let j = joints[i];
                let w = weights[i];
                // Dual quaternion blend when every influence is rigid.
                let all_rigid = (0..4).all(|k| w[k] <= 0.0 || dqs.get(j[k] as usize).is_some_and(|d| d.is_some()));
                if all_rigid {
                    let mut real = Vec4::ZERO;
                    let mut dual = Vec4::ZERO;
                    let mut pivot: Option<Quat> = None;
                    for k in 0..4 {
                        if w[k] <= 0.0 {
                            continue;
                        }
                        let (r, d) = dqs[j[k] as usize].unwrap();
                        let s = match pivot {
                            Some(p0) if p0.dot(r) < 0.0 => -w[k],
                            None => {
                                pivot = Some(r);
                                w[k]
                            }
                            _ => w[k],
                        };
                        real += Vec4::from(r) * s;
                        dual += Vec4::from(d) * s;
                    }
                    let len = real.length();
                    if len > 1e-8 {
                        let r = Quat::from_vec4(real / len);
                        let d = Quat::from_vec4(dual / len);
                        let t = (d * 2.0) * r.conjugate();
                        *p = r * *p + Vec3::new(t.x, t.y, t.z);
                        *nr = (r * *nr).normalize_or(Vec3::Y);
                        return;
                    }
                }
                let mut acc = Vec3::ZERO;
                let mut accn = Vec3::ZERO;
                for k in 0..4 {
                    let wk = w[k];
                    if wk <= 0.0 {
                        continue;
                    }
                    let ji = j[k] as usize;
                    if ji >= nj {
                        continue;
                    }
                    acc += skin[ji].transform_point3(*p) * wk;
                    accn += (nmats[ji] * *nr) * wk;
                }
                *p = acc;
                *nr = accn.normalize_or(Vec3::Y);
            });
    } else {
        let nm = Mat3::from_mat4(rigid).inverse().transpose();
        out.positions.par_iter_mut().with_min_len(4096).for_each(|p| *p = rigid.transform_point3(*p));
        out.normals.par_iter_mut().with_min_len(4096).for_each(|n| *n = (nm * *n).normalize_or(Vec3::Y));
    }
}

/// Twist helper bones (e.g. MakeHuman `upperarm02`, `lowerarm02`) that should share the twist
/// of their main limb bone: (main bone, helper, fraction of the twist the main bone keeps).
pub fn twist_split_pairs(skel: &crate::skeleton::Skeleton, hm: &crate::humanoid::HumanoidMap) -> Vec<(usize, usize, f32)> {
    use crate::humanoid::HumanBone as H;
    let mut out = vec![];
    if !hm.is_valid() {
        return out;
    }
    for (main, next, keep) in [
        (H::LeftUpperArm, H::LeftLowerArm, 0.5),
        (H::RightUpperArm, H::RightLowerArm, 0.5),
        (H::LeftLowerArm, H::LeftHand, 0.2),
        (H::RightLowerArm, H::RightHand, 0.2),
        (H::LeftUpperLeg, H::LeftLowerLeg, 0.5),
        (H::RightUpperLeg, H::RightLowerLeg, 0.5),
        (H::LeftLowerLeg, H::LeftFoot, 0.4),
        (H::RightLowerLeg, H::RightFoot, 0.4),
    ] {
        let (Some(m), Some(n)) = (hm.get(main), hm.get(next)) else { continue };
        // A single unmapped bone between the main bone and the next joint, lying on the main
        // bone's axis, is a twist helper.
        let Some(h) = skel.bones[n].parent.filter(|p| *p != m && skel.bones[*p].parent == Some(m) && hm.human_of(*p).is_none()) else { continue };
        let t = skel.bones[h].rest.translation;
        let axis = skel.bones[m].axis;
        if t.length() < 1e-6 || t.normalize().dot(axis) < 0.98 {
            continue;
        }
        out.push((m, h, keep));
    }
    out
}

/// Move part of each main limb bone's twist (about its own axis) onto its twist helper. Every
/// other bone keeps exactly the same world transform, so this only changes how skin deforms.
pub fn apply_twist_split(skel: &crate::skeleton::Skeleton, pairs: &[(usize, usize, f32)], locals: &[crate::transform::Transform]) -> Vec<crate::transform::Transform> {
    let mut out = locals.to_vec();
    for &(m, h, keep) in pairs {
        let axis = skel.bones[m].axis;
        let delta = (skel.bones[m].rest.rotation.inverse() * out[m].rotation).normalize();
        let delta = if delta.w < 0.0 { -delta } else { delta };
        let v = Vec3::new(delta.x, delta.y, delta.z);
        let twist = 2.0 * v.dot(axis).atan2(delta.w);
        let x = twist * (1.0 - keep);
        if x.abs() < 1e-5 {
            continue;
        }
        let r = Quat::from_axis_angle(axis, x);
        out[m].rotation = (out[m].rotation * r.inverse()).normalize();
        out[h].rotation = (r * out[h].rotation).normalize();
        out[h].translation = r * out[h].translation;
    }
    out
}

#[cfg(test)]
mod twist_tests {
    use super::*;

    #[test]
    fn twist_split_keeps_joints_in_place() {
        let m = crate::makehuman::build(&crate::makehuman::HumanParams::default());
        let pairs = twist_split_pairs(&m.skeleton, &m.humanoid);
        assert!(pairs.len() >= 4, "found {} twist helpers", pairs.len());
        let mut locals = m.skeleton.rest_locals();
        for &(b, _, _) in &pairs {
            let ax = m.skeleton.bones[b].axis;
            locals[b].rotation = locals[b].rotation * Quat::from_rotation_x(0.4) * Quat::from_axis_angle(ax, 1.1);
        }
        let split = apply_twist_split(&m.skeleton, &pairs, &locals);
        let a = m.skeleton.evaluate(&locals, None).world;
        let b = m.skeleton.evaluate(&split, None).world;
        let helpers: Vec<usize> = pairs.iter().flat_map(|p| [p.0, p.1]).collect();
        for i in 0..a.len() {
            if helpers.contains(&i) {
                // Position never changes; only the twist about the bone axis does.
                assert!((a[i].w_axis - b[i].w_axis).length() < 1e-4, "bone {i} moved");
                continue;
            }
            assert!(a[i].abs_diff_eq(b[i], 1e-4), "bone {i} ({}) changed", m.skeleton.bones[i].name);
        }
    }
}

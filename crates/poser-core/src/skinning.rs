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

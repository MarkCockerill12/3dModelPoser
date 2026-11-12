//! Humanoid auto-rigging: marker guessing, skeleton fitting and bone-heat skin weights
//! (heat diffusion after Baran & Popovic, "Automatic Rigging and Animation of 3D Characters").
use anyhow::{Result, anyhow};
use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet};
use glam::{Mat4, Vec3};
use poser_core::humanoid::{HumanBone as H, HumanoidMap};
use poser_core::model::Model;
use poser_core::skeleton::{Bone, Skeleton};
use poser_core::transform::Transform;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Mixamo-style landmark markers (model space).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Markers {
    pub chin: Vec3,
    pub left_wrist: Vec3,
    pub right_wrist: Vec3,
    pub left_elbow: Vec3,
    pub right_elbow: Vec3,
    pub left_knee: Vec3,
    pub right_knee: Vec3,
    pub groin: Vec3,
    pub floor: f32,
    pub top: f32,
}

pub const MARKER_NAMES: [&str; 8] = ["Chin", "Left wrist", "Right wrist", "Left elbow", "Right elbow", "Left knee", "Right knee", "Groin"];

impl Markers {
    pub fn get(&self, i: usize) -> Vec3 {
        [self.chin, self.left_wrist, self.right_wrist, self.left_elbow, self.right_elbow, self.left_knee, self.right_knee, self.groin][i]
    }

    pub fn set(&mut self, i: usize, v: Vec3) {
        match i {
            0 => self.chin = v,
            1 => self.left_wrist = v,
            2 => self.right_wrist = v,
            3 => self.left_elbow = v,
            4 => self.right_elbow = v,
            5 => self.left_knee = v,
            6 => self.right_knee = v,
            _ => self.groin = v,
        }
    }

    /// Mirror one side onto the other (keeps the rig symmetric while editing).
    pub fn symmetrize_from(&mut self, i: usize, center_x: f32) {
        let m = |v: Vec3| Vec3::new(2.0 * center_x - v.x, v.y, v.z);
        match i {
            1 => self.right_wrist = m(self.left_wrist),
            2 => self.left_wrist = m(self.right_wrist),
            3 => self.right_elbow = m(self.left_elbow),
            4 => self.left_elbow = m(self.right_elbow),
            5 => self.right_knee = m(self.left_knee),
            6 => self.left_knee = m(self.right_knee),
            _ => {}
        }
    }
}

/// All rest-pose vertices of a model in model space (subsampled to at most `max`).
fn sample_points(model: &Model, max: usize) -> Vec<Vec3> {
    let all: Vec<Vec3> = model.rest_positions().into_iter().flatten().collect();
    if all.len() <= max {
        return all;
    }
    let step = all.len().div_ceil(max);
    all.into_iter().step_by(step).collect()
}

/// Estimate marker positions for an upright character facing +Z (T- or A-pose).
pub fn guess_markers(model: &Model) -> Markers {
    let pts = sample_points(model, 60_000);
    let (lo, hi) = pts.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), p| (a.min(*p), b.max(*p)));
    let h = (hi.y - lo.y).max(1e-3);
    let y = |f: f32| lo.y + f * h;
    // Body centre line from the torso band.
    let torso: Vec<&Vec3> = pts.iter().filter(|p| p.y > y(0.55) && p.y < y(0.7)).collect();
    let cx = if torso.is_empty() { (lo.x + hi.x) * 0.5 } else { torso.iter().map(|p| p.x).sum::<f32>() / torso.len() as f32 };
    let band_center = |y0: f32, y1: f32, side: f32| -> Option<Vec3> {
        let v: Vec<&Vec3> = pts.iter().filter(|p| p.y > y0 && p.y < y1 && (p.x - cx) * side > 0.0).collect();
        if v.is_empty() {
            return None;
        }
        let n = v.len() as f32;
        Some(v.iter().fold(Vec3::ZERO, |a, p| a + **p) / n)
    };
    let cz = torso.iter().map(|p| p.z).sum::<f32>() / torso.len().max(1) as f32;
    // Chin: front of the head at ~87% height.
    let chin_y = y(0.865);
    let head: Vec<&Vec3> = pts.iter().filter(|p| (p.y - chin_y).abs() < h * 0.02 && (p.x - cx).abs() < h * 0.05).collect();
    let chin_z = head.iter().map(|p| p.z).fold(cz, f32::max);
    let chin = Vec3::new(cx, chin_y, (chin_z + cz) * 0.5 + (chin_z - cz) * 0.3);
    // Hands: extreme points left/right above the hips (arms out or down).
    let shoulder_y = y(0.81);
    let arm_pts = |side: f32| -> Vec3 {
        pts.iter()
            .filter(|p| p.y > y(0.35))
            .copied()
            .max_by(|a, b| ((a.x - cx) * side).total_cmp(&((b.x - cx) * side)))
            .unwrap_or(Vec3::new(cx + side * h * 0.4, shoulder_y, cz))
    };
    let mut out = Markers {
        chin,
        left_wrist: Vec3::ZERO,
        right_wrist: Vec3::ZERO,
        left_elbow: Vec3::ZERO,
        right_elbow: Vec3::ZERO,
        left_knee: Vec3::ZERO,
        right_knee: Vec3::ZERO,
        groin: Vec3::new(cx, y(0.47), cz),
        floor: lo.y,
        top: hi.y,
    };
    for side in [1.0f32, -1.0] {
        let tip = arm_pts(side);
        let shoulder = Vec3::new(cx + side * h * 0.11, shoulder_y, cz);
        let dir = (tip - shoulder).normalize_or(Vec3::X * side);
        let reach = (tip - shoulder).length();
        // Fingertip -> wrist: hand is ~19% of shoulder-to-fingertip.
        let wrist = shoulder + dir * reach * 0.8;
        let elbow = shoulder + dir * reach * 0.43 - Vec3::Z * h * 0.01;
        let knee = band_center(y(0.26), y(0.31), side).unwrap_or(Vec3::new(cx + side * h * 0.05, y(0.285), cz));
        if side > 0.0 {
            out.left_wrist = wrist;
            out.left_elbow = elbow;
            out.left_knee = knee;
        } else {
            out.right_wrist = wrist;
            out.right_elbow = elbow;
            out.right_knee = knee;
        }
    }
    // Groin: top of the empty gap between the legs on the centre line.
    let step = 0.008;
    let mut f = 0.2;
    let mut top_gap = None;
    let mut seen_gap = false;
    while f < 0.62 {
        let occupied = pts.iter().any(|p| (p.x - cx).abs() < h * 0.008 && p.y >= y(f) && p.y < y(f + step));
        if !occupied {
            seen_gap = true;
            top_gap = Some(f + step);
        } else if seen_gap {
            break;
        }
        f += step;
    }
    // Clothing (coats, skirts) can hide the crotch: keep it in the anatomical range.
    let gy = top_gap.map(y).unwrap_or(y(0.47)).clamp(y(0.42), y(0.5));
    out.groin = Vec3::new(cx, gy, cz);
    out
}

/// +1 when the character faces +Z, -1 when it faces -Z (toes point forward).
pub fn facing_sign(model: &Model) -> f32 {
    let pts = sample_points(model, 60_000);
    let (lo, hi) = pts.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), p| (a.min(*p), b.max(*p)));
    let h = (hi.y - lo.y).max(1e-3);
    let mean_z = |y0: f32, y1: f32| {
        let v: Vec<f32> = pts.iter().filter(|p| p.y >= lo.y + y0 * h && p.y < lo.y + y1 * h).map(|p| p.z).collect();
        if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 }
    };
    let feet = mean_z(0.0, 0.03);
    let shins = mean_z(0.12, 0.2);
    if feet >= shins { 1.0 } else { -1.0 }
}

/// Copy of `model` with all geometry baked into model space and turned to face +Z.
pub fn prepare_for_rigging(model: &Model) -> Model {
    let mut out = model.clone();
    let rest = model.rest_positions();
    let pm = model.skeleton.evaluate(&model.skeleton.rest_locals(), None);
    let flip = facing_sign(model) < 0.0;
    let rot = if flip { glam::Quat::from_rotation_y(std::f32::consts::PI) } else { glam::Quat::IDENTITY };
    for (mi, m) in out.meshes.iter_mut().enumerate() {
        let rigid = match m.rigid_bone {
            Some(b) if b < pm.world.len() => pm.world[b] * m.rigid_transform,
            _ => m.rigid_transform,
        };
        let nm = glam::Mat3::from_mat4(if m.is_skinned() { Mat4::IDENTITY } else { rigid }).inverse().transpose();
        m.positions = rest[mi].iter().map(|p| rot * *p).collect();
        for n in &mut m.normals {
            *n = (rot * (nm * *n)).normalize_or(Vec3::Y);
        }
        m.rigid_bone = None;
        m.rigid_transform = Mat4::IDENTITY;
        m.joints.clear();
        m.weights.clear();
        m.morphs.clear();
    }
    out.skeleton = Skeleton::default();
    out.humanoid = HumanoidMap::default();
    out.animations.clear();
    out
}

/// Bone layout derived from markers: (name, parent, position, humanoid bone).
pub fn skeleton_from_markers(m: &Markers) -> (Skeleton, HumanoidMap) {
    skeleton_from_markers_with_hands(m, None)
}

/// Skeleton from markers; with mesh points, finger bones are fitted to each hand.
pub fn skeleton_from_markers_with_hands(m: &Markers, points: Option<&[Vec3]>) -> (Skeleton, HumanoidMap) {
    let h = (m.top - m.floor).max(1e-3);
    let cx = (m.left_knee.x + m.right_knee.x + m.left_wrist.x + m.right_wrist.x) * 0.25;
    let cz = m.groin.z;
    let hips = Vec3::new(cx, m.groin.y + h * 0.05, cz);
    // Shoulder (upper arm head): back from the elbow by 1.15x the forearm length.
    let shoulder_of = |elbow: Vec3, wrist: Vec3| -> Vec3 {
        let fore = (wrist - elbow).length();
        let d = (elbow - wrist).normalize_or(Vec3::Y);
        let mut s = elbow + d * fore * 1.15;
        let min_x = h * 0.07;
        if (s.x - cx).abs() < min_x {
            s.x = cx + (s.x - cx).signum() * min_x;
        }
        s
    };
    let lsh = shoulder_of(m.left_elbow, m.left_wrist);
    let rsh = shoulder_of(m.right_elbow, m.right_wrist);
    let shoulder_y = (lsh.y + rsh.y) * 0.5;
    let neck = Vec3::new(cx, shoulder_y + (m.chin.y - shoulder_y) * 0.35, cz - h * 0.01);
    let head = Vec3::new(cx, m.chin.y + h * 0.01, (neck.z + m.chin.z) * 0.5 - h * 0.01);
    let top = Vec3::new(cx, m.top, head.z);
    let lerp = |a: Vec3, b: Vec3, t: f32| a + (b - a) * t;
    let spine = lerp(hips, neck, 0.2);
    let chest = lerp(hips, neck, 0.5);
    let upper = lerp(hips, neck, 0.78);
    let mut bones: Vec<(String, Option<usize>, Vec3, Option<H>)> = vec![];
    let mut add = |n: &str, p: Option<usize>, pos: Vec3, hb: Option<H>| {
        bones.push((n.to_string(), p, pos, hb));
        bones.len() - 1
    };
    let b_hips = add("Hips", None, hips, Some(H::Hips));
    let b_spine = add("Spine", Some(b_hips), spine, Some(H::Spine));
    let b_chest = add("Chest", Some(b_spine), chest, Some(H::Chest));
    let b_upper = add("UpperChest", Some(b_chest), upper, Some(H::UpperChest));
    let b_neck = add("Neck", Some(b_upper), neck, Some(H::Neck));
    let b_head = add("Head", Some(b_neck), head, Some(H::Head));
    add("HeadTop", Some(b_head), top, None);
    for (side, sh, elbow, wrist, knee) in [
        ("L", lsh, m.left_elbow, m.left_wrist, m.left_knee),
        ("R", rsh, m.right_elbow, m.right_wrist, m.right_knee),
    ] {
        let left = side == "L";
        let hb = |l: H| if left { l } else { l.mirror() };
        let sx = (sh.x - cx).signum();
        let clav = Vec3::new(cx + sx * h * 0.02, shoulder_y - h * 0.005, cz);
        let c = add(&format!("Shoulder.{side}"), Some(b_upper), clav, Some(hb(H::LeftShoulder)));
        let ua = add(&format!("UpperArm.{side}"), Some(c), sh, Some(hb(H::LeftUpperArm)));
        let la = add(&format!("LowerArm.{side}"), Some(ua), elbow, Some(hb(H::LeftLowerArm)));
        let hand = add(&format!("Hand.{side}"), Some(la), wrist, Some(hb(H::LeftHand)));
        let fdir = (wrist - elbow).normalize_or(Vec3::X * sx);
        match points.and_then(|pts| measure_hand(pts, wrist, fdir, h)) {
            Some((len, width)) => {
                let wv = fdir.cross(Vec3::Y).normalize_or(Vec3::Z) * fdir.x.signum();
                let fingers = [
                    (poser_core::humanoid::Finger::Index, 0.62, 0.95),
                    (poser_core::humanoid::Finger::Middle, 0.2, 1.0),
                    (poser_core::humanoid::Finger::Ring, -0.2, 0.94),
                    (poser_core::humanoid::Finger::Little, -0.6, 0.78),
                ];
                let sd = if left { poser_core::humanoid::Side::Left } else { poser_core::humanoid::Side::Right };
                for (f, off, lf) in fingers {
                    let total = len * 0.58 * lf;
                    let mut pos = wrist + fdir * len * 0.42 + wv * off * width * 0.5;
                    let mut parent = hand;
                    for (seg, frac) in [0.45, 0.3, 0.25].iter().enumerate() {
                        parent = add(&format!("{f:?}{seg}.{side}"), Some(parent), pos, Some(poser_core::humanoid::finger_bone(sd, f, seg).unwrap()));
                        pos += fdir * total * frac;
                    }
                    add(&format!("{f:?}Tip.{side}"), Some(parent), pos, None);
                }
                let tdir = (fdir * 0.6 + wv * 0.8).normalize();
                let mut pos = wrist + fdir * len * 0.1 + wv * width * 0.4;
                let mut parent = hand;
                for (seg, frac) in [0.38, 0.32, 0.3].iter().enumerate() {
                    parent = add(&format!("Thumb{seg}.{side}"), Some(parent), pos, Some(poser_core::humanoid::finger_bone(sd, poser_core::humanoid::Finger::Thumb, seg).unwrap()));
                    pos += tdir * len * 0.5 * frac;
                }
                add(&format!("ThumbTip.{side}"), Some(parent), pos, None);
            }
            None => {
                add(&format!("HandTip.{side}"), Some(hand), wrist + fdir * h * 0.1, None);
            }
        }
        let hip_j = Vec3::new(cx + (knee.x - cx) * 0.95, m.groin.y + h * 0.02, cz);
        let ul = add(&format!("UpperLeg.{side}"), Some(b_hips), hip_j, Some(hb(H::LeftUpperLeg)));
        let ll = add(&format!("LowerLeg.{side}"), Some(ul), knee, Some(hb(H::LeftLowerLeg)));
        let ankle = Vec3::new(knee.x, m.floor + h * 0.045, knee.z - h * 0.01);
        let ft = add(&format!("Foot.{side}"), Some(ll), ankle, Some(hb(H::LeftFoot)));
        let toes = Vec3::new(knee.x, m.floor + h * 0.012, ankle.z + h * 0.09);
        let t = add(&format!("Toes.{side}"), Some(ft), toes, Some(hb(H::LeftToes)));
        add(&format!("ToeTip.{side}"), Some(t), toes + Vec3::Z * h * 0.04, None);
    }
    let mut skel = Skeleton::default();
    let mut map = HumanoidMap::default();
    for (i, (n, p, pos, hb)) in bones.iter().enumerate() {
        let pw = p.map(|p| bones[p].2).unwrap_or(Vec3::ZERO);
        skel.bones.push(Bone::new(n.clone(), *p, Transform::from_translation(*pos - pw)));
        if let Some(hb) = hb {
            map.bones.insert(*hb, i);
        }
    }
    skel.recompute_inverse_bind();
    (skel, map)
}

/// (hand length, palm width) measured from mesh points beyond the wrist.
fn measure_hand(pts: &[Vec3], wrist: Vec3, dir: Vec3, h: f32) -> Option<(f32, f32)> {
    let wv = dir.cross(Vec3::Y).normalize_or(Vec3::Z);
    let near: Vec<(f32, f32)> = pts
        .iter()
        .filter_map(|p| {
            let d = *p - wrist;
            let t = d.dot(dir);
            let radial = (d - dir * t).length();
            (t > 0.0 && t < h * 0.14 && radial < h * 0.06).then(|| (t, d.dot(wv)))
        })
        .collect();
    if near.len() < 20 {
        return None;
    }
    let len = near.iter().map(|x| x.0).fold(0.0, f32::max);
    let band: Vec<f32> = near.iter().filter(|x| x.0 > len * 0.3 && x.0 < len * 0.5).map(|x| x.1).collect();
    let width = if band.is_empty() { len * 0.45 } else { band.iter().fold(f32::MIN, |a, b| a.max(*b)) - band.iter().fold(f32::MAX, |a, b| a.min(*b)) };
    (len > h * 0.05).then_some((len, width.clamp(len * 0.3, len * 0.7)))
}

/// Distance from p to segment ab.
fn seg_dist(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
    (a + ab * t - p).length()
}

/// Bone-heat skin weights for every vertex of `positions` (model space, all meshes concatenated).
/// `tris` index into `positions`. Returns per-vertex (joint, weight) top-4.
pub fn heat_weights(positions: &[Vec3], normals: &[Vec3], tris: &[[u32; 3]], skel: &Skeleton, deform: &[bool], region: &[Option<(Vec3, f32)>]) -> Result<Vec<([u16; 4], [f32; 4])>> {
    let n_in = positions.len();
    if n_in == 0 {
        return Ok(vec![]);
    }
    // 0) Connected parts (seams welded by position) - clusters never cross parts.
    let extent0 = positions.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), p| (a.min(*p), b.max(*p)));
    let extent0 = (extent0.1 - extent0.0).max_element().max(1e-6);
    let mut parent: Vec<u32> = (0..n_in as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    let union = |p: &mut Vec<u32>, a: u32, b: u32| {
        let (ra, rb) = (find(p, a), find(p, b));
        if ra != rb {
            p[ra as usize] = rb;
        }
    };
    for t in tris {
        union(&mut parent, t[0], t[1]);
        union(&mut parent, t[1], t[2]);
    }
    let eps = extent0 * 1e-5;
    let mut weld: HashMap<(i64, i64, i64), u32> = HashMap::new();
    for (i, p) in positions.iter().enumerate() {
        let k = ((p.x / eps) as i64, (p.y / eps) as i64, (p.z / eps) as i64);
        match weld.get(&k) {
            Some(&j) => union(&mut parent, i as u32, j),
            None => {
                weld.insert(k, i as u32);
            }
        }
    }
    let comp_of: Vec<u32> = (0..n_in as u32).map(|i| find(&mut parent, i)).collect();
    // 1) Vertex clustering keeps the solve small (also welds UV seams / split normals).
    let (lo, hi) = positions.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), p| (a.min(*p), b.max(*p)));
    let extent = (hi - lo).max_element().max(1e-6);
    let target = 30_000.0f32;
    let mut cell = extent / 400.0;
    let mut cluster_of: Vec<u32>;
    let mut centers: Vec<Vec3>;
    let mut nsum: Vec<Vec3>;
    loop {
        let mut map: HashMap<(u32, i32, i32, i32), u32> = HashMap::new();
        cluster_of = Vec::with_capacity(n_in);
        let mut sums: Vec<(Vec3, u32)> = vec![];
        nsum = vec![];
        for (vi, p) in positions.iter().enumerate() {
            let k = (comp_of[vi], ((p.x - lo.x) / cell) as i32, ((p.y - lo.y) / cell) as i32, ((p.z - lo.z) / cell) as i32);
            let id = *map.entry(k).or_insert_with(|| {
                sums.push((Vec3::ZERO, 0));
                nsum.push(Vec3::ZERO);
                (sums.len() - 1) as u32
            });
            nsum[id as usize] += normals.get(vi).copied().unwrap_or(Vec3::ZERO);
            sums[id as usize].0 += *p;
            sums[id as usize].1 += 1;
            cluster_of.push(id);
        }
        centers = sums.iter().map(|(s, c)| *s / *c as f32).collect();
        if centers.len() as f32 <= target * 1.3 || cell > extent {
            break;
        }
        cell *= (centers.len() as f32 / target).sqrt();
    }
    let n = centers.len();
    // 2) Graph Laplacian with inverse-length edge weights (robust on non-manifold meshes).
    let mut edges: HashMap<(u32, u32), f64> = HashMap::new();
    let mut area = vec![0.0f64; n];
    for t in tris {
        let c = [cluster_of[t[0] as usize], cluster_of[t[1] as usize], cluster_of[t[2] as usize]];
        if c[0] == c[1] || c[1] == c[2] || c[0] == c[2] {
            continue;
        }
        let (a, b, d) = (centers[c[0] as usize], centers[c[1] as usize], centers[c[2] as usize]);
        let ar = ((b - a).cross(d - a).length() * 0.5) as f64;
        for k in 0..3 {
            area[c[k] as usize] += ar / 3.0;
            let (i, j) = (c[k], c[(k + 1) % 3]);
            let key = if i < j { (i, j) } else { (j, i) };
            let len = (centers[i as usize] - centers[j as usize]).length().max(cell * 0.05) as f64;
            edges.insert(key, 1.0 / len);
        }
    }
    let mean_area = area.iter().sum::<f64>() / n as f64;
    for a in &mut area {
        if *a <= 0.0 {
            *a = mean_area.max(1e-12);
        }
    }
    // 3) Heat sources: nearest deforming bone segment(s).
    let world = skel.rest_world();
    let segs: Vec<(usize, Vec3, Vec3)> = (0..skel.len())
        .filter(|b| deform[*b])
        .map(|b| {
            let head = world[b].w_axis.truncate();
            let tail = head + world[b].transform_vector3(skel.bones[b].axis * skel.bones[b].length);
            (b, head, tail)
        })
        .collect();
    if segs.is_empty() {
        return Err(anyhow!("no deforming bones"));
    }
    let nb = segs.len();
    let mut h = vec![0.0f64; n];
    let mut p = vec![0.0f64; n * nb]; // column-major (vertex, bone)
    let closest = |p: Vec3, a: Vec3, b: Vec3| {
        let ab = b - a;
        let t = ((p - a).dot(ab) / ab.length_squared().max(1e-12)).clamp(0.0, 1.0);
        a + ab * t
    };
    for (i, c) in centers.iter().enumerate() {
        // Visibility approximation: a bone heats a vertex only if it lies behind the surface
        // (inside the body), i.e. against the vertex normal.
        let nrm = nsum[i].normalize_or_zero();
        let mut d: Vec<f32> = segs
            .iter()
            .map(|(bi, a, b)| {
                if let Some(Some((rc, rr))) = region.get(*bi) {
                    if c.distance(*rc) > *rr {
                        return f32::MAX;
                    }
                }
                let q = closest(*c, *a, *b);
                let dist = (q - *c).length();
                let facing = if nrm == Vec3::ZERO || dist < 1e-6 { -1.0 } else { nrm.dot((q - *c) / dist) };
                if facing > 0.25 { f32::MAX } else { dist }
            })
            .collect();
        if d.iter().all(|x| *x == f32::MAX) {
            d = segs.iter().map(|(_, a, b)| seg_dist(*c, *a, *b)).collect();
        }
        let dmin = d.iter().copied().fold(f32::MAX, f32::min).max(cell * 0.5);
        let near: Vec<usize> = (0..nb).filter(|k| d[*k] <= dmin * 1.05).collect();
        h[i] = 1.0 / (dmin as f64 * dmin as f64);
        for k in &near {
            p[k * n + i] = 1.0 / near.len() as f64;
        }
    }
    // 4) (L + M H) w = M H p
    let mut diag = vec![0.0f64; n];
    let mut trips: Vec<Triplet<usize, usize, f64>> = Vec::with_capacity(edges.len() * 2 + n);
    let scale = 1.0 / mean_area.max(1e-12); // keep magnitudes sane
    for ((i, j), w) in &edges {
        let (i, j) = (*i as usize, *j as usize);
        trips.push(Triplet::new(i, j, -*w));
        trips.push(Triplet::new(j, i, -*w));
        diag[i] += *w;
        diag[j] += *w;
    }
    for i in 0..n {
        trips.push(Triplet::new(i, i, diag[i] + area[i] * scale * h[i] + 1e-9));
    }
    let a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &trips).map_err(|e| anyhow!("matrix: {e:?}"))?;
    let llt = a.sp_cholesky(faer::Side::Lower).map_err(|e| anyhow!("cholesky: {e:?}"))?;
    let mut rhs = faer::Mat::<f64>::zeros(n, nb);
    for k in 0..nb {
        for i in 0..n {
            rhs[(i, k)] = area[i] * scale * h[i] * p[k * n + i];
        }
    }
    llt.solve_in_place(rhs.as_mut());
    // 5) Top-4 per cluster, then copy to the original vertices.
    let mut cw: Vec<([u16; 4], [f32; 4])> = Vec::with_capacity(n);
    for i in 0..n {
        let mut ws: Vec<(usize, f32)> = (0..nb)
            .map(|k| (segs[k].0, rhs[(i, k)].max(0.0) as f32))
            .filter(|(b, w)| *w > 0.01 && !matches!(region.get(*b), Some(Some((rc, rr))) if centers[i].distance(*rc) > *rr))
            .collect();
        ws.sort_by(|a, b| b.1.total_cmp(&a.1));
        ws.truncate(4);
        let s: f32 = ws.iter().map(|w| w.1).sum();
        let mut j = [0u16; 4];
        let mut w = [0f32; 4];
        if s <= 1e-6 {
            // Fallback: nearest bone.
            let c = centers[i];
            let best = segs.iter().min_by(|a, b| seg_dist(c, a.1, a.2).total_cmp(&seg_dist(c, b.1, b.2))).unwrap();
            j[0] = best.0 as u16;
            w[0] = 1.0;
        } else {
            for (k, (b, x)) in ws.into_iter().enumerate() {
                j[k] = b as u16;
                w[k] = x / s;
            }
        }
        cw.push((j, w));
    }
    let mut out: Vec<([u16; 4], [f32; 4])> = cluster_of.iter().map(|c| cw[*c as usize]).collect();
    // Small separate parts (props, eyes, teeth, buttons) move rigidly with their nearest bone.
    // Components are found on the original topology, welding split seam vertices by position.
    let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();
    for i in 0..n_in {
        groups.entry(comp_of[i]).or_default().push(i);
    }
    let biggest = groups.values().map(|g| g.len()).max().unwrap_or(0);
    for g in groups.values() {
        if g.len() == biggest || g.len() * 12 > n_in {
            continue;
        }
        let mut best = (f32::MAX, segs[0].0);
        for &v in g {
            for (b, a, t) in &segs {
                let d = seg_dist(positions[v], *a, *t);
                if d < best.0 {
                    best = (d, *b);
                }
            }
        }
        for &v in g {
            out[v] = ([best.1 as u16, 0, 0, 0], [1.0, 0.0, 0.0, 0.0]);
        }
    }
    let _ = &mut cw;
    Ok(out)
}

/// Rig an unrigged (or re-rig any) model as a humanoid. Returns a new model.
pub fn auto_rig(model: &Model, markers: &Markers) -> Result<Model> {
    let pts = sample_points(model, 200_000);
    let (skel, mut map) = skeleton_from_markers_with_hands(markers, Some(&pts));
    let mut out = model.clone();
    // Bring every mesh into model space at rest (drops any previous rig).
    let rest = model.rest_positions();
    let old_pm = model.skeleton.evaluate(&model.skeleton.rest_locals(), None);
    for (mi, m) in out.meshes.iter_mut().enumerate() {
        let rigid = match m.rigid_bone {
            Some(b) if b < old_pm.world.len() => old_pm.world[b] * m.rigid_transform,
            _ => m.rigid_transform,
        };
        let nm = glam::Mat3::from_mat4(if m.is_skinned() { Mat4::IDENTITY } else { rigid }).inverse().transpose();
        m.positions = rest[mi].clone();
        if !model.meshes[mi].is_skinned() {
            for n in &mut m.normals {
                *n = (nm * *n).normalize_or(Vec3::Y);
            }
        }
        m.rigid_bone = None;
        m.rigid_transform = Mat4::IDENTITY;
    }
    // Concatenate for the solve.
    let mut all = vec![];
    let mut alln = vec![];
    let mut tris = vec![];
    let mut offsets = vec![];
    for m in &out.meshes {
        let base = all.len() as u32;
        offsets.push(all.len());
        all.extend_from_slice(&m.positions);
        alln.extend_from_slice(&m.normals);
        for t in m.indices.chunks_exact(3) {
            tris.push([t[0] + base, t[1] + base, t[2] + base]);
        }
    }
    let deform: Vec<bool> = skel.bones.iter().map(|b| !b.name.contains("Tip") && !b.name.contains("HeadTop")).collect();
    let _ = &pts;
    let mut skel = skel;
    let prefer: Vec<Option<usize>> = (0..skel.len()).map(|i| map.human_of(i).and_then(|h| h.tail_child()).and_then(|t| map.get(t))).collect();
    skel.compute_axes(|i| prefer[i]);
    // Finger bones only influence the hand they belong to.
    let world = skel.rest_world();
    let region: Vec<Option<(Vec3, f32)>> = (0..skel.len())
        .map(|i| {
            let h = map.human_of(i)?;
            if !h.is_finger() {
                return None;
            }
            let hand = map.get(if h.side() == poser_core::humanoid::Side::Left { H::LeftHand } else { H::RightHand })?;
            let c = world[hand].w_axis.truncate();
            let r = (world[i].w_axis.truncate() - c).length() * 1.6 + skel.bones[i].length * 2.0;
            Some((c, r))
        })
        .collect();
    let weights = heat_weights(&all, &alln, &tris, &skel, &deform, &region)?;
    for (mi, m) in out.meshes.iter_mut().enumerate() {
        let o = offsets[mi];
        m.joints = (0..m.positions.len()).map(|i| weights[o + i].0).collect();
        m.weights = (0..m.positions.len()).map(|i| weights[o + i].1).collect();
        m.morphs.clear();
        m.linear_blend = true;
    }
    map.compute_tpose(&skel);
    out.skeleton = skel;
    out.humanoid = map;
    out.animations.clear();
    out.name = format!("{} (rigged)", model.name);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use poser_core::mannequin::{MannequinParams, build};

    #[test]
    fn rigs_a_mannequin_shell() {
        // Strip the mannequin's rig and auto-rig it again.
        let mut m = build(&MannequinParams::default());
        let rest = m.rest_positions();
        for (mesh, r) in m.meshes.iter_mut().zip(rest) {
            mesh.positions = r;
            mesh.joints.clear();
            mesh.weights.clear();
        }
        m.skeleton = Skeleton::default();
        m.humanoid = HumanoidMap::default();
        let mk = guess_markers(&m);
        assert!(mk.left_wrist.x > 0.4 && mk.right_wrist.x < -0.4, "{mk:?}");
        let r = auto_rig(&m, &mk).unwrap();
        assert!(r.humanoid.is_valid());
        assert!(r.meshes[0].is_skinned());
        // A vertex near the left wrist should be driven by the arm/hand.
        let la = r.humanoid.get(H::LeftHand).unwrap();
        let lf = r.humanoid.get(H::LeftLowerArm).unwrap();
        let i = r.meshes[0].positions.iter().enumerate().min_by(|a, b| a.1.distance(mk.left_wrist).total_cmp(&b.1.distance(mk.left_wrist))).unwrap().0;
        let j = r.meshes[0].joints[i];
        assert!(j.contains(&(la as u16)) || j.contains(&(lf as u16)), "{j:?}");
    }
}

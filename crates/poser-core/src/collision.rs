//! Simple body self-collision for Guided posing: a capsule is fitted to the skin around each
//! main humanoid bone, and limbs are tested against the torso, head, other limbs and the floor.
use crate::glam::{Mat4, Vec3};
use crate::humanoid::HumanBone as H;
use crate::model::Model;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Torso,
    Head,
    LeftArm,
    RightArm,
    LeftLeg,
    RightLeg,
}

#[derive(Clone, Debug)]
pub struct Capsule {
    pub human: H,
    pub bone: usize,
    /// Next joint along the limb; the capsule runs between fractions `ta`..`tb` of bone->end.
    pub end: Option<usize>,
    pub ta: f32,
    pub tb: f32,
    /// Endpoints in bone space (used when `end` is None).
    pub la: Vec3,
    pub lb: Vec3,
    pub radius: f32,
    pub part: Part,
}

#[derive(Clone, Copy, Debug)]
pub struct Contact {
    /// Capsule pair (indices into `caps`).
    pub a: usize,
    /// `None` = floor.
    pub b: Option<usize>,
    /// Penetration depth (model units).
    pub depth: f32,
    /// Direction that pushes capsule `a` out.
    pub normal: Vec3,
    /// Closest point on capsule `a`'s axis.
    pub point: Vec3,
}

#[derive(Clone, Debug, Default)]
pub struct BodyCollider {
    pub caps: Vec<Capsule>,
    /// Pairs to test (a, b); either side may be the one that moves.
    pub pairs: Vec<(usize, usize)>,
}

/// Fraction of the fitted radius used for contact, so skin can touch and squash a little.
const SOFT: f32 = 0.85;

const LIST: &[(H, Option<H>, Part)] = &[
    (H::Hips, Some(H::Spine), Part::Torso),
    (H::Spine, Some(H::Chest), Part::Torso),
    (H::Chest, Some(H::UpperChest), Part::Torso),
    (H::UpperChest, Some(H::Neck), Part::Torso),
    (H::Head, None, Part::Head),
    (H::LeftUpperArm, Some(H::LeftLowerArm), Part::LeftArm),
    (H::LeftLowerArm, Some(H::LeftHand), Part::LeftArm),
    (H::LeftHand, None, Part::LeftArm),
    (H::RightUpperArm, Some(H::RightLowerArm), Part::RightArm),
    (H::RightLowerArm, Some(H::RightHand), Part::RightArm),
    (H::RightHand, None, Part::RightArm),
    (H::LeftUpperLeg, Some(H::LeftLowerLeg), Part::LeftLeg),
    (H::LeftLowerLeg, Some(H::LeftFoot), Part::LeftLeg),
    (H::LeftFoot, Some(H::LeftToes), Part::LeftLeg),
    (H::RightUpperLeg, Some(H::RightLowerLeg), Part::RightLeg),
    (H::RightLowerLeg, Some(H::RightFoot), Part::RightLeg),
    (H::RightFoot, Some(H::RightToes), Part::RightLeg),
];

fn pos(m: &Mat4) -> Vec3 {
    m.w_axis.truncate()
}

fn percentile(v: &mut [f32], p: f32) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f32 * p).round() as usize]
}

/// Closest points between segments p1-q1 and p2-q2 (returns s, t parameters).
fn closest_segments(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (f32, f32) {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);
    if a <= 1e-12 && e <= 1e-12 {
        return (0.0, 0.0);
    }
    if a <= 1e-12 {
        return (0.0, (f / e).clamp(0.0, 1.0));
    }
    let c = d1.dot(r);
    if e <= 1e-12 {
        return ((-c / a).clamp(0.0, 1.0), 0.0);
    }
    let b = d1.dot(d2);
    let denom = a * e - b * b;
    let mut s = if denom > 1e-12 { ((b * f - c * e) / denom).clamp(0.0, 1.0) } else { 0.0 };
    let mut t = (b * s + f) / e;
    if t < 0.0 {
        t = 0.0;
        s = (-c / a).clamp(0.0, 1.0);
    } else if t > 1.0 {
        t = 1.0;
        s = ((b - c) / a).clamp(0.0, 1.0);
    }
    (s, t)
}

impl BodyCollider {
    /// Fit capsules to a humanoid model's rest-pose skin. None for non-humanoids.
    pub fn build(model: &Model) -> Option<Self> {
        let hm = &model.humanoid;
        let sk = &model.skeleton;
        if !hm.is_valid() || sk.bones.is_empty() {
            return None;
        }
        let rest = sk.rest_world();
        // Which capsule owns each skeleton bone's vertices (nearest main humanoid ancestor).
        let cap_of_human = |h: H| LIST.iter().position(|(x, _, _)| *x == h);
        let owner: Vec<Option<usize>> = (0..sk.len())
            .map(|b| {
                let mut x = Some(b);
                while let Some(i) = x {
                    if let Some(h) = hm.human_of(i) {
                        let h = match h {
                            H::Jaw | H::LeftEye | H::RightEye => H::Head,
                            H::LeftToes => H::LeftFoot,
                            H::RightToes => H::RightFoot,
                            H::Neck | H::LeftShoulder | H::RightShoulder => return None,
                            h => h,
                        };
                        return cap_of_human(h);
                    }
                    x = sk.bones[i].parent;
                }
                None
            })
            .collect();
        let mut pts: Vec<Vec<Vec3>> = vec![vec![]; LIST.len()];
        for (mesh, ps) in model.meshes.iter().zip(model.rest_positions()) {
            for (vi, p) in ps.iter().enumerate() {
                let b = if mesh.is_skinned() {
                    let (j, w) = (mesh.joints[vi], mesh.weights[vi]);
                    let k = (0..4).max_by(|a, b| w[*a].total_cmp(&w[*b])).unwrap_or(0);
                    if w[k] < 0.5 {
                        continue;
                    }
                    j[k] as usize
                } else {
                    match mesh.rigid_bone {
                        Some(b) => b,
                        None => continue,
                    }
                };
                if let Some(Some(c)) = owner.get(b) {
                    pts[*c].push(*p);
                }
            }
        }
        // Body axes at rest: left-right from the shoulders/hips, front-back perpendicular.
        let lr = |l: H, r: H| match (hm.get(l), hm.get(r)) {
            (Some(l), Some(r)) => Some(pos(&rest[l]) - pos(&rest[r])),
            _ => None,
        };
        let lateral = lr(H::LeftUpperArm, H::RightUpperArm).or(lr(H::LeftUpperLeg, H::RightUpperLeg)).unwrap_or(Vec3::X).normalize_or(Vec3::X);
        let up = (Vec3::Y - lateral * lateral.y).normalize_or(Vec3::Y);
        let depth_axis = lateral.cross(up).normalize_or(Vec3::Z);
        let mut caps = vec![];
        for (i, (h, end_h, part)) in LIST.iter().enumerate() {
            let Some(bone) = hm.get(*h) else { continue };
            let end_h = match (h, end_h) {
                (H::Chest, Some(_)) if hm.get(H::UpperChest).is_none() => Some(H::Neck),
                _ => *end_h,
            };
            let end = end_h.and_then(|e| hm.get(e));
            let head = pos(&rest[bone]);
            let tail = match end {
                Some(e) => pos(&rest[e]),
                None => {
                    let b = &sk.bones[bone];
                    head + rest[bone].transform_vector3(b.axis * b.length.max(1e-4))
                }
            };
            let d = tail - head;
            let len2 = d.length_squared();
            let p = &pts[i];
            if p.len() < 8 || len2 < 1e-10 {
                continue;
            }
            if *part == Part::Torso {
                // The torso is wide and shallow: a horizontal capsule across the body per slice.
                let n = p.len() as f32;
                let center = p.iter().copied().sum::<Vec3>() / n;
                let mut dx: Vec<f32> = p.iter().map(|v| (*v - center).dot(lateral).abs()).collect();
                let mut dz: Vec<f32> = p.iter().map(|v| (*v - center).dot(depth_axis).abs()).collect();
                let radius = percentile(&mut dz, 0.7);
                let half = (percentile(&mut dx, 0.85) - radius).max(0.0);
                let inv = rest[bone].inverse();
                caps.push(Capsule {
                    human: *h,
                    bone,
                    end: None,
                    ta: 0.0,
                    tb: 1.0,
                    la: inv.transform_point3(center - lateral * half),
                    lb: inv.transform_point3(center + lateral * half),
                    radius,
                    part: *part,
                });
                continue;
            }
            let mut ts: Vec<f32> = p.iter().map(|v| (*v - head).dot(d) / len2).collect();
            let mut rs: Vec<f32> = p.iter().zip(&ts).map(|(v, t)| (*v - (head + d * *t)).length()).collect();
            let radius = percentile(&mut rs, 0.4);
            let (t_lo, t_hi) = (percentile(&mut ts, 0.05), percentile(&mut ts, 0.95));
            let pull = radius / len2.sqrt();
            let (mut ta, mut tb) = (t_lo + pull, t_hi - pull);
            if ta > tb {
                ta = (t_lo + t_hi) * 0.5;
                tb = ta;
            }
            let inv = rest[bone].inverse();
            caps.push(Capsule {
                human: *h,
                bone,
                end,
                ta,
                tb,
                la: inv.transform_point3(head + d * ta),
                lb: inv.transform_point3(head + d * tb),
                radius,
                part: *part,
            });
        }
        // Pairs worth testing (adjacent parts that always overlap are skipped).
        let idx = |h: H| caps.iter().position(|c: &Capsule| c.human == h);
        let mut pairs = vec![];
        let mut add = |a: Option<usize>, b: Option<usize>| {
            if let (Some(a), Some(b)) = (a, b) {
                if !pairs.contains(&(a, b)) && !pairs.contains(&(b, a)) {
                    pairs.push((a, b));
                }
            }
        };
        let torso = [H::Hips, H::Spine, H::Chest, H::UpperChest];
        for (ua, la, hand, oua, ola, ohand) in [
            (H::LeftUpperArm, H::LeftLowerArm, H::LeftHand, H::RightUpperArm, H::RightLowerArm, H::RightHand),
            (H::RightUpperArm, H::RightLowerArm, H::RightHand, H::LeftUpperArm, H::LeftLowerArm, H::LeftHand),
        ] {
            for x in [la, hand] {
                for t in torso.iter().chain(&[H::Head, oua, ola, ohand, H::LeftUpperLeg, H::RightUpperLeg, H::LeftLowerLeg, H::RightLowerLeg]) {
                    add(idx(x), idx(*t));
                }
            }
            add(idx(ua), idx(H::Head));
        }
        for (ul, ll, ft, oul, oll, oft) in [
            (H::LeftUpperLeg, H::LeftLowerLeg, H::LeftFoot, H::RightUpperLeg, H::RightLowerLeg, H::RightFoot),
            (H::RightUpperLeg, H::RightLowerLeg, H::RightFoot, H::LeftUpperLeg, H::LeftLowerLeg, H::LeftFoot),
        ] {
            add(idx(ul), idx(oul));
            for x in [ll, ft] {
                for t in [oul, oll, oft, H::Head].iter().chain(&torso) {
                    add(idx(x), idx(*t));
                }
            }
            add(idx(ul), idx(H::Chest));
            add(idx(ul), idx(H::UpperChest));
        }
        Some(Self { caps, pairs })
    }

    /// Capsule axis endpoints for a pose (model space).
    pub fn segment(&self, c: &Capsule, world: &[Mat4]) -> (Vec3, Vec3) {
        match c.end {
            Some(e) => {
                let (h, t) = (pos(&world[c.bone]), pos(&world[e]));
                (h.lerp(t, c.ta), h.lerp(t, c.tb))
            }
            None => (world[c.bone].transform_point3(c.la), world[c.bone].transform_point3(c.lb)),
        }
    }

    /// All penetrating pairs, plus floor contacts when `floor` = (point, up normal) is given.
    /// Only contacts involving at least one capsule whose bone is in `moving` are reported,
    /// with `a` being the moving side.
    pub fn contacts(&self, world: &[Mat4], floor: Option<(Vec3, Vec3)>, moving: &dyn Fn(usize) -> bool) -> Vec<Contact> {
        let segs: Vec<(Vec3, Vec3)> = self.caps.iter().map(|c| self.segment(c, world)).collect();
        let mut out = vec![];
        for &(i, j) in &self.pairs {
            let (mi, mj) = (moving(self.caps[i].bone), moving(self.caps[j].bone));
            if !mi && !mj {
                continue;
            }
            let (a, b) = if mi { (i, j) } else { (j, i) };
            let (p1, q1) = segs[a];
            let (p2, q2) = segs[b];
            let (s, t) = closest_segments(p1, q1, p2, q2);
            let ca = p1.lerp(q1, s);
            let cb = p2.lerp(q2, t);
            let r = (self.caps[a].radius + self.caps[b].radius) * SOFT;
            let dv = ca - cb;
            let dist = dv.length();
            if dist < r {
                let normal = if dist > 1e-6 { dv / dist } else { Vec3::Y };
                out.push(Contact { a, b: Some(b), depth: r - dist, normal, point: ca });
            }
        }
        if let Some((fp, n)) = floor {
            for (i, c) in self.caps.iter().enumerate() {
                if !moving(c.bone) {
                    continue;
                }
                let (p, q) = segs[i];
                let (lp, dp) = ((p - fp).dot(n), (q - fp).dot(n));
                let (low, point) = if lp < dp { (lp, p) } else { (dp, q) };
                let depth = c.radius * SOFT - low;
                if depth > 0.0 {
                    out.push(Contact { a: i, b: None, depth, normal: n, point });
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_closest() {
        let (s, t) = closest_segments(Vec3::ZERO, Vec3::X, Vec3::new(0.5, 1.0, -1.0), Vec3::new(0.5, 1.0, 1.0));
        assert!((s - 0.5).abs() < 1e-5 && (t - 0.5).abs() < 1e-5);
    }
}

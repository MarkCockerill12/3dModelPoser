//! Procedural art mannequin (segmented doll) generated from proportion parameters.
//! Every body segment is rigidly bound to one bone, like a wooden drawing mannequin,
//! so posing never produces skinning artefacts and every part is clickable.
use crate::humanoid::{Finger, HumanBone, HumanoidMap, Side, finger_bone};
use crate::model::{Material, Mesh, Model, SourceFormat};
use crate::skeleton::{Bone, Skeleton};
use crate::transform::Transform;
use glam::{Mat4, Vec3, Vec4};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MannequinParams {
    pub height: f32,
    /// Total height divided by head height (anime ~6-7, realistic 7.5, heroic 8-9, chibi 2-4).
    pub heads_tall: f32,
    /// 0 = male, 1 = female.
    pub femininity: f32,
    pub muscle: f32,
    pub weight: f32,
    pub shoulder_width: f32,
    pub hip_width: f32,
    pub neck_length: f32,
    pub torso_length: f32,
    pub arm_length: f32,
    pub leg_length: f32,
    pub hand_size: f32,
    pub foot_size: f32,
    pub head_width: f32,
    pub breast: f32,
    /// Show joint balls between segments (wooden mannequin look).
    pub joint_balls: bool,
}

impl Default for MannequinParams {
    fn default() -> Self {
        Self {
            height: 1.75,
            heads_tall: 7.5,
            femininity: 0.0,
            muscle: 0.5,
            weight: 0.5,
            shoulder_width: 1.0,
            hip_width: 1.0,
            neck_length: 1.0,
            torso_length: 1.0,
            arm_length: 1.0,
            leg_length: 1.0,
            hand_size: 1.0,
            foot_size: 1.0,
            head_width: 1.0,
            breast: 0.0,
            joint_balls: true,
        }
    }
}

impl MannequinParams {
    pub fn female() -> Self {
        Self { height: 1.66, femininity: 1.0, muscle: 0.35, breast: 0.5, shoulder_width: 0.95, hip_width: 1.05, ..Default::default() }
    }

    /// Named body presets (Design Doll / Magic Poser style).
    pub fn presets() -> Vec<(&'static str, MannequinParams)> {
        let m = MannequinParams::default;
        let f = MannequinParams::female;
        vec![
            ("Male (realistic)", m()),
            ("Female (realistic)", f()),
            ("Heroic male (8.5 heads)", MannequinParams { heads_tall: 8.5, height: 1.88, muscle: 0.9, shoulder_width: 1.18, ..m() }),
            ("Heroic female (8.5 heads)", MannequinParams { heads_tall: 8.5, height: 1.78, muscle: 0.6, leg_length: 1.06, ..f() }),
            ("Anime male", MannequinParams { heads_tall: 7.0, height: 1.75, muscle: 0.35, weight: 0.35, leg_length: 1.08, head_width: 0.95, ..m() }),
            ("Anime female", MannequinParams { heads_tall: 6.5, height: 1.58, weight: 0.4, leg_length: 1.1, head_width: 0.95, breast: 0.4, ..f() }),
            ("Teen", MannequinParams { heads_tall: 6.8, height: 1.6, muscle: 0.3, weight: 0.4, shoulder_width: 0.92, ..m() }),
            ("Child", MannequinParams { heads_tall: 5.2, height: 1.15, muscle: 0.2, weight: 0.45, shoulder_width: 0.9, hand_size: 0.95, ..m() }),
            ("Chibi (3 heads)", MannequinParams { heads_tall: 3.0, height: 0.95, muscle: 0.1, weight: 0.6, arm_length: 0.85, leg_length: 0.8, hand_size: 1.1, foot_size: 1.15, ..m() }),
            ("Muscular", MannequinParams { muscle: 1.0, weight: 0.65, shoulder_width: 1.2, height: 1.85, ..m() }),
            ("Heavy", MannequinParams { muscle: 0.4, weight: 1.0, hip_width: 1.15, ..m() }),
            ("Skinny", MannequinParams { muscle: 0.15, weight: 0.1, shoulder_width: 0.92, ..m() }),
            ("Elderly", MannequinParams { muscle: 0.2, weight: 0.6, height: 1.68, heads_tall: 7.2, ..m() }),
            ("Long-legged fashion", MannequinParams { heads_tall: 9.0, height: 1.8, weight: 0.3, leg_length: 1.15, ..f() }),
        ]
    }
}

struct Builder {
    mesh: Mesh,
}

impl Builder {
    fn new() -> Self {
        Self { mesh: Mesh { name: "Mannequin".into(), rigid_transform: Mat4::IDENTITY, ..Default::default() } }
    }

    /// Lofted superellipse tube along `axis`, rigidly bound to `bone`.
    /// `profile(t) -> (rx, rz, ox, oz)` gives half-widths along `sx`/`sz` and centre offsets.
    #[allow(clippy::too_many_arguments)]
    fn loft(
        &mut self,
        bone: usize,
        origin: Vec3,
        axis: Vec3,
        sx: Vec3,
        sz: Vec3,
        length: f32,
        rings: usize,
        segs: usize,
        exponent: f32,
        cap: f32,
        profile: impl Fn(f32) -> (f32, f32, f32, f32),
    ) {
        let m = &mut self.mesh;
        let base = m.positions.len() as u32;
        let rings = rings.max(3);
        for i in 0..=rings {
            let s = i as f32 / rings as f32;
            // Denser rings at both ends for smooth caps.
            let t = (1.0 - (std::f32::consts::PI * s).cos()) * 0.5;
            let (rx, rz, ox, oz) = profile(t);
            // Rounded ends: superellipse cap.
            let u = (2.0 * t - 1.0).abs();
            let k = (1.0 - u.powf(cap)).max(0.0).powf(1.0 / cap);
            let c = origin + axis * (t * length) + sx * ox + sz * oz;
            for j in 0..segs {
                let a = j as f32 / segs as f32 * std::f32::consts::TAU;
                let (sa, ca) = a.sin_cos();
                let e = 2.0 / exponent;
                let x = ca.signum() * ca.abs().powf(e) * rx * k;
                let z = sa.signum() * sa.abs().powf(e) * rz * k;
                m.positions.push(c + sx * x + sz * z);
                m.joints.push([bone as u16, 0, 0, 0]);
                m.weights.push([1.0, 0.0, 0.0, 0.0]);
                m.part_ids.push(bone as u16);
            }
        }
        let s = segs as u32;
        for i in 0..rings as u32 {
            for j in 0..s {
                let a = base + i * s + j;
                let b = base + i * s + (j + 1) % s;
                let c = base + (i + 1) * s + j;
                let d = base + (i + 1) * s + (j + 1) % s;
                // Winding: outward normals for right-handed (sx, axis, sz) frames.
                m.indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
    }

    fn ellipsoid(&mut self, bone: usize, center: Vec3, r: Vec3, segs: usize) {
        self.loft(bone, center - Vec3::Y * r.y, Vec3::Y, Vec3::X, Vec3::Z, 2.0 * r.y, segs, segs, 2.0, 2.0, |_| (r.x, r.z, 0.0, 0.0));
    }

    fn sphere(&mut self, bone: usize, center: Vec3, r: f32) {
        self.ellipsoid(bone, center, Vec3::splat(r), 12);
    }
}

/// Bone layout + geometry. Returns a complete model (T-pose rest, humanoid mapped).
pub fn build(p: &MannequinParams) -> Model {
    let h = p.height.max(0.3);
    let f = p.femininity.clamp(0.0, 1.0);
    let mu = p.muscle.clamp(0.0, 1.0);
    let we = p.weight.clamp(0.0, 1.0);
    let head_h = h / p.heads_tall.clamp(2.0, 10.0);
    // Body below the chin is scaled to fill the remaining height.
    let body_h = h - head_h;
    let k = body_h / (h * 0.87);
    let wk = k.sqrt(); // widths scale less than heights (chibi stays chunky)
    let lerp = |a: f32, b: f32| a + (b - a) * f;
    let girth = (0.85 + 0.3 * we + 0.15 * mu) * wk;
    let limb = (0.82 + 0.25 * we + 0.3 * mu) * wk;

    // Vertical landmarks (fractions of a 7.5-head figure, scaled into the body height).
    let leg_scale = p.leg_length;
    let torso_scale = p.torso_length;
    let y = |frac: f32| frac * h * k;
    let ankle_y = y(0.045) * leg_scale;
    let knee_y = ankle_y + y(0.24) * leg_scale;
    let hip_y = knee_y + y(0.245) * leg_scale;
    let spine_y = hip_y + y(0.06) * torso_scale;
    let chest_y = spine_y + y(0.08) * torso_scale;
    let upper_chest_y = chest_y + y(0.07) * torso_scale;
    let neck_y = upper_chest_y + y(0.095) * torso_scale;
    let head_y = neck_y + y(0.045) * p.neck_length;
    let chin_y = head_y - head_h * 0.08;
    let top_y = chin_y + head_h;
    // Re-scale everything vertically so the final height matches `h` exactly.
    let s_fix = h / top_y.max(1e-3);
    let fy = |v: f32| v * s_fix;
    let (ankle_y, knee_y, hip_y, spine_y, chest_y, upper_chest_y, neck_y, head_y, chin_y) =
        (fy(ankle_y), fy(knee_y), fy(hip_y), fy(spine_y), fy(chest_y), fy(upper_chest_y), fy(neck_y), fy(head_y), fy(chin_y));
    let head_h = head_h * s_fix;
    let hk = h * k * s_fix; // scale for widths/lengths

    let shoulder_x = hk * lerp(0.118, 0.103) * p.shoulder_width;
    let hip_x = hk * lerp(0.052, 0.058) * p.hip_width;
    let upper_arm = hk * 0.186 * p.arm_length;
    let forearm = hk * 0.15 * p.arm_length;
    let palm = hk * 0.058 * p.hand_size;
    let arm_y = upper_chest_y + (neck_y - upper_chest_y) * 0.55;
    let foot_len = hk * 0.152 * p.foot_size;

    // ---- Skeleton ------------------------------------------------------------------------
    let mut bones: Vec<(String, Option<usize>, Vec3, Option<HumanBone>)> = vec![];
    let mut add = |name: &str, parent: Option<usize>, world: Vec3, hb: Option<HumanBone>| {
        bones.push((name.to_string(), parent, world, hb));
        bones.len() - 1
    };
    let hips = add("Hips", None, Vec3::new(0.0, hip_y, 0.0), Some(HumanBone::Hips));
    let spine = add("Spine", Some(hips), Vec3::new(0.0, spine_y, -0.004 * hk), Some(HumanBone::Spine));
    let chest = add("Chest", Some(spine), Vec3::new(0.0, chest_y, -0.006 * hk), Some(HumanBone::Chest));
    let upper_chest = add("UpperChest", Some(chest), Vec3::new(0.0, upper_chest_y, -0.008 * hk), Some(HumanBone::UpperChest));
    let neck = add("Neck", Some(upper_chest), Vec3::new(0.0, neck_y, -0.012 * hk), Some(HumanBone::Neck));
    let head = add("Head", Some(neck), Vec3::new(0.0, head_y, -0.006 * hk), Some(HumanBone::Head));
    let mut side_bones: HashMap<(Side, &str), usize> = HashMap::new();
    for side in [Side::Left, Side::Right] {
        let sx = if side == Side::Left { 1.0 } else { -1.0 };
        let sfx = if side == Side::Left { ".L" } else { ".R" };
        let hb = |l: HumanBone| if side == Side::Left { l } else { l.mirror() };
        let sh = add(&format!("Shoulder{sfx}"), Some(upper_chest), Vec3::new(sx * 0.02 * hk, arm_y, -0.006 * hk), Some(hb(HumanBone::LeftShoulder)));
        let ua = add(&format!("UpperArm{sfx}"), Some(sh), Vec3::new(sx * shoulder_x, arm_y, -0.01 * hk), Some(hb(HumanBone::LeftUpperArm)));
        let la = add(&format!("LowerArm{sfx}"), Some(ua), Vec3::new(sx * (shoulder_x + upper_arm), arm_y, -0.014 * hk), Some(hb(HumanBone::LeftLowerArm)));
        let wrist_x = shoulder_x + upper_arm + forearm;
        let hand = add(&format!("Hand{sfx}"), Some(la), Vec3::new(sx * wrist_x, arm_y, -0.01 * hk), Some(hb(HumanBone::LeftHand)));
        // Fingers
        let fl = [(Finger::Index, 0.86, 0.33), (Finger::Middle, 0.95, 0.11), (Finger::Ring, 0.89, -0.11), (Finger::Little, 0.72, -0.33)];
        let palm_w = palm * 0.84;
        for (finger, len, zf) in fl {
            let total = palm * len;
            let base = Vec3::new(sx * (wrist_x + palm), arm_y - 0.002 * hk, -0.01 * hk + zf * palm_w);
            let segs = [0.45, 0.3, 0.25];
            let mut parent = hand;
            let mut pos = base;
            for (si, frac) in segs.iter().enumerate() {
                let hbone = finger_bone(side, finger, si).unwrap();
                let nm = format!("{:?}{}{sfx}", finger, ["Proximal", "Intermediate", "Distal"][si]);
                parent = add(&nm, Some(parent), pos, Some(hbone));
                pos += Vec3::new(sx * total * frac, 0.0, 0.0);
            }
            side_bones.insert((side, finger_tip_key(finger)), add(&format!("{:?}Tip{sfx}", finger), Some(parent), pos, None));
        }
        // Thumb: points forward-out at 45 degrees.
        let tdir = Vec3::new(sx, 0.0, 1.0).normalize();
        let mut pos = Vec3::new(sx * (wrist_x + palm * 0.1), arm_y - 0.01 * hk, -0.01 * hk + palm_w * 0.32);
        let mut parent = hand;
        for (si, frac) in [0.38, 0.32, 0.27].iter().enumerate() {
            let hbone = finger_bone(side, Finger::Thumb, si).unwrap();
            let nm = format!("Thumb{}{sfx}", ["Metacarpal", "Proximal", "Distal"][si]);
            parent = add(&nm, Some(parent), pos, Some(hbone));
            pos += tdir * palm * 0.95 * frac;
        }
        side_bones.insert((side, "thumbtip"), add(&format!("ThumbTip{sfx}"), Some(parent), pos, None));
        // Legs
        let ul = add(&format!("UpperLeg{sfx}"), Some(hips), Vec3::new(sx * hip_x, hip_y - 0.02 * hk, 0.0), Some(hb(HumanBone::LeftUpperLeg)));
        let ll = add(&format!("LowerLeg{sfx}"), Some(ul), Vec3::new(sx * hip_x * 0.92, knee_y, 0.004 * hk), Some(hb(HumanBone::LeftLowerLeg)));
        let ft = add(&format!("Foot{sfx}"), Some(ll), Vec3::new(sx * hip_x * 0.9, ankle_y, -0.004 * hk), Some(hb(HumanBone::LeftFoot)));
        let toes = add(&format!("Toes{sfx}"), Some(ft), Vec3::new(sx * hip_x * 0.93, 0.012 * hk, foot_len * 0.62), Some(hb(HumanBone::LeftToes)));
        side_bones.insert((side, "toetip"), add(&format!("ToeTip{sfx}"), Some(toes), Vec3::new(sx * hip_x * 0.95, 0.01 * hk, foot_len * 0.82), None));
        side_bones.insert((side, "upperarm"), ua);
        side_bones.insert((side, "lowerarm"), la);
        side_bones.insert((side, "hand"), hand);
        side_bones.insert((side, "shoulder"), sh);
        side_bones.insert((side, "upperleg"), ul);
        side_bones.insert((side, "lowerleg"), ll);
        side_bones.insert((side, "foot"), ft);
        side_bones.insert((side, "toes"), toes);
    }
    let head_top = add("HeadTop", Some(head), Vec3::new(0.0, chin_y + head_h, -0.01 * hk), None);

    let mut skel = Skeleton::default();
    let mut human = HumanoidMap::default();
    for (i, (name, parent, world, hb)) in bones.iter().enumerate() {
        let pw = parent.map(|p| bones[p].2).unwrap_or(Vec3::ZERO);
        skel.bones.push(Bone::new(name.clone(), *parent, Transform::from_translation(*world - pw)));
        if let Some(hb) = hb {
            human.bones.insert(*hb, i);
        }
    }
    skel.recompute_inverse_bind();
    let prefer: Vec<Option<usize>> = (0..skel.len())
        .map(|i| human.human_of(i).and_then(|h| h.tail_child()).and_then(|t| human.get(t)))
        .collect();
    skel.compute_axes(|i| prefer[i]);
    human.compute_tpose(&skel);
    let wp = |b: usize| bones[b].2;

    // ---- Geometry -------------------------------------------------------------------------
    let mut b = Builder::new();
    let x = Vec3::X;
    let yv = Vec3::Y;
    let z = Vec3::Z;
    let balls = p.joint_balls;

    // Head (egg with jaw), chin at t=0.
    let hw = head_h * 0.37 * p.head_width;
    let hd = head_h * 0.45;
    let hp = wp(head);
    b.loft(head, Vec3::new(0.0, chin_y, hp.z), yv, x, z, head_h, 18, 22, 2.3, 2.2, |t| {
        let jaw = (t / 0.5).min(1.0);
        let rx = hw * (0.55 + 0.45 * jaw.powf(0.6));
        let rz = hd * (0.62 + 0.38 * jaw.powf(0.5));
        let oz = head_h * (0.1 * (1.0 - t).powf(1.5) - 0.04 * t);
        (rx, rz, 0.0, oz)
    });
    // Nose: orientation cue.
    b.ellipsoid(head, Vec3::new(0.0, chin_y + head_h * 0.33, hp.z + hd * 0.78), Vec3::new(head_h * 0.06, head_h * 0.1, head_h * 0.07), 8);
    // Ears
    for sx in [-1.0, 1.0] {
        b.ellipsoid(head, Vec3::new(sx * hw * 0.98, chin_y + head_h * 0.47, hp.z - hd * 0.05), Vec3::new(head_h * 0.03, head_h * 0.11, head_h * 0.06), 8);
    }
    // Neck
    let nr = hk * 0.031 * girth * lerp(1.1, 0.9);
    let n0 = wp(neck);
    b.loft(neck, n0 - yv * hk * 0.02, yv, x, z, (wp(head).y - n0.y) + head_h * 0.25, 8, 16, 2.0, 6.0, |_| (nr, nr * 0.95, 0.0, 0.0));

    // Torso: pelvis (hips), abdomen (spine), ribcage (chest), upper chest.
    let pel_w = hk * lerp(0.088, 0.1) * p.hip_width * girth;
    let pel_d = hk * lerp(0.062, 0.066) * girth;
    let crotch_y = wp(hips).y - hk * 0.075;
    b.loft(hips, Vec3::new(0.0, crotch_y, 0.0), yv, x, z, wp(spine).y - crotch_y + hk * 0.02, 12, 22, 2.4, 3.0, |t| {
        let w = pel_w * (0.8 + 0.2 * (t * 2.0).min(1.0)) * (1.0 - 0.12 * t);
        (w, pel_d * (0.92 + 0.08 * (1.0 - t)), 0.0, -hk * 0.006 * (1.0 - t))
    });
    let waist_w = hk * lerp(0.08, 0.07) * girth * (0.85 + 0.3 * we);
    let waist_d = hk * 0.055 * girth * (0.85 + 0.35 * we);
    let s0 = wp(spine);
    b.loft(spine, s0, yv, x, z, wp(chest).y - s0.y + hk * 0.02, 10, 22, 2.3, 4.0, |t| {
        (waist_w * (1.0 + 0.08 * t), waist_d, 0.0, hk * 0.006 * we * (1.0 - t))
    });
    let rib_w = hk * lerp(0.092, 0.08) * girth * p.shoulder_width.sqrt();
    let rib_d = hk * lerp(0.062, 0.055) * girth;
    let c0 = wp(chest);
    b.loft(chest, c0 - yv * hk * 0.01, yv, x, z, wp(upper_chest).y - c0.y + hk * 0.03, 10, 22, 2.3, 4.0, |t| {
        (rib_w * (0.92 + 0.08 * t), rib_d * (0.95 + 0.05 * t), 0.0, hk * 0.004 * t)
    });
    let uc = wp(upper_chest);
    let sh_w = shoulder_x * 0.82;
    b.loft(upper_chest, uc - yv * hk * 0.01, yv, x, z, wp(neck).y - uc.y + hk * 0.015, 10, 22, 2.6, 3.0, |t| {
        let w = rib_w + (sh_w - rib_w) * (t * 1.6).min(1.0);
        let w = w * (1.0 - 0.45 * ((t - 0.65).max(0.0) / 0.35).powf(1.5));
        (w, rib_d * (1.0 - 0.25 * t) * (1.0 + 0.12 * mu * (1.0 - f)), 0.0, hk * 0.006 * (1.0 - t))
    });
    // Breasts
    if p.breast > 0.01 {
        let r = hk * (0.022 + 0.03 * p.breast);
        for sx in [-1.0, 1.0] {
            let c = Vec3::new(sx * rib_w * 0.45, chest_y + (upper_chest_y - chest_y) * 0.55, rib_d * 0.82 + c0.z);
            b.ellipsoid(upper_chest, c, Vec3::new(r * 1.05, r * 0.95, r * 0.85), 12);
        }
    }

    for side in [Side::Left, Side::Right] {
        let sx = if side == Side::Left { 1.0 } else { -1.0 };
        let ax = x * sx;
        let (fx, fz) = (yv, z);
        let get = |k: &str| side_bones[&(side, k)];
        // Shoulder / clavicle
        let shb = get("shoulder");
        let s0 = wp(shb);
        let ua = get("upperarm");
        b.loft(shb, s0, ax, fx, fz, wp(ua).x.abs() - s0.x.abs(), 6, 12, 2.0, 3.0, |t| {
            let r = hk * 0.022 * girth * (1.0 + 0.4 * t);
            (r, r * 1.2, 0.0, 0.0)
        });
        // Upper arm with deltoid/biceps bulge.
        let ua0 = wp(ua);
        let la = get("lowerarm");
        let r0 = hk * 0.034 * limb * lerp(1.05, 0.9);
        let r1 = hk * 0.024 * limb * lerp(1.0, 0.88);
        b.loft(ua, ua0 - ax * hk * 0.01, ax, fx, fz, upper_arm + hk * 0.012, 14, 18, 2.0, 3.5, |t| {
            let bulge = (std::f32::consts::PI * t).sin() * hk * 0.004 * (0.5 + mu);
            let r = r0 + (r1 - r0) * t + bulge;
            (r, r * 0.95, 0.0, 0.0)
        });
        if balls {
            b.sphere(ua, ua0, r0 * 0.95);
        }
        // Forearm (flattened), tapering to the wrist.
        let la0 = wp(la);
        let hand = get("hand");
        let w1 = hk * 0.018 * limb;
        b.loft(la, la0 - ax * hk * 0.005, ax, fx, fz, forearm + hk * 0.008, 14, 18, 2.2, 3.5, |t| {
            let bulge = (std::f32::consts::PI * (t * 0.8)).sin() * hk * 0.005 * (0.4 + mu);
            let r = r1 * 0.98 + (w1 - r1 * 0.98) * t + bulge * (1.0 - t);
            (r * 0.8, r * 1.05, 0.0, 0.0)
        });
        if balls {
            b.sphere(la, la0, r1 * 0.92);
        }
        // Palm (boxy).
        let h0 = wp(hand);
        let pw = palm * 0.42;
        let pt = hk * 0.012 * (0.9 + 0.2 * we);
        b.loft(hand, h0 - ax * hk * 0.004, ax, fx, fz, palm * 1.04, 8, 16, 3.5, 4.0, |t| {
            (pt * (1.0 - 0.15 * t), pw * (0.75 + 0.25 * (t * 2.5).min(1.0)), -hk * 0.002, 0.0)
        });
        if balls {
            b.sphere(hand, h0, w1 * 0.95);
        }
        // Fingers
        for finger in Finger::ALL {
            let mut segs: Vec<usize> = (0..3).map(|s| human.get(finger_bone(side, finger, s).unwrap()).unwrap()).collect();
            let tip = if finger == Finger::Thumb { get("thumbtip") } else { get(finger_tip_key(finger)) };
            segs.push(tip);
            for s in 0..3 {
                let a = wp(segs[s]);
                let c = wp(segs[s + 1]);
                let len = (c - a).length();
                let dir = (c - a) / len.max(1e-6);
                let base_r = hk * if finger == Finger::Thumb { 0.0085 } else { 0.0068 } * (0.9 + 0.2 * we) * p.hand_size;
                let r = base_r * (1.0 - 0.12 * s as f32);
                let gx = (yv - dir * dir.dot(yv)).normalize_or(z);
                let gz = dir.cross(gx).normalize_or(x);
                b.loft(segs[s], a - dir * r * 0.4, dir, gx, gz, len + r * 0.6, 6, 10, 2.2, 3.0, |t| {
                    let rr = r * (1.0 - 0.15 * t);
                    (rr * 0.85, rr, 0.0, 0.0)
                });
            }
        }
        // Thigh
        let ul = get("upperleg");
        let ll = get("lowerleg");
        let ft = get("foot");
        let tl = wp(ul).distance(wp(ll));
        let t0 = hk * lerp(0.056, 0.062) * limb;
        let t1 = hk * 0.034 * limb;
        let dir = (wp(ll) - wp(ul)).normalize();
        let (lx, lz) = (z, x);
        b.loft(ul, wp(ul) - dir * hk * 0.02, dir, lx, lz, tl + hk * 0.025, 14, 20, 2.0, 3.5, |t| {
            let bulge = (std::f32::consts::PI * t).sin() * hk * 0.004 * (0.5 + mu);
            let r = t0 + (t1 - t0) * t.powf(0.8) + bulge;
            (r, r * 0.97, 0.0, 0.0)
        });
        if balls {
            b.sphere(ul, wp(ul), t0 * 0.85);
        }
        // Shin with calf
        let sl = wp(ll).distance(wp(ft));
        let c1 = hk * 0.02 * limb;
        let dir2 = (wp(ft) - wp(ll)).normalize();
        b.loft(ll, wp(ll) - dir2 * hk * 0.008, dir2, lx, lz, sl + hk * 0.01, 14, 18, 2.0, 3.5, |t| {
            let calf = (std::f32::consts::PI * (t * 1.4).min(1.0)).sin() * hk * 0.008 * (0.5 + mu);
            let r = t1 * 0.95 + (c1 - t1 * 0.95) * t;
            (r + calf * 0.4, r + calf, 0.0, -calf * 0.6)
        });
        if balls {
            b.sphere(ll, wp(ll), t1 * 0.9);
            b.sphere(ft, wp(ft), c1 * 0.95);
        }
        // Foot: from heel to ball along +Z, flat sole.
        let fa = wp(ft);
        let heel_z = -foot_len * 0.22;
        let ball_z = wp(get("toes")).z;
        let fw = hk * 0.026 * p.foot_size;
        let (ffx, ffz) = (x, yv);
        let ankle_h = fa.y;
        b.loft(ft, Vec3::new(fa.x, 0.0, heel_z), z, ffx, ffz, ball_z - heel_z + hk * 0.01, 12, 16, 2.6, 4.0, |t| {
            let hh = (ankle_h * 0.55) * (1.0 - t * 0.7);
            let w = fw * (0.75 + 0.3 * t);
            (w, hh, 0.0, hh)
        });
        // Toes
        let tb = get("toes");
        let ta = wp(tb);
        let toe_len = wp(get("toetip")).z - ta.z;
        b.loft(tb, Vec3::new(ta.x, 0.0, ta.z - hk * 0.005), z, ffx, ffz, toe_len + hk * 0.005, 8, 14, 2.6, 3.0, |t| {
            let hh = hk * 0.011 * (1.0 - 0.3 * t);
            let w = fw * 1.02 * (1.0 - 0.15 * t);
            (w, hh, 0.0, hh)
        });
    }
    let _ = head_top;
    let mut mesh = b.mesh;
    mesh.compute_normals();
    mesh.sanitize();
    let mut model = Model::empty("Mannequin", SourceFormat::Procedural);
    model.materials = vec![Material { name: "Mannequin".into(), base_color: Vec4::new(0.85, 0.82, 0.78, 1.0), ..Default::default() }];
    model.meshes = vec![mesh];
    model.skeleton = skel;
    model.humanoid = human;
    model
}

fn finger_tip_key(f: Finger) -> &'static str {
    match f {
        Finger::Thumb => "thumbtip",
        Finger::Index => "indextip",
        Finger::Middle => "middletip",
        Finger::Ring => "ringtip",
        Finger::Little => "littletip",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_valid_humanoid() {
        for (_, p) in MannequinParams::presets() {
            let m = build(&p);
            assert!(m.humanoid.is_valid());
            assert!(m.meshes[0].positions.len() > 1000);
            let (lo, hi) = m.rest_bounds();
            assert!((hi.y - p.height).abs() < p.height * 0.08, "height {} vs {}", hi.y - lo.y, p.height);
        }
    }
}

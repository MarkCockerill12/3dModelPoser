//! Canonical humanoid skeleton (VRM 1.0 bone set), rig auto-detection, T-pose normalisation
//! and joint limits.
use crate::skeleton::Skeleton;
use glam::{EulerRot, Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

macro_rules! human_bones {
    ($($v:ident = $s:literal),* $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        pub enum HumanBone { $($v),* }
        impl HumanBone {
            pub const ALL: &'static [HumanBone] = &[$(HumanBone::$v),*];
            pub fn name(self) -> &'static str { match self { $(HumanBone::$v => $s),* } }
            pub fn from_name(s: &str) -> Option<Self> { match s { $($s => Some(HumanBone::$v),)* _ => None } }
        }
    };
}

human_bones! {
    Hips = "hips", Spine = "spine", Chest = "chest", UpperChest = "upperChest", Neck = "neck", Head = "head",
    LeftEye = "leftEye", RightEye = "rightEye", Jaw = "jaw",
    LeftUpperLeg = "leftUpperLeg", LeftLowerLeg = "leftLowerLeg", LeftFoot = "leftFoot", LeftToes = "leftToes",
    RightUpperLeg = "rightUpperLeg", RightLowerLeg = "rightLowerLeg", RightFoot = "rightFoot", RightToes = "rightToes",
    LeftShoulder = "leftShoulder", LeftUpperArm = "leftUpperArm", LeftLowerArm = "leftLowerArm", LeftHand = "leftHand",
    RightShoulder = "rightShoulder", RightUpperArm = "rightUpperArm", RightLowerArm = "rightLowerArm", RightHand = "rightHand",
    LeftThumbMetacarpal = "leftThumbMetacarpal", LeftThumbProximal = "leftThumbProximal", LeftThumbDistal = "leftThumbDistal",
    LeftIndexProximal = "leftIndexProximal", LeftIndexIntermediate = "leftIndexIntermediate", LeftIndexDistal = "leftIndexDistal",
    LeftMiddleProximal = "leftMiddleProximal", LeftMiddleIntermediate = "leftMiddleIntermediate", LeftMiddleDistal = "leftMiddleDistal",
    LeftRingProximal = "leftRingProximal", LeftRingIntermediate = "leftRingIntermediate", LeftRingDistal = "leftRingDistal",
    LeftLittleProximal = "leftLittleProximal", LeftLittleIntermediate = "leftLittleIntermediate", LeftLittleDistal = "leftLittleDistal",
    RightThumbMetacarpal = "rightThumbMetacarpal", RightThumbProximal = "rightThumbProximal", RightThumbDistal = "rightThumbDistal",
    RightIndexProximal = "rightIndexProximal", RightIndexIntermediate = "rightIndexIntermediate", RightIndexDistal = "rightIndexDistal",
    RightMiddleProximal = "rightMiddleProximal", RightMiddleIntermediate = "rightMiddleIntermediate", RightMiddleDistal = "rightMiddleDistal",
    RightRingProximal = "rightRingProximal", RightRingIntermediate = "rightRingIntermediate", RightRingDistal = "rightRingDistal",
    RightLittleProximal = "rightLittleProximal", RightLittleIntermediate = "rightLittleIntermediate", RightLittleDistal = "rightLittleDistal",
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Center,
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finger {
    Thumb,
    Index,
    Middle,
    Ring,
    Little,
}

impl Finger {
    pub const ALL: [Finger; 5] = [Finger::Thumb, Finger::Index, Finger::Middle, Finger::Ring, Finger::Little];
}

use HumanBone as H;

impl HumanBone {
    pub fn side(self) -> Side {
        let n = self.name();
        if n.starts_with("left") {
            Side::Left
        } else if n.starts_with("right") {
            Side::Right
        } else {
            Side::Center
        }
    }

    pub fn mirror(self) -> HumanBone {
        let n = self.name();
        let m = if let Some(r) = n.strip_prefix("left") {
            format!("right{r}")
        } else if let Some(r) = n.strip_prefix("right") {
            format!("left{r}")
        } else {
            return self;
        };
        HumanBone::from_name(&m).unwrap_or(self)
    }

    /// Canonical parent in the humanoid hierarchy.
    pub fn parent(self) -> Option<HumanBone> {
        Some(match self {
            H::Hips => return None,
            H::Spine => H::Hips,
            H::Chest => H::Spine,
            H::UpperChest => H::Chest,
            H::Neck => H::UpperChest,
            H::Head => H::Neck,
            H::LeftEye | H::RightEye | H::Jaw => H::Head,
            H::LeftUpperLeg | H::RightUpperLeg => H::Hips,
            H::LeftLowerLeg => H::LeftUpperLeg,
            H::RightLowerLeg => H::RightUpperLeg,
            H::LeftFoot => H::LeftLowerLeg,
            H::RightFoot => H::RightLowerLeg,
            H::LeftToes => H::LeftFoot,
            H::RightToes => H::RightFoot,
            H::LeftShoulder | H::RightShoulder => H::UpperChest,
            H::LeftUpperArm => H::LeftShoulder,
            H::RightUpperArm => H::RightShoulder,
            H::LeftLowerArm => H::LeftUpperArm,
            H::RightLowerArm => H::RightUpperArm,
            H::LeftHand => H::LeftLowerArm,
            H::RightHand => H::RightLowerArm,
            b => {
                let (side, finger, seg) = b.finger_info()?;
                if seg == 0 {
                    if side == Side::Left { H::LeftHand } else { H::RightHand }
                } else {
                    finger_bone(side, finger, seg - 1)?
                }
            }
        })
    }

    /// (side, finger, segment 0..3) for finger bones.
    pub fn finger_info(self) -> Option<(Side, Finger, usize)> {
        let i = FINGER_TABLE.iter().position(|b| *b == self)?;
        let side = if i < 15 { Side::Left } else { Side::Right };
        let f = Finger::ALL[(i % 15) / 3];
        Some((side, f, i % 3))
    }

    pub fn is_finger(self) -> bool {
        self.finger_info().is_some()
    }

    /// Bone the tail of this bone points at (used for bone axis + T-pose).
    pub fn tail_child(self) -> Option<HumanBone> {
        Some(match self {
            H::Hips => H::Spine,
            H::Spine => H::Chest,
            H::Chest => H::UpperChest,
            H::UpperChest => H::Neck,
            H::Neck => H::Head,
            H::LeftUpperLeg => H::LeftLowerLeg,
            H::LeftLowerLeg => H::LeftFoot,
            H::LeftFoot => H::LeftToes,
            H::RightUpperLeg => H::RightLowerLeg,
            H::RightLowerLeg => H::RightFoot,
            H::RightFoot => H::RightToes,
            H::LeftShoulder => H::LeftUpperArm,
            H::LeftUpperArm => H::LeftLowerArm,
            H::LeftLowerArm => H::LeftHand,
            H::LeftHand => H::LeftMiddleProximal,
            H::RightShoulder => H::RightUpperArm,
            H::RightUpperArm => H::RightLowerArm,
            H::RightLowerArm => H::RightHand,
            H::RightHand => H::RightMiddleProximal,
            b => {
                let (side, f, seg) = b.finger_info()?;
                finger_bone(side, f, seg + 1)?
            }
        })
    }

    /// Canonical bone direction in the normalised T-pose (character faces +Z, left = +X, up = +Y).
    pub fn tpose_direction(self) -> Option<Vec3> {
        let x = match self.side() {
            Side::Left => 1.0,
            Side::Right => -1.0,
            Side::Center => 0.0,
        };
        Some(match self {
            H::Hips | H::Spine | H::Chest | H::UpperChest | H::Neck | H::Head => Vec3::Y,
            H::LeftUpperLeg | H::LeftLowerLeg | H::RightUpperLeg | H::RightLowerLeg => -Vec3::Y,
            H::LeftUpperArm | H::LeftLowerArm | H::LeftHand | H::RightUpperArm | H::RightLowerArm | H::RightHand => {
                Vec3::new(x, 0.0, 0.0)
            }
            b if b.is_finger() => {
                let (_, f, _) = b.finger_info().unwrap();
                if f == Finger::Thumb {
                    Vec3::new(x * 0.7071, 0.0, 0.7071)
                } else {
                    Vec3::new(x, 0.0, 0.0)
                }
            }
            _ => return None,
        })
    }

    /// Default joint limits in degrees, normalised frame (see [`JointLimit`]).
    pub fn default_limit(self) -> Option<JointLimit> {
        let l = |x: (f32, f32), y: (f32, f32), z: (f32, f32)| JointLimit { x, y, z };
        let left = match self.side() {
            Side::Right => self.mirror(),
            _ => self,
        };
        let lim = match left {
            H::Hips => return None,
            H::Spine | H::Chest | H::UpperChest => l((-30.0, 45.0), (-35.0, 35.0), (-30.0, 30.0)),
            H::Neck => l((-40.0, 50.0), (-60.0, 60.0), (-35.0, 35.0)),
            H::Head => l((-45.0, 45.0), (-60.0, 60.0), (-35.0, 35.0)),
            H::LeftEye => l((-25.0, 25.0), (-35.0, 35.0), (-1.0, 1.0)),
            H::Jaw => l((-2.0, 35.0), (-8.0, 8.0), (-5.0, 5.0)),
            H::LeftShoulder => l((-20.0, 20.0), (-25.0, 20.0), (-15.0, 35.0)),
            H::LeftUpperArm => l((-95.0, 95.0), (-120.0, 45.0), (-95.0, 110.0)),
            H::LeftLowerArm => l((-95.0, 95.0), (-155.0, 8.0), (-8.0, 8.0)),
            H::LeftHand => l((-25.0, 25.0), (-35.0, 35.0), (-85.0, 75.0)),
            H::LeftUpperLeg => l((-125.0, 35.0), (-50.0, 50.0), (-35.0, 75.0)),
            H::LeftLowerLeg => l((-3.0, 155.0), (-10.0, 10.0), (-5.0, 5.0)),
            H::LeftFoot => l((-35.0, 55.0), (-30.0, 30.0), (-30.0, 30.0)),
            H::LeftToes => l((-60.0, 35.0), (-5.0, 5.0), (-5.0, 5.0)),
            b if b.is_finger() => {
                let (_, f, seg) = b.finger_info().unwrap();
                if f == Finger::Thumb {
                    l((-60.0, 60.0), (-70.0, 50.0), (-60.0, 60.0))
                } else {
                    match seg {
                        0 => l((-10.0, 10.0), (-25.0, 25.0), (-95.0, 25.0)),
                        1 => l((-5.0, 5.0), (-5.0, 5.0), (-110.0, 5.0)),
                        _ => l((-5.0, 5.0), (-5.0, 5.0), (-90.0, 10.0)),
                    }
                }
            }
            _ => return None,
        };
        Some(if self.side() == Side::Right { lim.mirrored() } else { lim })
    }
}

const FINGER_TABLE: [HumanBone; 30] = [
    H::LeftThumbMetacarpal, H::LeftThumbProximal, H::LeftThumbDistal,
    H::LeftIndexProximal, H::LeftIndexIntermediate, H::LeftIndexDistal,
    H::LeftMiddleProximal, H::LeftMiddleIntermediate, H::LeftMiddleDistal,
    H::LeftRingProximal, H::LeftRingIntermediate, H::LeftRingDistal,
    H::LeftLittleProximal, H::LeftLittleIntermediate, H::LeftLittleDistal,
    H::RightThumbMetacarpal, H::RightThumbProximal, H::RightThumbDistal,
    H::RightIndexProximal, H::RightIndexIntermediate, H::RightIndexDistal,
    H::RightMiddleProximal, H::RightMiddleIntermediate, H::RightMiddleDistal,
    H::RightRingProximal, H::RightRingIntermediate, H::RightRingDistal,
    H::RightLittleProximal, H::RightLittleIntermediate, H::RightLittleDistal,
];

pub fn finger_bone(side: Side, finger: Finger, seg: usize) -> Option<HumanBone> {
    if seg > 2 || side == Side::Center {
        return None;
    }
    let fi = Finger::ALL.iter().position(|f| *f == finger)?;
    let base = if side == Side::Left { 0 } else { 15 };
    Some(FINGER_TABLE[base + fi * 3 + seg])
}

/// Joint rotation limits (degrees) as Euler ranges in the normalised (T-pose, world-aligned) frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointLimit {
    pub x: (f32, f32),
    pub y: (f32, f32),
    pub z: (f32, f32),
}

impl JointLimit {
    /// Mirror across the YZ plane: Y and Z rotations change sign.
    pub fn mirrored(&self) -> Self {
        Self { x: self.x, y: (-self.y.1, -self.y.0), z: (-self.z.1, -self.z.0) }
    }

    pub fn clamp(&self, q: Quat) -> Quat {
        let (z, y, x) = q.to_euler(EulerRot::ZYX);
        let c = |v: f32, r: (f32, f32)| v.to_degrees().clamp(r.0, r.1).to_radians();
        Quat::from_euler(EulerRot::ZYX, c(z, self.z), c(y, self.y), c(x, self.x))
    }

    /// 0 = comfortably inside, 1 = at the limit (used for joint-stress display).
    pub fn stress(&self, q: Quat) -> f32 {
        let (z, y, x) = q.to_euler(EulerRot::ZYX);
        let s = |v: f32, r: (f32, f32)| {
            let v = v.to_degrees();
            let span = (r.1 - r.0).max(1.0);
            let mid = (r.0 + r.1) * 0.5;
            ((v - mid).abs() / (span * 0.5)).clamp(0.0, 1.5)
        };
        s(x, self.x).max(s(y, self.y)).max(s(z, self.z))
    }
}

/// Mapping canonical bone -> skeleton bone index, plus the normalisation data.
#[derive(Clone, Debug, Default)]
pub struct HumanoidMap {
    pub bones: HashMap<HumanBone, usize>,
    /// World-space rest rotation of every skeleton bone after T-pose correction.
    pub tpose_world: Vec<Quat>,
    /// Local rotations that turn the rest pose into the T-pose.
    pub tpose_local: Vec<Quat>,
    /// Hips height in T-pose (model units), used for retarget scaling.
    pub hips_height: f32,
}

impl HumanoidMap {
    pub fn is_valid(&self) -> bool {
        [H::Hips, H::Spine, H::Head, H::LeftUpperArm, H::RightUpperArm, H::LeftUpperLeg, H::RightUpperLeg]
            .iter()
            .all(|b| self.bones.contains_key(b))
    }

    pub fn get(&self, b: HumanBone) -> Option<usize> {
        self.bones.get(&b).copied()
    }

    /// Reverse lookup: skeleton bone -> canonical bone.
    pub fn human_of(&self, bone: usize) -> Option<HumanBone> {
        self.bones.iter().find(|(_, i)| **i == bone).map(|(h, _)| *h)
    }

    pub fn reverse(&self, n: usize) -> Vec<Option<HumanBone>> {
        let mut r = vec![None; n];
        for (h, i) in &self.bones {
            if *i < n {
                r[*i] = Some(*h);
            }
        }
        r
    }

    /// Compute T-pose correction + normalisation frames. Requires the map to be filled.
    pub fn compute_tpose(&mut self, skel: &Skeleton) {
        let n = skel.len();
        let mut locals = skel.rest_locals();
        let rev = self.reverse(n);
        let order: Vec<usize> = (0..n).collect();
        for &i in &order {
            let Some(h) = rev[i] else { continue };
            let Some(dir) = h.tpose_direction() else { continue };
            let Some(tail_h) = h.tail_child() else { continue };
            let Some(tail) = self.get(tail_h).or_else(|| {
                // Fall back to the first mapped descendant.
                skel.bones[i].children.first().copied()
            }) else {
                continue;
            };
            if !skel.is_ancestor(i, tail) {
                continue;
            }
            let w = skel.evaluate(&locals, None).world;
            let head = w[i].w_axis.truncate();
            let tailp = w[tail].w_axis.truncate();
            let cur = (tailp - head).normalize_or_zero();
            if cur == Vec3::ZERO {
                continue;
            }
            // Only straighten spine and limbs if they are reasonably off.
            let delta = Quat::from_rotation_arc(cur, dir);
            let parent_rot = match skel.bones[i].parent {
                Some(p) => crate::skeleton::mat_rotation(&w[p]),
                None => crate::skeleton::mat_rotation(&skel.root_transform),
            };
            locals[i].rotation =
                crate::skeleton::apply_world_rotation(locals[i].rotation, parent_rot, delta);
        }
        let w = skel.evaluate(&locals, None).world;
        self.tpose_world = w.iter().map(crate::skeleton::mat_rotation).collect();
        self.tpose_local = locals.iter().map(|t| t.rotation).collect();
        self.hips_height = self.get(H::Hips).map(|h| w[h].w_axis.y).unwrap_or(1.0).abs().max(1e-3);
    }

    /// Normalised rotation (world-aligned T-pose frame) -> skeleton local rotation.
    pub fn to_local(&self, skel: &Skeleton, bone: usize, normalized: Quat) -> Quat {
        let wb = self.tpose_world[bone];
        let wp = match skel.bones[bone].parent {
            Some(p) => self.tpose_world[p],
            None => crate::skeleton::mat_rotation(&skel.root_transform),
        };
        (wp.inverse() * normalized * wb).normalize()
    }

    /// Skeleton local rotation -> normalised rotation.
    pub fn to_normalized(&self, skel: &Skeleton, bone: usize, local: Quat) -> Quat {
        let wb = self.tpose_world[bone];
        let wp = match skel.bones[bone].parent {
            Some(p) => self.tpose_world[p],
            None => crate::skeleton::mat_rotation(&skel.root_transform),
        };
        (wp * local * wb.inverse()).normalize()
    }
}

// ---------------------------------------------------------------------------------------------
// Auto-detection from bone names + hierarchy.

fn split_side(raw: &str) -> (Side, String) {
    let mut s = raw.to_string();
    // Strip namespaces like "mixamorig:" / "Armature|".
    if let Some(p) = s.rfind([':', '|']) {
        s = s[p + 1..].to_string();
    }
    // Japanese side markers.
    if s.starts_with('左') {
        return (Side::Left, s.trim_start_matches('左').to_string());
    }
    if s.starts_with('右') {
        return (Side::Right, s.trim_start_matches('右').to_string());
    }
    let lower = s.to_lowercase();
    let tokens: Vec<String> = lower
        .split(|c: char| c == '_' || c == '.' || c == ' ' || c == '-')
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect();
    let mut side = Side::Center;
    let mut kept = vec![];
    for (i, t) in tokens.iter().enumerate() {
        let first = i == 0;
        let last = i == tokens.len() - 1;
        if (t == "l" || t == "left") && (first || last || tokens.len() > 2) {
            side = Side::Left;
            continue;
        }
        if (t == "r" || t == "right") && (first || last || tokens.len() > 2) {
            side = Side::Right;
            continue;
        }
        kept.push(t.clone());
    }
    let mut core = kept.join("");
    // CamelCase forms: LeftArm, RightUpLeg, leftHand...
    if side == Side::Center {
        for (p, sd) in [("left", Side::Left), ("right", Side::Right)] {
            if let Some(r) = core.strip_prefix(p) {
                side = sd;
                core = r.to_string();
                break;
            }
            if let Some(r) = core.strip_suffix(p) {
                side = sd;
                core = r.to_string();
                break;
            }
        }
    }
    // Strip common rig prefixes.
    for pre in ["mixamorig", "jbipc", "jbipl", "jbipr", "jbip", "ccbase", "def", "org", "bip01", "bip001", "bip", "valvebiped", "npc"] {
        if core.starts_with(pre) && core.len() > pre.len() + 1 {
            core = core[pre.len()..].to_string();
        }
    }
    // VRoid: J_Bip_L_UpperArm -> tokens [j, bip, upperarm] side L.
    if let Some(r) = core.strip_prefix("jbip") {
        core = r.to_string();
    }
    (side, core)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Hips,
    Neck,
    Head,
    Eye,
    Jaw,
    Shoulder,
    UpperArm,
    LowerArm,
    Hand,
    UpperLeg,
    LowerLeg,
    Foot,
    Toes,
    Finger(Finger),
    Twist,
    Spine,
}

fn classify(core: &str) -> Option<Kind> {
    let has = |p: &str| core.contains(p);
    if has("twist") || has("roll") || has("捩") || has("share") || has("helper") || has("ik") && core.len() <= 6 {
        return Some(Kind::Twist);
    }
    if has("end") || has("nub") || has("tip") || has("先") && !has("つま先") {
        return None;
    }
    // Japanese (MMD)
    let jp = [
        ("センター", Kind::Hips),
        ("手首", Kind::Hand),
        ("足首", Kind::Foot),
        ("つま先", Kind::Toes),
        ("首", Kind::Neck),
        ("頭", Kind::Head),
        ("目", Kind::Eye),
        ("肩", Kind::Shoulder),
        ("ひじ", Kind::LowerArm),
        ("ひざ", Kind::LowerLeg),
        ("親指", Kind::Finger(Finger::Thumb)),
        ("人差指", Kind::Finger(Finger::Index)),
        ("中指", Kind::Finger(Finger::Middle)),
        ("薬指", Kind::Finger(Finger::Ring)),
        ("小指", Kind::Finger(Finger::Little)),
        ("上半身", Kind::Spine),
    ];
    for (k, v) in jp {
        if core.contains(k) {
            return Some(v);
        }
    }
    if core == "腕" {
        return Some(Kind::UpperArm);
    }
    if core == "足" {
        return Some(Kind::UpperLeg);
    }
    // Fingers (before hand/arm because names like "handthumb1").
    let fingers = [
        (["thumb", "finger0", "pollex"], Finger::Thumb),
        (["index", "finger1", "pointer"], Finger::Index),
        (["middle", "finger2", "mid"], Finger::Middle),
        (["ring", "finger3", "fourth"], Finger::Ring),
        (["pinky", "little", "finger4"], Finger::Little),
    ];
    for (keys, f) in fingers {
        if keys.iter().any(|k| has(k)) {
            let weak_mid = f == Finger::Middle && !has("middle") && !has("finger2");
            if weak_mid && !(core.starts_with("mid") || has("handmid") || has("fmid")) {
                continue;
            }
            return Some(Kind::Finger(f));
        }
    }
    if has("eye") && !has("brow") && !has("lid") && !has("lash") {
        return Some(Kind::Eye);
    }
    if has("jaw") || has("chin") {
        return Some(Kind::Jaw);
    }
    if has("head") && !has("headtop") {
        return Some(Kind::Head);
    }
    if has("neck") {
        return Some(Kind::Neck);
    }
    if core == "hips" || core == "hip" || core == "pelvis" || core == "root_hips" || core == "hipsbone" {
        return Some(Kind::Hips);
    }
    if has("clavicle") || has("collar") || has("shoulder") && !has("upper") {
        return Some(Kind::Shoulder);
    }
    if has("forearm") || has("lowerarm") || has("elbow") || has("lowarm") || has("loarm") {
        return Some(Kind::LowerArm);
    }
    if has("upperarm") || has("uparm") || core == "arm" || core == "humerus" || core == "upper_arm" {
        return Some(Kind::UpperArm);
    }
    if has("hand") || has("wrist") || has("palm") && false {
        return Some(Kind::Hand);
    }
    if has("upleg") || has("thigh") || has("upperleg") || has("femur") {
        return Some(Kind::UpperLeg);
    }
    if has("calf") || has("shin") || has("lowerleg") || has("knee") || has("lowleg") || core == "leg" {
        return Some(Kind::LowerLeg);
    }
    if has("toe") || has("ball") {
        return Some(Kind::Toes);
    }
    if has("foot") || has("ankle") {
        return Some(Kind::Foot);
    }
    if has("spine") || has("chest") || has("torso") || has("waist") || has("abdomen") || has("ribcage") {
        return Some(Kind::Spine);
    }
    None
}

fn depth(skel: &Skeleton, mut b: usize) -> usize {
    let mut d = 0;
    while let Some(p) = skel.bones[b].parent {
        d += 1;
        b = p;
    }
    d
}

/// Detect the humanoid mapping from names and hierarchy. Returns an empty map on failure.
pub fn detect_humanoid(skel: &Skeleton) -> HumanoidMap {
    let n = skel.len();
    let mut map = HumanoidMap::default();
    if n == 0 {
        return map;
    }
    let info: Vec<(Side, Option<Kind>)> = skel
        .bones
        .iter()
        .map(|b| {
            let (s, c) = split_side(&b.name);
            (s, classify(&c))
        })
        .collect();

    // Pick the shallowest bone of each (kind, side).
    let pick = |kind: Kind, side: Side| -> Option<usize> {
        (0..n)
            .filter(|i| info[*i].1 == Some(kind) && info[*i].0 == side)
            .min_by_key(|i| depth(skel, *i))
    };

    fn set(map: &mut HumanoidMap, h: HumanBone, b: Option<usize>) {
        if let Some(b) = b {
            map.bones.insert(h, b);
        }
    }

    for (side, hand, lower, upper, shoulder, up_leg, low_leg, foot, toes) in [
        (Side::Left, H::LeftHand, H::LeftLowerArm, H::LeftUpperArm, H::LeftShoulder, H::LeftUpperLeg, H::LeftLowerLeg, H::LeftFoot, H::LeftToes),
        (Side::Right, H::RightHand, H::RightLowerArm, H::RightUpperArm, H::RightShoulder, H::RightUpperLeg, H::RightLowerLeg, H::RightFoot, H::RightToes),
    ] {
        let hand_b = pick(Kind::Hand, side);
        let mut lower_b = pick(Kind::LowerArm, side);
        let mut upper_b = pick(Kind::UpperArm, side);
        // Structural fill-in walking up from the hand, skipping twist bones.
        if let Some(h) = hand_b {
            let mut chain = vec![];
            let mut b = h;
            while let Some(p) = skel.bones[b].parent {
                if info[p].1 != Some(Kind::Twist) {
                    chain.push(p);
                }
                b = p;
                if chain.len() >= 4 {
                    break;
                }
            }
            if lower_b.is_none_or(|l| !skel.is_ancestor(l, h)) {
                lower_b = chain.first().copied();
            }
            if upper_b.is_none_or(|u| lower_b.is_some_and(|l| !skel.is_ancestor(u, l))) {
                upper_b = lower_b.and_then(|l| {
                    let mut b = l;
                    while let Some(p) = skel.bones[b].parent {
                        if info[p].1 != Some(Kind::Twist) && info[p].1 != Some(Kind::LowerArm) {
                            return Some(p);
                        }
                        b = p;
                    }
                    None
                });
            }
        }
        let shoulder_b = pick(Kind::Shoulder, side).filter(|s| upper_b.is_some_and(|u| skel.is_ancestor(*s, u)));
        set(&mut map, hand, hand_b);
        set(&mut map, lower, lower_b);
        set(&mut map, upper, upper_b);
        set(&mut map, shoulder, shoulder_b);

        let foot_b = pick(Kind::Foot, side);
        let mut low_leg_b = pick(Kind::LowerLeg, side);
        let mut up_leg_b = pick(Kind::UpperLeg, side);
        if let Some(f) = foot_b {
            if low_leg_b.is_none_or(|l| !skel.is_ancestor(l, f)) {
                low_leg_b = skel.bones[f].parent;
            }
            if let Some(l) = low_leg_b {
                if up_leg_b.is_none_or(|u| !skel.is_ancestor(u, l)) || up_leg_b == low_leg_b {
                    let mut b = l;
                    up_leg_b = None;
                    while let Some(p) = skel.bones[b].parent {
                        if info[p].1 != Some(Kind::Twist) {
                            up_leg_b = Some(p);
                            break;
                        }
                        b = p;
                    }
                }
            }
        }
        set(&mut map, foot, foot_b);
        set(&mut map, low_leg, low_leg_b);
        set(&mut map, up_leg, up_leg_b);
        set(&mut map, toes, pick(Kind::Toes, side).filter(|t| foot_b.is_some_and(|f| skel.is_ancestor(f, *t))));

        // Fingers: chains below the hand.
        if let Some(h) = hand_b {
            for f in Finger::ALL {
                let mut cands: Vec<usize> = (0..n)
                    .filter(|i| info[*i].1 == Some(Kind::Finger(f)) && skel.is_ancestor(h, *i))
                    .collect();
                cands.sort_by_key(|i| depth(skel, *i));
                // Skip metacarpals for non-thumb fingers.
                if f != Finger::Thumb {
                    cands.retain(|i| {
                        let nm = skel.bones[*i].name.to_lowercase();
                        !(nm.contains("metacarpal") || nm.contains("meta") || nm.contains("palm"))
                    });
                }
                // Follow a single chain from the shallowest candidate.
                if let Some(&first) = cands.first() {
                    let mut chain = vec![first];
                    let mut b = first;
                    while chain.len() < 3 {
                        let next = skel.bones[b]
                            .children
                            .iter()
                            .copied()
                            .find(|c| info[*c].1 == Some(Kind::Finger(f)));
                        match next {
                            Some(c) => {
                                chain.push(c);
                                b = c;
                            }
                            None => break,
                        }
                    }
                    for (seg, bone) in chain.into_iter().enumerate() {
                        set(&mut map, finger_bone(side, f, seg).unwrap(), Some(bone));
                    }
                }
            }
        }
        // Eyes
        set(&mut map, if side == Side::Left { H::LeftEye } else { H::RightEye }, pick(Kind::Eye, side));
    }

    // Head / neck
    let head = pick(Kind::Head, Side::Center);
    set(&mut map, H::Head, head);
    let neck = (0..n)
        .filter(|i| info[*i].1 == Some(Kind::Neck) && head.is_some_and(|h| skel.is_ancestor(*i, h)))
        .min_by_key(|i| depth(skel, *i))
        .or_else(|| head.and_then(|h| skel.bones[h].parent));
    set(&mut map, H::Neck, neck);
    set(&mut map, H::Jaw, pick(Kind::Jaw, Side::Center));

    // Hips: lowest common ancestor of the legs and the neck (handles MMD center/groove).
    let lul = map.get(H::LeftUpperLeg);
    let rul = map.get(H::RightUpperLeg);
    let mut hips = pick(Kind::Hips, Side::Center);
    if let (Some(l), Some(r), Some(nk)) = (lul, rul, neck) {
        let anc = |mut b: usize| {
            let mut v = vec![b];
            while let Some(p) = skel.bones[b].parent {
                v.push(p);
                b = p;
            }
            v
        };
        let al = anc(l);
        let ar = anc(r);
        let an = anc(nk);
        let lca = al.iter().copied().find(|b| ar.contains(b) && an.contains(b));
        let ok = hips.is_some_and(|h| skel.is_ancestor(h, l) && skel.is_ancestor(h, r) && skel.is_ancestor(h, nk));
        if !ok {
            hips = lca;
        }
    }
    set(&mut map, H::Hips, hips);

    // Spine chain: bones between hips and neck.
    if let (Some(h), Some(nk)) = (hips, neck) {
        if let Some(mut chain) = skel.chain(h, nk) {
            chain.pop(); // neck itself
            chain.retain(|b| info[*b].1 != Some(Kind::Twist));
            // Prefer bones that sit above the hips (ignore MMD lower-body siblings - not in chain anyway).
            match chain.len() {
                0 => {}
                1 => set(&mut map, H::Spine, Some(chain[0])),
                2 => {
                    set(&mut map, H::Spine, Some(chain[0]));
                    set(&mut map, H::Chest, Some(chain[1]));
                }
                k => {
                    set(&mut map, H::Spine, Some(chain[0]));
                    set(&mut map, H::Chest, Some(chain[k / 2]));
                    set(&mut map, H::UpperChest, Some(chain[k - 1]));
                }
            }
        }
    }
    // Shoulders must hang off the upper spine; reparent canonical parent expectations are loose.
    if !map.is_valid() {
        log::warn!("humanoid detection incomplete: {} bones mapped", map.bones.len());
    }
    map
}

/// Swap side markers in a bone name (for mirroring non-humanoid bones).
pub fn mirror_name(name: &str) -> Option<String> {
    let pairs = [
        ("Left", "Right"), ("left", "right"), ("LEFT", "RIGHT"),
        (".L", ".R"), ("_L", "_R"), (".l", ".r"), ("_l", "_r"), ("L_", "R_"), ("l_", "r_"), (" L ", " R "),
        ("左", "右"),
    ];
    for (a, b) in pairs {
        if name.contains(a) {
            return Some(name.replacen(a, b, 1));
        }
        if name.contains(b) {
            return Some(name.replacen(b, a, 1));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn side_split() {
        assert_eq!(split_side("mixamorig:LeftForeArm"), (Side::Left, "forearm".into()));
        assert_eq!(split_side("upperarm_r").0, Side::Right);
        assert_eq!(split_side("J_Bip_L_UpperArm"), (Side::Left, "upperarm".into()));
        assert_eq!(split_side("thigh.L").0, Side::Left);
        assert_eq!(split_side("左ひじ"), (Side::Left, "ひじ".into()));
        assert_eq!(classify("forearm"), Some(Kind::LowerArm));
        assert_eq!(classify("handindex1"), Some(Kind::Finger(Finger::Index)));
        assert_eq!(classify("upleg"), Some(Kind::UpperLeg));
    }

    #[test]
    fn finger_parents() {
        assert_eq!(H::LeftIndexIntermediate.parent(), Some(H::LeftIndexProximal));
        assert_eq!(H::RightThumbMetacarpal.parent(), Some(H::RightHand));
        assert_eq!(H::LeftHand.mirror(), H::RightHand);
    }
}

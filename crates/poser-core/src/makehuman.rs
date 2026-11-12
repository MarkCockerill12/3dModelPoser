//! Realistic parametric human built from MakeHuman's CC0 base mesh, default rig, skin weights and
//! macro targets (packed by scripts/mh_convert.py into assets/makehuman/body.bin.gz).
use crate::humanoid::{Finger, HumanBone as H, HumanoidMap, Side, finger_bone};
use crate::model::{Material, Mesh, Model, SourceFormat};
use crate::skeleton::{Bone, Skeleton};
use crate::transform::Transform;
use glam::{Mat4, Vec3, Vec4};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;
use std::sync::OnceLock;

static PACKED: &[u8] = include_bytes!("../../../assets/makehuman/body.bin.gz");

struct Target {
    idx: Vec<u32>,
    delta: Vec<Vec3>,
}

struct BoneDef {
    name: String,
    parent: Option<usize>,
    head: usize,
    tail: usize,
    weights: Vec<(u32, f32)>,
}

struct Asset {
    base: Vec<Vec3>,
    tris: Vec<[u32; 3]>,
    joints: Vec<Vec<u32>>,
    bones: Vec<BoneDef>,
    targets: HashMap<String, Target>,
}

fn asset() -> &'static Asset {
    static A: OnceLock<Asset> = OnceLock::new();
    A.get_or_init(|| parse().expect("embedded MakeHuman asset is corrupt"))
}

fn parse() -> Option<Asset> {
    let mut raw = vec![];
    flate2::read::GzDecoder::new(PACKED).read_to_end(&mut raw).ok()?;
    let mut p = 4usize;
    let b = &raw;
    let u32_ = |p: &mut usize| {
        let v = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
        *p += 4;
        v
    };
    let f32_ = |p: &mut usize| {
        let v = f32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
        *p += 4;
        v
    };
    let str_ = |p: &mut usize| {
        let n = u16::from_le_bytes([b[*p], b[*p + 1]]) as usize;
        let s = String::from_utf8_lossy(&b[*p + 2..*p + 2 + n]).to_string();
        *p += 2 + n;
        s
    };
    if &b[0..4] != b"MHB1" {
        return None;
    }
    let nv = u32_(&mut p) as usize;
    let mut base = Vec::with_capacity(nv);
    for _ in 0..nv {
        base.push(Vec3::new(f32_(&mut p), f32_(&mut p), f32_(&mut p)));
    }
    let nt = u32_(&mut p) as usize;
    let mut tris = Vec::with_capacity(nt);
    for _ in 0..nt {
        tris.push([u32_(&mut p), u32_(&mut p), u32_(&mut p)]);
    }
    let nj = u32_(&mut p) as usize;
    let mut joints = Vec::with_capacity(nj);
    for _ in 0..nj {
        let _name = str_(&mut p);
        let n = u32_(&mut p) as usize;
        joints.push((0..n).map(|_| u32_(&mut p)).collect());
    }
    let nb = u32_(&mut p) as usize;
    let mut bones = Vec::with_capacity(nb);
    for _ in 0..nb {
        let name = str_(&mut p);
        let parent = i32::from_le_bytes(b[p..p + 4].try_into().unwrap());
        p += 4;
        let head = u32_(&mut p) as usize;
        let tail = u32_(&mut p) as usize;
        let nw = u32_(&mut p) as usize;
        let weights = (0..nw).map(|_| (u32_(&mut p), f32_(&mut p))).collect();
        bones.push(BoneDef { name, parent: (parent >= 0).then_some(parent as usize), head, tail, weights });
    }
    let ntg = u32_(&mut p) as usize;
    let mut targets = HashMap::new();
    for _ in 0..ntg {
        let name = str_(&mut p);
        let n = u32_(&mut p) as usize;
        let mut t = Target { idx: Vec::with_capacity(n), delta: Vec::with_capacity(n) };
        for _ in 0..n {
            t.idx.push(u32_(&mut p));
            let q = |p: &mut usize| {
                let v = i16::from_le_bytes([b[*p], b[*p + 1]]);
                *p += 2;
                v as f32 / 20000.0
            };
            t.delta.push(Vec3::new(q(&mut p), q(&mut p), q(&mut p)));
        }
        targets.insert(name, t);
    }
    Some(Asset { base, tris, joints, bones, targets })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HumanParams {
    /// 0 = female, 1 = male.
    pub gender: f32,
    /// 0..1 (0.5 = 25 years, 1 = 90 years).
    pub age: f32,
    pub muscle: f32,
    pub weight: f32,
    /// Final standing height in metres (uniform scale).
    pub height: f32,
    pub african: f32,
    pub asian: f32,
    pub caucasian: f32,
    pub breast_size: f32,
    pub breast_firmness: f32,
    /// Stylisation (bone length/girth multipliers, applied as figure shapes).
    pub head_size: f32,
    pub neck_length: f32,
    pub torso_length: f32,
    pub arm_length: f32,
    pub leg_length: f32,
    pub hand_size: f32,
    pub foot_size: f32,
    pub shoulder_width: f32,
}

impl Default for HumanParams {
    fn default() -> Self {
        Self {
            gender: 1.0,
            age: 0.5,
            muscle: 0.5,
            weight: 0.5,
            height: 1.78,
            african: 1.0 / 3.0,
            asian: 1.0 / 3.0,
            caucasian: 1.0 / 3.0,
            breast_size: 0.5,
            breast_firmness: 0.5,
            head_size: 1.0,
            neck_length: 1.0,
            torso_length: 1.0,
            arm_length: 1.0,
            leg_length: 1.0,
            hand_size: 1.0,
            foot_size: 1.0,
            shoulder_width: 1.0,
        }
    }
}

pub fn age_to_years(a: f32) -> f32 {
    if a < 0.5 { 1.0 + 24.0 * a / 0.5 } else { 25.0 + 65.0 * (a - 0.5) / 0.5 }
}

pub fn years_to_age(y: f32) -> f32 {
    if y < 25.0 { ((y - 1.0) / 24.0 * 0.5).max(0.0) } else { (0.5 + (y - 25.0) / 65.0 * 0.5).min(1.0) }
}

impl HumanParams {
    pub fn female() -> Self {
        Self { gender: 0.0, height: 1.66, ..Default::default() }
    }

    pub fn presets() -> Vec<(&'static str, HumanParams)> {
        let m = Self::default;
        let f = Self::female;
        let ya = years_to_age;
        vec![
            ("Male", m()),
            ("Female", f()),
            ("Athletic male", HumanParams { muscle: 0.85, weight: 0.45, height: 1.83, ..m() }),
            ("Athletic female", HumanParams { muscle: 0.75, weight: 0.4, height: 1.7, ..f() }),
            ("Bodybuilder", HumanParams { muscle: 1.0, weight: 0.75, height: 1.82, shoulder_width: 1.08, ..m() }),
            ("Slim male", HumanParams { muscle: 0.3, weight: 0.15, ..m() }),
            ("Slim female", HumanParams { muscle: 0.3, weight: 0.15, ..f() }),
            ("Heavy male", HumanParams { muscle: 0.4, weight: 1.0, ..m() }),
            ("Curvy female", HumanParams { weight: 0.8, breast_size: 0.8, ..f() }),
            ("Elderly man", HumanParams { age: ya(75.0), muscle: 0.3, height: 1.72, ..m() }),
            ("Elderly woman", HumanParams { age: ya(75.0), muscle: 0.3, height: 1.6, ..f() }),
            ("Teen boy", HumanParams { age: ya(14.0), height: 1.62, muscle: 0.4, weight: 0.4, ..m() }),
            ("Teen girl", HumanParams { age: ya(14.0), height: 1.57, weight: 0.4, ..f() }),
            ("Child", HumanParams { age: ya(8.0), height: 1.28, gender: 0.5, ..m() }),
            ("Toddler", HumanParams { age: ya(2.5), height: 0.92, gender: 0.5, ..m() }),
            ("Heroic male (comic)", HumanParams { muscle: 0.95, weight: 0.5, height: 1.92, head_size: 0.88, leg_length: 1.08, shoulder_width: 1.12, ..m() }),
            ("Heroic female (comic)", HumanParams { muscle: 0.7, weight: 0.4, height: 1.8, head_size: 0.9, leg_length: 1.1, breast_size: 0.7, ..f() }),
            ("Anime male", HumanParams { muscle: 0.4, weight: 0.3, height: 1.76, head_size: 1.12, leg_length: 1.1, neck_length: 1.1, ..m() }),
            ("Anime female", HumanParams { weight: 0.3, height: 1.6, head_size: 1.18, leg_length: 1.12, neck_length: 1.1, hand_size: 0.9, ..f() }),
            ("Chibi", HumanParams { age: ya(6.0), height: 1.0, gender: 0.5, head_size: 1.6, leg_length: 0.8, arm_length: 0.85, ..m() }),
            ("Fashion model", HumanParams { weight: 0.15, height: 1.8, leg_length: 1.12, neck_length: 1.08, ..f() }),
        ]
    }

    /// Bone-shape multipliers (length, girth) for humanoid bones.
    pub fn shapes(&self) -> Vec<(H, f32, f32)> {
        let mut v = vec![
            (H::Head, self.head_size, self.head_size),
            (H::Neck, self.neck_length, 1.0),
            (H::Spine, self.torso_length, 1.0),
            (H::Chest, self.torso_length, 1.0),
            (H::LeftShoulder, self.shoulder_width, 1.0),
            (H::RightShoulder, self.shoulder_width, 1.0),
        ];
        for s in [Side::Left, Side::Right] {
            let b = |h: H| if s == Side::Left { h } else { h.mirror() };
            v.push((b(H::LeftUpperArm), self.arm_length, 1.0));
            v.push((b(H::LeftLowerArm), self.arm_length, 1.0));
            v.push((b(H::LeftUpperLeg), self.leg_length, 1.0));
            v.push((b(H::LeftLowerLeg), self.leg_length, 1.0));
            v.push((b(H::LeftHand), self.hand_size, self.hand_size));
            v.push((b(H::LeftFoot), self.foot_size, self.foot_size));
        }
        v
    }

    fn target_weights(&self) -> Vec<(String, f32)> {
        let (g, a, m, w) = (self.gender.clamp(0.0, 1.0), self.age.clamp(0.0, 1.0), self.muscle.clamp(0.0, 1.0), self.weight.clamp(0.0, 1.0));
        let genders = [("female", 1.0 - g), ("male", g)];
        let (baby, child, young, old) = if a < 0.5 {
            let young = ((a - 0.1875) * 3.2).max(0.0);
            (( 1.0 - a * 5.333).max(0.0), ((5.333 * a).min(1.0) - young).max(0.0), young, 0.0)
        } else {
            let old = (a * 2.0 - 1.0).max(0.0);
            (0.0, 0.0, 1.0 - old, old)
        };
        let ages = [("baby", baby), ("child", child), ("young", young), ("old", old)];
        let tri = |x: f32| {
            let lo = (1.0 - 2.0 * x).max(0.0);
            let hi = (2.0 * x - 1.0).max(0.0);
            [("min", lo), ("average", 1.0 - lo - hi), ("max", hi)]
        };
        let mut out = vec![];
        for (gn, gw) in genders {
            for (an, aw) in ages {
                if gw * aw < 1e-4 {
                    continue;
                }
                for (mn, mw) in tri(m) {
                    for (wn, ww) in tri(w) {
                        let x = gw * aw * mw * ww;
                        if x > 1e-4 {
                            out.push((format!("universal-{gn}-{an}-{mn}muscle-{wn}weight"), x));
                        }
                    }
                }
                let rs = (self.african + self.asian + self.caucasian).max(1e-4);
                for (rn, rw) in [("african", self.african), ("asian", self.asian), ("caucasian", self.caucasian)] {
                    let x = gw * aw * rw / rs;
                    if x > 1e-4 {
                        out.push((format!("{rn}-{gn}-{an}"), x));
                    }
                }
            }
        }
        let female = 1.0 - g;
        if female > 1e-3 {
            for (cn, cw) in tri(self.breast_size) {
                for (fname, fw) in tri(self.breast_firmness) {
                    let x = female * cw * fw;
                    if x > 1e-4 && !(cn == "average" && fname == "average") {
                        out.push((format!("breast/female-young-averagemuscle-averageweight-{cn}cup-{fname}firmness"), x));
                    }
                }
            }
        }
        out
    }
}

fn humanoid_names() -> Vec<(H, &'static str)> {
    let mut v = vec![
        (H::Hips, "root"),
        (H::Spine, "spine04"),
        (H::Chest, "spine02"),
        (H::UpperChest, "spine01"),
        (H::Neck, "neck01"),
        (H::Head, "head"),
        (H::Jaw, "jaw"),
        (H::LeftEye, "eye.L"),
        (H::RightEye, "eye.R"),
    ];
    for (s, x) in [(Side::Left, "L"), (Side::Right, "R")] {
        let b = |h: H| if s == Side::Left { h } else { h.mirror() };
        let n = |base: &str| -> &'static str { Box::leak(format!("{base}.{x}").into_boxed_str()) };
        v.push((b(H::LeftShoulder), n("clavicle")));
        v.push((b(H::LeftUpperArm), n("upperarm01")));
        v.push((b(H::LeftLowerArm), n("lowerarm01")));
        v.push((b(H::LeftHand), n("wrist")));
        v.push((b(H::LeftUpperLeg), n("upperleg01")));
        v.push((b(H::LeftLowerLeg), n("lowerleg01")));
        v.push((b(H::LeftFoot), n("foot")));
        for (fi, f) in Finger::ALL.iter().enumerate() {
            for seg in 0..3 {
                v.push((finger_bone(s, *f, seg).unwrap(), n(&format!("finger{}-{}", fi + 1, seg + 1))));
            }
        }
    }
    v
}

/// Build the realistic human. Stylisation multipliers are NOT baked (see [`HumanParams::shapes`]).
pub fn build(p: &HumanParams) -> Model {
    let a = asset();
    let mut pos = a.base.clone();
    for (name, w) in p.target_weights() {
        if let Some(t) = a.targets.get(&name) {
            for (i, d) in t.idx.iter().zip(&t.delta) {
                if let Some(v) = pos.get_mut(*i as usize) {
                    *v += *d * w;
                }
            }
        }
    }
    // Ground and scale to the requested height (body vertices only).
    let mut used = vec![false; pos.len()];
    for t in &a.tris {
        for i in t {
            used[*i as usize] = true;
        }
    }
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for (v, u) in pos.iter().zip(&used) {
        if *u {
            lo = lo.min(v.y);
            hi = hi.max(v.y);
        }
    }
    let s = p.height.max(0.3) / (hi - lo).max(1e-3);
    for v in &mut pos {
        *v = Vec3::new(v.x * s, (v.y - lo) * s, v.z * s);
    }
    let joint_pos: Vec<Vec3> = a
        .joints
        .iter()
        .map(|vs| vs.iter().map(|i| pos[*i as usize]).sum::<Vec3>() / vs.len().max(1) as f32)
        .collect();
    // Skeleton (identity rotations, positions from joint helpers).
    let mut skel = Skeleton::default();
    for b in &a.bones {
        let head = joint_pos[b.head];
        let ph = b.parent.map(|q| joint_pos[a.bones[q].head]).unwrap_or(Vec3::ZERO);
        skel.bones.push(Bone::new(b.name.clone(), b.parent, Transform::from_translation(head - ph)));
    }
    skel.recompute_inverse_bind();
    skel.rebuild_children();
    for (i, b) in a.bones.iter().enumerate() {
        let d = joint_pos[b.tail] - joint_pos[b.head];
        let l = d.length();
        if l > 1e-5 {
            skel.bones[i].axis = d / l;
            skel.bones[i].length = l;
        }
    }
    let mut map = HumanoidMap::default();
    for (h, n) in humanoid_names() {
        if let Some(i) = skel.find(n) {
            map.bones.insert(h, i);
        }
    }
    map.compute_tpose(&skel);
    // Body mesh (compact).
    let mut remap = vec![u32::MAX; pos.len()];
    let mut mesh = Mesh { name: "Body".into(), rigid_transform: Mat4::IDENTITY, ..Default::default() };
    for (i, u) in used.iter().enumerate() {
        if *u {
            remap[i] = mesh.positions.len() as u32;
            mesh.positions.push(pos[i]);
        }
    }
    mesh.indices = a.tris.iter().flat_map(|t| t.iter().map(|i| remap[*i as usize])).collect();
    let mut per: Vec<Vec<(u16, f32)>> = vec![vec![]; mesh.positions.len()];
    for (bi, b) in a.bones.iter().enumerate() {
        for (v, w) in &b.weights {
            let r = remap.get(*v as usize).copied().unwrap_or(u32::MAX);
            if r != u32::MAX && *w > 0.0 {
                per[r as usize].push((bi as u16, *w));
            }
        }
    }
    let root = skel.find("root").unwrap_or(0) as u16;
    for ws in &mut per {
        ws.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut j = [root, 0, 0, 0];
        let mut w = [1.0, 0.0, 0.0, 0.0];
        let s: f32 = ws.iter().take(4).map(|x| x.1).sum();
        if s > 1e-6 {
            w = [0.0; 4];
            for (k, (b, x)) in ws.iter().take(4).enumerate() {
                j[k] = *b;
                w[k] = *x / s;
            }
        }
        mesh.joints.push(j);
        mesh.weights.push(w);
    }
    mesh.compute_normals();
    mesh.sanitize();
    let mut model = Model::empty("Human", SourceFormat::Procedural);
    model.materials = vec![
        Material { name: "Skin".into(), base_color: Vec4::new(0.86, 0.72, 0.64, 1.0), ..Default::default() },
        Material { name: "Eyes".into(), base_color: Vec4::new(0.95, 0.95, 0.95, 1.0), ..Default::default() },
    ];
    model.meshes.push(mesh);
    // Simple eyeballs in the sockets.
    for eye in ["eye.L", "eye.R"] {
        if let Some(e) = skel.find(eye) {
            let c = skel.rest_world()[e].w_axis.truncate();
            let r = 0.0118 * (p.height / 1.75).max(0.5);
            let mut m = Mesh { name: eye.into(), material: 1, rigid_bone: Some(e), rigid_transform: Mat4::from_translation(Vec3::ZERO), ..Default::default() };
            let n = 12;
            for i in 0..=n {
                let th = std::f32::consts::PI * i as f32 / n as f32;
                for j in 0..=n {
                    let ph = std::f32::consts::TAU * j as f32 / n as f32;
                    let d = Vec3::new(th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin());
                    m.positions.push(d * r);
                    m.normals.push(d);
                }
            }
            for i in 0..n as u32 {
                for j in 0..n as u32 {
                    let a0 = i * (n as u32 + 1) + j;
                    let b0 = a0 + n as u32 + 1;
                    m.indices.extend_from_slice(&[a0, b0, a0 + 1, a0 + 1, b0, b0 + 1]);
                }
            }
            let _ = c;
            m.sanitize();
            model.meshes.push(m);
        }
    }
    model.skeleton = skel;
    model.humanoid = map;
    model
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_all_presets() {
        for (name, p) in HumanParams::presets() {
            let m = build(&p);
            assert!(m.humanoid.is_valid(), "{name}");
            assert!(m.humanoid.bones.len() >= 53, "{name}");
            let (lo, hi) = m.rest_bounds();
            assert!((hi.y - lo.y - p.height).abs() < 0.05, "{name}: {}", hi.y - lo.y);
        }
    }
}

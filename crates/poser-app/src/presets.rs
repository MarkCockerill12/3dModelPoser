//! Built-in pose and hand presets (portable HumanPoses in the normalised frame).
//! Convention: character faces +Z, its left side is +X, up is +Y. Poses are written for the
//! left side and mirrored for the right.
use poser_core::glam::{Quat, Vec3};
use poser_core::humanoid::{Finger, HumanBone as H, Side, finger_bone};
use poser_core::pose::HumanPose;

fn rx(d: f32) -> Quat {
    Quat::from_rotation_x(d.to_radians())
}
fn ry(d: f32) -> Quat {
    Quat::from_rotation_y(d.to_radians())
}
fn rz(d: f32) -> Quat {
    Quat::from_rotation_z(d.to_radians())
}
fn mirror(q: Quat) -> Quat {
    Quat::from_xyzw(q.x, -q.y, -q.z, q.w)
}
/// Left upper arm pointing along `dir`, twisted about itself (degrees, + = external).
fn arm(dir: Vec3, twist: f32) -> Quat {
    let d = dir.normalize();
    Quat::from_axis_angle(d, twist.to_radians()) * Quat::from_rotation_arc(Vec3::X, d)
}
/// Upper leg pointing along `dir`.
fn leg(dir: Vec3, twist: f32) -> Quat {
    let d = dir.normalize();
    Quat::from_axis_angle(d, twist.to_radians()) * Quat::from_rotation_arc(Vec3::NEG_Y, d)
}
/// Elbow flex (left), forward = positive.
fn elbow(d: f32) -> Quat {
    ry(-d)
}
/// Knee flex (backwards), positive.
fn knee(d: f32) -> Quat {
    rx(d)
}

struct PB(HumanPose);

impl PB {
    fn new(name: &str, cat: &str) -> Self {
        PB(HumanPose { name: name.into(), category: cat.into(), ..Default::default() })
    }
    fn set(mut self, b: H, q: Quat) -> Self {
        self.0.rotations.insert(b, q);
        self
    }
    /// Same rotation on both sides (left given, right mirrored).
    fn sym(self, left: H, q: Quat) -> Self {
        self.set(left, q).set(left.mirror(), mirror(q))
    }
    /// Right-side bone from a left-side definition.
    fn right(self, left: H, q: Quat) -> Self {
        self.set(left.mirror(), mirror(q))
    }
    fn hips(mut self, off: Vec3, q: Quat) -> Self {
        self.0.hips_offset = Some(off);
        self.0.rotations.insert(H::Hips, q);
        self
    }
    fn hand(mut self, side: Side, h: &HumanPose) -> Self {
        let src = if side == Side::Left { h.clone() } else { h.mirrored() };
        for (b, q) in src.rotations {
            if b.is_finger() && b.side() == side {
                self.0.rotations.insert(b, q);
            }
        }
        self
    }
    fn done(self) -> HumanPose {
        self.0
    }
}

/// Finger curl preset for the left hand: curls (thumb, index, middle, ring, little) 0..1.
pub fn hand_shape(name: &str, curl: [f32; 5], spread: f32, opposition: f32) -> HumanPose {
    let mut p = HumanPose { name: name.into(), category: "Hand".into(), ..Default::default() };
    let spreads = [0.0, 12.0, 2.0, -7.0, -15.0];
    for (i, f) in Finger::ALL.iter().enumerate() {
        let c = curl[i];
        if *f == Finger::Thumb {
            let t = Vec3::new(1.0, 0.0, 1.0).normalize();
            let axis = t.cross(Vec3::NEG_Y).normalize();
            let opp_axis = Vec3::NEG_Y;
            let meta = Quat::from_axis_angle(opp_axis, (opposition * 28.0).to_radians())
                * Quat::from_axis_angle(axis, (c * 22.0 + opposition * 18.0).to_radians());
            p.rotations.insert(finger_bone(Side::Left, *f, 0).unwrap(), meta);
            p.rotations.insert(finger_bone(Side::Left, *f, 1).unwrap(), Quat::from_axis_angle(axis, (c * 45.0).to_radians()));
            p.rotations.insert(finger_bone(Side::Left, *f, 2).unwrap(), Quat::from_axis_angle(axis, (c * 65.0).to_radians()));
        } else {
            let curl_axis = Vec3::NEG_Z;
            let sp = Quat::from_axis_angle(Vec3::NEG_Y, (spreads[i] * spread).to_radians());
            p.rotations.insert(finger_bone(Side::Left, *f, 0).unwrap(), sp * Quat::from_axis_angle(curl_axis, (c * 82.0).to_radians()));
            p.rotations.insert(finger_bone(Side::Left, *f, 1).unwrap(), Quat::from_axis_angle(curl_axis, (c * 98.0).to_radians()));
            p.rotations.insert(finger_bone(Side::Left, *f, 2).unwrap(), Quat::from_axis_angle(curl_axis, (c * 72.0).to_radians()));
        }
    }
    p
}

pub fn hand_presets() -> Vec<HumanPose> {
    vec![
        hand_shape("Relaxed", [0.2, 0.15, 0.22, 0.28, 0.36], 0.15, 0.2),
        hand_shape("Flat", [0.0; 5], 0.0, 0.0),
        hand_shape("Spread", [0.0; 5], 1.0, -0.2),
        hand_shape("Fist", [0.8, 1.0, 1.0, 1.0, 1.0], 0.0, 0.7),
        hand_shape("Point", [0.7, 0.0, 1.0, 1.0, 1.0], 0.0, 0.6),
        hand_shape("Peace", [0.75, 0.0, 0.0, 1.0, 1.0], 0.8, 0.7),
        hand_shape("Thumbs up", [0.0, 1.0, 1.0, 1.0, 1.0], 0.0, -0.3),
        hand_shape("Grip", [0.5, 0.62, 0.65, 0.68, 0.7], 0.0, 0.7),
        hand_shape("OK / pinch", [0.55, 0.62, 0.1, 0.05, 0.05], 0.45, 0.85),
        hand_shape("Claw", [0.4, 0.55, 0.55, 0.55, 0.55], 0.8, 0.4),
        hand_shape("Cupped", [0.3, 0.3, 0.32, 0.34, 0.36], 0.05, 0.4),
        hand_shape("Rock", [0.6, 0.0, 1.0, 1.0, 0.0], 0.6, 0.7),
        hand_shape("Pen grip", [0.45, 0.45, 0.6, 0.85, 0.9], 0.1, 0.8),
        hand_shape("Three", [0.75, 0.0, 0.0, 0.0, 1.0], 0.5, 0.7),
        hand_shape("Karate chop", [0.1, 0.0, 0.0, 0.0, 0.0], 0.0, 0.5),
    ]
}

pub fn body_presets() -> Vec<HumanPose> {
    let relaxed = hand_presets()[0].clone();
    let fist = hand_presets()[3].clone();
    let mut v = vec![
        PB::new("T-pose", "Basic").done(),
        PB::new("A-pose", "Basic").sym(H::LeftUpperArm, rz(-45.0)).done(),
        PB::new("Relaxed stand", "Standing")
            .sym(H::LeftUpperArm, arm(Vec3::new(0.18, -1.0, 0.02), 0.0))
            .sym(H::LeftLowerArm, elbow(12.0))
            .sym(H::LeftUpperLeg, leg(Vec3::new(0.04, -1.0, 0.0), 0.0))
            .hand(Side::Left, &relaxed)
            .hand(Side::Right, &relaxed)
            .done(),
        PB::new("Contrapposto", "Standing")
            .hips(Vec3::new(-0.03, -0.01, 0.0), rz(-5.0) * ry(6.0))
            .set(H::Spine, rz(3.0))
            .set(H::Chest, rz(4.0) * ry(-4.0))
            .set(H::Neck, rz(-3.0))
            .set(H::Head, rz(-5.0) * ry(8.0))
            .sym(H::LeftUpperArm, arm(Vec3::new(0.15, -1.0, 0.0), 0.0))
            .sym(H::LeftLowerArm, elbow(14.0))
            .set(H::LeftUpperLeg, leg(Vec3::new(0.12, -1.0, 0.12), 0.0))
            .set(H::LeftLowerLeg, knee(18.0))
            .set(H::LeftFoot, rx(-6.0))
            .right(H::LeftUpperLeg, leg(Vec3::new(-0.02, -1.0, -0.02), 0.0))
            .hand(Side::Left, &relaxed)
            .hand(Side::Right, &relaxed)
            .done(),
        PB::new("Walking", "Motion")
            .hips(Vec3::new(0.0, -0.03, 0.0), ry(-6.0))
            .set(H::Chest, ry(10.0))
            .set(H::LeftUpperLeg, leg(Vec3::new(0.03, -0.94, 0.33), 0.0))
            .set(H::LeftLowerLeg, knee(12.0))
            .set(H::LeftFoot, rx(-12.0))
            .right(H::LeftUpperLeg, leg(Vec3::new(0.03, -0.96, -0.28), 0.0))
            .right(H::LeftLowerLeg, knee(28.0))
            .right(H::LeftFoot, rx(18.0))
            .set(H::LeftUpperArm, arm(Vec3::new(0.15, -0.97, -0.22), 0.0))
            .set(H::LeftLowerArm, elbow(15.0))
            .right(H::LeftUpperArm, arm(Vec3::new(0.15, -0.95, 0.28), 0.0))
            .right(H::LeftLowerArm, elbow(30.0))
            .hand(Side::Left, &relaxed)
            .hand(Side::Right, &relaxed)
            .done(),
        PB::new("Running", "Motion")
            .hips(Vec3::new(0.0, -0.06, 0.0), rx(10.0) * ry(-8.0))
            .set(H::Spine, rx(6.0))
            .set(H::Chest, ry(14.0))
            .set(H::Head, rx(-12.0))
            .set(H::LeftUpperLeg, leg(Vec3::new(0.04, -0.55, 0.83), 0.0))
            .set(H::LeftLowerLeg, knee(95.0))
            .set(H::LeftFoot, rx(10.0))
            .right(H::LeftUpperLeg, leg(Vec3::new(0.03, -0.86, -0.5), 0.0))
            .right(H::LeftLowerLeg, knee(65.0))
            .right(H::LeftFoot, rx(30.0))
            .set(H::LeftUpperArm, arm(Vec3::new(0.2, -0.8, -0.55), -40.0))
            .set(H::LeftLowerArm, elbow(85.0))
            .right(H::LeftUpperArm, arm(Vec3::new(0.2, -0.75, 0.6), -40.0))
            .right(H::LeftLowerArm, elbow(95.0))
            .hand(Side::Left, &fist)
            .hand(Side::Right, &fist)
            .done(),
        PB::new("Sitting (chair)", "Sitting")
            .hips(Vec3::new(0.0, -0.46, -0.08), rx(-4.0))
            .set(H::Spine, rx(4.0))
            .sym(H::LeftUpperLeg, leg(Vec3::new(0.1, -0.05, 1.0), 0.0))
            .sym(H::LeftLowerLeg, knee(88.0))
            .sym(H::LeftUpperArm, arm(Vec3::new(0.18, -0.9, 0.35), -20.0))
            .sym(H::LeftLowerArm, elbow(55.0))
            .hand(Side::Left, &relaxed)
            .hand(Side::Right, &relaxed)
            .done(),
        PB::new("Sitting cross-legged", "Sitting")
            .hips(Vec3::new(0.0, -0.84, -0.05), rx(-6.0))
            .set(H::Spine, rx(10.0))
            .set(H::Chest, rx(4.0))
            .sym(H::LeftUpperLeg, leg(Vec3::new(0.75, -0.08, 0.65), 40.0))
            .sym(H::LeftLowerLeg, knee(140.0))
            .sym(H::LeftUpperArm, arm(Vec3::new(0.3, -0.85, 0.3), -20.0))
            .sym(H::LeftLowerArm, elbow(40.0))
            .hand(Side::Left, &relaxed)
            .hand(Side::Right, &relaxed)
            .done(),
        PB::new("Kneeling", "Sitting")
            .hips(Vec3::new(0.0, -0.47, 0.0), rx(0.0))
            .sym(H::LeftUpperLeg, leg(Vec3::new(0.06, -1.0, 0.05), 0.0))
            .sym(H::LeftLowerLeg, knee(92.0))
            .sym(H::LeftFoot, rx(45.0))
            .sym(H::LeftUpperArm, arm(Vec3::new(0.15, -1.0, 0.1), 0.0))
            .sym(H::LeftLowerArm, elbow(20.0))
            .hand(Side::Left, &relaxed)
            .hand(Side::Right, &relaxed)
            .done(),
        PB::new("One knee down", "Sitting")
            .hips(Vec3::new(0.0, -0.45, 0.0), ry(0.0))
            .set(H::LeftUpperLeg, leg(Vec3::new(0.08, -0.08, 1.0), 0.0))
            .set(H::LeftLowerLeg, knee(90.0))
            .right(H::LeftUpperLeg, leg(Vec3::new(0.05, -1.0, -0.1), 0.0))
            .right(H::LeftLowerLeg, knee(90.0))
            .right(H::LeftFoot, rx(40.0))
            .set(H::LeftUpperArm, arm(Vec3::new(0.25, -0.8, 0.5), -30.0))
            .set(H::LeftLowerArm, elbow(50.0))
            .right(H::LeftUpperArm, arm(Vec3::new(0.15, -1.0, 0.0), 0.0))
            .hand(Side::Left, &relaxed)
            .hand(Side::Right, &relaxed)
            .done(),
        PB::new("Crouching", "Sitting")
            .hips(Vec3::new(0.0, -0.56, -0.1), rx(20.0))
            .set(H::Spine, rx(12.0))
            .set(H::Head, rx(-20.0))
            .sym(H::LeftUpperLeg, leg(Vec3::new(0.3, -0.25, 0.9), 0.0))
            .sym(H::LeftLowerLeg, knee(135.0))
            .sym(H::LeftFoot, rx(-25.0))
            .sym(H::LeftUpperArm, arm(Vec3::new(0.25, -0.6, 0.75), -30.0))
            .sym(H::LeftLowerArm, elbow(25.0))
            .hand(Side::Left, &relaxed)
            .hand(Side::Right, &relaxed)
            .done(),
        PB::new("Jumping", "Motion")
            .hips(Vec3::new(0.0, 0.35, 0.0), rx(-6.0))
            .set(H::Spine, rx(-6.0))
            .set(H::LeftUpperLeg, leg(Vec3::new(0.1, -0.75, 0.6), 0.0))
            .set(H::LeftLowerLeg, knee(110.0))
            .right(H::LeftUpperLeg, leg(Vec3::new(0.06, -0.95, 0.15), 0.0))
            .right(H::LeftLowerLeg, knee(80.0))
            .sym(H::LeftFoot, rx(30.0))
            .sym(H::LeftUpperArm, arm(Vec3::new(0.45, 0.88, 0.1), 0.0))
            .sym(H::LeftLowerArm, elbow(15.0))
            .hand(Side::Left, &hand_presets()[2])
            .hand(Side::Right, &hand_presets()[2])
            .done(),
        PB::new("Fighting stance", "Action")
            .hips(Vec3::new(0.0, -0.08, 0.0), ry(-28.0))
            .set(H::Chest, ry(-8.0))
            .set(H::Head, ry(30.0) * rx(8.0))
            .set(H::LeftUpperLeg, leg(Vec3::new(0.2, -0.95, 0.25), 0.0))
            .set(H::LeftLowerLeg, knee(25.0))
            .right(H::LeftUpperLeg, leg(Vec3::new(0.12, -0.95, -0.25), 0.0))
            .right(H::LeftLowerLeg, knee(30.0))
            .sym(H::LeftUpperArm, arm(Vec3::new(0.35, -0.6, 0.72), -60.0))
            .sym(H::LeftLowerArm, elbow(125.0))
            .hand(Side::Left, &fist)
            .hand(Side::Right, &fist)
            .done(),
        PB::new("Waving", "Gesture")
            .sym(H::LeftUpperLeg, leg(Vec3::new(0.04, -1.0, 0.0), 0.0))
            .set(H::LeftUpperArm, arm(Vec3::new(0.15, -1.0, 0.05), 0.0))
            .set(H::LeftLowerArm, elbow(10.0))
            .right(H::LeftUpperArm, arm(Vec3::new(0.95, 0.3, 0.1), -90.0))
            .right(H::LeftLowerArm, elbow(95.0))
            .set(H::Head, rz(6.0))
            .hand(Side::Left, &relaxed)
            .hand(Side::Right, &hand_presets()[1])
            .done(),
        PB::new("Reaching up", "Gesture")
            .set(H::Spine, rx(-6.0))
            .set(H::Chest, rx(-6.0))
            .set(H::Head, rx(-25.0))
            .sym(H::LeftUpperArm, arm(Vec3::new(0.2, 1.0, 0.12), 0.0))
            .sym(H::LeftLowerArm, elbow(5.0))
            .sym(H::LeftFoot, rx(25.0))
            .hips(Vec3::new(0.0, 0.04, 0.0), Quat::IDENTITY)
            .hand(Side::Left, &hand_presets()[2])
            .hand(Side::Right, &hand_presets()[2])
            .done(),
        PB::new("Lying on back", "Lying")
            .hips(Vec3::new(0.0, -0.86, -0.6), rx(-90.0))
            .sym(H::LeftUpperArm, arm(Vec3::new(0.25, -1.0, 0.0), 0.0))
            .sym(H::LeftUpperLeg, leg(Vec3::new(0.05, -1.0, 0.0), 0.0))
            .sym(H::LeftFoot, rx(30.0))
            .hand(Side::Left, &relaxed)
            .hand(Side::Right, &relaxed)
            .done(),
        PB::new("Hero landing", "Action")
            .hips(Vec3::new(0.0, -0.6, 0.0), rx(25.0))
            .set(H::Spine, rx(15.0))
            .set(H::Head, rx(-30.0))
            .set(H::LeftUpperLeg, leg(Vec3::new(0.25, -0.3, 0.9), 0.0))
            .set(H::LeftLowerLeg, knee(130.0))
            .right(H::LeftUpperLeg, leg(Vec3::new(0.15, -0.9, -0.4), 0.0))
            .right(H::LeftLowerLeg, knee(120.0))
            .right(H::LeftFoot, rx(45.0))
            .set(H::LeftUpperArm, arm(Vec3::new(0.7, -0.4, -0.55), 0.0))
            .right(H::LeftUpperArm, arm(Vec3::new(0.1, -0.85, 0.5), -30.0))
            .right(H::LeftLowerArm, elbow(20.0))
            .hand(Side::Left, &hand_presets()[2])
            .hand(Side::Right, &fist)
            .done(),
    ];
    for p in &mut v {
        if p.category.is_empty() {
            p.category = "Pose".into();
        }
    }
    v
}

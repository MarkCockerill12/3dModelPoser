//! Pose from photo: offline 2D keypoints (poser-ai) lifted to 3D with the figure's bone lengths.
use crate::app::PoserApp;
use poser_ai::*;
use poser_core::glam::{Mat3, Quat, Vec2, Vec3};
use poser_core::humanoid::HumanBone as H;
use poser_core::pose::{HumanPose, extract_human_pose};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};

pub struct PhotoJob {
    pub figure_id: u64,
    pub path: PathBuf,
    pub rx: Receiver<anyhow::Result<PhotoResult>>,
}

/// Body keypoints, both hands (person's left, right) and the image size.
pub type PhotoResult = (Vec<Keypoint>, [Option<poser_ai::HandShape>; 2], u32, u32);

/// Run body + hand estimation on an image (on a worker thread).
pub fn estimate_photo(rgb: &image::RgbImage) -> anyhow::Result<PhotoResult> {
    let (w, h) = rgb.dimensions();
    let k = poser_ai::estimate(rgb)?;
    let hands = poser_ai::estimate_hands(rgb, &k);
    Ok((k, hands, w, h))
}

fn frame(x: Vec3, y: Vec3) -> Quat {
    let y = y.normalize_or(Vec3::Y);
    let x = (x - y * x.dot(y)).normalize_or(Vec3::X);
    let z = x.cross(y);
    Quat::from_mat3(&Mat3::from_cols(x, y, z)).normalize()
}

/// Lift keypoints to 3D (camera looks down -Z at the person; +Z = towards camera).
fn lift(k: &[Keypoint], len: &dyn Fn(H, H) -> f32) -> Vec<Vec3> {
    let p = |i: usize| Vec2::new(k[i].pos.x, -k[i].pos.y);
    let mid = |a: usize, b: usize| (p(a) + p(b)) * 0.5;
    // Pixels per metre from the least foreshortened long segment.
    let pairs = [
        (L_SHOULDER, L_ELBOW, H::LeftUpperArm, H::LeftLowerArm),
        (R_SHOULDER, R_ELBOW, H::RightUpperArm, H::RightLowerArm),
        (L_ELBOW, L_WRIST, H::LeftLowerArm, H::LeftHand),
        (R_ELBOW, R_WRIST, H::RightLowerArm, H::RightHand),
        (L_HIP, L_KNEE, H::LeftUpperLeg, H::LeftLowerLeg),
        (R_HIP, R_KNEE, H::RightUpperLeg, H::RightLowerLeg),
        (L_KNEE, L_ANKLE, H::LeftLowerLeg, H::LeftFoot),
        (R_KNEE, R_ANKLE, H::RightLowerLeg, H::RightFoot),
    ];
    let mut s: f32 = 1.0;
    for (a, b, ha, hb) in pairs {
        let l = len(ha, hb);
        if l > 1e-3 && k[a].score > 0.3 && k[b].score > 0.3 {
            s = s.max((p(b) - p(a)).length() / l);
        }
    }
    let torso_px = (mid(L_SHOULDER, R_SHOULDER) - mid(L_HIP, R_HIP)).length();
    let torso_m = len(H::Hips, H::Neck).max(1e-3);
    s = s.max(torso_px / torso_m);
    let depth = |d2: Vec2, l: f32, sign: f32| -> Vec3 {
        let d = d2 / s;
        let z = (l * l - d.length_squared()).max(0.0).sqrt();
        Vec3::new(d.x, d.y, z * sign)
    };
    // Body turn: nose to the person's left (screen right) => turned towards their left.
    let sh_mid = mid(L_SHOULDER, R_SHOULDER);
    let facing_away = p(L_SHOULDER).x < p(R_SHOULDER).x;
    let turn = if (p(NOSE).x - sh_mid.x) > 0.0 { 1.0 } else { -1.0 } * if facing_away { -1.0 } else { 1.0 };
    let mut out = vec![Vec3::ZERO; 17];
    let hip_mid3 = Vec3::ZERO;
    let half_hip = len(H::LeftUpperLeg, H::RightUpperLeg) * 0.5;
    let half_sh = len(H::LeftUpperArm, H::RightUpperArm) * 0.5;
    let hip_lr = depth(p(L_HIP) - mid(L_HIP, R_HIP), half_hip, -turn);
    out[L_HIP] = hip_mid3 + hip_lr;
    out[R_HIP] = hip_mid3 - hip_lr;
    let torso = depth(sh_mid - mid(L_HIP, R_HIP), torso_m, 1.0) * Vec3::new(1.0, 1.0, 0.3);
    let sh3 = hip_mid3 + torso;
    let sh_lr = depth(p(L_SHOULDER) - sh_mid, half_sh, -turn);
    out[L_SHOULDER] = sh3 + sh_lr;
    out[R_SHOULDER] = sh3 - sh_lr;
    let mut chain = |from: usize, to: usize, l: f32, sign: f32| {
        out[to] = out[from] + depth(p(to) - p(from), l, sign);
    };
    chain(L_SHOULDER, L_ELBOW, len(H::LeftUpperArm, H::LeftLowerArm), 1.0);
    chain(R_SHOULDER, R_ELBOW, len(H::RightUpperArm, H::RightLowerArm), 1.0);
    chain(L_ELBOW, L_WRIST, len(H::LeftLowerArm, H::LeftHand), 1.0);
    chain(R_ELBOW, R_WRIST, len(H::RightLowerArm, H::RightHand), 1.0);
    chain(L_HIP, L_KNEE, len(H::LeftUpperLeg, H::LeftLowerLeg), 1.0);
    chain(R_HIP, R_KNEE, len(H::RightUpperLeg, H::RightLowerLeg), 1.0);
    chain(L_KNEE, L_ANKLE, len(H::LeftLowerLeg, H::LeftFoot), -1.0);
    chain(R_KNEE, R_ANKLE, len(H::RightLowerLeg, H::RightFoot), -1.0);
    // Head points: place from shoulders in the image plane, nose a little forward.
    let head_scale = 1.0 / s;
    for i in [NOSE, L_EYE, R_EYE, L_EAR, R_EAR] {
        let d = (p(i) - sh_mid) * head_scale;
        out[i] = sh3 + Vec3::new(d.x, d.y, if i == NOSE { 0.08 } else { 0.0 });
    }
    out
}

impl PoserApp {
    pub fn photo_dialog(&mut self) {
        let Some(f) = self.sel.figure.filter(|f| self.scene.figures.get(*f).is_some_and(|x| x.model.humanoid.is_valid())) else {
            self.status = "Select a humanoid figure first (pose from photo needs a humanoid rig)".into();
            return;
        };
        let Some(path) = rfd::FileDialog::new().add_filter("Photos", &["png", "jpg", "jpeg", "bmp", "webp"]).set_title("Pose from photo").pick_file() else { return };
        let (tx, rx) = channel();
        let p2 = path.clone();
        std::thread::spawn(move || {
            let r = image::open(&p2).map_err(anyhow::Error::from).and_then(|img| estimate_photo(&img.to_rgb8()));
            let _ = tx.send(r);
        });
        self.photo_job = Some(PhotoJob { figure_id: self.scene.figures[f].id, path, rx });
        self.status = "Estimating pose from photo (offline)...".into();
    }

    pub fn poll_photo(&mut self, ctx: &egui::Context) {
        let Some(job) = self.photo_job.as_ref() else { return };
        ctx.request_repaint_after(std::time::Duration::from_millis(50));
        let Ok(res) = job.rx.try_recv() else { return };
        let job = self.photo_job.take().unwrap();
        let Some(idx) = self.scene.index_of(job.figure_id) else { return };
        let (kps, hands, _w, _h) = match res {
            Ok(x) => x,
            Err(e) => {
                self.status = format!("Pose from photo failed: {e:#}");
                return;
            }
        };
        let good = kps.iter().filter(|k| k.score > 0.3).count();
        if good < 8 {
            self.status = "Couldn't find a clear full-body person in that photo".into();
            return;
        }
        let fig = &self.scene.figures[idx];
        let m = fig.model.clone();
        // T-pose joint positions for bone lengths.
        let mut tl = m.skeleton.rest_locals();
        for (i, t) in tl.iter_mut().enumerate() {
            t.rotation = m.humanoid.tpose_local[i];
        }
        let tw = m.skeleton.evaluate(&tl, fig.shapes_opt()).world;
        let jp = |h: H| m.humanoid.get(h).map(|b| tw[b].w_axis.truncate());
        let len = |a: H, b: H| match (jp(a), jp(b)) {
            (Some(x), Some(y)) => (y - x).length(),
            _ => 0.3,
        };
        let pts = lift(&kps, &len);
        // Normalised world rotations per canonical bone.
        let dir = |a: usize, b: usize| (pts[b] - pts[a]).normalize_or(Vec3::NEG_Y);
        let hips_n = frame(pts[L_HIP] - pts[R_HIP], (pts[L_SHOULDER] + pts[R_SHOULDER]) * 0.5 - (pts[L_HIP] + pts[R_HIP]) * 0.5);
        let chest_n = frame(pts[L_SHOULDER] - pts[R_SHOULDER], (pts[L_SHOULDER] + pts[R_SHOULDER]) * 0.5 - (pts[L_HIP] + pts[R_HIP]) * 0.5);
        let ear_mid = (pts[L_EAR] + pts[R_EAR]) * 0.5;
        let sh_mid = (pts[L_SHOULDER] + pts[R_SHOULDER]) * 0.5;
        let head_up = ear_mid - sh_mid;
        let head_n = {
            let fwd = (pts[NOSE] - ear_mid).normalize_or(Vec3::Z);
            let up = head_up.normalize_or(Vec3::Y);
            frame(up.cross(fwd), up)
        };
        let limb = |h: H, a: usize, b: usize| -> Quat {
            let t = h.tpose_direction().unwrap_or(Vec3::NEG_Y);
            Quat::from_rotation_arc(t, dir(a, b))
        };
        let mut n: std::collections::HashMap<H, Quat> = std::collections::HashMap::new();
        n.insert(H::Hips, hips_n);
        n.insert(H::Spine, hips_n.slerp(chest_n, 0.35));
        n.insert(H::Chest, hips_n.slerp(chest_n, 0.7));
        n.insert(H::UpperChest, chest_n);
        n.insert(H::Neck, chest_n.slerp(head_n, 0.5));
        n.insert(H::Head, head_n);
        n.insert(H::LeftShoulder, chest_n);
        n.insert(H::RightShoulder, chest_n);
        n.insert(H::LeftUpperArm, limb(H::LeftUpperArm, L_SHOULDER, L_ELBOW));
        n.insert(H::RightUpperArm, limb(H::RightUpperArm, R_SHOULDER, R_ELBOW));
        n.insert(H::LeftLowerArm, limb(H::LeftLowerArm, L_ELBOW, L_WRIST));
        n.insert(H::RightLowerArm, limb(H::RightLowerArm, R_ELBOW, R_WRIST));
        n.insert(H::LeftUpperLeg, limb(H::LeftUpperLeg, L_HIP, L_KNEE));
        n.insert(H::RightUpperLeg, limb(H::RightUpperLeg, R_HIP, R_KNEE));
        n.insert(H::LeftLowerLeg, limb(H::LeftLowerLeg, L_KNEE, L_ANKLE));
        n.insert(H::RightLowerLeg, limb(H::RightLowerLeg, R_KNEE, R_ANKLE));
        // Hands/feet follow their parents.
        for (c, p) in [(H::LeftHand, H::LeftLowerArm), (H::RightHand, H::RightLowerArm), (H::LeftFoot, H::LeftLowerLeg), (H::RightFoot, H::RightLowerLeg)] {
            let v = n[&p];
            n.insert(c, v);
        }
        let parent_n = |h: H| -> Quat {
            let mut p = h.parent();
            while let Some(x) = p {
                if let Some(q) = n.get(&x) {
                    return *q;
                }
                p = x.parent();
            }
            Quat::IDENTITY
        };
        let mut hp = HumanPose { name: "From photo".into(), ..Default::default() };
        for (h, q) in &n {
            if m.humanoid.get(*h).is_some() {
                hp.rotations.insert(*h, (parent_n(*h).inverse() * *q).normalize());
            }
        }
        // Fingers: from the detected hands, else keep the current ones.
        let cur = extract_human_pose(&m.skeleton, &m.humanoid, &fig.pose.locals);
        let mut found = 0;
        for (i, side) in [poser_core::humanoid::Side::Left, poser_core::humanoid::Side::Right].into_iter().enumerate() {
            match hands[i] {
                Some(hs) => {
                    found += 1;
                    let shape = crate::presets::hand_shape("photo", hs.curl, hs.spread, hs.opposition);
                    let shape = if side == poser_core::humanoid::Side::Left { shape } else { shape.mirrored() };
                    for (b, q) in shape.rotations {
                        if b.is_finger() && b.side() == side {
                            hp.rotations.insert(b, q);
                        }
                    }
                }
                None => {
                    for (b, q) in cur.rotations.iter() {
                        if b.is_finger() && b.side() == side {
                            hp.rotations.insert(*b, *q);
                        }
                    }
                }
            }
        }
        self.push_undo("Pose from photo");
        self.apply_library_pose(idx, &hp);
        // Show the photo behind the figure, seen from the front.
        if let Ok(img) = image::open(&job.path) {
            let img = img.to_rgba8();
            let (w, h) = img.dimensions();
            let ci = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &img);
            let tex = ctx.load_texture("reference", ci, egui::TextureOptions::LINEAR);
            self.ref_image = Some(crate::app::RefImage { tex, size: egui::vec2(w as f32, h as f32), opacity: 0.45, in_front: false, scale: 1.0, offset: egui::Vec2::ZERO, visible: true });
        }
        let now = self.last_time;
        self.set_view(poser_core::camera::ViewAxis::Front, now);
        self.status = format!(
            "Pose estimated from {} ({} of 17 body points confident, {found} of 2 hands). Depth is a best guess: refine with the rings.",
            job.path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(),
            good
        );
        self.needs_render = true;
    }
}

//! Body collision, floor contact, props + attachments, and turntable export.
use crate::app::{IkKind, PoserApp, Selection};
use poser_core::collision::BodyCollider;
use poser_core::glam::{Mat4, Quat, Vec3};
use poser_core::humanoid::HumanBone as H;
use poser_core::ik;
use poser_core::props::PropKind;
use poser_core::scene::Attachment;
use poser_core::skeleton::mat_rotation;
use poser_core::transform::Transform;
use std::path::{Path, PathBuf};
use std::sync::Arc;

type Baseline = Vec<(usize, Option<usize>, f32)>;

fn pos(m: &Mat4) -> Vec3 {
    m.w_axis.truncate()
}

impl PoserApp {
    // ---- Collision ------------------------------------------------------------------------

    fn collisions_on(&self) -> bool {
        self.guided() && self.opts.collisions
    }

    /// Body collider for a figure (fitted once per model, rebuilt when the model changes).
    pub fn collider(&mut self, f: usize) -> Option<Arc<BodyCollider>> {
        let fig = self.scene.figures.get(f)?;
        let key = Arc::as_ptr(&fig.model) as usize;
        if let Some((k, c)) = self.colliders.get(&fig.id) {
            if *k == key {
                return c.clone();
            }
        }
        let c = BodyCollider::build(&fig.model).map(Arc::new);
        self.colliders.insert(fig.id, (key, c.clone()));
        c
    }

    /// The floor (world y = 0) in the figure's model space.
    fn floor_plane(&self, f: usize) -> Option<(Vec3, Vec3)> {
        if !self.opts.floor {
            return None;
        }
        let fig = &self.scene.figures[f];
        if fig.attachment.is_some() {
            return None;
        }
        let inv = fig.transform.to_mat4().inverse();
        Some((inv.transform_point3(Vec3::ZERO), inv.transform_vector3(Vec3::Y).normalize_or(Vec3::Y)))
    }

    fn subtree(&self, f: usize, root: usize) -> Vec<bool> {
        let sk = &self.scene.figures[f].model.skeleton;
        (0..sk.len()).map(|b| b == root || sk.is_ancestor(root, b)).collect()
    }

    fn world_of(&self, f: usize) -> Vec<Mat4> {
        self.scene.figures[f].evaluate().world
    }

    /// Penetrations of the limb below `root` (current pose).
    fn contacts_below(&mut self, f: usize, root: usize) -> Vec<poser_core::collision::Contact> {
        let Some(col) = self.collider(f) else { return vec![] };
        let moving = self.subtree(f, root);
        let world = self.world_of(f);
        col.contacts(&world, self.floor_plane(f), &|b| moving.get(b).copied().unwrap_or(false))
    }

    /// Debug: penetration of the limb below `root` regardless of settings (floor included).
    pub fn raw_penetration(&mut self, f: usize, root: usize) -> (f32, usize) {
        let Some(col) = self.collider(f) else { return (0.0, 0) };
        let moving = self.subtree(f, root);
        let world = self.world_of(f);
        let inv = self.scene.figures[f].transform.to_mat4().inverse();
        let floor = Some((inv.transform_point3(Vec3::ZERO), inv.transform_vector3(Vec3::Y).normalize()));
        let cs = col.contacts(&world, floor, &|b| moving.get(b).copied().unwrap_or(false));
        (cs.iter().map(|c| c.depth).sum(), cs.len())
    }

    /// Existing contacts (so poses that already touch are not forced apart).
    pub fn contact_baseline(&mut self, f: usize, root: usize) -> Baseline {
        if !self.collisions_on() {
            return vec![];
        }
        self.contacts_below(f, root).iter().map(|c| (c.a, c.b, c.depth)).collect()
    }

    /// Total penetration depth of the limb below `root`, beyond the baseline.
    pub fn penetration(&mut self, f: usize, root: usize) -> f32 {
        if !self.collisions_on() {
            return 0.0;
        }
        self.contacts_below(f, root).iter().map(|c| c.depth).sum()
    }

    /// Deepest penetration of the limb below `root` beyond the baseline (0 when collision is off).
    pub fn collision_excess(&mut self, f: usize, root: usize, base: &Baseline) -> f32 {
        if !self.collisions_on() {
            return 0.0;
        }
        self.contacts_below(f, root)
            .iter()
            .map(|c| c.depth - base.iter().find(|(a, b, _)| *a == c.a && *b == c.b).map(|x| x.2).unwrap_or(0.0).max(0.0))
            .fold(0.0, f32::max)
    }

    /// Push the limb below `root` out of the body / floor by rotating `root` (Guided mode).
    pub fn resolve_collisions(&mut self, f: usize, root: usize, base: &Baseline) {
        if !self.collisions_on() {
            return;
        }
        for _ in 0..10 {
            let cs = self.contacts_below(f, root);
            let excess = |c: &poser_core::collision::Contact| {
                let b = base.iter().find(|(a, b, _)| *a == c.a && *b == c.b).map(|x| x.2).unwrap_or(0.0);
                c.depth - b.max(0.0)
            };
            let Some(c) = cs.iter().max_by(|x, y| excess(x).total_cmp(&excess(y))) else { return };
            let e = excess(c);
            if e < 1e-4 {
                return;
            }
            let world = self.world_of(f);
            let lever = c.point - pos(&world[root]);
            let axis = lever.cross(c.normal);
            if axis.length_squared() < 1e-10 || lever.length_squared() < 1e-8 {
                return;
            }
            let angle = (e / lever.length() * 1.1).min(0.3);
            let fig = &mut self.scene.figures[f];
            let model = fig.model.clone();
            ik::rotate_bone_world(&model.skeleton, &world, &mut fig.pose.locals, root, Quat::from_axis_angle(axis.normalize(), angle));
            self.clamp_bone(f, root);
        }
    }

    /// The joint that is rotated to push a dragged limb out of the body.
    pub fn collision_root(kind: &IkKind) -> Option<usize> {
        match kind {
            IkKind::Limb { root, .. } | IkKind::Swing { root, .. } => Some(*root),
            IkKind::Aim { bone } => Some(*bone),
            _ => None,
        }
    }

    /// Guided arm/leg IK that respects anatomy: the elbow/knee stays on the side it pointed to at
    /// the start of the drag (or `hint`), the upper bone gets exactly the twist that lines the
    /// hinge up with the reach plane, and the lower bone bends on its hinge only. Returns false
    /// when the model lacks the data (caller falls back to plain two-bone IK).
    pub fn guided_two_bone(&mut self, f: usize, root: usize, mid: usize, end: usize, target: Vec3, hint: Vec3) -> bool {
        let fig = &self.scene.figures[f];
        let model = fig.model.clone();
        let (sk, hm) = (&model.skeleton, &model.humanoid);
        if hm.tpose_world.len() != sk.len() {
            return false;
        }
        let (Some(hr), Some(hmid)) = (hm.human_of(root), hm.human_of(mid)) else { return false };
        let (Some(u0), Some(lim)) = (hr.tpose_direction(), hmid.default_limit()) else { return false };
        // Hinge axis = the widest non-twist axis of the lower joint; bend sign = its larger side.
        let ranges = [lim.x, lim.y, lim.z];
        let Some(hi) = (0..3).filter(|i| *i != lim.twist as usize).max_by(|a, b| {
            let w = |r: (f32, f32)| r.1 - r.0;
            w(ranges[*a]).total_cmp(&w(ranges[*b]))
        }) else {
            return false;
        };
        let h0 = [Vec3::X, Vec3::Y, Vec3::Z][hi];
        let sign = if ranges[hi].1.abs() > ranges[hi].0.abs() { 1.0 } else { -1.0 };
        let bent = Quat::from_axis_angle(h0, sign * 1.0) * u0;
        let s0 = (bent - u0 * bent.dot(u0)).normalize_or_zero();
        if s0 == Vec3::ZERO {
            return false;
        }
        let shapes = fig.shapes_opt().map(|s| s.to_vec());
        let w = sk.evaluate(&fig.pose.locals, shapes.as_deref()).world;
        let (a, b, c) = (pos(&w[root]), pos(&w[mid]), pos(&w[end]));
        let (lab, lbc) = ((b - a).length(), (c - b).length());
        if lab < 1e-6 || lbc < 1e-6 {
            return false;
        }
        let to = target - a;
        let eps = 1e-3 * (lab + lbc);
        let dist = to.length().clamp((lab - lbc).abs() + eps, lab + lbc - eps);
        let dir = to.normalize_or(Vec3::NEG_Y);
        let perp = |v: Vec3| v - dir * v.dot(dir);
        // Elbow / knee direction from anatomy: opposite the side the hinge bends towards, in the
        // upper bone's current orientation (so it also keeps its twist from the drag start).
        let nw_root = mat_rotation(&w[root]) * hm.tpose_world[root].inverse();
        let mut pole = perp(-(nw_root * s0));
        if pole.length() < 0.05 {
            pole = perp(hint);
        }
        let pole0 = pole.normalize_or(dir.any_orthonormal_vector());
        let cos_a = ((lab * lab + dist * dist - lbc * lbc) / (2.0 * lab * dist)).clamp(-1.0, 1.0);
        let sin_a = (1.0 - cos_a * cos_a).sqrt();
        let t = a + dir * dist;
        let frame = |u: Vec3, s: Vec3| poser_core::glam::Mat3::from_cols(u, s, u.cross(s));
        let root_lim = self.limit_for(f, root);
        let parent_nw = sk.bones[root].parent.map(|p| mat_rotation(&w[p]) * hm.tpose_world[p].inverse()).unwrap_or(Quat::IDENTITY);
        // Elbow on a circle around the reach line: try the natural direction first, then swing
        // it round until the shoulder/hip can actually do it.
        let solve = |pole: Vec3| {
            let e = a + dir * (lab * cos_a) + pole * (lab * sin_a);
            let d1 = (e - a).normalize();
            let f1 = (t - e).normalize();
            let mut s1 = f1 - d1 * f1.dot(d1);
            if s1.length() < 1e-4 {
                s1 = -(pole - d1 * pole.dot(d1));
            }
            let s1 = s1.normalize_or(d1.any_orthonormal_vector());
            let q = Quat::from_mat3(&(frame(d1, s1) * frame(u0, s0).transpose())).normalize();
            let theta = d1.dot(f1).clamp(-1.0, 1.0).acos();
            let q_mid = (q * Quat::from_axis_angle(h0, sign * theta)).normalize();
            let viol = root_lim.map(|l| {
                let n = (parent_nw.inverse() * q).normalize();
                n.angle_between(l.clamp(n))
            });
            (q, q_mid, viol.unwrap_or(0.0))
        };
        let mut best = solve(pole0);
        if best.2 > 0.02 {
            for k in 1..=12 {
                for sgn in [1.0f32, -1.0] {
                    let pole = Quat::from_axis_angle(dir, sgn * k as f32 * 15f32.to_radians()) * pole0;
                    let cand = solve(pole);
                    if cand.2 + 1e-3 < best.2 {
                        best = cand;
                    }
                }
                if best.2 <= 0.02 {
                    break;
                }
            }
        }
        let (q, q_mid, _) = best;
        let fig = &mut self.scene.figures[f];
        let set = |locals: &mut Vec<poser_core::Transform>, bone: usize, nw: Quat| {
            let wcur = sk.evaluate(locals, shapes.as_deref()).world;
            let pr = sk.bones[bone].parent.map(|p| mat_rotation(&wcur[p])).unwrap_or(mat_rotation(&sk.root_transform));
            locals[bone].rotation = (pr.inverse() * nw * hm.tpose_world[bone]).normalize();
        };
        set(&mut fig.pose.locals, root, q);
        set(&mut fig.pose.locals, mid, q_mid);
        true
    }

    // ---- Floor ----------------------------------------------------------------------------

    /// Move a figure up or down so its lowest point rests on the floor.
    pub fn drop_to_floor(&mut self, f: usize) {
        let Some(fig) = self.scene.figures.get(f) else { return };
        if fig.attachment.is_some() {
            return;
        }
        let Some(c) = self.cache.figs.get(&fig.id) else { return };
        let mut low = f32::MAX;
        for m in &c.meshes {
            for p in &m.positions {
                low = low.min(c.world.transform_point3(*p).y);
            }
        }
        if low == f32::MAX || low.abs() < 1e-5 {
            return;
        }
        self.push_undo("Drop to floor");
        self.scene.figures[f].transform.translation.y -= low;
        self.needs_render = true;
        self.status = "Dropped to the floor".into();
    }

    // ---- Props ----------------------------------------------------------------------------

    pub fn add_prop(&mut self, kind: PropKind) -> usize {
        self.push_undo("Add prop");
        let holder = self.sel.figure.filter(|i| self.scene.figures.get(*i).is_some_and(|f| f.model.humanoid.is_valid()));
        let model = Arc::new(poser_core::props::build(kind));
        let idx = self.scene.add_model(model);
        let id = self.scene.figures[idx].id;
        self.scene.figures[idx].name = kind.name().to_string();
        self.props.insert(id, kind);
        match holder {
            Some(h) if kind.handheld() => {
                self.put_in_hand(idx, h, false);
            }
            _ => {
                // Beside / in front of the selected figure.
                let base = holder.map(|h| self.scene.figures[h].transform.translation).unwrap_or(Vec3::ZERO);
                let off = match kind {
                    PropKind::Chair | PropKind::Stool | PropKind::Bench | PropKind::Block => Vec3::new(0.0, 0.0, -0.45),
                    PropKind::Wall => Vec3::new(0.0, 0.0, -0.8),
                    PropKind::Table | PropKind::Stairs => Vec3::new(0.0, 0.0, 0.8),
                    _ => Vec3::new(0.6, 0.0, 0.3),
                };
                self.scene.figures[idx].transform.translation = Vec3::new(base.x, 0.0, base.z) + off;
            }
        }
        self.sel = Selection { figure: Some(idx), bone: None };
        self.needs_render = true;
        self.status = format!("Added {}. Move it with W / rotate with E; attach it to a hand in the Pose tab.", kind.name());
        idx
    }

    /// Attach `idx` to a figure's bone (or detach with None), keeping it where it is.
    pub fn set_attachment(&mut self, idx: usize, to: Option<(usize, usize)>) {
        self.push_undo("Attach");
        let id = self.scene.figures[idx].id;
        self.attach_sync.remove(&id);
        self.scene.figures[idx].attachment = to.map(|(p, bone)| Attachment { parent: self.scene.figures[p].id, bone, offset: Transform::IDENTITY });
        self.sync_attachments();
        self.needs_render = true;
    }

    /// Attach a hand-held prop to a figure's hand, placed across the palm, and curl the fingers.
    pub fn put_in_hand(&mut self, idx: usize, holder: usize, left: bool) {
        let hf = &self.scene.figures[holder];
        let hm = &hf.model.humanoid;
        let side = |l: H, r: H| if left { l } else { r };
        let Some(hand) = hm.get(side(H::LeftHand, H::RightHand)) else { return };
        let w = hf.evaluate().world;
        let fw = hf.transform.to_mat4();
        let wp = |b: Option<usize>| b.map(|b| fw.transform_point3(pos(&w[b])));
        let hand_p = fw.transform_point3(pos(&w[hand]));
        let index = wp(hm.get(side(H::LeftIndexProximal, H::RightIndexProximal)));
        let little = wp(hm.get(side(H::LeftLittleProximal, H::RightLittleProximal)));
        let middle = wp(hm.get(side(H::LeftMiddleProximal, H::RightMiddleProximal)));
        let (grip_axis, center) = match (index, little, middle) {
            (Some(i), Some(l), Some(m)) => ((i - l).normalize_or(Vec3::Y), hand_p.lerp(m, 0.75)),
            _ => {
                let dir = fw.transform_vector3(mat_rotation(&w[hand]) * hf.model.skeleton.bones[hand].axis).normalize_or(Vec3::X);
                (dir.any_orthonormal_vector(), hand_p + dir * 0.06)
            }
        };
        let finger_dir = middle.map(|m| (m - hand_p).normalize_or(Vec3::Z)).unwrap_or(Vec3::Z);
        // Prop +Y along the grip (thumb side), +Z along the fingers.
        let r1 = Quat::from_rotation_arc(Vec3::Y, grip_axis);
        let z1 = r1 * Vec3::Z;
        let fz = (finger_dir - grip_axis * finger_dir.dot(grip_axis)).normalize_or(z1);
        let r2 = Quat::from_axis_angle(grip_axis, z1.cross(fz).dot(grip_axis).atan2(z1.dot(fz)));
        let t = &mut self.scene.figures[idx].transform;
        t.translation = center;
        t.rotation = (r2 * r1).normalize();
        let holder_id = self.scene.figures[holder].id;
        let id = self.scene.figures[idx].id;
        self.attach_sync.remove(&id);
        self.scene.figures[idx].attachment = Some(Attachment { parent: holder_id, bone: hand, offset: Transform::IDENTITY });
        self.sync_attachments();
        // Curl the fingers round it.
        if let Some(mut grip) = crate::presets::hand_presets().into_iter().find(|p| p.name == "Grip") {
            if !left {
                grip = grip.mirrored();
            }
            self.apply_pose_to(holder, &grip, false);
        }
    }

    /// Keep attached objects following their bone. Editing an attached object's transform
    /// (gizmo, placement fields) updates its offset from the bone instead.
    pub fn sync_attachments(&mut self) {
        if !self.scene.figures.iter().any(|f| f.attachment.is_some()) {
            return;
        }
        let evals: Vec<Option<poser_core::skeleton::PoseMatrices>> = self.scene.figures.iter().map(|f| Some(f.evaluate())).collect();
        for i in 0..self.scene.figures.len() {
            let Some(a) = self.scene.figures[i].attachment else { continue };
            let Some(pi) = self.scene.index_of(a.parent).filter(|p| *p != i) else {
                self.scene.figures[i].attachment = None;
                continue;
            };
            let Some(bm) = evals[pi].as_ref().and_then(|e| e.world.get(a.bone)).copied() else {
                self.scene.figures[i].attachment = None;
                continue;
            };
            let parent = self.scene.figure_world(pi, &evals) * bm;
            let id = self.scene.figures[i].id;
            let t = self.scene.figures[i].transform;
            let tr = Mat4::from_rotation_translation(t.rotation, t.translation);
            let mut a2 = a;
            if self.attach_sync.get(&id) != Some(&t) {
                a2.offset = Transform::from_mat4(&(parent.inverse() * tr));
            }
            let (_, r, p) = (parent * a2.offset.to_mat4()).to_scale_rotation_translation();
            let fig = &mut self.scene.figures[i];
            fig.attachment = Some(a2);
            let new_t = Transform { translation: p, rotation: r.normalize(), scale: t.scale };
            if new_t != t {
                fig.transform = new_t;
                self.needs_render = true;
            }
            self.attach_sync.insert(id, new_t);
        }
    }

    // ---- Turntable ------------------------------------------------------------------------

    /// Render the scene from `n` angles around it into one sheet (and optionally one file each).
    pub fn export_turntable(&mut self, path: PathBuf, n: u32, separate: bool) {
        let n = n.clamp(2, 36);
        let cols = if n <= 4 { n } else if n <= 8 { 4 } else { 6 };
        let rows = n.div_ceil(cols);
        let (ew, eh) = (self.export.width.clamp(16, 8192), self.export.height.clamp(16, 8192));
        let scale = (8192.0 / (ew * cols) as f32).min(8192.0 / (eh * rows) as f32).min(1.0);
        let (tw, th) = (((ew as f32 * scale) as u32).max(16), ((eh as f32 * scale) as u32).max(16));
        let (center, _) = self.cache.scene_bounds(&self.scene);
        let base = self.cam;
        let mut sheet = image::RgbaImage::new(tw * cols, th * rows);
        for k in 0..n {
            let mut cam = base;
            cam.pivot = center;
            cam.rotation = (Quat::from_rotation_y(k as f32 / n as f32 * std::f32::consts::TAU) * base.rotation).normalize();
            if separate {
                let Some(img) = self.render_camera(&cam, ew, eh) else { break };
                let _ = img.save(numbered(&path, k + 1));
            }
            let Some(img) = self.render_camera(&cam, tw, th) else {
                self.status = "Export failed: could not read the image back from the GPU".into();
                return;
            };
            image::imageops::overlay(&mut sheet, &img, ((k % cols) * tw) as i64, ((k / cols) * th) as i64);
        }
        self.status = match sheet.save(&path) {
            Ok(()) => format!("Saved {n}-angle turntable {}", path.display()),
            Err(e) => format!("Save failed: {e}"),
        };
        self.needs_render = true;
        self.last_rendered = None;
    }

    fn render_camera(&mut self, cam: &poser_core::camera::Camera, w: u32, h: u32) -> Option<image::RgbaImage> {
        let (center, radius) = self.cache.scene_bounds(&self.scene);
        let draws = crate::viewport::build_draws(&self.scene, &self.cache, true);
        let r = self.renderer.as_mut()?;
        r.use_slot(2);
        let desc = poser_render::FrameDesc {
            camera: cam,
            settings: &self.scene.render,
            lights: &self.scene.lights,
            width: w,
            height: h,
            transparent: self.export.transparent,
            bounds_center: center,
            bounds_radius: radius,
        };
        r.render(&desc, &draws);
        let (w, h, px) = r.read_output()?;
        image::RgbaImage::from_raw(w, h, px)
    }
}

fn numbered(path: &Path, k: u32) -> PathBuf {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("turntable");
    path.with_file_name(format!("{stem}_{k:02}.png"))
}

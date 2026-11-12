//! Application state and high-level commands.
use crate::deform::DeformCache;
use crate::gizmo::Ring;
use crate::presets;
use poser_core::camera::{Camera, Projection, ViewAxis};
use poser_core::glam::{Quat, Vec3};
use poser_core::humanoid::{HumanBone, JointLimit};
use poser_core::mannequin::{self, MannequinParams};
use poser_core::model::Model;
use poser_core::pose::{HumanPose, Pose, apply_human_pose, extract_human_pose};
use poser_core::scene::{AnimState, RenderSettings, Scene, UndoStack};
use poser_core::skeleton::mat_rotation;
use poser_render::Renderer;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tool {
    Pose,
    Move,
    Rotate,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PropsTab {
    Pose,
    Hands,
    Body,
    Library,
    Camera,
    Render,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Selection {
    pub figure: Option<usize>,
    pub bone: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
pub struct Pin {
    /// Model-space position and rotation of the pinned bone.
    pub pos: Vec3,
    pub rot: Quat,
}

#[derive(Clone, Debug)]
pub enum IkKind {
    Limb { root: usize, mid: usize, end: usize, hint: Vec3 },
    Swing { root: usize, end: usize },
    Hips,
    Ccd { chain: Vec<usize>, effector: usize },
    Aim { bone: usize },
}

#[derive(Clone, Debug)]
pub enum Drag {
    Ring { ring: Ring, axis_world: Vec3, last_angle: f32, tangent: Option<egui::Vec2>, last_pos: egui::Pos2 },
    Ik { figure: usize, kind: IkKind, effector: usize, grab_offset: Vec3, plane_point: Vec3 },
    MoveFigure { figure: usize, grab_offset: Vec3, plane_point: Vec3 },
    RotateFigure { figure: usize },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModalKind {
    Grab,
    Rotate,
    Scale,
}

#[derive(Clone, Debug)]
pub struct Modal {
    pub kind: ModalKind,
    pub figure: usize,
    pub bone: Option<usize>,
    pub start_pose: Pose,
    pub start_shapes: Vec<poser_core::skeleton::BoneShape>,
    pub start_transform: poser_core::Transform,
    pub start_mouse: egui::Pos2,
    pub accum_angle: f32,
    pub last_angle: f32,
    /// (axis index 0..3, local)
    pub constraint: Option<(usize, bool)>,
    pub typed: String,
    pub trackball: bool,
}

pub struct Options {
    pub local_axes: bool,
    pub limits: bool,
    pub emulate_3button: bool,
    pub touchpad_orbit: bool,
    pub show_rings: bool,
    pub keep_feet_planted: bool,
}

/// Animation borrowed from another model and replayed through the humanoid mapping.
pub struct Retarget {
    pub source: Arc<Model>,
    pub clip: usize,
    pub time: f32,
    pub playing: bool,
}

pub struct ExportSettings {
    pub width: u32,
    pub height: u32,
    pub transparent: bool,
    pub open: bool,
}

pub struct PoserApp {
    pub scene: Scene,
    pub undo: UndoStack,
    pub sel: Selection,
    pub hover: Option<(usize, Option<usize>)>,
    pub hover_ring: Option<Ring>,
    pub cam: Camera,
    pub cam_anim: Option<(Camera, Camera, f64)>,
    pub saved_views: Vec<(String, Camera)>,
    pub tool: Tool,
    pub drag: Option<Drag>,
    pub modal: Option<Modal>,
    pub cache: DeformCache,
    pub renderer: Option<Renderer>,
    pub tex: Option<egui::TextureId>,
    pub needs_render: bool,
    pub last_rendered: Option<(Camera, RenderSettings, (u32, u32), Vec<poser_core::scene::Light>)>,
    pub pending_undo_done: bool,
    pub panel_editing: bool,
    pub drag_prev_x: Option<f32>,
    pub hand_sliders: HashMap<(u64, bool), [f32; 7]>,
    pub retargets: HashMap<u64, Retarget>,
    pub retarget_loading: Vec<(u64, Receiver<(PathBuf, anyhow::Result<Model>)>)>,
    pub bone_filter: String,
    pub morph_filter: String,
    pub new_pose_name: String,
    pub opts: Options,
    pub clipboard: Option<HumanPose>,
    pub importing: Vec<Receiver<(PathBuf, anyhow::Result<Model>)>>,
    pub status: String,
    pub props_tab: PropsTab,
    pub show_left: bool,
    pub show_right: bool,
    pub show_timeline: bool,
    pub pins: HashMap<(u64, usize), Pin>,
    pub mannequins: HashMap<u64, MannequinParams>,
    pub export: ExportSettings,
    pub help_open: bool,
    pub frame_ms: f32,
    pub body_presets: Vec<HumanPose>,
    pub hand_presets: Vec<HumanPose>,
    pub user_poses: Vec<HumanPose>,
    pub last_time: f64,
}

impl PoserApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        crate::theme::apply(&cc.egui_ctx);
        let renderer = cc.wgpu_render_state.as_ref().map(|rs| Renderer::new(&rs.device, &rs.queue));
        let mut app = Self {
            scene: Scene::default(),
            undo: UndoStack::new(200),
            sel: Selection::default(),
            hover: None,
            hover_ring: None,
            cam: Camera::default(),
            cam_anim: None,
            saved_views: vec![],
            tool: Tool::Pose,
            drag: None,
            modal: None,
            cache: DeformCache::default(),
            renderer,
            tex: None,
            needs_render: true,
            last_rendered: None,
            pending_undo_done: false,
            panel_editing: false,
            drag_prev_x: None,
            hand_sliders: HashMap::new(),
            retargets: HashMap::new(),
            retarget_loading: vec![],
            bone_filter: String::new(),
            morph_filter: String::new(),
            new_pose_name: "My pose".into(),
            opts: Options {
                local_axes: true,
                limits: true,
                emulate_3button: true,
                touchpad_orbit: true,
                show_rings: true,
                keep_feet_planted: true,
            },
            clipboard: None,
            importing: vec![],
            status: "Welcome! MMB orbit, Shift+MMB pan, wheel zoom (Alt+LMB orbit on touchpads). Click a body part to pose it.".into(),
            props_tab: PropsTab::Pose,
            show_left: true,
            show_right: true,
            show_timeline: true,
            pins: HashMap::new(),
            mannequins: HashMap::new(),
            export: ExportSettings { width: 1920, height: 1080, transparent: true, open: false },
            help_open: false,
            frame_ms: 0.0,
            body_presets: presets::body_presets(),
            hand_presets: presets::hand_presets(),
            user_poses: load_user_poses(),
            last_time: 0.0,
        };
        let idx = app.add_mannequin(MannequinParams::default(), "Mannequin");
        app.sel = Selection { figure: Some(idx), bone: None };
        app.frame_all(false);
        for a in std::env::args().skip(1) {
            let p = PathBuf::from(a);
            if p.is_file() {
                app.import_file(p);
            }
        }
        app
    }

    // ---- Scene helpers -------------------------------------------------------------------

    pub fn push_undo(&mut self, label: &str) {
        self.undo.push(label, &self.scene);
    }

    pub fn undo(&mut self) {
        if let Some(l) = self.undo.undo(&mut self.scene) {
            self.status = format!("Undo: {l}");
            self.validate_selection();
        }
    }

    pub fn redo(&mut self) {
        if let Some(l) = self.undo.redo(&mut self.scene) {
            self.status = format!("Redo: {l}");
            self.validate_selection();
        }
    }

    pub fn validate_selection(&mut self) {
        if self.sel.figure.is_some_and(|f| f >= self.scene.figures.len()) {
            self.sel = Selection::default();
        }
        if let (Some(f), Some(b)) = (self.sel.figure, self.sel.bone) {
            if b >= self.scene.figures[f].model.skeleton.len() {
                self.sel.bone = None;
            }
        }
        self.hover = None;
        self.needs_render = true;
    }

    pub fn add_mannequin(&mut self, params: MannequinParams, name: &str) -> usize {
        let model = Arc::new(mannequin::build(&params));
        let idx = self.scene.add_model(model);
        let fig = &mut self.scene.figures[idx];
        fig.name = name.to_string();
        let id = fig.id;
        // Place beside existing figures.
        let n = self.scene.figures.len();
        if n > 1 {
            let x = self.scene.figures.iter().take(n - 1).map(|f| f.transform.translation.x).fold(f32::MIN, f32::max);
            self.scene.figures[idx].transform.translation.x = x + 0.8;
        }
        self.mannequins.insert(id, params);
        let relaxed = self.body_presets.iter().find(|p| p.name == "Relaxed stand").cloned();
        if let Some(p) = relaxed {
            self.apply_pose_to(idx, &p, true);
        }
        idx
    }

    /// Rebuild a procedural mannequin with new proportions, keeping its pose.
    pub fn rebuild_mannequin(&mut self, idx: usize, params: MannequinParams) {
        let fig = &self.scene.figures[idx];
        let pose = extract_human_pose(&fig.model.skeleton, &fig.model.humanoid, &fig.pose.locals);
        let model = Arc::new(mannequin::build(&params));
        let id = fig.id;
        let fig = &mut self.scene.figures[idx];
        fig.pose = Pose::tpose(&model.skeleton, &model.humanoid);
        fig.shapes = vec![Default::default(); model.skeleton.len()];
        fig.model = model;
        fig.touch();
        self.mannequins.insert(id, params);
        self.apply_pose_to(idx, &pose, true);
        self.refresh_pins(idx);
    }

    pub fn import_dialog(&mut self) {
        let files = rfd::FileDialog::new()
            .add_filter("3D models", poser_io::IMPORT_EXTENSIONS)
            .add_filter("All files", &["*"])
            .set_title("Import model")
            .pick_files();
        for f in files.unwrap_or_default() {
            self.import_file(f);
        }
    }

    pub fn import_file(&mut self, path: PathBuf) {
        let (tx, rx) = channel();
        self.status = format!("Importing {} ...", path.display());
        std::thread::spawn(move || {
            let r = poser_io::load_model(&path);
            let _ = tx.send((path, r));
        });
        self.importing.push(rx);
    }

    pub fn poll_imports(&mut self, ctx: &egui::Context) {
        if self.importing.is_empty() {
            return;
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(50));
        let mut done = vec![];
        let mut results = vec![];
        for (i, rx) in self.importing.iter().enumerate() {
            if let Ok(r) = rx.try_recv() {
                done.push(i);
                results.push(r);
            }
        }
        for (path, res) in results {
                match res {
                    Ok(model) => {
                        self.push_undo("Import");
                        let info = format!(
                            "Imported {}: {} verts, {} bones{}{}{}",
                            path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(),
                            model.vertex_count(),
                            model.skeleton.len(),
                            if model.humanoid.is_valid() { ", humanoid rig detected" } else if model.has_skeleton() { ", custom rig" } else { ", no rig (use Rig > Auto-rig)" },
                            if model.animations.is_empty() { String::new() } else { format!(", {} animation(s)", model.animations.len()) },
                            {
                                let (f, t, _) = model.rest_extent();
                                let h = t - f;
                                if h > 3.2 || h < 0.35 { format!(" (rescaled from {h:.1} units)") } else { String::new() }
                            },
                        );
                        // Stand it on the floor and bring absurd unit scales to human size.
                        let (floor, top, center) = model.rest_extent();
                        let h = (top - floor).max(1e-4);
                        let scale = if h > 3.2 || h < 0.35 { 1.75 / h } else { 1.0 };
                        let idx = self.scene.add_model(Arc::new(model));
                        let n = self.scene.figures.len();
                        let x = if n > 1 {
                            self.scene.figures.iter().take(n - 1).map(|f| f.transform.translation.x).fold(f32::MIN, f32::max) + 0.9
                        } else {
                            0.0
                        };
                        let t = &mut self.scene.figures[idx].transform;
                        t.scale = Vec3::splat(scale);
                        t.translation = Vec3::new(x - center.x * scale, -floor * scale, -center.z * scale);
                        if !self.scene.figures[idx].model.animations.is_empty() {
                            self.scene.figures[idx].anim = Some(AnimState { clip: 0, time: 0.0, playing: false, looping: true, speed: 1.0 });
                        }
                        self.sel = Selection { figure: Some(idx), bone: None };
                        self.status = info;
                        self.frame_all(true);
                    }
                    Err(e) => {
                        self.status = format!("Import failed for {}: {e:#}", path.display());
                    }
                }
        }
        for i in done.into_iter().rev() {
            self.importing.remove(i);
        }
        self.needs_render = true;
    }

    pub fn delete_selected(&mut self) {
        if let Some(f) = self.sel.figure {
            self.push_undo("Delete");
            let id = self.scene.figures[f].id;
            self.scene.figures.remove(f);
            self.pins.retain(|(fid, _), _| *fid != id);
            self.sel = Selection::default();
            self.needs_render = true;
        }
    }

    pub fn duplicate_selected(&mut self) {
        if let Some(f) = self.sel.figure {
            self.push_undo("Duplicate");
            let mut fig = self.scene.figures[f].clone();
            fig.id = self.scene.alloc_id();
            fig.name = format!("{} copy", fig.name);
            fig.transform.translation.x += 0.8;
            fig.touch();
            if let Some(p) = self.mannequins.get(&self.scene.figures[f].id).cloned() {
                self.mannequins.insert(fig.id, p);
            }
            self.scene.figures.push(fig);
            self.sel = Selection { figure: Some(self.scene.figures.len() - 1), bone: None };
        }
    }

    // ---- Pose helpers ---------------------------------------------------------------------

    pub fn apply_pose_to(&mut self, idx: usize, pose: &HumanPose, reset_missing: bool) {
        let fig = &mut self.scene.figures[idx];
        let m = fig.model.clone();
        if !m.humanoid.is_valid() {
            return;
        }
        let shapes = fig.shapes_opt().map(|s| s.to_vec());
        apply_human_pose(&m.skeleton, &m.humanoid, pose, &mut fig.pose.locals, shapes.as_deref(), reset_missing);
        fig.anim.iter_mut().for_each(|a| a.playing = false);
        fig.touch();
        self.refresh_pins(idx);
        self.needs_render = true;
    }

    pub fn current_human_pose(&self, idx: usize) -> Option<HumanPose> {
        let f = self.scene.figures.get(idx)?;
        if !f.model.humanoid.is_valid() {
            return None;
        }
        Some(extract_human_pose(&f.model.skeleton, &f.model.humanoid, &f.pose.locals))
    }

    /// Copy one side onto the other (true = left -> right).
    pub fn mirror_pose(&mut self, idx: usize, left_to_right: bool) {
        let Some(p) = self.current_human_pose(idx) else { return };
        self.push_undo("Mirror pose");
        let src_side = if left_to_right { poser_core::humanoid::Side::Left } else { poser_core::humanoid::Side::Right };
        let m = p.mirrored().filtered(|b| b.side() != src_side && b.side() != poser_core::humanoid::Side::Center);
        self.apply_pose_to(idx, &m, false);
    }

    pub fn flip_pose(&mut self, idx: usize) {
        let Some(p) = self.current_human_pose(idx) else { return };
        self.push_undo("Flip pose");
        self.apply_pose_to(idx, &p.mirrored(), false);
    }

    pub fn reset_pose(&mut self, idx: usize, tpose: bool) {
        self.push_undo("Reset pose");
        let fig = &mut self.scene.figures[idx];
        fig.pose = if tpose && fig.model.humanoid.is_valid() {
            Pose::tpose(&fig.model.skeleton, &fig.model.humanoid)
        } else {
            Pose::rest(&fig.model.skeleton)
        };
        fig.anim.iter_mut().for_each(|a| a.playing = false);
        fig.touch();
        self.refresh_pins(idx);
    }

    pub fn copy_pose(&mut self) {
        if let Some(p) = self.sel.figure.and_then(|f| self.current_human_pose(f)) {
            self.clipboard = Some(p);
            self.status = "Pose copied".into();
        }
    }

    pub fn paste_pose(&mut self, flipped: bool) {
        let (Some(f), Some(p)) = (self.sel.figure, self.clipboard.clone()) else { return };
        self.push_undo("Paste pose");
        let p = if flipped { p.mirrored() } else { p };
        self.apply_pose_to(f, &p, true);
        self.status = "Pose pasted".into();
    }

    pub fn limit_for(&self, idx: usize, bone: usize) -> Option<JointLimit> {
        let f = &self.scene.figures[idx];
        f.model.humanoid.human_of(bone)?.default_limit()
    }

    /// Clamp a bone to its joint limits (normalised frame).
    pub fn clamp_bone(&mut self, idx: usize, bone: usize) {
        if !self.opts.limits {
            return;
        }
        let Some(lim) = self.limit_for(idx, bone) else { return };
        let fig = &mut self.scene.figures[idx];
        let m = &fig.model;
        if m.humanoid.tpose_world.len() != m.skeleton.len() {
            return;
        }
        let n = m.humanoid.to_normalized(&m.skeleton, bone, fig.pose.locals[bone].rotation);
        let c = lim.clamp(n);
        fig.pose.locals[bone].rotation = m.humanoid.to_local(&m.skeleton, bone, c);
    }

    pub fn toggle_pin(&mut self, idx: usize, bone: usize) {
        let id = self.scene.figures[idx].id;
        if self.pins.remove(&(id, bone)).is_none() {
            let w = self.scene.figures[idx].evaluate().world;
            self.pins.insert((id, bone), Pin { pos: w[bone].w_axis.truncate(), rot: mat_rotation(&w[bone]) });
            self.status = "Pinned (stays in place while you pose other parts)".into();
        } else {
            self.status = "Unpinned".into();
        }
        self.needs_render = true;
    }

    /// Re-record pin positions after a pose was replaced wholesale.
    pub fn refresh_pins(&mut self, idx: usize) {
        let id = self.scene.figures[idx].id;
        if !self.pins.keys().any(|(f, _)| *f == id) {
            return;
        }
        let w = self.scene.figures[idx].evaluate().world;
        for ((f, b), p) in self.pins.iter_mut() {
            if *f == id && *b < w.len() {
                p.pos = w[*b].w_axis.truncate();
                p.rot = mat_rotation(&w[*b]);
            }
        }
    }

    /// Limb chain (root, mid, end, bend hint) ending at a humanoid hand/foot.
    pub fn limb_chain(&self, idx: usize, end: usize) -> Option<(usize, usize, usize, Vec3)> {
        let m = &self.scene.figures[idx].model;
        let h = m.humanoid.human_of(end)?;
        use HumanBone as H;
        let (r, mid, hint) = match h {
            H::LeftHand => (H::LeftUpperArm, H::LeftLowerArm, Vec3::new(0.0, 0.0, -1.0)),
            H::RightHand => (H::RightUpperArm, H::RightLowerArm, Vec3::new(0.0, 0.0, -1.0)),
            H::LeftFoot => (H::LeftUpperLeg, H::LeftLowerLeg, Vec3::new(0.0, 0.0, 1.0)),
            H::RightFoot => (H::RightUpperLeg, H::RightLowerLeg, Vec3::new(0.0, 0.0, 1.0)),
            _ => return None,
        };
        Some((m.humanoid.get(r)?, m.humanoid.get(mid)?, end, hint))
    }

    /// Keep pinned hands/feet in place (two-bone IK + restore world rotation).
    pub fn solve_pins(&mut self, idx: usize, skip_bone: Option<usize>) {
        let id = self.scene.figures[idx].id;
        let pins: Vec<(usize, Pin)> = self.pins.iter().filter(|((f, _), _)| *f == id).map(|((_, b), p)| (*b, *p)).collect();
        for (bone, pin) in pins {
            if Some(bone) == skip_bone {
                continue;
            }
            let Some((r, m, e, hint)) = self.limb_chain(idx, bone) else { continue };
            let fig = &mut self.scene.figures[idx];
            let model = fig.model.clone();
            let shapes = fig.shapes_opt().map(|s| s.to_vec());
            poser_core::ik::two_bone_ik(&model.skeleton, shapes.as_deref(), &mut fig.pose.locals, r, m, e, pin.pos, hint);
            let w = model.skeleton.evaluate(&fig.pose.locals, shapes.as_deref()).world;
            let pr = model.skeleton.bones[e].parent.map(|p| mat_rotation(&w[p])).unwrap_or(Quat::IDENTITY);
            fig.pose.locals[e].rotation = (pr.inverse() * pin.rot).normalize();
        }
    }

    // ---- Camera ---------------------------------------------------------------------------

    pub fn animate_camera_to(&mut self, target: Camera, now: f64) {
        self.cam_anim = Some((self.cam, target, now));
    }

    pub fn set_view(&mut self, axis: ViewAxis, now: f64) {
        let mut t = self.cam;
        t.set_view_axis(axis);
        self.animate_camera_to(t, now);
    }

    pub fn tick_camera(&mut self, ctx: &egui::Context) {
        if let Some((a, b, t0)) = self.cam_anim {
            let now = ctx.input(|i| i.time);
            let t = ((now - t0) / 0.22).clamp(0.0, 1.0) as f32;
            let s = t * t * (3.0 - 2.0 * t);
            self.cam = a.lerp(&b, s);
            if t >= 1.0 {
                self.cam_anim = None;
            } else {
                ctx.request_repaint();
            }
        }
    }

    pub fn frame_all(&mut self, animate: bool) {
        self.cache.update(&self.scene, None, None);
        let (c, r) = self.cache.scene_bounds(&self.scene);
        let mut t = self.cam;
        t.frame(c, r);
        if animate {
            self.animate_camera_to(t, self.last_time);
        } else {
            self.cam = t;
        }
    }

    pub fn frame_selected(&mut self) {
        let Some(f) = self.sel.figure else {
            self.frame_all(true);
            return;
        };
        let fig = &self.scene.figures[f];
        let Some(c) = self.cache.figs.get(&fig.id) else { return };
        let (center, radius) = match self.sel.bone {
            Some(b) if b < c.pose.world.len() => {
                let p = c.world.transform_point3(c.pose.world[b].w_axis.truncate());
                (p, (fig.model.skeleton.bones[b].length * 2.5).max(0.12))
            }
            _ => (c.world.transform_point3(c.center), c.radius),
        };
        let mut t = self.cam;
        t.frame(center, radius);
        self.animate_camera_to(t, self.last_time);
    }

    pub fn toggle_ortho(&mut self) {
        self.cam.projection = match self.cam.projection {
            Projection::Orthographic => Projection::Perspective,
            _ => Projection::Orthographic,
        };
        self.needs_render = true;
    }

    // ---- Animation --------------------------------------------------------------------------

    pub fn tick_animation(&mut self, ctx: &egui::Context, dt: f32) {
        let mut any = false;
        for fig in &mut self.scene.figures {
            let Some(a) = fig.anim.as_mut() else { continue };
            if !a.playing {
                continue;
            }
            let Some(clip) = fig.model.animations.get(a.clip) else { continue };
            a.time += dt * a.speed;
            if a.time > clip.duration {
                if a.looping {
                    a.time = if clip.duration > 0.0 { a.time % clip.duration } else { 0.0 };
                } else {
                    a.time = clip.duration;
                    a.playing = false;
                }
            }
            any = true;
        }
        let idxs: Vec<usize> = (0..self.scene.figures.len()).filter(|i| self.scene.figures[*i].anim.as_ref().is_some_and(|a| a.playing)).collect();
        for i in idxs {
            self.sample_animation(i);
        }
        if any {
            ctx.request_repaint();
        }
    }

    pub fn retarget_dialog(&mut self, fig_id: u64) {
        let Some(path) = rfd::FileDialog::new().add_filter("Animated models", poser_io::IMPORT_EXTENSIONS).set_title("Animation source (Mixamo FBX, GLB, VRM...)").pick_file() else { return };
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let r = poser_io::load_model(&path);
            let _ = tx.send((path, r));
        });
        self.retarget_loading.push((fig_id, rx));
        self.status = "Loading animation source...".into();
    }

    pub fn poll_retargets(&mut self, ctx: &egui::Context) {
        if self.retarget_loading.is_empty() {
            return;
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(50));
        let mut done = vec![];
        for (i, (fid, rx)) in self.retarget_loading.iter().enumerate() {
            if let Ok((path, res)) = rx.try_recv() {
                done.push(i);
                match res {
                    Ok(m) if m.humanoid.is_valid() && !m.animations.is_empty() => {
                        self.status = format!("Retargeting {} clip(s) from {}", m.animations.len(), path.display());
                        self.retargets.insert(*fid, Retarget { source: Arc::new(m), clip: 0, time: 0.0, playing: true });
                    }
                    Ok(m) if m.animations.is_empty() => self.status = "That file has no animations".into(),
                    Ok(_) => self.status = "Animation source has no recognisable humanoid rig".into(),
                    Err(e) => self.status = format!("Load failed: {e:#}"),
                }
            }
        }
        for i in done.into_iter().rev() {
            self.retarget_loading.remove(i);
        }
    }

    pub fn tick_retargets(&mut self, ctx: &egui::Context, dt: f32) {
        let ids: Vec<u64> = self.retargets.keys().copied().collect();
        for id in ids {
            let Some(idx) = self.scene.index_of(id) else {
                self.retargets.remove(&id);
                continue;
            };
            let rt = self.retargets.get_mut(&id).unwrap();
            if !rt.playing {
                continue;
            }
            let dur = rt.source.animations.get(rt.clip).map(|c| c.duration).unwrap_or(0.0);
            rt.time = if dur > 0.0 { (rt.time + dt) % dur } else { 0.0 };
            self.sample_retarget(idx);
            ctx.request_repaint();
        }
    }

    /// Evaluate the retarget source at its time and apply it to the figure.
    pub fn sample_retarget(&mut self, idx: usize) {
        let id = self.scene.figures[idx].id;
        let Some(rt) = self.retargets.get(&id) else { return };
        let src = rt.source.clone();
        let Some(clip) = src.animations.get(rt.clip) else { return };
        let mut locals = src.skeleton.rest_locals();
        clip.sample_into(rt.time, &mut locals);
        let hp = extract_human_pose(&src.skeleton, &src.humanoid, &locals);
        self.apply_pose_to(idx, &hp, true);
    }

    /// Apply the figure's current animation time to its pose.
    pub fn sample_animation(&mut self, idx: usize) {
        let fig = &mut self.scene.figures[idx];
        let Some(a) = fig.anim.clone() else { return };
        let Some(clip) = fig.model.animations.get(a.clip) else { return };
        let mut locals = fig.model.skeleton.rest_locals();
        clip.sample_into(a.time, &mut locals);
        fig.pose.locals = locals;
        for mc in &clip.morph_channels {
            if let Some(w) = mc.weights.sample_weights(a.time) {
                for (name, v) in mc.names.iter().zip(w) {
                    if let Some(i) = fig.morph_names.iter().position(|n| n == name) {
                        fig.morph_weights[i] = v;
                    }
                }
            }
        }
        fig.touch();
        self.needs_render = true;
    }

    // ---- Export -----------------------------------------------------------------------------

    pub fn export_image(&mut self, path: PathBuf) {
        let Some(r) = self.renderer.as_mut() else { return };
        let (w, h) = (self.export.width.clamp(16, 8192), self.export.height.clamp(16, 8192));
        let (center, radius) = self.cache.scene_bounds(&self.scene);
        let draws = crate::viewport::build_draws(&self.scene, &self.cache, true);
        let desc = poser_render::FrameDesc {
            camera: &self.cam,
            settings: &self.scene.render,
            lights: &self.scene.lights,
            width: w,
            height: h,
            transparent: self.export.transparent,
            bounds_center: center,
            bounds_radius: radius,
        };
        r.render(&desc, &draws);
        let res = r.read_output();
        self.needs_render = true;
        match res {
            Some((w, h, px)) => match image::RgbaImage::from_raw(w, h, px).map(|img| img.save(&path)) {
                Some(Ok(())) => self.status = format!("Saved {}", path.display()),
                Some(Err(e)) => self.status = format!("Save failed: {e}"),
                None => self.status = "Save failed: bad image buffer".into(),
            },
            None => self.status = "Export failed: could not read the image back from the GPU".into(),
        }
        self.last_rendered = None;
    }

    pub fn save_user_pose(&mut self, name: String) {
        if let Some(mut p) = self.sel.figure.and_then(|f| self.current_human_pose(f)) {
            p.name = name;
            p.category = "My poses".into();
            self.user_poses.push(p);
            save_user_poses(&self.user_poses);
            self.status = "Pose saved to library".into();
        }
    }
}

fn user_pose_path() -> Option<PathBuf> {
    let base = std::env::var_os("APPDATA").map(PathBuf::from)?;
    Some(base.join("3DModelPoser").join("poses.json"))
}

pub fn load_user_poses() -> Vec<HumanPose> {
    user_pose_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save_user_poses(p: &[HumanPose]) {
    if let Some(path) = user_pose_path() {
        let _ = std::fs::create_dir_all(path.parent().unwrap());
        if let Ok(s) = serde_json::to_string_pretty(p) {
            let _ = std::fs::write(path, s);
        }
    }
}

impl eframe::App for PoserApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let now = ctx.input(|i| i.time);
        let dt = if self.last_time > 0.0 { (now - self.last_time) as f32 } else { 0.016 };
        self.last_time = now;
        self.frame_ms = self.frame_ms * 0.9 + dt * 1000.0 * 0.1;
        self.poll_imports(&ctx);
        self.poll_retargets(&ctx);
        self.handle_dropped_files(&ctx);
        self.handle_shortcuts(&ctx);
        self.tick_animation(&ctx, dt.min(0.1));
        self.tick_retargets(&ctx, dt.min(0.1));
        self.tick_camera(&ctx);
        crate::panels::top_bar(self, ui);
        crate::panels::status_bar(self, ui);
        if self.show_timeline {
            crate::panels::timeline(self, ui);
        }
        if self.show_left {
            crate::panels::toolbar(self, ui);
        }
        if self.show_right {
            crate::panels::right_panel(self, ui);
        }
        crate::panels::windows(self, &ctx);
        egui::CentralPanel::no_frame().show(ui, |ui| self.viewport(ui, frame));
    }
}

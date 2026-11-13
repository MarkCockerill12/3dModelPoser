//! Application state and high-level commands.
use crate::deform::DeformCache;
use crate::gizmo::Ring;
use crate::presets;
use poser_core::camera::{Camera, Projection, ViewAxis};
use poser_core::glam::{Quat, Vec3};
use poser_core::humanoid::{HumanBone, JointLimit};
use poser_core::makehuman::{self, HumanParams};
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
#[allow(dead_code)]
pub enum Tool {
    Pose,
    Move,
    Rotate,
}

/// Which transform gizmo is shown (W / E-R / S like Godot and Blender).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GizmoMode {
    Move,
    Rotate,
    Scale,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PropsTab {
    Rig,
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
    Translate { bone: usize },
}

#[derive(Clone, Debug)]
pub enum Drag {
    Ring { ring: Ring, axis_world: Vec3, last_angle: f32, tangent: Option<egui::Vec2>, last_pos: egui::Pos2 },
    /// IK / FK drag. Re-solved every frame from `start` with the total pointer movement, so it is
    /// smooth and fully reversible; nothing moves until the pointer travels a few pixels.
    Ik { figure: usize, kind: IkKind, effector: usize, start: Vec<poser_core::Transform>, start_target: Vec3, plane_point: Vec3, press: egui::Pos2, started: bool },
    MoveFigure { figure: usize, grab_offset: Vec3, plane_point: Vec3 },
    RotateFigure { figure: usize },
    Handle { handle: crate::gizmo::Handle, mode: GizmoMode, last: egui::Pos2 },
    NavGizmo { start: egui::Pos2, last: egui::Pos2, moved: bool },
    Orbit { last: egui::Pos2, pan: bool },
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PoseMode {
    /// Design Doll feel: anatomical limits, IK stays natural ("on rails").
    Guided,
    /// Blender feel: no limits, any bone can rotate or move anywhere.
    Free,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScrollMode {
    /// Mouse wheel zooms, precision-touchpad scrolling orbits.
    Auto,
    /// Every scroll zooms (smooth-scrolling mice).
    AlwaysZoom,
    /// Every scroll orbits (Shift pans, Ctrl zooms).
    AlwaysOrbit,
}

pub struct Options {
    pub mode: PoseMode,
    pub scroll: ScrollMode,
    pub drag_empty_orbits: bool,
    pub dual_quaternion: bool,
    /// Viewport resolution multiplier (lower = faster, less memory).
    pub render_scale: f32,
    /// Rotating one spine joint bends the whole spine smoothly.
    pub spine_spread: bool,
    /// Allow selecting every bone (face, twist, toes...) instead of only the main posing joints.
    pub all_bones: bool,
    pub local_axes: bool,
    pub limits: bool,
    pub emulate_3button: bool,
    pub show_rings: bool,
    pub keep_feet_planted: bool,
    /// Guided: limbs can't pass through the body.
    pub collisions: bool,
    /// Guided: nothing goes through the floor (world y = 0).
    pub floor: bool,
}

/// Interactive auto-rig session (marker placement).
pub struct RigSession {
    pub figure_id: u64,
    /// Model baked to model space facing +Z (what gets rigged).
    pub prepared: Arc<Model>,
    pub markers: poser_rig::Markers,
    pub symmetric: bool,
    pub drag: Option<usize>,
    pub hover: Option<usize>,
    pub computing: Option<Receiver<anyhow::Result<Model>>>,
}

/// Animation borrowed from another model and replayed through the humanoid mapping.
pub struct Retarget {
    pub source: Arc<Model>,
    pub clip: usize,
    pub time: f32,
    pub playing: bool,
}

pub struct RefImage {
    pub tex: egui::TextureHandle,
    pub size: egui::Vec2,
    pub opacity: f32,
    pub in_front: bool,
    pub scale: f32,
    pub offset: egui::Vec2,
    pub visible: bool,
}

pub struct ExportSettings {
    pub width: u32,
    pub height: u32,
    pub transparent: bool,
    pub open: bool,
    pub turntable_angles: u32,
    pub turntable_separate: bool,
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
    pub humans: HashMap<u64, HumanParams>,
    pub export: ExportSettings,
    pub help_open: bool,
    pub frame_ms: f32,
    pub body_presets: Vec<HumanPose>,
    pub hand_presets: Vec<HumanPose>,
    pub user_poses: Vec<HumanPose>,
    pub last_time: f64,
    /// Debug: (output png, frames to wait) for --preset-sheet.
    pub selftest: Option<(PathBuf, u32)>,
    pub rig: Option<RigSession>,
    pub sheet_model: Option<PathBuf>,
    pub rig_markers: HashMap<u64, poser_rig::Markers>,
    /// Free mode: Ctrl+drag a bone moves it instead of rotating.
    pub drag_translate: bool,
    pub ctrl_down: bool,
    pub press_pos: egui::Pos2,
    pub ref_image: Option<RefImage>,
    pub thumbs: crate::thumbs::Thumbs,
    /// Picture-in-picture camera (None = off).
    pub inset: Option<Camera>,
    pub inset_tex: Option<egui::TextureId>,
    pub armature: Option<crate::armature::ArmatureEdit>,
    pub photo_job: Option<crate::photo::PhotoJob>,
    pub gizmo: GizmoMode,
    pub hover_handle: Option<crate::gizmo::Handle>,
    /// Debug: export the viewport image after N frames, then quit.
    pub export_after: Option<(PathBuf, u32)>,
    /// Body colliders per figure id, keyed by model pointer.
    pub colliders: HashMap<u64, (usize, Option<Arc<poser_core::collision::BodyCollider>>)>,
    /// Built-in props per figure id.
    pub props: HashMap<u64, poser_core::props::PropKind>,
    /// Last transform written to each attached figure (detects user edits).
    pub attach_sync: HashMap<u64, poser_core::transform::Transform>,
    /// Debug: (output, frames to wait) for --prop-test (turntable of a prop scene).
    pub turntable_after: Option<(PathBuf, u32)>,
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
                mode: PoseMode::Guided,
                scroll: ScrollMode::Auto,
                drag_empty_orbits: true,
                dual_quaternion: true,
                render_scale: 1.0,
                spine_spread: true,
                all_bones: false,
                local_axes: true,
                limits: true,
                emulate_3button: true,
                show_rings: true,
                keep_feet_planted: true,
                collisions: true,
                floor: true,
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
            humans: HashMap::new(),
            export: ExportSettings { width: 1920, height: 1080, transparent: true, open: false, turntable_angles: 8, turntable_separate: false },
            help_open: false,
            frame_ms: 0.0,
            body_presets: presets::body_presets(),
            hand_presets: presets::hand_presets(),
            user_poses: load_user_poses(),
            last_time: 0.0,
            selftest: None,
            rig: None,
            sheet_model: None,
            rig_markers: HashMap::new(),
            drag_translate: false,
            ctrl_down: false,
            press_pos: egui::Pos2::ZERO,
            ref_image: None,
            thumbs: Default::default(),
            inset: None,
            inset_tex: None,
            armature: None,
            photo_job: None,
            gizmo: GizmoMode::Rotate,
            hover_handle: None,
            export_after: None,
            colliders: HashMap::new(),
            props: HashMap::new(),
            attach_sync: HashMap::new(),
            turntable_after: None,
        };
        let idx = app.add_human(HumanParams::default(), "Human");
        app.sel = Selection { figure: Some(idx), bone: None };
        app.frame_all(false);
        let args: Vec<String> = std::env::args().skip(1).collect();
        match std::env::var("POSER_TAB").as_deref() {
            Ok("library") => app.props_tab = PropsTab::Library,
            Ok("body") => app.props_tab = PropsTab::Body,
            Ok("hands") => app.props_tab = PropsTab::Hands,
            _ => {}
        }
        if std::env::var_os("POSER_INSET").is_some() {
            let mut c = app.cam;
            c.set_view_axis(ViewAxis::Right);
            c.projection = Projection::Orthographic;
            app.inset = Some(c);
        }
        if let Ok(g) = std::env::var("POSER_GIZMO") {
            app.gizmo = match g.as_str() { "move" => GizmoMode::Move, "scale" => GizmoMode::Scale, _ => GizmoMode::Rotate };
            app.sel.bone = app.scene.figures[idx].model.humanoid.get(poser_core::humanoid::HumanBone::LeftLowerArm);
        }
        if std::env::var_os("POSER_LBS").is_some() {
            poser_core::skinning::DUAL_QUATERNION.store(false, std::sync::atomic::Ordering::Relaxed);
            app.opts.dual_quaternion = false;
        }
        if let Some(i) = args.iter().position(|a| a == "--photo-test") {
            let input = PathBuf::from(&args[i + 1]);
            let out = PathBuf::from(&args[i + 2]);
            let (tx, rx) = channel();
            let img = image::open(&input).expect("photo").to_rgb8();
            let (w, h) = img.dimensions();
            let _ = tx.send(poser_ai::estimate(&img).map(|k| (k, w, h)));
            app.photo_job = Some(crate::photo::PhotoJob { figure_id: app.scene.figures[0].id, path: input, rx });
            app.export_after = Some((out, 8));
            app.export.transparent = false;
            app.export.width = 1600;
            app.export.height = 1000;
            return app;
        }
        if let Some(i) = args.iter().position(|a| a == "--prop-test") {
            // Debug: a posed figure holding a sword, seated props around, rendered as a turntable.
            if let Some(p) = app.body_presets.iter().find(|p| p.name == "Relaxed stand").cloned() {
                app.apply_pose_to(idx, &p, true);
            }
            app.sel = Selection { figure: Some(idx), bone: None };
            app.add_prop(poser_core::props::PropKind::Sword);
            app.sel = Selection { figure: Some(idx), bone: None };
            app.add_prop(poser_core::props::PropKind::Chair);
            app.sel = Selection { figure: Some(idx), bone: None };
            app.add_prop(poser_core::props::PropKind::Table);
            app.sel = Selection::default();
            app.scene.render.show_joints = false;
            app.export.width = 800;
            app.export.height = 900;
            app.export.transparent = false;
            app.turntable_after = Some((PathBuf::from(&args[i + 1]), 4));
            return app;
        }
        if let Some(i) = args.iter().position(|a| a == "--preset-sheet") {
            app.sheet_model = args.get(i + 2).map(PathBuf::from);
            app.build_preset_sheet(args.get(i + 1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("presets.png")));
            return app;
        }
        for a in args {
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

    fn apply_human_shapes(&mut self, idx: usize, p: &HumanParams) {
        let fig = &mut self.scene.figures[idx];
        let map = fig.model.humanoid.clone();
        fig.shapes = vec![Default::default(); fig.model.skeleton.len()];
        for (h, l, g) in p.shapes() {
            if let Some(b) = map.get(h) {
                fig.shapes[b].length = l;
                fig.shapes[b].girth = g;
            }
        }
        fig.touch();
    }

    /// Add the realistic MakeHuman-based body.
    pub fn add_human(&mut self, params: HumanParams, name: &str) -> usize {
        let model = Arc::new(makehuman::build(&params));
        let idx = self.scene.add_model(model);
        let n = self.scene.figures.len();
        if n > 1 {
            let x = self.scene.figures.iter().take(n - 1).map(|f| f.transform.translation.x).fold(f32::MIN, f32::max);
            self.scene.figures[idx].transform.translation.x = x + 0.8;
        }
        self.scene.figures[idx].name = name.to_string();
        let id = self.scene.figures[idx].id;
        self.apply_human_shapes(idx, &params);
        self.humans.insert(id, params);
        if let Some(p) = self.body_presets.iter().find(|p| p.name == "Relaxed stand").cloned() {
            self.apply_pose_to(idx, &p, true);
        }
        idx
    }

    pub fn rebuild_human(&mut self, idx: usize, params: HumanParams) {
        let fig = &self.scene.figures[idx];
        let pose = extract_human_pose(&fig.model.skeleton, &fig.model.humanoid, &fig.pose.locals);
        let id = fig.id;
        let model = Arc::new(makehuman::build(&params));
        let fig = &mut self.scene.figures[idx];
        fig.pose = Pose::tpose(&model.skeleton, &model.humanoid);
        fig.model = model;
        self.apply_human_shapes(idx, &params);
        self.humans.insert(id, params);
        self.apply_pose_to(idx, &pose, true);
        self.refresh_pins(idx);
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
                        if !model.textures.is_empty() && self.scene.render.shading == poser_core::scene::Shading::Solid {
                            self.scene.render.shading = poser_core::scene::Shading::Textured;
                        }
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
            if let Some(p) = self.humans.get(&self.scene.figures[f].id).cloned() {
                self.humans.insert(fig.id, p);
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

/// Map a clicked bone to the joint the user should pose: the nearest main humanoid bone
    /// (face, twist, breast and toe helper bones belong to their parent joint).
    pub fn posable_bone(&self, idx: usize, bone: usize) -> usize {
        let m = &self.scene.figures[idx].model;
        if self.opts.all_bones || !m.humanoid.is_valid() {
            return bone;
        }
        let mut b = bone;
        loop {
            if m.humanoid.human_of(b).is_some() {
                return b;
            }
            match m.skeleton.bones[b].parent {
                Some(p) => b = p,
                None => return bone,
            }
        }
    }

    pub fn limit_for(&self, idx: usize, bone: usize) -> Option<JointLimit> {
        let f = &self.scene.figures[idx];
        f.model.humanoid.human_of(bone)?.default_limit()
    }

    /// Clamp a bone to its joint limits (normalised frame).
    pub fn clamp_bone(&mut self, idx: usize, bone: usize) {
        if !self.opts.limits || self.opts.mode == PoseMode::Free {
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

    /// How far (radians) a bone is outside its Guided joint limits (0 when inside / Free mode).
    pub fn limit_violation(&self, idx: usize, bone: usize) -> f32 {
        if !self.opts.limits || self.opts.mode == PoseMode::Free {
            return 0.0;
        }
        let Some(lim) = self.limit_for(idx, bone) else { return 0.0 };
        let fig = &self.scene.figures[idx];
        let m = &fig.model;
        if m.humanoid.tpose_world.len() != m.skeleton.len() {
            return 0.0;
        }
        let n = m.humanoid.to_normalized(&m.skeleton, bone, fig.pose.locals[bone].rotation);
        n.angle_between(lim.clamp(n))
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

    pub fn guided(&self) -> bool {
        self.opts.mode == PoseMode::Guided && self.opts.limits
    }

    /// Clamp a set of canonical bones (Guided mode only).
    pub fn clamp_humans(&mut self, idx: usize, bones: &[HumanBone]) {
        let ids: Vec<usize> = bones.iter().filter_map(|h| self.scene.figures[idx].model.humanoid.get(*h)).collect();
        for b in ids {
            self.clamp_bone(idx, b);
        }
    }

    /// Natural shoulder rhythm: raising an arm lifts the collarbone a little (Guided mode), keeping
    /// the arm pointing where it was. Stops raised arms looking dislocated.
    pub fn shoulder_rhythm(&mut self, idx: usize) {
        if !self.guided() {
            return;
        }
        let fig = &mut self.scene.figures[idx];
        let m = fig.model.clone();
        let hm = &m.humanoid;
        if !hm.is_valid() || hm.tpose_world.len() != m.skeleton.len() {
            return;
        }
        let sk = &m.skeleton;
        let shapes = fig.shapes_opt().map(|s| s.to_vec());
        for (sh, ua, ch, side) in [
            (HumanBone::LeftShoulder, HumanBone::LeftUpperArm, HumanBone::UpperChest, 1.0f32),
            (HumanBone::RightShoulder, HumanBone::RightUpperArm, HumanBone::UpperChest, -1.0f32),
        ] {
            let (Some(sb), Some(ub)) = (hm.get(sh), hm.get(ua)) else { continue };
            let cb = hm.get(ch).or(hm.get(HumanBone::Chest)).unwrap_or(sb);
            let w = sk.evaluate(&fig.pose.locals, shapes.as_deref()).world;
            let n = |b: usize| (mat_rotation(&w[b]) * hm.tpose_world[b].inverse()).normalize();
            let dir = n(ub) * Vec3::new(side, 0.0, 0.0);
            let local = n(cb).inverse() * dir;
            // Scapulohumeral rhythm: the shoulder girdle takes roughly a third of the arm's
            // elevation and follows it forwards (protraction) or back (retraction), so the
            // deltoid and chest skin aren't stretched round a fixed shoulder.
            let elev = local.y.clamp(-1.0, 1.0).asin();
            let horiz = (local.x * local.x + local.z * local.z).sqrt();
            let fwd = local.z.atan2(local.x * side);
            let raise = ((elev + 0.2) * 0.32).clamp(0.0, 0.55);
            let protract = (fwd * 0.22 * horiz).clamp(-0.3, 0.45);
            let total = Quat::from_rotation_z(raise * side) * Quat::from_rotation_y(-protract * side);
            // Shoulder-blade helper between collarbone and upper arm (MakeHuman `shoulder01`).
            let blade = sk.bones[ub].parent.filter(|p| *p != sb && hm.human_of(*p).is_none() && sk.bones[*p].parent == Some(sb));
            let keep = mat_rotation(&w[ub]);
            match blade {
                Some(bb) => {
                    fig.pose.locals[sb].rotation = hm.to_local(sk, sb, Quat::IDENTITY.slerp(total, 0.6));
                    fig.pose.locals[bb].rotation = hm.to_local(sk, bb, Quat::IDENTITY.slerp(total, 0.4));
                }
                None => fig.pose.locals[sb].rotation = hm.to_local(sk, sb, total),
            }
            let w2 = sk.evaluate(&fig.pose.locals, shapes.as_deref()).world;
            let pr = sk.bones[ub].parent.map(|p| mat_rotation(&w2[p])).unwrap_or(Quat::IDENTITY);
            fig.pose.locals[ub].rotation = (pr.inverse() * keep).normalize();
        }
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
        r.use_slot(2);
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

    /// Debug aid: render every body + hand preset on a mannequin into one contact sheet.
    pub fn build_preset_sheet(&mut self, out: PathBuf) {
        self.selftest = Some((out, 3));
    }

    fn render_preset_sheet(&mut self, out: &std::path::Path) {
        self.scene.figures.clear();
        self.pins.clear();
        let idx = match self.sheet_model.clone() {
            Some(path) => {
                let mut m = match poser_io::load_model(&path) {
                    Ok(m) => m,
                    Err(e) => {
                        println!("load failed: {e:#}");
                        return;
                    }
                };
                if !m.humanoid.is_valid() {
                    let prep = poser_rig::prepare_for_rigging(&m);
                    let mk = poser_rig::guess_markers(&prep);
                    m = poser_rig::auto_rig(&prep, &mk).expect("auto-rig");
                    println!("auto-rigged");
                }
                let (floor, top, center) = m.rest_extent();
                let h = (top - floor).max(1e-4);
                let s = 1.75 / h;
                let idx = self.scene.add_model(Arc::new(m));
                let t = &mut self.scene.figures[idx].transform;
                t.scale = Vec3::splat(s);
                t.translation = Vec3::new(-center.x * s, -floor * s, -center.z * s);
                self.scene.render.shading = poser_core::scene::Shading::Textured;
                idx
            }
            None => {
                let mut hp = HumanParams::default();
                if let Ok(o) = std::env::var("POSER_OUTFIT") {
                    let mut parts = o.split(';');
                    hp.clothes = parts.next().unwrap_or("").split(',').filter(|x| !x.is_empty()).map(String::from).collect();
                    hp.hair = parts.next().filter(|x| !x.is_empty()).map(String::from);
                }
                self.add_human(hp, "Sheet")
            }
        };
        let has_fingers = self.scene.figures[idx].model.humanoid.get(poser_core::humanoid::HumanBone::LeftIndexProximal).is_some();
        self.scene.render.show_joints = false;
        self.sel = Selection::default();
        let drag_test = std::env::var_os("POSER_DRAGTEST").is_some();
        let mut poses = if drag_test { self.drag_test_poses(idx) } else { self.body_presets.clone() };
        for h in if has_fingers && !drag_test { self.hand_presets.clone() } else { vec![] } {
            // Hands shown on a figure holding both hands up in front of the face.
            let mut p = self.body_presets.iter().find(|p| p.name == "T-pose").cloned().unwrap_or_default();
            p.name = format!("hand {}", h.name);
            p.rotations.insert(poser_core::humanoid::HumanBone::LeftUpperArm, Quat::from_rotation_arc(Vec3::X, Vec3::new(0.3, -0.2, 0.93).normalize()));
            p.rotations.insert(poser_core::humanoid::HumanBone::LeftLowerArm, Quat::from_rotation_y(-100f32.to_radians()));
            p.rotations.insert(poser_core::humanoid::HumanBone::RightUpperArm, Quat::from_rotation_z(80f32.to_radians()));
            for (b, q) in h.rotations.iter() {
                p.rotations.insert(*b, *q);
            }
            poses.push(p);
        }
        let (tw, th) = if drag_test { (560u32, 700u32) } else { (360u32, 450u32) };
        let cols = if drag_test { 4u32 } else { 7u32 };
        let rows = (poses.len() as u32).div_ceil(cols);
        let mut sheet = image::RgbaImage::new(tw * cols, th * rows);
        for (i, p) in poses.iter().enumerate() {
            self.apply_pose_to(idx, p, true);
            self.cache.update(&self.scene, None, None);
            let hand = p.name.starts_with("hand ");
            let (c, r) = self.cache.scene_bounds(&self.scene);
            self.cam = Camera::default();
            if hand {
                self.cam.set_yaw_pitch(40f32.to_radians(), -15f32.to_radians());
                let fig = &self.scene.figures[idx];
                let cc = &self.cache.figs[&fig.id];
                let hb = fig.model.humanoid.get(poser_core::humanoid::HumanBone::LeftHand).unwrap();
                let hp = cc.world.transform_point3(cc.pose.world[hb].w_axis.truncate());
                self.cam.frame(hp + Vec3::new(0.0, 0.03, 0.05), 0.12);
            } else {
                self.cam.set_yaw_pitch(30f32.to_radians(), -10f32.to_radians());
                self.cam.frame(c, r * if drag_test { 0.62 } else { 0.85 });
                // Debug: POSER_FOCUS=yaw_degrees frames the left shoulder/arm close up.
                if let Some(yaw) = std::env::var("POSER_FOCUS").ok().and_then(|v| v.parse::<f32>().ok()).filter(|_| drag_test) {
                    let fig = &self.scene.figures[idx];
                    let cc = &self.cache.figs[&fig.id];
                    if let Some(b) = fig.model.humanoid.get(poser_core::humanoid::HumanBone::LeftUpperArm) {
                        let p = cc.world.transform_point3(cc.pose.world[b].w_axis.truncate());
                        self.cam.set_yaw_pitch(yaw.to_radians(), -10f32.to_radians());
                        self.cam.frame(p + Vec3::new(-0.05, -0.1, 0.0), 0.32);
                    }
                }
            }
            let draws = crate::viewport::build_draws(&self.scene, &self.cache, true);
            let desc = poser_render::FrameDesc {
                camera: &self.cam,
                settings: &self.scene.render,
                lights: &self.scene.lights,
                width: tw,
                height: th,
                transparent: false,
                bounds_center: c,
                bounds_radius: r,
            };
            let Some(rd) = self.renderer.as_mut() else { return };
            rd.use_slot(2);
            rd.render(&desc, &draws);
            for f in self.cache.figs.values_mut() {
                f.deform_dirty = false;
            }
            if let Some((w, h, px)) = rd.read_output() {
                if let Some(img) = image::RgbaImage::from_raw(w, h, px) {
                    let (x, y) = ((i as u32 % cols) * tw, (i as u32 / cols) * th);
                    image::imageops::overlay(&mut sheet, &img, x as i64, y as i64);
                }
            }
            println!("{:>3}: {}", i, p.name);
        }
        let _ = sheet.save(out);
    }

    /// Begin auto-rigging the selected figure: bake its geometry and guess markers.
    pub fn start_rig(&mut self, idx: usize) {
        let fig = &self.scene.figures[idx];
        let prepared = poser_rig::prepare_for_rigging(&fig.model);
        let markers = poser_rig::guess_markers(&prepared);
        let id = fig.id;
        // Swap the figure to the prepared (unrigged, +Z facing) model so markers line up.
        self.push_undo("Auto-rig");
        let prepared = Arc::new(prepared);
        let f = &mut self.scene.figures[idx];
        f.model = prepared.clone();
        f.pose = Pose::rest(&prepared.skeleton);
        f.shapes = vec![];
        f.morph_names.clear();
        f.morph_weights.clear();
        f.anim = None;
        f.touch();
        self.pins.retain(|(fid, _), _| *fid != id);
        self.rig = Some(RigSession { figure_id: id, prepared, markers, symmetric: true, drag: None, hover: None, computing: None });
        self.sel = Selection { figure: Some(idx), bone: None };
        self.props_tab = PropsTab::Rig;
        let now = self.last_time;
        self.set_view(ViewAxis::Front, now);
        self.cam.projection = Projection::Orthographic;
        self.status = "Auto-rig: drag the coloured markers onto chin, wrists, elbows, knees and groin (check the side view too), then Apply".into();
    }

    pub fn apply_rig(&mut self) {
        let Some(r) = self.rig.as_mut() else { return };
        if r.computing.is_some() {
            return;
        }
        let (tx, rx) = channel();
        let model = r.prepared.clone();
        let markers = r.markers;
        std::thread::spawn(move || {
            let _ = tx.send(poser_rig::auto_rig(&model, &markers));
        });
        r.computing = Some(rx);
        self.status = "Computing skin weights...".into();
    }

    pub fn poll_rig(&mut self) {
        let Some(r) = self.rig.as_mut() else { return };
        let Some(rx) = r.computing.as_ref() else { return };
        let Ok(res) = rx.try_recv() else { return };
        let id = r.figure_id;
        let markers = r.markers;
        self.rig = None;
        self.cam.projection = Projection::Perspective;
        match (res, self.scene.index_of(id)) {
            (Ok(model), Some(idx)) => {
                let model = Arc::new(model);
                let f = &mut self.scene.figures[idx];
                f.pose = Pose::rest(&model.skeleton);
                f.shapes = vec![Default::default(); model.skeleton.len()];
                f.morph_names = model.morph_names();
                f.morph_weights = vec![0.0; f.morph_names.len()];
                f.model = model;
                f.touch();
                self.rig_markers.insert(id, markers);
                self.props_tab = PropsTab::Pose;
                self.status = "Rigged! Click a body part to pose it. Presets, IK and hands (no fingers yet) all work now.".into();
            }
            (Err(e), _) => self.status = format!("Auto-rig failed: {e:#}"),
            _ => {}
        }
        self.needs_render = true;
    }

    /// Export the selected figure rigged (skeleton + weights + morphs), in its current pose.
    pub fn export_rigged_dialog(&mut self) {
        let Some(f) = self.sel.figure else {
            self.status = "Select a figure to export".into();
            return;
        };
        let fig = &self.scene.figures[f];
        if !fig.model.has_skeleton() {
            self.status = "This model has no skeleton: use Export posed model, or auto-rig it first".into();
            return;
        }
        let ext = ["glb", "fbx"];
        let Some(path) = rfd::FileDialog::new().add_filter("glTF binary (.glb)", &ext[..1]).add_filter("FBX (.fbx)", &ext[1..]).set_file_name(format!("{}.glb", fig.name)).save_file() else { return };
        let morphs: Vec<(String, f32)> = fig.morph_names.iter().cloned().zip(fig.morph_weights.iter().copied()).collect();
        let is_fbx = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("fbx"));
        let r = if is_fbx {
            poser_io::export_fbx::write_fbx(&path, &fig.model, &fig.shapes, &fig.pose.locals, &fig.transform)
        } else {
            poser_io::export_rigged::write_rigged_glb(&path, &fig.model, &fig.shapes, &fig.pose.locals, &morphs, &fig.transform)
        };
        self.status = match r {
            Ok(()) => format!("Exported rigged {} to {}", fig.name, path.display()),
            Err(e) => format!("Export failed: {e:#}"),
        };
    }

    /// Export posed figures as static meshes (GLB or OBJ).
    pub fn export_model_dialog(&mut self, only_selected: bool) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("glTF binary (.glb)", &["glb"])
            .add_filter("Wavefront OBJ (.obj)", &["obj"])
            .set_file_name("pose.glb")
            .set_title("Export posed model")
            .save_file()
        else {
            return;
        };
        let sel = self.sel.figure.map(|f| (f, self.sel.bone));
        self.cache.update(&self.scene, sel, self.hover);
        let mut meshes = vec![];
        for (fi, f) in self.scene.figures.iter().enumerate() {
            if !f.visible || (only_selected && self.sel.figure != Some(fi)) {
                continue;
            }
            let Some(c) = self.cache.figs.get(&f.id) else { continue };
            let nm = poser_core::glam::Mat3::from_mat4(c.world).inverse().transpose();
            for (mi, mesh) in f.model.meshes.iter().enumerate() {
                let Some(d) = c.meshes.get(mi) else { continue };
                let mat = &f.model.materials[mesh.material.min(f.model.materials.len() - 1)];
                meshes.push(poser_io::export::ExportMesh {
                    name: format!("{}_{}", f.name, mesh.name),
                    positions: d.positions.iter().map(|p| c.world.transform_point3(*p)).collect(),
                    normals: d.normals.iter().map(|n| (nm * *n).normalize_or(Vec3::Y)).collect(),
                    uvs: &mesh.uvs,
                    indices: &mesh.indices,
                    material: mat,
                    texture: mat.base_color_texture.and_then(|t| f.model.textures.get(t)),
                });
            }
        }
        let obj = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("obj"));
        let r = if obj { poser_io::export::write_obj(&path, &meshes) } else { poser_io::export::write_glb(&path, &meshes) };
        self.status = match r {
            Ok(()) => format!("Exported {} mesh(es) to {}", meshes.len(), path.display()),
            Err(e) => format!("Export failed: {e:#}"),
        };
    }

    pub fn load_ref_image(&mut self, ctx: &egui::Context) {
        let Some(p) = rfd::FileDialog::new().add_filter("Images", &["png", "jpg", "jpeg", "bmp", "tga"]).pick_file() else { return };
        match image::open(&p) {
            Ok(img) => {
                let img = img.to_rgba8();
                let (w, h) = img.dimensions();
                let ci = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &img);
                let tex = ctx.load_texture("reference", ci, egui::TextureOptions::LINEAR);
                self.ref_image = Some(RefImage { tex, size: egui::vec2(w as f32, h as f32), opacity: 0.6, in_front: false, scale: 1.0, offset: egui::Vec2::ZERO, visible: true });
                self.needs_render = true;
            }
            Err(e) => self.status = format!("Could not open image: {e}"),
        }
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
        self.poll_rig();
        self.poll_armature();
        self.poll_photo(&ctx);
        if self.armature.as_ref().is_some_and(|a| a.computing.is_some()) {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
        if self.rig.as_ref().is_some_and(|r| r.computing.is_some()) {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
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
        if let Some((path, n)) = self.export_after.take() {
            if n == 0 {
                // Side-by-side: front with photo, plus a 3/4 view to judge depth.
                self.inset = Some({
                    let mut c = self.cam;
                    c.set_yaw_pitch(60f32.to_radians(), -10f32.to_radians());
                    c
                });
                self.export_image(path);
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            } else {
                self.export_after = Some((path, n - 1));
                ctx.request_repaint();
            }
        }
        if let Some((path, n)) = self.turntable_after.take() {
            if n == 0 {
                self.frame_all(false);
                self.cam_anim = None;
                self.cam.set_yaw_pitch(0.0, -12f32.to_radians());
                self.export_turntable(path, 8, false);
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            } else {
                self.turntable_after = Some((path, n - 1));
                ctx.request_repaint();
            }
        }
        if let Some((path, n)) = self.selftest.take() {
            if n == 0 {
                self.render_preset_sheet(&path);
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            } else {
                self.selftest = Some((path, n - 1));
                ctx.request_repaint();
            }
        }
        self.sync_attachments();
        egui::CentralPanel::no_frame().show(ui, |ui| self.viewport(ui, frame));
    }
}

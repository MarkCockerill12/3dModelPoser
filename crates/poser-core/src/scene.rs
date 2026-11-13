//! Scene document: figures/props, lights, render settings, undo.
use crate::model::Model;
use crate::pose::Pose;
use crate::skeleton::{BoneShape, PoseMatrices};
use crate::transform::Transform;
use glam::{Mat4, Vec3, Vec4};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static REVISION: AtomicU64 = AtomicU64::new(1);

/// Globally unique, monotonically increasing revision number (cache keys).
pub fn next_revision() -> u64 {
    REVISION.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AnimState {
    pub clip: usize,
    pub time: f32,
    pub playing: bool,
    pub looping: bool,
    pub speed: f32,
}

/// Where an object is attached (props held in hands etc.).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Attachment {
    pub parent: u64,
    pub bone: usize,
    /// Object transform relative to the bone's world matrix.
    pub offset: Transform,
}

/// A recorded "delete parts" edit: (mesh, island) pairs as found by `edit::mesh_islands`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartEdit {
    pub islands: Vec<(usize, u32)>,
    /// Made after auto-rigging (re-applied after the rig when a scene is loaded).
    #[serde(default)]
    pub after_rig: bool,
}

/// A model instance in the scene (character or prop).
#[derive(Clone, Debug)]
pub struct Figure {
    pub id: u64,
    pub name: String,
    pub model: Arc<Model>,
    pub transform: Transform,
    pub pose: Pose,
    pub shapes: Vec<BoneShape>,
    /// Indexed like `model.morph_names()`.
    pub morph_names: Vec<String>,
    pub morph_weights: Vec<f32>,
    pub visible: bool,
    pub pinned: Vec<usize>,
    pub limits_enabled: bool,
    pub tint: Option<Vec4>,
    pub anim: Option<AnimState>,
    pub attachment: Option<Attachment>,
    /// Pieces deleted from the model, in order (so scenes can redo them on load).
    pub part_edits: Vec<PartEdit>,
    /// Bumped whenever something that affects deformation changes.
    pub revision: u64,
    /// Generator parameters for procedural figures (serialised JSON), so they can be rebuilt.
    pub procedural: Option<String>,
}

impl Figure {
    pub fn new(id: u64, model: Arc<Model>) -> Self {
        let pose = if model.humanoid.is_valid() && model.humanoid.tpose_local.len() == model.skeleton.len() {
            Pose::rest(&model.skeleton)
        } else {
            Pose::rest(&model.skeleton)
        };
        let morph_names = model.morph_names();
        Self {
            id,
            name: model.name.clone(),
            shapes: vec![BoneShape::default(); model.skeleton.len()],
            morph_weights: vec![0.0; morph_names.len()],
            morph_names,
            pose,
            model,
            transform: Transform::IDENTITY,
            visible: true,
            pinned: vec![],
            limits_enabled: true,
            tint: None,
            anim: None,
            attachment: None,
            part_edits: vec![],
            revision: next_revision(),
            procedural: None,
        }
    }

    pub fn touch(&mut self) {
        self.revision = next_revision();
    }

    pub fn shapes_opt(&self) -> Option<&[BoneShape]> {
        if self.shapes.iter().all(|s| s.is_identity()) { None } else { Some(&self.shapes) }
    }

    pub fn evaluate(&self) -> PoseMatrices {
        self.model.skeleton.evaluate(&self.pose.locals, self.shapes_opt())
    }

    /// Morph weights in a mesh's own target order.
    pub fn mesh_morph_weights(&self, mesh: usize) -> Vec<f32> {
        self.model.meshes[mesh]
            .morphs
            .iter()
            .map(|t| {
                self.morph_names
                    .iter()
                    .position(|n| *n == t.name)
                    .map(|i| self.morph_weights[i])
                    .unwrap_or(0.0)
            })
            .collect()
    }

    pub fn is_character(&self) -> bool {
        self.model.has_skeleton()
    }

    /// Height of the figure (rest bounds, scaled).
    pub fn height(&self) -> f32 {
        let (lo, hi) = self.model.rest_bounds();
        (hi.y - lo.y) * self.transform.scale.y
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LightKind {
    Directional,
    Point,
    Spot,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Light {
    pub name: String,
    pub kind: LightKind,
    pub color: Vec3,
    pub intensity: f32,
    /// Azimuth/elevation (radians) for directional + spot direction.
    pub azimuth: f32,
    pub elevation: f32,
    pub position: Vec3,
    pub range: f32,
    pub spot_angle: f32,
    pub enabled: bool,
}

impl Light {
    pub fn key() -> Self {
        Self {
            name: "Key Light".into(),
            kind: LightKind::Directional,
            color: Vec3::ONE,
            intensity: 2.6,
            azimuth: 35f32.to_radians(),
            elevation: 45f32.to_radians(),
            position: Vec3::new(1.5, 2.5, 2.0),
            range: 10.0,
            spot_angle: 35f32.to_radians(),
            enabled: true,
        }
    }

    /// Direction light travels *towards* (world space).
    pub fn direction(&self) -> Vec3 {
        let (sa, ca) = self.azimuth.sin_cos();
        let (se, ce) = self.elevation.sin_cos();
        -Vec3::new(sa * ce, se, ca * ce).normalize()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shading {
    /// Clay/solid with simple lighting (Blender Solid).
    Solid,
    /// Textured with lighting.
    Textured,
    /// Two-tone cel shading for drawing reference.
    Toon,
    /// Flat silhouette.
    Silhouette,
    /// Semi-transparent see-through.
    XRay,
    /// Show normals-based shading without lights (good for form reading).
    Matcap,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RenderSettings {
    pub shading: Shading,
    pub outline: bool,
    pub outline_width: f32,
    pub outline_color: Vec4,
    pub shadows: bool,
    pub ambient: f32,
    pub background_top: Vec4,
    pub background_bottom: Vec4,
    pub show_grid: bool,
    pub show_floor: bool,
    pub show_bones: bool,
    pub show_joints: bool,
    pub joint_stress: bool,
    pub toon_threshold: f32,
    pub toon_shadow: Vec4,
    pub clay_color: Vec4,
    pub rim: f32,
    pub eye_level_line: bool,
    pub thirds_grid: bool,
    pub perspective_guides: bool,
    pub msaa: u32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            shading: Shading::Solid,
            outline: true,
            outline_width: 1.5,
            outline_color: Vec4::new(0.05, 0.05, 0.06, 1.0),
            shadows: true,
            ambient: 0.35,
            background_top: Vec4::new(0.24, 0.24, 0.25, 1.0),
            background_bottom: Vec4::new(0.16, 0.16, 0.17, 1.0),
            show_grid: true,
            show_floor: true,
            show_bones: false,
            show_joints: true,
            joint_stress: false,
            toon_threshold: 0.45,
            toon_shadow: Vec4::new(0.55, 0.52, 0.58, 1.0),
            clay_color: Vec4::new(0.80, 0.78, 0.75, 1.0),
            rim: 0.25,
            eye_level_line: false,
            thirds_grid: false,
            perspective_guides: false,
            msaa: 4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Scene {
    pub figures: Vec<Figure>,
    pub lights: Vec<Light>,
    pub render: RenderSettings,
    pub next_id: u64,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            figures: vec![],
            lights: vec![Light::key(), {
                let mut l = Light::key();
                l.name = "Fill Light".into();
                l.intensity = 0.7;
                l.azimuth = -120f32.to_radians();
                l.elevation = 20f32.to_radians();
                l
            }],
            render: RenderSettings::default(),
            next_id: 1,
        }
    }
}

impl Scene {
    pub fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn add_model(&mut self, model: Arc<Model>) -> usize {
        let id = self.alloc_id();
        self.figures.push(Figure::new(id, model));
        self.figures.len() - 1
    }

    pub fn index_of(&self, id: u64) -> Option<usize> {
        self.figures.iter().position(|f| f.id == id)
    }

    /// World matrix of a figure, resolving attachments (one level deep chains are fine;
    /// cycles are cut).
    pub fn figure_world(&self, idx: usize, evals: &[Option<PoseMatrices>]) -> Mat4 {
        let f = &self.figures[idx];
        let local = f.transform.to_mat4();
        if let Some(a) = f.attachment {
            if let Some(pi) = self.index_of(a.parent).filter(|p| *p != idx) {
                let parent = &self.figures[pi];
                if parent.attachment.is_none_or(|pa| pa.parent != f.id) {
                    let pw = self.figure_world(pi, evals);
                    if let Some(Some(pm)) = evals.get(pi) {
                        if let Some(bm) = pm.world.get(a.bone) {
                            return pw * *bm * a.offset.to_mat4() * Mat4::from_scale(f.transform.scale);
                        }
                    }
                }
            }
        }
        local
    }
}

/// Snapshot undo (models are Arc-shared so snapshots are cheap).
#[derive(Default)]
pub struct UndoStack {
    past: Vec<(String, Vec<Figure>, Vec<Light>, RenderSettings)>,
    future: Vec<(String, Vec<Figure>, Vec<Light>, RenderSettings)>,
    pub max: usize,
}

impl UndoStack {
    pub fn new(max: usize) -> Self {
        Self { past: vec![], future: vec![], max }
    }

    /// Record the scene state *before* a change.
    pub fn push(&mut self, label: impl Into<String>, scene: &Scene) {
        self.past.push((label.into(), scene.figures.clone(), scene.lights.clone(), scene.render.clone()));
        if self.past.len() > self.max {
            self.past.remove(0);
        }
        self.future.clear();
    }

    pub fn undo(&mut self, scene: &mut Scene) -> Option<String> {
        let (label, f, l, r) = self.past.pop()?;
        self.future.push((label.clone(), std::mem::replace(&mut scene.figures, f), std::mem::replace(&mut scene.lights, l), std::mem::replace(&mut scene.render, r)));
        for fig in &mut scene.figures {
            fig.touch();
        }
        Some(label)
    }

    pub fn redo(&mut self, scene: &mut Scene) -> Option<String> {
        let (label, f, l, r) = self.future.pop()?;
        self.past.push((label.clone(), std::mem::replace(&mut scene.figures, f), std::mem::replace(&mut scene.lights, l), std::mem::replace(&mut scene.render, r)));
        for fig in &mut scene.figures {
            fig.touch();
        }
        Some(label)
    }

    /// Drop the most recent snapshot (cancelled operation).
    pub fn discard_last(&mut self) {
        self.past.pop();
    }

    pub fn can_undo(&self) -> bool {
        !self.past.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.future.is_empty()
    }
}

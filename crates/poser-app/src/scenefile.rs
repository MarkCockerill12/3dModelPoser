//! .poser scene files (JSON): figures reference their source (mannequin parameters or a model
//! file + optional auto-rig markers) plus pose, proportions, morphs, placement, lights and camera.
use crate::app::{PoserApp, Selection};
use anyhow::{Context, Result, bail};
use poser_core::camera::Camera;
use poser_core::glam::Vec4;
use poser_core::mannequin::{self, MannequinParams};
use poser_core::scene::{Light, RenderSettings};
use poser_core::skeleton::BoneShape;
use poser_core::transform::Transform;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Serialize, Deserialize)]
pub enum FigureSource {
    Mannequin(MannequinParams),
    File { path: PathBuf, rig: Option<poser_rig::Markers> },
}

#[derive(Serialize, Deserialize)]
pub struct FigureFile {
    pub name: String,
    pub source: FigureSource,
    pub transform: Transform,
    pub locals: Vec<Transform>,
    pub shapes: Vec<BoneShape>,
    pub morphs: Vec<(String, f32)>,
    pub visible: bool,
    pub tint: Option<Vec4>,
    #[serde(default)]
    pub pins: Vec<usize>,
}

#[derive(Serialize, Deserialize)]
pub struct SceneFile {
    pub version: u32,
    pub figures: Vec<FigureFile>,
    pub lights: Vec<Light>,
    pub render: RenderSettings,
    pub camera: Camera,
    #[serde(default)]
    pub saved_views: Vec<(String, Camera)>,
}

impl PoserApp {
    pub fn save_scene_dialog(&mut self) {
        let Some(path) = rfd::FileDialog::new().add_filter("Poser scene", &["poser"]).set_file_name("scene.poser").save_file() else { return };
        self.status = match self.save_scene(&path) {
            Ok(()) => format!("Saved {}", path.display()),
            Err(e) => format!("Save failed: {e:#}"),
        };
    }

    pub fn open_scene_dialog(&mut self) {
        let Some(path) = rfd::FileDialog::new().add_filter("Poser scene", &["poser"]).pick_file() else { return };
        self.status = match self.load_scene(&path) {
            Ok(n) => format!("Opened {} ({n} figures)", path.display()),
            Err(e) => format!("Open failed: {e:#}"),
        };
    }

    pub fn save_scene(&self, path: &Path) -> Result<()> {
        let mut figures = vec![];
        for f in &self.scene.figures {
            let source = if let Some(p) = self.mannequins.get(&f.id) {
                FigureSource::Mannequin(p.clone())
            } else if let Some(src) = f.model.path.clone() {
                FigureSource::File { path: src, rig: self.rig_markers.get(&f.id).copied() }
            } else {
                continue;
            };
            figures.push(FigureFile {
                name: f.name.clone(),
                source,
                transform: f.transform,
                locals: f.pose.locals.clone(),
                shapes: f.shapes.clone(),
                morphs: f.morph_names.iter().cloned().zip(f.morph_weights.iter().copied()).filter(|(_, w)| w.abs() > 1e-5).collect(),
                visible: f.visible,
                tint: f.tint,
                pins: self.pins.keys().filter(|(id, _)| *id == f.id).map(|(_, b)| *b).collect(),
            });
        }
        let file = SceneFile {
            version: 1,
            figures,
            lights: self.scene.lights.clone(),
            render: self.scene.render.clone(),
            camera: self.cam,
            saved_views: self.saved_views.clone(),
        };
        std::fs::write(path, serde_json::to_string_pretty(&file)?).context("writing scene")?;
        Ok(())
    }

    pub fn load_scene(&mut self, path: &Path) -> Result<usize> {
        let text = std::fs::read_to_string(path)?;
        let file: SceneFile = serde_json::from_str(&text).context("not a valid .poser file")?;
        if file.version > 1 {
            bail!("scene was saved by a newer version");
        }
        self.push_undo("Open scene");
        self.scene.figures.clear();
        self.pins.clear();
        self.mannequins.clear();
        self.rig_markers.clear();
        self.retargets.clear();
        let base = path.parent().map(Path::to_path_buf).unwrap_or_default();
        let mut missing = vec![];
        for ff in file.figures {
            let (model, params, markers) = match &ff.source {
                FigureSource::Mannequin(p) => (Arc::new(mannequin::build(p)), Some(p.clone()), None),
                FigureSource::File { path: p, rig } => {
                    let p = if p.is_file() { p.clone() } else { base.join(p.file_name().unwrap_or_default()) };
                    match poser_io::load_model(&p) {
                        Ok(mut m) => {
                            if let Some(mk) = rig {
                                let prep = poser_rig::prepare_for_rigging(&m);
                                m = poser_rig::auto_rig(&prep, mk)?;
                                m.path = Some(p.clone());
                            }
                            (Arc::new(m), None, *rig)
                        }
                        Err(_) => {
                            missing.push(p.display().to_string());
                            continue;
                        }
                    }
                }
            };
            let idx = self.scene.add_model(model.clone());
            let fig = &mut self.scene.figures[idx];
            fig.name = ff.name;
            fig.transform = ff.transform;
            if ff.locals.len() == model.skeleton.len() {
                fig.pose.locals = ff.locals;
            }
            if ff.shapes.len() == model.skeleton.len() {
                fig.shapes = ff.shapes;
            }
            for (n, w) in ff.morphs {
                if let Some(i) = fig.morph_names.iter().position(|x| *x == n) {
                    fig.morph_weights[i] = w;
                }
            }
            fig.visible = ff.visible;
            fig.tint = ff.tint;
            fig.touch();
            let id = fig.id;
            if let Some(p) = params {
                self.mannequins.insert(id, p);
            }
            if let Some(m) = markers {
                self.rig_markers.insert(id, m);
            }
            for b in ff.pins {
                if b < model.skeleton.len() {
                    self.toggle_pin(idx, b);
                }
            }
        }
        self.scene.lights = file.lights;
        self.scene.render = file.render;
        self.cam = file.camera;
        self.saved_views = file.saved_views;
        self.sel = Selection::default();
        self.needs_render = true;
        if !missing.is_empty() {
            self.status = format!("Missing model files: {}", missing.join(", "));
        }
        Ok(self.scene.figures.len())
    }

    pub fn new_scene(&mut self) {
        self.push_undo("New scene");
        self.scene.figures.clear();
        self.pins.clear();
        self.retargets.clear();
        let i = self.add_mannequin(MannequinParams::default(), "Mannequin");
        self.sel = Selection { figure: Some(i), bone: None };
        self.cam = Camera::default();
        self.frame_all(false);
    }
}

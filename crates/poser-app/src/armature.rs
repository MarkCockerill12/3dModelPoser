//! Manual skeleton editing (Blender "armature edit" style): add / extrude / move / delete bones,
//! set parents and humanoid roles, symmetrize, then bind with automatic weights.
use crate::app::{PoserApp, PropsTab, Selection};
use crate::gizmo::ViewCtx;
use egui::{Color32, Pos2, RichText, Stroke};
use poser_core::glam::{Mat4, Vec3};
use poser_core::humanoid::{HumanBone, HumanoidMap};
use poser_core::model::Model;
use poser_core::skeleton::{Bone, Skeleton};
use poser_core::transform::Transform;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

#[derive(Clone)]
pub struct EditBone {
    pub name: String,
    pub parent: Option<usize>,
    /// Model-space joint position.
    pub head: Vec3,
    pub role: Option<HumanBone>,
}

pub struct ArmatureEdit {
    pub figure_id: u64,
    pub prepared: Arc<Model>,
    pub bones: Vec<EditBone>,
    pub selected: Option<usize>,
    pub drag: Option<usize>,
    pub computing: Option<Receiver<anyhow::Result<Model>>>,
}

impl ArmatureEdit {
    fn world(&self, app: &PoserApp) -> Option<Mat4> {
        let f = app.scene.index_of(self.figure_id)?;
        Some(app.cache.figs.get(&app.scene.figures[f].id)?.world)
    }

    fn unique_name(&self, base: &str) -> String {
        if !self.bones.iter().any(|b| b.name == base) {
            return base.to_string();
        }
        (1..).map(|i| format!("{base}.{i:03}")).find(|n| !self.bones.iter().any(|b| b.name == *n)).unwrap()
    }

    pub fn delete(&mut self, i: usize) {
        let parent = self.bones[i].parent;
        for b in &mut self.bones {
            if b.parent == Some(i) {
                b.parent = parent;
            }
        }
        self.bones.remove(i);
        for b in &mut self.bones {
            if let Some(p) = b.parent.as_mut() {
                if *p > i {
                    *p -= 1;
                }
            }
        }
        self.selected = None;
    }

    /// Copy every left-side bone (+X, name containing L/Left) to the right side.
    pub fn symmetrize(&mut self, center_x: f32) {
        let n = self.bones.len();
        let mut map: Vec<Option<usize>> = vec![None; n];
        for i in 0..n {
            let b = self.bones[i].clone();
            if b.head.x <= center_x + 1e-4 {
                continue;
            }
            let Some(mirror_name) = poser_core::humanoid::mirror_name(&b.name) else { continue };
            let head = Vec3::new(2.0 * center_x - b.head.x, b.head.y, b.head.z);
            let existing = self.bones.iter().position(|x| x.name == mirror_name);
            let role = b.role.map(|r| r.mirror());
            let idx = match existing {
                Some(e) => {
                    self.bones[e].head = head;
                    self.bones[e].role = role.or(self.bones[e].role);
                    e
                }
                None => {
                    self.bones.push(EditBone { name: mirror_name, parent: b.parent, head, role });
                    self.bones.len() - 1
                }
            };
            map[i] = Some(idx);
        }
        // Re-target parents of mirrored bones to mirrored parents.
        for i in 0..n {
            if let Some(m) = map[i] {
                if let Some(p) = self.bones[i].parent {
                    self.bones[m].parent = Some(map[p].unwrap_or(p));
                }
            }
        }
    }

    pub fn to_skeleton(&self) -> (Skeleton, HumanoidMap) {
        // Parents before children.
        let n = self.bones.len();
        let mut order = vec![];
        let mut placed = vec![false; n];
        for _ in 0..=n {
            for i in 0..n {
                if !placed[i] && self.bones[i].parent.is_none_or(|p| placed[p] || p >= n) {
                    placed[i] = true;
                    order.push(i);
                }
            }
        }
        let mut new_of = vec![0; n];
        for (k, o) in order.iter().enumerate() {
            new_of[*o] = k;
        }
        let mut skel = Skeleton::default();
        let mut map = HumanoidMap::default();
        for &o in &order {
            let b = &self.bones[o];
            let p = b.parent.filter(|p| *p < n).map(|p| new_of[p]);
            let ph = b.parent.filter(|p| *p < n).map(|p| self.bones[p].head).unwrap_or(Vec3::ZERO);
            skel.bones.push(Bone::new(b.name.clone(), p, Transform::from_translation(b.head - ph)));
            if let Some(r) = b.role {
                map.bones.insert(r, skel.bones.len() - 1);
            }
        }
        skel.recompute_inverse_bind();
        if !map.is_valid() {
            let det = poser_core::humanoid::detect_humanoid(&skel);
            for (h, b) in det.bones {
                map.bones.entry(h).or_insert(b);
            }
        }
        (skel, map)
    }
}

impl PoserApp {
    pub fn start_armature_edit(&mut self, idx: usize) {
        let fig = &self.scene.figures[idx];
        // Existing skeleton -> editable bones (rest pose positions).
        let rest = fig.model.skeleton.rest_world();
        let mut bones: Vec<EditBone> = fig
            .model
            .skeleton
            .bones
            .iter()
            .enumerate()
            .map(|(i, b)| EditBone { name: b.name.clone(), parent: b.parent, head: rest[i].w_axis.truncate(), role: fig.model.humanoid.human_of(i) })
            .collect();
        let prepared = poser_rig::prepare_for_rigging(&fig.model);
        if poser_rig::facing_sign(&fig.model) < 0.0 {
            // Preparation turned the mesh to face +Z: keep the bones consistent.
            for b in &mut bones {
                b.head = Vec3::new(-b.head.x, b.head.y, -b.head.z);
            }
        }
        let id = fig.id;
        self.push_undo("Edit skeleton");
        let prepared = Arc::new(prepared);
        let f = &mut self.scene.figures[idx];
        f.model = prepared.clone();
        f.pose = poser_core::pose::Pose::rest(&prepared.skeleton);
        f.shapes.clear();
        f.morph_names.clear();
        f.morph_weights.clear();
        f.anim = None;
        f.touch();
        self.pins.retain(|(fid, _), _| *fid != id);
        self.armature = Some(ArmatureEdit { figure_id: id, prepared, bones, selected: None, drag: None, computing: None });
        self.sel = Selection { figure: Some(idx), bone: None };
        self.props_tab = PropsTab::Rig;
        self.scene.render.shading = poser_core::scene::Shading::XRay;
        self.status = "Skeleton edit: click a joint to select, drag to move, E = extrude child, Ctrl+click = add bone, X = delete".into();
    }

    pub fn apply_armature(&mut self) {
        let Some(a) = self.armature.as_mut() else { return };
        if a.bones.is_empty() {
            self.status = "Add at least one bone first".into();
            return;
        }
        let (skel, map) = a.to_skeleton();
        let model = a.prepared.clone();
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(poser_rig::bind_skeleton(&model, skel, map));
        });
        a.computing = Some(rx);
        self.status = "Computing skin weights...".into();
    }

    pub fn poll_armature(&mut self) {
        let Some(a) = self.armature.as_ref() else { return };
        let Some(rx) = a.computing.as_ref() else { return };
        let Ok(res) = rx.try_recv() else { return };
        let id = a.figure_id;
        self.armature = None;
        self.scene.render.shading = poser_core::scene::Shading::Solid;
        match (res, self.scene.index_of(id)) {
            (Ok(m), Some(idx)) => {
                let m = Arc::new(m);
                let f = &mut self.scene.figures[idx];
                f.pose = poser_core::pose::Pose::rest(&m.skeleton);
                f.shapes = vec![Default::default(); m.skeleton.len()];
                f.morph_names = m.morph_names();
                f.morph_weights = vec![0.0; f.morph_names.len()];
                f.model = m;
                f.touch();
                self.props_tab = PropsTab::Pose;
                self.status = "Skeleton bound. Pose it with the rings and G/R/S.".into();
            }
            (Err(e), _) => self.status = format!("Binding failed: {e:#}"),
            _ => {}
        }
        self.needs_render = true;
    }

    /// Viewport interaction while editing. Returns true when handled.
    pub fn armature_pointer(&mut self, ui: &egui::Ui, view: &ViewCtx, pointer: Option<Pos2>) -> bool {
        let Some(w) = self.armature.as_ref().and_then(|a| a.world(self)) else { return false };
        let (pressed, down, released, ctrl, keys) = ui.input(|i| {
            (
                i.pointer.button_pressed(egui::PointerButton::Primary),
                i.pointer.button_down(egui::PointerButton::Primary),
                i.pointer.button_released(egui::PointerButton::Primary),
                i.modifiers.ctrl,
                i.events.iter().filter_map(|e| match e {
                    egui::Event::Key { key, pressed: true, repeat: false, .. } => Some(*key),
                    _ => None,
                }).collect::<Vec<_>>(),
            )
        });
        let pick = |a: &ArmatureEdit, p: Pos2| {
            a.bones
                .iter()
                .enumerate()
                .filter_map(|(i, b)| view.project(w.transform_point3(b.head)).map(|s| (i, (s - p).length())))
                .filter(|(_, d)| *d < 10.0)
                .min_by(|x, y| x.1.total_cmp(&y.1))
                .map(|x| x.0)
        };
        let surface = self.pick(view, pointer.unwrap_or_default()).map(|h| h.2);
        let a = self.armature.as_mut().unwrap();
        if !ui.ctx().egui_wants_keyboard_input() {
            for k in keys {
                match k {
                    egui::Key::E => {
                        if let (Some(s), Some(p)) = (a.selected, pointer) {
                            let from = w.transform_point3(a.bones[s].head);
                            let to = w.inverse().transform_point3(view.drag_point(p, from));
                            let name = a.unique_name(&format!("{}.child", a.bones[s].name));
                            a.bones.push(EditBone { name, parent: Some(s), head: to, role: None });
                            a.selected = Some(a.bones.len() - 1);
                            a.drag = a.selected;
                        }
                    }
                    egui::Key::X | egui::Key::Delete => {
                        if let Some(s) = a.selected {
                            a.delete(s);
                        }
                    }
                    _ => {}
                }
            }
        }
        if pressed {
            if let Some(p) = pointer {
                if ctrl {
                    // Add a bone where the cursor hits the mesh (inside the body: push in a bit).
                    let base = a.selected.map(|s| w.transform_point3(a.bones[s].head));
                    let hit = surface.map(|h| h - view.cam.forward() * -0.0).or_else(|| base.map(|b| view.drag_point(p, b)));
                    if let Some(hw) = hit {
                        let head = w.inverse().transform_point3(hw + view.cam.forward() * 0.04);
                        let name = a.unique_name("Bone");
                        a.bones.push(EditBone { name, parent: a.selected, head, role: None });
                        a.selected = Some(a.bones.len() - 1);
                    }
                } else {
                    a.selected = pick(a, p);
                    a.drag = a.selected;
                }
            }
        }
        if let (Some(i), true, Some(p)) = (a.drag, down, pointer) {
            let cur = w.transform_point3(a.bones[i].head);
            a.bones[i].head = w.inverse().transform_point3(view.drag_point(p, cur));
        }
        if released {
            a.drag = None;
        }
        true
    }

    pub fn draw_armature(&self, painter: &egui::Painter, view: &ViewCtx) {
        let Some(a) = self.armature.as_ref() else { return };
        let Some(w) = a.world(self) else { return };
        for (i, b) in a.bones.iter().enumerate() {
            let Some(h) = view.project(w.transform_point3(b.head)) else { continue };
            if let Some(p) = b.parent.and_then(|p| a.bones.get(p)) {
                if let Some(ph) = view.project(w.transform_point3(p.head)) {
                    crate::gizmo::bone(painter, ph, h, if a.selected == Some(i) { crate::theme::ACTIVE } else { Color32::from_gray(230) });
                }
            }
            let sel = a.selected == Some(i);
            painter.circle_filled(h, if sel { 6.0 } else { 4.5 }, if sel { crate::theme::ACTIVE } else if b.role.is_some() { Color32::from_rgb(120, 200, 255) } else { Color32::WHITE });
            painter.circle_stroke(h, if sel { 6.0 } else { 4.5 }, Stroke::new(1.0, Color32::BLACK));
        }
    }

    pub fn armature_panel(&mut self, ui: &mut egui::Ui) {
        let Some(a) = self.armature.as_mut() else { return };
        ui.label(RichText::new("Edit skeleton").strong());
        ui.label(RichText::new("Click a joint to select, drag to move. E: extrude a child to the cursor. Ctrl+click the mesh: add a bone. X: delete.").weak());
        let busy = a.computing.is_some();
        let mut template = false;
        let mut sym = false;
        let mut clear = false;
        ui.horizontal(|ui| {
            template = ui.button("Humanoid template").on_hover_text("Replace with a full humanoid skeleton fitted to the mesh").clicked();
            sym = ui.button("Symmetrize L->R").clicked();
            clear = ui.button("Clear").clicked();
        });
        if template {
            let mk = poser_rig::guess_markers(&a.prepared);
            let (sk, map) = poser_rig::skeleton_from_markers(&mk);
            let w = sk.rest_world();
            a.bones = sk
                .bones
                .iter()
                .enumerate()
                .map(|(i, b)| EditBone { name: b.name.clone(), parent: b.parent, head: w[i].w_axis.truncate(), role: map.human_of(i) })
                .collect();
            a.selected = None;
        }
        if sym {
            let cx = a.bones.iter().find(|b| b.parent.is_none()).map(|b| b.head.x).unwrap_or(0.0);
            a.symmetrize(cx);
        }
        if clear {
            a.bones.clear();
            a.selected = None;
        }
        egui::ScrollArea::vertical().id_salt("armbones").max_height(160.0).show(ui, |ui| {
            for i in 0..a.bones.len() {
                if ui.selectable_label(a.selected == Some(i), &a.bones[i].name).clicked() {
                    a.selected = Some(i);
                }
            }
        });
        if let Some(s) = a.selected.filter(|s| *s < a.bones.len()) {
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Name");
                ui.text_edit_singleline(&mut a.bones[s].name);
            });
            let names: Vec<String> = a.bones.iter().map(|b| b.name.clone()).collect();
            let mut parent = a.bones[s].parent;
            egui::ComboBox::from_label("Parent").selected_text(parent.map(|p| names[p].clone()).unwrap_or_else(|| "(root)".into())).show_ui(ui, |ui| {
                ui.selectable_value(&mut parent, None, "(root)");
                for (i, n) in names.iter().enumerate() {
                    if i != s {
                        ui.selectable_value(&mut parent, Some(i), n);
                    }
                }
            });
            a.bones[s].parent = parent;
            let mut role = a.bones[s].role;
            egui::ComboBox::from_label("Humanoid role").selected_text(role.map(|r| r.name().to_string()).unwrap_or_else(|| "none".into())).show_ui(ui, |ui| {
                ui.selectable_value(&mut role, None, "none");
                for h in HumanBone::ALL {
                    ui.selectable_value(&mut role, Some(*h), h.name());
                }
            });
            a.bones[s].role = role;
            let mut p = a.bones[s].head.to_array();
            ui.horizontal(|ui| {
                for (k, v) in p.iter_mut().enumerate() {
                    ui.add(egui::DragValue::new(v).speed(0.002).prefix(["x ", "y ", "z "][k]));
                }
            });
            a.bones[s].head = Vec3::from_array(p);
        }
        ui.separator();
        let mut apply = false;
        let mut cancel = false;
        ui.horizontal(|ui| {
            apply = ui.add_enabled(!busy, egui::Button::new(RichText::new("Apply (automatic weights)").strong())).clicked();
            if busy {
                ui.spinner();
            }
            cancel = ui.add_enabled(!busy, egui::Button::new("Cancel")).clicked();
        });
        if apply {
            self.apply_armature();
        }
        if cancel {
            self.armature = None;
            self.scene.render.shading = poser_core::scene::Shading::Solid;
            self.undo();
        }
    }
}

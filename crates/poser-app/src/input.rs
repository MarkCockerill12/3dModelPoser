//! Blender-style keymap: navigation, global shortcuts and modal G/R/S transforms.
use crate::app::{Modal, ModalKind, PoserApp};
use crate::gizmo::{ViewCtx, screen_angle, wrap_angle};
use egui::{Key, Modifiers, MouseWheelUnit, PointerButton, Response};
use poser_core::camera::ViewAxis;
use poser_core::glam::{Quat, Vec3};
use poser_core::ik;
use poser_core::skeleton::mat_rotation;

const ORBIT_SPEED: f32 = 0.0085;

impl PoserApp {
    pub fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let files: Vec<std::path::PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());
        for f in files {
            self.import_file(f);
        }
    }

    /// Shortcuts that work anywhere (unless a text field has focus).
    pub fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let cmd = |k: Key| ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, k));
        let cmd_shift = |k: Key| ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, k));
        if cmd_shift(Key::Z) || cmd(Key::Y) {
            self.redo();
        } else if cmd(Key::Z) {
            self.undo();
        }
        if cmd(Key::I) {
            self.import_dialog();
        }
        if cmd(Key::N) {
            self.new_scene();
        }
        if cmd(Key::O) {
            self.open_scene_dialog();
        }
        if cmd(Key::S) {
            self.save_scene_dialog();
        }
        if cmd_shift(Key::V) {
            self.paste_pose(true);
        } else if cmd(Key::V) {
            self.paste_pose(false);
        }
        if cmd(Key::C) {
            self.copy_pose();
        }
        if cmd_shift(Key::S) {
            self.export.open = true;
        }
        if ctx.input(|i| i.key_pressed(Key::F1)) {
            self.help_open = !self.help_open;
        }
    }

    /// Viewport navigation. Returns true if the pointer interaction was consumed.
    pub fn handle_navigation(&mut self, ui: &egui::Ui, resp: &Response, view: &ViewCtx) -> bool {
        let mods = ui.input(|i| i.modifiers);
        let mut consumed = false;
        let nav = |app: &mut Self, delta: egui::Vec2, shift: bool, ctrl: bool| {
            if shift {
                app.cam.pan(delta.x, delta.y, view.size().y);
            } else if ctrl {
                app.cam.zoom((delta.y * 0.01).exp());
            } else {
                app.cam.orbit(delta.x * ORBIT_SPEED, delta.y * ORBIT_SPEED);
            }
            app.cam_anim = None;
        };
        if resp.dragged_by(PointerButton::Middle) {
            nav(self, resp.drag_delta(), mods.shift, mods.ctrl);
            consumed = true;
        }
        if resp.dragged_by(PointerButton::Secondary) && self.modal.is_none() {
            nav(self, resp.drag_delta(), mods.shift, mods.ctrl);
            consumed = true;
        }
        if self.opts.emulate_3button && mods.alt && resp.dragged_by(PointerButton::Primary) && self.drag.is_none() {
            nav(self, resp.drag_delta(), mods.shift, mods.ctrl);
            consumed = true;
        }
        if resp.hovered() || resp.contains_pointer() {
            let events = ui.input(|i| i.events.clone());
            for e in events {
                match e {
                    egui::Event::MouseWheel { unit, delta, modifiers, .. } => {
                        let unit = match self.opts.scroll {
                            crate::app::ScrollMode::AlwaysZoom => MouseWheelUnit::Line,
                            crate::app::ScrollMode::AlwaysOrbit => MouseWheelUnit::Point,
                            crate::app::ScrollMode::Auto => unit,
                        };
                        let delta = if self.opts.scroll == crate::app::ScrollMode::AlwaysZoom && unit == MouseWheelUnit::Line && delta.y.abs() > 3.0 { delta / 50.0 } else { delta };
                        match unit {
                            MouseWheelUnit::Point if !modifiers.ctrl => {
                                // Precision touchpad two-finger scroll: orbit (Shift = pan).
                                if modifiers.shift {
                                    self.cam.pan(-delta.x, -delta.y, view.size().y);
                                } else {
                                    self.cam.orbit(-delta.x * ORBIT_SPEED, -delta.y * ORBIT_SPEED);
                                }
                            }
                            MouseWheelUnit::Point => self.cam.zoom((-delta.y * 0.004).exp()),
                            MouseWheelUnit::Line => self.cam.zoom(0.88f32.powf(delta.y)),
                            MouseWheelUnit::Page => self.cam.zoom(0.6f32.powf(delta.y)),
                        }
                        self.cam_anim = None;
                        consumed = true;
                    }
                    egui::Event::Zoom(z) => {
                        self.cam.zoom(1.0 / z.max(0.01));
                        self.cam_anim = None;
                        consumed = true;
                    }
                    _ => {}
                }
            }
        }
        consumed
    }

    /// Keyboard commands while the pointer is over the viewport.
    pub fn handle_viewport_keys(&mut self, ui: &egui::Ui, view: &ViewCtx, pointer: Option<egui::Pos2>) {
        if ui.ctx().egui_wants_keyboard_input() {
            return;
        }
        let now = ui.input(|i| i.time);
        let (mods, pressed): (Modifiers, Vec<Key>) = ui.input(|i| {
            (
                i.modifiers,
                i.events
                    .iter()
                    .filter_map(|e| match e {
                        egui::Event::Key { key, pressed: true, repeat, .. } if !*repeat || matches!(key, Key::Num2 | Key::Num4 | Key::Num6 | Key::Num8) => Some(*key),
                        _ => None,
                    })
                    .collect(),
            )
        });
        if self.modal.is_some() {
            return;
        }
        for k in pressed {
            let ctrl = mods.ctrl;
            let shift = mods.shift;
            let alt = mods.alt;
            match k {
                // Numpad / number-row views (Blender emulate numpad).
                Key::Num1 => self.set_view(if ctrl { ViewAxis::Back } else { ViewAxis::Front }, now),
                Key::Num3 => self.set_view(if ctrl { ViewAxis::Left } else { ViewAxis::Right }, now),
                Key::Num7 => self.set_view(if ctrl { ViewAxis::Bottom } else { ViewAxis::Top }, now),
                Key::Num5 => self.toggle_ortho(),
                Key::Num9 => {
                    let mut t = self.cam;
                    t.orbit(std::f32::consts::PI, 0.0);
                    self.animate_camera_to(t, now);
                }
                Key::Num4 | Key::Num6 | Key::Num8 | Key::Num2 => {
                    let step = 15f32.to_radians();
                    let (dx, dy) = match k {
                        Key::Num4 => (-step, 0.0),
                        Key::Num6 => (step, 0.0),
                        Key::Num8 => (0.0, -step),
                        _ => (0.0, step),
                    };
                    if ctrl {
                        let px = 40.0;
                        self.cam.pan(dx.signum() * px, dy.signum() * px, view.size().y);
                    } else if shift && dy == 0.0 {
                        self.cam.roll += dx;
                    } else {
                        self.cam.orbit(dx, dy);
                    }
                }
                Key::Num0 if !ctrl => self.frame_selected(),
                Key::Home => self.frame_all(true),
                Key::Period => self.frame_selected(),
                Key::F if !ctrl => self.frame_selected(),
                Key::N if !ctrl => self.show_right = !self.show_right,
                Key::T if !ctrl => self.show_left = !self.show_left,
                Key::Delete | Key::X if !ctrl && !alt && self.armature.is_none() => self.delete_selected(),
                Key::D if shift => self.duplicate_selected(),
                Key::P if !ctrl => {
                    if let (Some(f), Some(b)) = (self.sel.figure, self.sel.bone) {
                        self.toggle_pin(f, b);
                    }
                }
                Key::M if !ctrl && !shift => {
                    if let Some(f) = self.sel.figure {
                        self.flip_pose(f);
                    }
                }
                Key::H if alt => {
                    for f in &mut self.scene.figures {
                        f.visible = true;
                    }
                    self.needs_render = true;
                }
                Key::H => {
                    if let Some(f) = self.sel.figure {
                        self.scene.figures[f].visible = false;
                        self.needs_render = true;
                    }
                }
                Key::A if alt => self.sel = Default::default(),
                Key::Escape => {
                    if self.sel.bone.is_some() {
                        self.sel.bone = None;
                    } else {
                        self.sel = Default::default();
                    }
                }
                Key::R | Key::G | Key::S if alt => {
                    if let (Some(f), Some(b)) = (self.sel.figure, self.sel.bone) {
                        self.push_undo("Clear transform");
                        let fig = &mut self.scene.figures[f];
                        let rest = fig.model.skeleton.bones[b].rest;
                        match k {
                            Key::R => fig.pose.locals[b].rotation = rest.rotation,
                            Key::G => fig.pose.locals[b].translation = rest.translation,
                            _ => fig.shapes[b] = Default::default(),
                        }
                        fig.touch();
                    } else if let Some(f) = self.sel.figure {
                        self.push_undo("Clear transform");
                        let t = &mut self.scene.figures[f].transform;
                        match k {
                            Key::R => t.rotation = Quat::IDENTITY,
                            Key::G => t.translation = Vec3::ZERO,
                            _ => t.scale = Vec3::ONE,
                        }
                    }
                }
                Key::W if !ctrl && !alt => {
                    self.gizmo = crate::app::GizmoMode::Move;
                    self.tool = crate::app::Tool::Pose;
                    self.status = "Move gizmo (W): drag an arrow to move along that axis, the square to move freely".into();
                    self.needs_render = true;
                }
                Key::E if !ctrl && !alt => {
                    self.gizmo = crate::app::GizmoMode::Rotate;
                    self.tool = crate::app::Tool::Pose;
                    self.needs_render = true;
                }
                Key::R | Key::S if !ctrl && self.armature.is_none() && self.rig.is_none() && {
                    let want = if k == Key::R { crate::app::GizmoMode::Rotate } else { crate::app::GizmoMode::Scale };
                    // First press switches gizmo; pressing again starts the free (Blender) modal transform.
                    let switch = self.gizmo != want;
                    if switch {
                        self.gizmo = want;
                        self.tool = crate::app::Tool::Pose;
                        self.needs_render = true;
                        self.status = if want == crate::app::GizmoMode::Rotate {
                            "Rotate gizmo (R/E): drag a ring. Press R again for free rotate (Blender style)".into()
                        } else {
                            "Scale gizmo (S): Y arrow = length, X/Z = thickness, centre = both. Press S again for free scale".into()
                        };
                    }
                    !switch
                } => {
                    if let (Some(f), Some(p)) = (self.sel.figure, pointer) {
                        let kind = if k == Key::R { ModalKind::Rotate } else { ModalKind::Scale };
                        self.begin_modal(kind, f, p, view);
                    }
                }
                Key::G if !ctrl && self.armature.is_none() && self.rig.is_none() => {
                    if let (Some(f), Some(p)) = (self.sel.figure, pointer) {
                        let kind = ModalKind::Grab;
                        self.begin_modal(kind, f, p, view);
                    }
                }
                _ => {}
            }
        }
    }

    pub fn begin_modal(&mut self, kind: ModalKind, figure: usize, mouse: egui::Pos2, view: &ViewCtx) {
        self.push_undo(match kind {
            ModalKind::Grab => "Grab",
            ModalKind::Rotate => "Rotate",
            ModalKind::Scale => "Scale",
        });
        let fig = &self.scene.figures[figure];
        let center = self.modal_center(figure, self.sel.bone);
        let c = view.project(center).unwrap_or(mouse);
        self.modal = Some(Modal {
            kind,
            figure,
            bone: self.sel.bone,
            start_pose: fig.pose.clone(),
            start_shapes: fig.shapes.clone(),
            start_transform: fig.transform,
            start_mouse: mouse,
            accum_angle: 0.0,
            last_angle: screen_angle(c, mouse),
            constraint: None,
            typed: String::new(),
            trackball: false,
        });
        self.status = "G/R/S: move mouse, X/Y/Z to constrain (twice = local), type a value, Enter/LMB confirm, Esc/RMB cancel".into();
    }

    /// World-space pivot of the modal operation.
    pub fn modal_center(&self, figure: usize, bone: Option<usize>) -> Vec3 {
        let fig = &self.scene.figures[figure];
        let Some(c) = self.cache.figs.get(&fig.id) else { return fig.transform.translation };
        match bone {
            Some(b) if b < c.pose.world.len() => c.world.transform_point3(c.pose.world[b].w_axis.truncate()),
            _ => c.world.transform_point3(Vec3::ZERO),
        }
    }

    /// Drive an active modal transform. Returns true while the modal is active.
    pub fn update_modal(&mut self, ui: &egui::Ui, view: &ViewCtx, pointer: Option<egui::Pos2>) -> bool {
        let Some(mut m) = self.modal.take() else { return false };
        let (pressed, typed_chars, lmb, rmb) = ui.input(|i| {
            let keys: Vec<Key> = i.events.iter().filter_map(|e| match e {
                egui::Event::Key { key, pressed: true, repeat: false, .. } => Some(*key),
                _ => None,
            }).collect();
            let text: String = i.events.iter().filter_map(|e| match e {
                egui::Event::Text(t) => Some(t.clone()),
                _ => None,
            }).collect();
            (keys, text, i.pointer.button_pressed(PointerButton::Primary), i.pointer.button_pressed(PointerButton::Secondary))
        });
        let mut confirm = lmb;
        let mut cancel = rmb;
        for k in pressed {
            match k {
                Key::Enter => confirm = true,
                Key::Escape => cancel = true,
                Key::X | Key::Y | Key::Z => {
                    let a = match k {
                        Key::X => 0,
                        Key::Y => 1,
                        _ => 2,
                    };
                    m.constraint = match m.constraint {
                        Some((b, false)) if b == a => Some((a, true)),
                        Some((b, true)) if b == a => None,
                        _ => Some((a, false)),
                    };
                }
                Key::R if m.kind == ModalKind::Rotate => m.trackball = !m.trackball,
                Key::Backspace => {
                    m.typed.pop();
                }
                _ => {}
            }
        }
        for ch in typed_chars.chars() {
            if ch.is_ascii_digit() || ch == '.' || ch == '-' {
                m.typed.push(ch);
            }
        }
        if cancel {
            let fig = &mut self.scene.figures[m.figure];
            fig.pose = m.start_pose.clone();
            fig.shapes = m.start_shapes.clone();
            fig.transform = m.start_transform;
            fig.touch();
            self.undo.discard_last();
            self.status = "Cancelled".into();
            self.needs_render = true;
            return true;
        }
        let Some(p) = pointer else {
            self.modal = Some(m);
            return true;
        };
        let typed: Option<f32> = m.typed.parse().ok();
        // Reset to start, then apply the full transform for this frame.
        {
            let fig = &mut self.scene.figures[m.figure];
            fig.pose = m.start_pose.clone();
            fig.shapes = m.start_shapes.clone();
            fig.transform = m.start_transform;
        }
        let center = {
            let fig = &self.scene.figures[m.figure];
            let w = fig.model.skeleton.evaluate(&m.start_pose.locals, fig.shapes_opt()).world;
            let fw = m.start_transform.to_mat4();
            match m.bone {
                Some(b) => fw.transform_point3(w[b].w_axis.truncate()),
                None => m.start_transform.translation,
            }
        };
        let cs = view.project(center).unwrap_or(m.start_mouse);
        let fig_rot = m.start_transform.rotation;
        let bone_rot_world = m.bone.map(|b| {
            let fig = &self.scene.figures[m.figure];
            let w = fig.model.skeleton.evaluate(&m.start_pose.locals, None).world;
            fig_rot * mat_rotation(&w[b])
        });
        let axis_world = |m: &Modal| -> Option<Vec3> {
            m.constraint.map(|(a, local)| {
                let v = [Vec3::X, Vec3::Y, Vec3::Z][a];
                if local { bone_rot_world.unwrap_or(fig_rot) * v } else { v }
            })
        };
        match m.kind {
            ModalKind::Rotate => {
                let a = screen_angle(cs, p);
                m.accum_angle += wrap_angle(a - m.last_angle);
                m.last_angle = a;
                let to_cam = (view.cam.eye() - center).normalize_or(Vec3::Z);
                let (axis, angle) = if m.trackball {
                    let d = p - m.start_mouse;
                    let ax = (view.cam.up() * d.x + view.cam.right() * d.y).normalize_or(Vec3::Y);
                    (ax, d.length() * 0.01)
                } else {
                    let axis = axis_world(&m).unwrap_or(to_cam);
                    let s = if axis.dot(to_cam) >= 0.0 { -1.0 } else { 1.0 };
                    (axis, typed.map(|t| t.to_radians()).unwrap_or(m.accum_angle * s))
                };
                let q_world = Quat::from_axis_angle(axis, angle);
                match m.bone {
                    Some(b) => {
                        if m.trackball {
                            let fig = &mut self.scene.figures[m.figure];
                            let model = fig.model.clone();
                            let q_model = fig_rot.inverse() * q_world * fig_rot;
                            let w = model.skeleton.evaluate(&fig.pose.locals, fig.shapes_opt()).world;
                            ik::rotate_bone_world(&model.skeleton, &w, &mut fig.pose.locals, b, q_model);
                            self.clamp_bone(m.figure, b);
                        } else {
                            // Single axis: stays on that axis, stopping at the joint limit.
                            self.rotate_on_axis(m.figure, &[b], (fig_rot.inverse() * axis).normalize(), angle);
                        }
                        self.solve_pins(m.figure, Some(b));
                    }
                    None => {
                        let t = &mut self.scene.figures[m.figure].transform;
                        t.rotation = (q_world * m.start_transform.rotation).normalize();
                    }
                }
                self.status = format!("Rotate {:.1}°{}", angle.to_degrees(), constraint_label(&m));
            }
            ModalKind::Grab => {
                let start_w = view.drag_point(m.start_mouse, center);
                let now_w = view.drag_point(p, center);
                let mut d = now_w - start_w;
                if let Some(ax) = axis_world(&m) {
                    d = ax * typed.unwrap_or(d.dot(ax));
                }
                match m.bone {
                    Some(b) => self.grab_bone(m.figure, b, d),
                    None => self.scene.figures[m.figure].transform.translation = m.start_transform.translation + d,
                }
                self.status = format!("Move {:.3} {:.3} {:.3}{}", d.x, d.y, d.z, constraint_label(&m));
            }
            ModalKind::Scale => {
                let d0 = (m.start_mouse - cs).length().max(1.0);
                let d1 = (p - cs).length();
                let s = typed.unwrap_or(d1 / d0).clamp(0.05, 20.0);
                match m.bone {
                    Some(b) => {
                        let fig = &mut self.scene.figures[m.figure];
                        let mut sh = m.start_shapes[b];
                        match m.constraint.map(|c| c.0) {
                            Some(1) => sh.length *= s,
                            Some(_) => sh.girth *= s,
                            None => {
                                sh.length *= s;
                                sh.girth *= s;
                            }
                        }
                        fig.shapes[b] = sh;
                    }
                    None => self.scene.figures[m.figure].transform.scale = m.start_transform.scale * s,
                }
                self.status = format!("Scale {:.3}{}", s, constraint_label(&m));
            }
        }
        self.scene.figures[m.figure].touch();
        self.needs_render = true;
        if confirm {
            self.status = "Done".into();
            if let Some(b) = m.bone {
                // Re-pin a moved pinned effector at its new place.
                let id = self.scene.figures[m.figure].id;
                if self.pins.contains_key(&(id, b)) {
                    let w = self.scene.figures[m.figure].evaluate().world;
                    self.pins.insert((id, b), crate::app::Pin { pos: w[b].w_axis.truncate(), rot: mat_rotation(&w[b]) });
                }
            }
            return true;
        }
        self.modal = Some(m);
        true
    }

    /// Move a bone by a world delta: IK for effectors, translation for roots, FK aim otherwise.
    pub fn grab_bone(&mut self, figure: usize, bone: usize, delta_world: Vec3) {
        let fig = &self.scene.figures[figure];
        let fig_rot = fig.transform.rotation;
        let scale = fig.transform.scale.max_element().max(1e-6);
        let d_model = fig_rot.inverse() * delta_world / scale;
        let model = fig.model.clone();
        let shapes = fig.shapes_opt().map(|s| s.to_vec());
        let w = model.skeleton.evaluate(&fig.pose.locals, shapes.as_deref()).world;
        let pos = w[bone].w_axis.truncate();
        let target = pos + d_model;
        let human = model.humanoid.human_of(bone);
        use poser_core::humanoid::HumanBone as H;
        if let Some((r, mi, e, hint)) = self.limb_chain(figure, bone) {
            let fig = &mut self.scene.figures[figure];
            let keep = mat_rotation(&w[e]);
            ik::two_bone_ik(&model.skeleton, shapes.as_deref(), &mut fig.pose.locals, r, mi, e, target, hint);
            let w2 = model.skeleton.evaluate(&fig.pose.locals, shapes.as_deref()).world;
            let pr = model.skeleton.bones[e].parent.map(|p| mat_rotation(&w2[p])).unwrap_or(Quat::IDENTITY);
            fig.pose.locals[e].rotation = (pr.inverse() * keep).normalize();
        } else if human == Some(H::Hips) || model.skeleton.bones[bone].parent.is_none() {
            // Remember where the feet are so they stay planted.
            let feet: Vec<(usize, poser_core::glam::Vec3, Quat)> = if self.opts.keep_feet_planted {
                [H::LeftFoot, H::RightFoot]
                    .iter()
                    .filter_map(|h| model.humanoid.get(*h))
                    .map(|f| (f, w[f].w_axis.truncate(), mat_rotation(&w[f])))
                    .collect()
            } else {
                vec![]
            };
            let fig = &mut self.scene.figures[figure];
            let pw = match model.skeleton.bones[bone].parent {
                Some(p) => w[p],
                None => model.skeleton.root_transform,
            };
            let local_d = pw.inverse().transform_vector3(d_model);
            fig.pose.locals[bone].translation += local_d;
            for (f, p, r) in feet {
                if let Some((root, mid, end, hint)) = self.limb_chain(figure, f) {
                    let fig = &mut self.scene.figures[figure];
                    ik::two_bone_ik(&model.skeleton, shapes.as_deref(), &mut fig.pose.locals, root, mid, end, p, hint);
                    let w2 = model.skeleton.evaluate(&fig.pose.locals, shapes.as_deref()).world;
                    let pr = model.skeleton.bones[end].parent.map(|x| mat_rotation(&w2[x])).unwrap_or(Quat::IDENTITY);
                    fig.pose.locals[end].rotation = (pr.inverse() * r).normalize();
                }
            }
        } else if self.opts.mode == crate::app::PoseMode::Free {
            let fig = &mut self.scene.figures[figure];
            let pw = match model.skeleton.bones[bone].parent {
                Some(p) => w[p],
                None => model.skeleton.root_transform,
            };
            fig.pose.locals[bone].translation += pw.inverse().transform_vector3(d_model);
        } else {
            let fig = &mut self.scene.figures[figure];
            if let Some(p) = model.skeleton.bones[bone].parent {
                ik::aim_bone(&model.skeleton, shapes.as_deref(), &mut fig.pose.locals, p, target);
            }
        }
        self.solve_pins(figure, Some(bone));
    }
}

fn constraint_label(m: &Modal) -> String {
    match m.constraint {
        Some((a, local)) => format!("  along {}{}", ["X", "Y", "Z"][a], if local { " (local)" } else { " (global)" }),
        None => String::new(),
    }
}

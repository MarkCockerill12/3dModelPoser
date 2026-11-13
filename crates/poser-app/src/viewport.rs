//! 3D viewport: rendering, picking, posing tools and overlays.
use crate::app::{Drag, GizmoMode, IkKind, PoserApp, Selection, Tool};
use crate::gizmo::{AxisHandles, Handle};
use crate::deform::DeformCache;
use crate::gizmo::{self, Ring, RingSet, ViewCtx, screen_angle, wrap_angle};
use crate::theme;
use egui::{Color32, PointerButton, Pos2, Rect, Sense, Stroke};
use egui_wgpu::wgpu;
use poser_core::glam::{Quat, Vec3, Vec4};
use poser_core::humanoid::HumanBone as H;
use poser_core::ik;
use poser_core::picking::{Hit, hit_bone, pick_mesh};
use poser_core::scene::{Scene, Shading};
use poser_core::skeleton::mat_rotation;
use poser_render::{MeshDraw, to_linear};

const AXIS_WIDGET_OFFSET: egui::Vec2 = egui::vec2(-62.0, 62.0);

pub fn build_draws<'a>(scene: &'a Scene, cache: &'a DeformCache, export: bool) -> Vec<MeshDraw<'a>> {
    static ZEROS: std::sync::OnceLock<std::sync::Mutex<Vec<Vec4>>> = std::sync::OnceLock::new();
    let _ = &ZEROS;
    let s = &scene.render;
    let mut out = vec![];
    for (fi, f) in scene.figures.iter().enumerate() {
        if !f.visible {
            continue;
        }
        let Some(c) = cache.figs.get(&f.id) else { continue };
        for (mi, mesh) in f.model.meshes.iter().enumerate() {
            let Some(d) = c.meshes.get(mi) else { continue };
            let mat = &f.model.materials[mesh.material.min(f.model.materials.len() - 1)];
            let tint = f.tint.unwrap_or(s.clay_color);
            let (color, use_texture) = match s.shading {
                Shading::Textured => (mat.base_color, true),
                Shading::Silhouette => (to_linear(s.outline_color), false),
                _ => (to_linear(tint) * poser_core::glam::Vec4::new(mat.clay_shade, mat.clay_shade, mat.clay_shade, 1.0), false),
            };
            out.push(MeshDraw {
                key: (f.id, mi),
                model_key: std::sync::Arc::as_ptr(&f.model) as *const u8 as usize,
                indices: &mesh.indices,
                uvs: &mesh.uvs,
                positions: &d.positions,
                normals: &d.normals,
                deform_dirty: c.deform_dirty || export,
                overlay: if export { None } else if c.overlay_dirty { c.overlays.get(mi).map(|v| v.as_slice()) } else { None },
                material: mat,
                textures: &f.model.textures,
                world: c.world,
                color,
                use_texture,
                object_id: (fi as u32 + 1) * 256 + mi as u32 % 256,
                cast_shadow: true,
            });
        }
    }
    out
}

impl PoserApp {
    pub fn viewport(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let rect = ui.available_rect_before_wrap();
        let resp = ui.interact(rect, ui.scope_id().with("viewport"), Sense::click_and_drag());
        let view = ViewCtx { cam: self.cam, rect };
        let pointer = ui.input(|i| i.pointer.hover_pos()).filter(|p| rect.contains(*p));

        let modal = self.update_modal(ui, &view, pointer);
        if !modal {
            let nav = self.handle_navigation(ui, &resp, &view);
            if resp.hovered() || resp.contains_pointer() {
                self.handle_viewport_keys(ui, &view, pointer);
            }
            if !nav {
                self.handle_pointer_tools(ui, &resp, &view, pointer);
            }
        } else {
            ui.ctx().request_repaint();
        }
        let view = ViewCtx { cam: self.cam, rect };
        let sel = self.sel.figure.map(|f| (f, self.sel.bone));
        let changed = self.cache.update(&self.scene, sel, self.hover);
        self.render_viewport(ui, frame, rect, changed);
        self.draw_overlays(ui, &view);
    }

    fn render_viewport(&mut self, ui: &egui::Ui, frame: &mut eframe::Frame, rect: Rect, changed: bool) {
        let ppp = ui.ctx().pixels_per_point() * self.opts.render_scale;
        let w = ((rect.width() * ppp).round() as u32).max(1);
        let h = ((rect.height() * ppp).round() as u32).max(1);
        let stale = match &self.last_rendered {
            Some((c, s, sz, lights)) => *c != self.cam || *s != self.scene.render || *sz != (w, h) || *lights != self.scene.lights,
            None => true,
        };
        if (self.needs_render || changed || stale || self.tex.is_none()) && self.renderer.is_some() {
            let (center, radius) = self.cache.scene_bounds(&self.scene);
            let draws = build_draws(&self.scene, &self.cache, false);
            let desc = poser_render::FrameDesc {
                camera: &self.cam,
                settings: &self.scene.render,
                lights: &self.scene.lights,
                width: w,
                height: h,
                transparent: self.ref_image.as_ref().is_some_and(|r| r.visible && !r.in_front),
                bounds_center: center,
                bounds_radius: radius,
            };
            let renderer = self.renderer.as_mut().unwrap();
            renderer.use_slot(0);
            let view = renderer.render(&desc, &draws);
            if let Some(rs) = frame.wgpu_render_state() {
                let mut r = rs.renderer.write();
                match self.tex {
                    None => self.tex = Some(r.register_native_texture(&rs.device, view, wgpu::FilterMode::Linear)),
                    Some(id) => r.update_egui_texture_from_wgpu_texture(&rs.device, view, wgpu::FilterMode::Linear, id),
                }
            }
            // Picture-in-picture camera.
            if let (Some(icam), Some(rs)) = (self.inset, frame.wgpu_render_state()) {
                let (iw, ih) = ((w as f32 * 0.3) as u32, (h as f32 * 0.3) as u32);
                let draws = build_draws(&self.scene, &self.cache, false);
                let idesc = poser_render::FrameDesc { camera: &icam, width: iw.max(16), height: ih.max(16), ..desc };
                let renderer = self.renderer.as_mut().unwrap();
                renderer.use_slot(1);
                let v = renderer.render(&idesc, &draws);
                let mut r = rs.renderer.write();
                match self.inset_tex {
                    None => self.inset_tex = Some(r.register_native_texture(&rs.device, v, wgpu::FilterMode::Linear)),
                    Some(id) => r.update_egui_texture_from_wgpu_texture(&rs.device, v, wgpu::FilterMode::Linear, id),
                }
            }
            for c in self.cache.figs.values_mut() {
                c.deform_dirty = false;
                c.overlay_dirty = false;
            }
            self.needs_render = false;
            self.last_rendered = Some((self.cam, self.scene.render.clone(), (w, h), self.scene.lights.clone()));
        }
        let ref_rect = self.ref_image.as_ref().filter(|r| r.visible).map(|r| {
            let fit = (rect.width() / r.size.x).min(rect.height() / r.size.y) * r.scale;
            (r.tex.id(), Rect::from_center_size(rect.center() + r.offset, r.size * fit), r.opacity, r.in_front)
        });
        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        if let Some((id, rr, op, false)) = ref_rect {
            ui.painter().rect_filled(rect, 0.0, Color32::from_gray(40));
            ui.painter().image(id, rr, uv, Color32::from_white_alpha((op * 255.0) as u8));
        }
        if let Some(t) = self.tex {
            ui.painter().image(t, rect, uv, Color32::WHITE);
        }
        if let Some((id, rr, op, true)) = ref_rect {
            ui.painter().image(id, rr, uv, Color32::from_white_alpha((op * 255.0) as u8));
        }
        if let (Some(_), Some(t)) = (self.inset, self.inset_tex) {
            let ir = Rect::from_min_size(rect.right_bottom() - egui::vec2(rect.width() * 0.3 + 12.0, rect.height() * 0.3 + 12.0), egui::vec2(rect.width() * 0.3, rect.height() * 0.3));
            ui.painter().image(t, ir, uv, Color32::WHITE);
            ui.painter().rect_stroke(ir, 3.0, Stroke::new(1.5, Color32::from_gray(200)), egui::StrokeKind::Outside);
        }
    }

    /// Pick the closest figure part under a screen point.
    pub fn pick(&self, view: &ViewCtx, p: Pos2) -> Option<(usize, Option<usize>, Vec3)> {
        let mut best: Option<Hit> = None;
        let local = view.local(p);
        for (fi, f) in self.scene.figures.iter().enumerate() {
            if !f.visible {
                continue;
            }
            let Some(c) = self.cache.figs.get(&f.id) else { continue };
            // Cheap reject with the projected bounding sphere.
            let wc = c.world.transform_point3(c.center);
            if let Some(sc) = view.project(wc) {
                let r_px = c.radius * c.world.x_axis.truncate().length() / view.world_per_point(wc);
                if (sc - p).length() > r_px * 1.6 + 20.0 {
                    continue;
                }
            }
            for (mi, m) in c.meshes.iter().enumerate() {
                pick_mesh(&view.cam, view.size(), local, c.world, &m.positions, &f.model.meshes[mi].indices, fi, mi, &mut best);
            }
        }
        let h = best?;
        let f = &self.scene.figures[h.figure];
        let bone = hit_bone(&f.model.meshes[h.mesh], &h).map(|b| self.posable_bone(h.figure, b));
        Some((h.figure, bone, h.point))
    }

    /// Move / scale handles for the selected bone (or the whole figure).
    pub fn handle_set(&self, view: &ViewCtx) -> Option<AxisHandles> {
        let f = self.sel.figure?;
        let fig = self.scene.figures.get(f)?;
        let c = self.cache.figs.get(&fig.id)?;
        match self.sel.bone {
            Some(b) => {
                let rs = self.ring_set(view)?;
                let _ = (c, b);
                // Move uses world axes (Blender default); scale follows the bone (Y = length).
                let axes = if self.gizmo == GizmoMode::Move { [Vec3::X, Vec3::Y, Vec3::Z] } else { rs.axes };
                Some(AxisHandles::new(rs.center, axes, view))
            }
            None => {
                let r = fig.transform.rotation;
                let axes = if self.opts.local_axes { [r * Vec3::X, r * Vec3::Y, r * Vec3::Z] } else { [Vec3::X, Vec3::Y, Vec3::Z] };
                Some(AxisHandles::new(fig.transform.translation, axes, view))
            }
        }
    }

    pub fn ring_set(&self, view: &ViewCtx) -> Option<RingSet> {
        let f = self.sel.figure?;
        let fig = self.scene.figures.get(f)?;
        let Some(b) = self.sel.bone else {
            // Whole figure: rings around its origin.
            let r = fig.transform.rotation;
            let mut rs = RingSet::for_bone(fig.transform.translation, r, Vec3::Y, false, view);
            if self.opts.local_axes {
                rs.axes = [r * Vec3::X, r * Vec3::Y, r * Vec3::Z];
            }
            return Some(rs);
        };
        let c = self.cache.figs.get(&fig.id)?;
        let bw = *c.pose.world.get(b)?;
        let center = c.world.transform_point3(bw.w_axis.truncate());
        let rot = fig.transform.rotation * mat_rotation(&bw);
        let mut rs = RingSet::for_bone(center, rot, fig.model.skeleton.bones[b].axis, self.opts.local_axes, view);
        if let Some(ax) = self.anatomical_axes(f, b) {
            let fr = fig.transform.rotation;
            rs.axes = ax.map(|a| (fr * a).normalize());
        }
        if let Some((root, mid, end)) = self.swing_chain_for(f, b) {
            let wp = |i: usize| c.world.transform_point3(c.pose.world[i].w_axis.truncate());
            let (a, m, e) = (wp(root), wp(mid), wp(end));
            let axis = (e - a).normalize_or_zero();
            if axis != Vec3::ZERO {
                let foot = a + axis * (m - a).dot(axis);
                let r = (m - foot).length().max(rs.radius * 0.35) * 1.25;
                rs.swing = Some((foot, axis, r));
            }
        }
        Some(rs)
    }

    /// Guided mode: the joint's own anatomical axes (model space) in X/Y/Z order. The twist axis
    /// runs along the bone; the other two are the bend axes the joint limits are defined on, so
    /// e.g. one elbow ring is the hinge, one the forearm twist, and the third is locked.
    pub fn anatomical_axes(&self, f: usize, b: usize) -> Option<[Vec3; 3]> {
        if !self.guided() {
            return None;
        }
        let lim = self.limit_for(f, b)?;
        let fig = self.scene.figures.get(f)?;
        let m = &fig.model;
        let hm = &m.humanoid;
        if hm.tpose_world.len() != m.skeleton.len() {
            return None;
        }
        let c = self.cache.figs.get(&fig.id)?;
        let nb = mat_rotation(c.pose.world.get(b)?) * hm.tpose_world[b].inverse();
        let np = match m.skeleton.bones[b].parent {
            Some(p) => mat_rotation(&c.pose.world[p]) * hm.tpose_world[p].inverse(),
            None => Quat::IDENTITY,
        };
        let e = [Vec3::X, Vec3::Y, Vec3::Z];
        Some(std::array::from_fn(|i| if i == lim.twist as usize { nb * e[i] } else { np * e[i] }.normalize()))
    }

    /// (root, mid, end) of the arm/leg containing `b` (for the swing donut).
    pub fn swing_chain_for(&self, f: usize, b: usize) -> Option<(usize, usize, usize)> {
        let hm = &self.scene.figures[f].model.humanoid;
        let h = hm.human_of(b)?;
        let chain = match h {
            H::LeftLowerArm | H::LeftHand => [H::LeftUpperArm, H::LeftLowerArm, H::LeftHand],
            H::RightLowerArm | H::RightHand => [H::RightUpperArm, H::RightLowerArm, H::RightHand],
            H::LeftLowerLeg | H::LeftFoot => [H::LeftUpperLeg, H::LeftLowerLeg, H::LeftFoot],
            H::RightLowerLeg | H::RightFoot => [H::RightUpperLeg, H::RightLowerLeg, H::RightFoot],
            _ => return None,
        };
        Some((hm.get(chain[0])?, hm.get(chain[1])?, hm.get(chain[2])?))
    }

    /// Spine bones that share a bend when one of them is rotated.
    fn spine_chain(&self, f: usize, b: usize) -> Vec<usize> {
        let hm = &self.scene.figures[f].model.humanoid;
        let spine = [H::Spine, H::Chest, H::UpperChest];
        if !hm.human_of(b).is_some_and(|h| spine.contains(&h)) {
            return vec![];
        }
        spine.iter().filter_map(|h| hm.get(*h)).collect()
    }

    fn axis_widget_center(&self, rect: Rect) -> Pos2 {
        rect.right_top() + AXIS_WIDGET_OFFSET
    }

    fn handle_pointer_tools(&mut self, ui: &egui::Ui, resp: &egui::Response, view: &ViewCtx, pointer: Option<Pos2>) {
        let (pressed, down, released, mods, latest) = ui.input(|i| {
            (
                i.pointer.button_pressed(PointerButton::Primary),
                i.pointer.button_down(PointerButton::Primary),
                i.pointer.button_released(PointerButton::Primary),
                i.modifiers,
                i.pointer.latest_pos(),
            )
        });
        self.ctrl_down = mods.ctrl;
        let alt_nav = self.opts.emulate_3button && mods.alt;
        if self.armature.is_some() {
            if resp.contains_pointer() || self.armature.as_ref().is_some_and(|a| a.drag.is_some()) {
                self.armature_pointer(ui, view, pointer.or(latest));
            }
            return;
        }
        if self.rig.is_some() {
            self.rig_pointer(view, pointer, latest, pressed && !alt_nav && resp.contains_pointer(), down, released);
            return;
        }
        if self.drag.is_none() {
            if let Some(p) = pointer {
                let gizmo_on = self.tool == Tool::Pose && self.sel.figure.is_some();
                let ring = if gizmo_on && self.gizmo == GizmoMode::Rotate && self.opts.show_rings { self.ring_set(view).and_then(|r| r.hit(view, p)) } else { None };
                let handle = if gizmo_on && self.gizmo != GizmoMode::Rotate { self.handle_set(view).and_then(|h| h.hit(view, p)) } else { None };
                let hover = if ring.is_some() || handle.is_some() { None } else { self.pick(view, p).map(|(f, b, _)| (f, b)) };
                if ring != self.hover_ring || hover != self.hover || handle != self.hover_handle {
                    self.hover_ring = ring;
                    self.hover_handle = handle;
                    self.hover = hover;
                    self.needs_render = true;
                }
            } else if self.hover.is_some() || self.hover_ring.is_some() || self.hover_handle.is_some() {
                self.hover = None;
                self.hover_ring = None;
                self.hover_handle = None;
                self.needs_render = true;
            }
        }
        if pressed && resp.contains_pointer() && !alt_nav {
            if let Some(p) = pointer {
                self.on_press(view, p, mods.shift);
            }
        }
        if down && self.drag.is_some() {
            if let Some(p) = latest {
                self.on_drag(view, p, mods.shift);
                ui.ctx().request_repaint();
            }
        }
        if released {
            if let Some(Drag::NavGizmo { start, moved: false, .. }) = self.drag {
                // A click (not a drag) on an axis bubble snaps to that view.
                let wc = self.axis_widget_center(view.rect);
                if let Some(h) = gizmo::axis_points(wc, &self.cam).into_iter().filter(|h| (h.1 - start).length() < 11.0).last() {
                    let now = self.last_time;
                    self.set_view(h.4, now);
                }
            }
            if self.drag.take().is_some() {
                self.pending_undo_done = false;
            }
        }
    }

    fn rig_world(&self) -> Option<poser_core::glam::Mat4> {
        let r = self.rig.as_ref()?;
        let fig = self.scene.figures.get(self.scene.index_of(r.figure_id)?)?;
        Some(self.cache.figs.get(&fig.id)?.world)
    }

    fn rig_pointer(&mut self, view: &ViewCtx, pointer: Option<Pos2>, latest: Option<Pos2>, pressed: bool, down: bool, released: bool) {
        let Some(w) = self.rig_world() else { return };
        let Some(r) = self.rig.as_mut() else { return };
        if r.drag.is_none() {
            r.hover = pointer.and_then(|p| {
                (0..8)
                    .filter_map(|i| view.project(w.transform_point3(r.markers.get(i))).map(|s| (i, (s - p).length())))
                    .filter(|(_, d)| *d < 12.0)
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|x| x.0)
            });
            if pressed {
                r.drag = r.hover;
            }
        }
        if let (Some(i), true, Some(p)) = (r.drag, down, latest) {
            let cur = w.transform_point3(r.markers.get(i));
            let np = w.inverse().transform_point3(view.drag_point(p, cur));
            r.markers.set(i, np);
            if r.symmetric {
                let cx = (r.markers.chin.x + r.markers.groin.x) * 0.5;
                r.markers.symmetrize_from(i, cx);
            }
        }
        if released {
            r.drag = None;
        }
    }

    fn draw_rig_markers(&self, painter: &egui::Painter, view: &ViewCtx) {
        let (Some(r), Some(w)) = (self.rig.as_ref(), self.rig_world()) else { return };
        let cols = [
            Color32::from_rgb(255, 210, 60),
            Color32::from_rgb(80, 200, 255),
            Color32::from_rgb(255, 110, 110),
            Color32::from_rgb(80, 200, 255),
            Color32::from_rgb(255, 110, 110),
            Color32::from_rgb(80, 200, 255),
            Color32::from_rgb(255, 110, 110),
            Color32::from_rgb(170, 255, 120),
        ];
        // Preview of the skeleton that will be built.
        let (sk, _) = poser_rig::skeleton_from_markers(&r.markers);
        let sw = sk.rest_world();
        for (i, b) in sk.bones.iter().enumerate() {
            if let Some(pa) = b.parent {
                let a = view.project(w.transform_point3(sw[pa].w_axis.truncate()));
                let c = view.project(w.transform_point3(sw[i].w_axis.truncate()));
                if let (Some(a), Some(c)) = (a, c) {
                    painter.line_segment([a, c], Stroke::new(2.0, Color32::from_rgba_unmultiplied(255, 255, 255, 150)));
                }
            }
        }
        for i in 0..8 {
            let Some(p) = view.project(w.transform_point3(r.markers.get(i))) else { continue };
            let hot = r.hover == Some(i) || r.drag == Some(i);
            let rad = if hot { 9.0 } else { 7.0 };
            painter.circle_filled(p, rad, cols[i]);
            painter.circle_stroke(p, rad, Stroke::new(1.5, Color32::BLACK));
            painter.text(p + egui::vec2(11.0, -9.0), egui::Align2::LEFT_CENTER, poser_rig::MARKER_NAMES[i], egui::FontId::proportional(12.0), Color32::WHITE);
        }
    }

    fn on_press(&mut self, view: &ViewCtx, p: Pos2, shift: bool) {
        self.drag_translate = self.ctrl_down;
        self.press_pos = p;
        // Navigation gizmo (click an axis bubble).
        let wc = self.axis_widget_center(view.rect);
        if (p - wc).length() < 50.0 {
            // Drag the navigation gizmo to orbit freely; click a bubble to snap.
            self.drag = Some(Drag::NavGizmo { start: p, last: p, moved: false });
            return;
        }
        self.pending_undo_done = false;
        if let Some(handle) = self.hover_handle {
            self.drag = Some(Drag::Handle { handle, mode: self.gizmo, last: p });
            return;
        }
        if self.tool == Tool::Pose {
            if let (Some(ring), Some(rs)) = (self.hover_ring, self.ring_set(view)) {
                let axis_world = rs.axis(ring, view);
                let c = view.project(rs.center).unwrap_or(p);
                // Tangent on screen at the grab point (for edge-on rings).
                let (o, d) = view.ray(p);
                let n = axis_world;
                let denom = d.dot(n);
                let hit = if denom.abs() > 1e-5 { o + d * ((rs.center - o).dot(n) / denom) } else { rs.center + view.cam.right() * rs.radius };
                let radial = (hit - rs.center).normalize_or(view.cam.right());
                let tan_w = n.cross(radial);
                let tangent = match (view.project(rs.center + radial * rs.radius), view.project(rs.center + radial * rs.radius + tan_w * rs.radius * 0.1)) {
                    (Some(a), Some(b)) => Some((b - a).normalized()),
                    _ => None,
                };
                self.drag = Some(Drag::Ring { ring, axis_world, last_angle: screen_angle(c, p), tangent, last_pos: p });
                return;
            }
        }
        match self.pick(view, p) {
            Some((f, bone, point)) => {
                let new_sel = Selection { figure: Some(f), bone: if self.tool == Tool::Pose { bone } else { None } };
                if new_sel != self.sel {
                    self.sel = new_sel;
                    self.needs_render = true;
                }
                self.drag = match self.tool {
                    Tool::Pose => bone.and_then(|b| self.begin_ik(f, b, point)),
                    Tool::Move => {
                        let t = self.scene.figures[f].transform.translation;
                        Some(Drag::MoveFigure { figure: f, grab_offset: t - point, plane_point: point })
                    }
                    Tool::Rotate => Some(Drag::RotateFigure { figure: f }),
                };
                let _ = shift;
            }
            None => {
                if self.sel != Selection::default() {
                    self.sel = Selection::default();
                    self.needs_render = true;
                }
                if self.opts.drag_empty_orbits {
                    self.drag = Some(Drag::Orbit { last: p, pan: shift });
                }
            }
        }
    }

    fn begin_ik(&self, f: usize, bone: usize, grab: Vec3) -> Option<Drag> {
        let fig = &self.scene.figures[f];
        let c = self.cache.figs.get(&fig.id)?;
        let m = &fig.model;
        let hm = &m.humanoid;
        let h = hm.human_of(bone);
        let get = |b: H| hm.get(b);
        let (kind, effector) = match h {
            Some(H::LeftHand | H::RightHand | H::LeftFoot | H::RightFoot) => {
                let (root, mid, end, hint) = self.limb_chain(f, bone)?;
                (IkKind::Limb { root, mid, end, hint }, bone)
            }
            Some(H::LeftLowerArm) => (IkKind::Swing { root: get(H::LeftUpperArm)?, end: get(H::LeftHand)? }, bone),
            Some(H::RightLowerArm) => (IkKind::Swing { root: get(H::RightUpperArm)?, end: get(H::RightHand)? }, bone),
            Some(H::LeftLowerLeg) => (IkKind::Swing { root: get(H::LeftUpperLeg)?, end: get(H::LeftFoot)? }, bone),
            Some(H::RightLowerLeg) => (IkKind::Swing { root: get(H::RightUpperLeg)?, end: get(H::RightFoot)? }, bone),
            Some(H::Hips) => (IkKind::Hips, bone),
            Some(H::Head | H::Neck | H::Jaw) => {
                let chain: Vec<usize> = [H::Chest, H::UpperChest, H::Neck].iter().filter_map(|b| get(*b)).collect();
                (IkKind::Ccd { chain, effector: get(H::Head)? }, get(H::Head)?)
            }
            Some(H::Spine | H::Chest | H::UpperChest) => {
                let chain: Vec<usize> = [H::Spine, H::Chest, H::UpperChest].iter().filter_map(|b| get(*b)).collect();
                let eff = get(H::Neck).or(get(H::Head))?;
                (IkKind::Ccd { chain, effector: eff }, eff)
            }
            Some(fh) if fh.is_finger() && !(self.opts.mode == crate::app::PoseMode::Free && self.drag_translate) => {
                // Drag a finger: the whole finger curls towards the cursor.
                let (side, finger, _) = fh.finger_info()?;
                let segs: Vec<usize> = (0..3).filter_map(|i| poser_core::humanoid::finger_bone(side, finger, i).and_then(|x| hm.get(x))).collect();
                let last = *segs.last()?;
                let tip = m.skeleton.bones[last].children.first().copied();
                let (chain, eff) = match tip {
                    Some(t) => (segs.clone(), t),
                    None => (segs[..segs.len() - 1].to_vec(), last),
                };
                (IkKind::Ccd { chain, effector: eff }, eff)
            }
            _ if self.opts.mode == crate::app::PoseMode::Free && self.drag_translate => (IkKind::Translate { bone }, bone),
            _ => {
                // FK aim: the bone's tail follows the cursor.
                let child = m.skeleton.bones[bone].children.first().copied();
                (IkKind::Aim { bone }, child.unwrap_or(bone))
            }
        };
        let start_target = c.pose.world.get(effector)?.w_axis.truncate();
        let plane_point = c.world.transform_point3(start_target);
        let _ = grab;
        Some(Drag::Ik { figure: f, kind, effector, start: fig.pose.locals.clone(), start_target, plane_point, press: self.press_pos, started: false })
    }

    /// Debug (POSER_DRAGTEST): simulate drags of the main handles by model-space offsets through
    /// the real IK path and return the resulting poses for a contact sheet.
    pub fn drag_test_poses(&mut self, f: usize) -> Vec<poser_core::pose::HumanPose> {
        let base = self.scene.figures[f].pose.locals.clone();
        if std::env::var_os("POSER_BONES").is_some() {
            let m = &self.scene.figures[f].model;
            for (i, b) in m.skeleton.bones.iter().enumerate() {
                let p = b.parent.map(|p| m.skeleton.bones[p].name.clone()).unwrap_or_default();
                println!("bone {i} {} <- {p} {:?}", b.name, m.humanoid.human_of(i));
            }
        }
        if std::env::var_os("POSER_NOCOL").is_some() {
            self.opts.collisions = false;
            self.opts.floor = false;
        }
        let tests: &[(H, Vec3)] = &[
            (H::LeftHand, Vec3::new(0.0, 0.7, 0.0)),
            (H::LeftHand, Vec3::new(-0.3, 0.2, 0.6)),
            (H::LeftHand, Vec3::new(-0.8, 0.3, 0.3)),
            (H::LeftHand, Vec3::new(0.1, -0.3, -0.5)),
            (H::LeftHand, Vec3::new(1.0, 1.0, 0.8)),
            (H::LeftHand, Vec3::new(-0.3, 0.6, -0.2)),
            (H::LeftHand, Vec3::new(-0.5, 0.45, 0.25)),
            (H::LeftFoot, Vec3::new(0.0, 0.5, 0.5)),
            (H::LeftFoot, Vec3::new(0.0, 1.2, 0.4)),
            (H::LeftFoot, Vec3::new(0.0, 0.3, -0.6)),
            (H::LeftFoot, Vec3::new(-0.6, 0.3, 0.0)),
            (H::LeftLowerArm, Vec3::new(0.0, 0.3, 0.3)),
            (H::LeftLowerLeg, Vec3::new(0.3, 0.0, 0.3)),
            (H::Head, Vec3::new(0.0, -0.3, 0.4)),
            (H::Chest, Vec3::new(0.3, -0.2, 0.3)),
            (H::Hips, Vec3::new(0.0, -0.4, 0.0)),
            // Collision cases: through the chest, through the other leg, through the floor.
            (H::LeftHand, Vec3::new(-0.45, 0.45, -0.05)),
            (H::LeftHand, Vec3::new(-0.35, 0.1, 0.0)),
            (H::LeftHand, Vec3::new(-0.3, 0.38, 0.0)),
            (H::LeftHand, Vec3::new(-0.28, 0.2, -0.04)),
            (H::LeftHand, Vec3::new(-0.3, 0.75, -0.05)),
            (H::LeftFoot, Vec3::new(-0.45, 0.15, 0.0)),
            (H::LeftFoot, Vec3::new(0.0, -0.3, 0.2)),
        ];
        // Ring rotation check: each axis step must rotate the bone about that axis only.
        for hb in [H::LeftUpperArm, H::LeftLowerArm, H::LeftUpperLeg, H::Chest, H::Neck] {
            let Some(b) = self.scene.figures[f].model.humanoid.get(hb) else { continue };
            for ring in 0..3 {
                self.scene.figures[f].pose.locals = base.clone();
                // Start from a bent pose so local axes are not aligned with the T-pose.
                if let Some(ua) = self.scene.figures[f].model.humanoid.get(H::LeftUpperArm) {
                    let q = self.scene.figures[f].pose.locals[ua].rotation;
                    self.scene.figures[f].pose.locals[ua].rotation = q * Quat::from_rotation_z(0.6) * Quat::from_rotation_x(0.4);
                }
                let sk = self.scene.figures[f].model.skeleton.clone();
                let w0 = sk.evaluate(&self.scene.figures[f].pose.locals, None).world;
                let r0 = mat_rotation(&w0[b]);
                self.scene.figures[f].touch();
                self.cache.update(&self.scene, None, None);
                let Some(axes) = self.anatomical_axes(f, b) else { continue };
                let ax = axes[ring];
                let mut total = 0.0;
                for _ in 0..40 {
                    self.rotate_on_axis(f, &[b], ax, 0.03);
                    total += 0.03;
                }
                let w1 = sk.evaluate(&self.scene.figures[f].pose.locals, None).world;
                let (d_axis, d_ang) = (mat_rotation(&w1[b]) * r0.inverse()).to_axis_angle();
                let off = if d_ang > 1e-3 { d_axis.angle_between(ax).min(d_axis.angle_between(-ax)).to_degrees() } else { 0.0 };
                println!("ring {hb:?} axis {ring}: turned {:.1} of {:.1} deg, off-axis {:.2} deg", d_ang.to_degrees(), (total as f32).to_degrees(), off);
            }
        }
        let mut out = vec![];
        for (hb, off) in tests {
            self.scene.figures[f].pose.locals = base.clone();
            self.scene.figures[f].touch();
            let start = base.clone();
            self.cache.update(&self.scene, None, None);
            let Some(bone) = self.scene.figures[f].model.humanoid.get(*hb) else { continue };
            let Some(Drag::Ik { kind, effector, start_target, .. }) = self.begin_ik(f, bone, Vec3::ZERO) else { continue };
            // Step the drag like the mouse does, re-solving from the start pose each frame.
            for k in 1..=20 {
                self.solve_drag(f, &kind, effector, start_target + *off * (k as f32 / 20.0), &start);
            }
            let fig = &self.scene.figures[f];
            let mut p = poser_core::pose::extract_human_pose(&fig.model.skeleton, &fig.model.humanoid, &fig.pose.locals);
            p.name = format!("{hb:?} {off:?}");
            let pen = Self::collision_root(&kind).map(|r| self.raw_penetration(f, r)).unwrap_or((0.0, 0));
            let reached = (self.scene.figures[f].evaluate().world[effector].w_axis.truncate() - (start_target + *off)).length();
            println!("drag {hb:?} {off:?}: penetration {:.3} m ({} contacts), miss {reached:.3} m", pen.0, pen.1);
            out.push(p);
        }
        self.scene.figures[f].pose.locals = base;
        self.scene.figures[f].touch();
        out
    }

    /// Rotate `bones` (split evenly) about one model-space axis, and nothing else: in Guided mode
    /// the rotation stops at the joint limit instead of being clamped sideways onto other axes.
    pub(crate) fn rotate_on_axis(&mut self, f: usize, bones: &[usize], axis: Vec3, phi: f32) {
        let model = self.scene.figures[f].model.clone();
        let sk = &model.skeleton;
        let before: Vec<_> = bones.iter().map(|b| self.scene.figures[f].pose.locals[*b]).collect();
        let worst = |app: &Self| bones.iter().map(|b| app.limit_violation(f, *b)).fold(0.0f32, f32::max);
        let v0 = worst(self);
        let root = bones[0];
        let p0 = self.penetration(f, root);
        let ok = |app: &mut Self| worst(app) <= v0 + 1e-3 && app.penetration(f, root) <= p0 + 1e-4;
        let apply = |app: &mut Self, t: f32| {
            let fig = &mut app.scene.figures[f];
            for (b, l) in bones.iter().zip(&before) {
                fig.pose.locals[*b] = *l;
            }
            let step = Quat::from_axis_angle(axis, phi * t / bones.len() as f32);
            for b in bones.iter().rev() {
                let w = sk.evaluate(&fig.pose.locals, fig.shapes_opt()).world;
                ik::rotate_bone_world(sk, &w, &mut fig.pose.locals, *b, step);
            }
        };
        apply(self, 1.0);
        if ok(self) {
            return;
        }
        // Find how far along this axis we can go before leaving the limits.
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        for _ in 0..10 {
            let mid = (lo + hi) * 0.5;
            apply(self, mid);
            if ok(self) { lo = mid } else { hi = mid }
        }
        apply(self, lo);
    }

    /// One IK drag step from the drag's start pose: solve, then (Guided) push the limb out of
    /// the body and floor so it slides along instead of passing through.
    fn solve_drag(&mut self, f: usize, kind: &IkKind, effector: usize, target: Vec3, start: &[poser_core::Transform]) {
        self.scene.figures[f].pose.locals = start.to_vec();
        let root = Self::collision_root(kind);
        let base = root.map(|r| self.contact_baseline(f, r)).unwrap_or_default();
        let from = self.scene.figures[f].evaluate().world.get(effector).map(|m| m.w_axis.truncate()).unwrap_or(target);
        let solve = |app: &mut Self, t: Vec3| {
            app.scene.figures[f].pose.locals = start.to_vec();
            app.apply_ik(f, kind, effector, t);
            match root {
                Some(r) => {
                    app.resolve_collisions(f, r, &base);
                    app.collision_excess(f, r, &base)
                }
                None => 0.0,
            }
        };
        if solve(self, target) <= 0.01 {
            return;
        }
        // Still inside the body (the joints can't slide round it): stop at the furthest point
        // along the drag that stays outside.
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        for _ in 0..7 {
            let mid = (lo + hi) * 0.5;
            if solve(self, from.lerp(target, mid)) <= 0.01 { lo = mid } else { hi = mid }
        }
        solve(self, from.lerp(target, lo));
    }

    fn ensure_undo(&mut self, label: &str) {
        if !self.pending_undo_done {
            self.push_undo(label);
            self.pending_undo_done = true;
        }
    }

    fn on_drag(&mut self, view: &ViewCtx, p: Pos2, shift: bool) {
        let Some(drag) = self.drag.clone() else { return };
        match drag {
            Drag::NavGizmo { start, last, moved } => {
                let d = p - last;
                self.cam.orbit(d.x * 0.012, d.y * 0.012);
                self.cam_anim = None;
                self.drag = Some(Drag::NavGizmo { start, last: p, moved: moved || (p - start).length() > 3.0 });
            }
            Drag::Handle { handle, mode, last } => {
                let Some(f) = self.sel.figure else { return };
                let Some(hs) = self.handle_set(view) else { return };
                let d = p - last;
                self.drag = Some(Drag::Handle { handle, mode, last: p });
                if d.length_sq() < 1e-6 {
                    return;
                }
                match mode {
                    GizmoMode::Move => {
                        self.ensure_undo("Move");
                        let delta = if handle == Handle::Center {
                            view.drag_point(p, hs.center) - view.drag_point(last, hs.center)
                        } else {
                            hs.axis(handle) * hs.axis_delta(view, handle, d)
                        };
                        match self.sel.bone {
                            Some(b) => {
                                self.grab_bone(f, b, delta);
                                self.scene.figures[f].touch();
                                self.stop_anim(f);
                            }
                            None => {
                                self.scene.figures[f].transform.translation += delta;
                                self.needs_render = true;
                            }
                        }
                    }
                    GizmoMode::Scale => {
                        self.ensure_undo("Scale");
                        let k = if handle == Handle::Center {
                            (1.0 - d.y * 0.006 + d.x * 0.006).clamp(0.8, 1.25)
                        } else {
                            (1.0 + hs.axis_delta(view, handle, d) / hs.len).clamp(0.8, 1.25)
                        };
                        match self.sel.bone {
                            Some(b) => {
                                let fig = &mut self.scene.figures[f];
                                let sh = &mut fig.shapes[b];
                                match handle {
                                    Handle::Y => sh.length = (sh.length * k).clamp(0.1, 5.0),
                                    Handle::X | Handle::Z => sh.girth = (sh.girth * k).clamp(0.1, 5.0),
                                    Handle::Center => {
                                        sh.length = (sh.length * k).clamp(0.1, 5.0);
                                        sh.girth = (sh.girth * k).clamp(0.1, 5.0);
                                    }
                                }
                                fig.touch();
                            }
                            None => {
                                let t = &mut self.scene.figures[f].transform;
                                t.scale = (t.scale * k).clamp(Vec3::splat(0.02), Vec3::splat(50.0));
                                self.needs_render = true;
                            }
                        }
                    }
                    GizmoMode::Rotate => {}
                }
            }
            Drag::Ring { ring, axis_world, last_angle, tangent, last_pos } => {
                let Some(f) = self.sel.figure else { return };
                let Some(rs) = self.ring_set(view) else { return };
                let Some(b) = self.sel.bone else {
                    // Rotate the whole figure.
                    let c = view.project(rs.center).unwrap_or(p);
                    let to_cam = (view.cam.eye() - rs.center).normalize_or(Vec3::Z);
                    let facing = axis_world.dot(to_cam);
                    let a = screen_angle(c, p);
                    let phi = if facing.abs() > 0.25 || ring == Ring::View {
                        -wrap_angle(a - last_angle) * facing.signum()
                    } else if let Some(t) = tangent {
                        (p - last_pos).dot(t) / gizmo::RING_PX
                    } else {
                        0.0
                    };
                    if phi.abs() > 1e-6 {
                        self.ensure_undo("Rotate figure");
                        let t = &mut self.scene.figures[f].transform;
                        t.rotation = (Quat::from_axis_angle(axis_world, phi) * t.rotation).normalize();
                        self.needs_render = true;
                    }
                    self.drag = Some(Drag::Ring { ring, axis_world, last_angle: a, tangent, last_pos: p });
                    return;
                };
                let rc = if ring == Ring::Swing { rs.swing.map(|s| s.0).unwrap_or(rs.center) } else { rs.center };
                let c = view.project(rc).unwrap_or(p);
                let to_cam = (view.cam.eye() - rc).normalize_or(Vec3::Z);
                let facing = axis_world.dot(to_cam);
                let a = screen_angle(c, p);
                let phi = if facing.abs() > 0.25 || ring == Ring::View {
                    -wrap_angle(a - last_angle) * facing.signum()
                } else if let Some(t) = tangent {
                    (p - last_pos).dot(t) / gizmo::RING_PX
                } else {
                    0.0
                } * if shift { 0.3 } else { 1.0 };
                if phi.abs() > 1e-6 && ring == Ring::Swing {
                    self.ensure_undo("Swing limb");
                    if let Some((root, _, end)) = self.swing_chain_for(f, b) {
                        let fig = &mut self.scene.figures[f];
                        let model = fig.model.clone();
                        let shapes = fig.shapes_opt().map(|s| s.to_vec());
                        ik::swing_chain(&model.skeleton, shapes.as_deref(), &mut fig.pose.locals, root, end, true, phi);
                        fig.touch();
                        self.stop_anim(f);
                    }
                } else if phi.abs() > 1e-6 && self.opts.spine_spread && self.spine_chain(f, b).len() > 1 {
                    self.ensure_undo("Bend spine");
                    let chain = self.spine_chain(f, b);
                    let ax = (self.scene.figures[f].transform.rotation.inverse() * axis_world).normalize();
                    self.rotate_on_axis(f, &chain, ax, phi);
                    self.solve_pins(f, Some(b));
                    self.scene.figures[f].touch();
                    self.stop_anim(f);
                } else if phi.abs() > 1e-6 {
                    self.ensure_undo("Rotate bone");
                    let ax = (self.scene.figures[f].transform.rotation.inverse() * axis_world).normalize();
                    self.rotate_on_axis(f, &[b], ax, phi);
                    // The arm keeps exactly the turn you asked for; the shoulder girdle follows it.
                    self.shoulder_rhythm(f);
                    self.solve_pins(f, Some(b));
                    self.scene.figures[f].touch();
                    self.stop_anim(f);
                }
                // Keep using the ring's initial axis for the whole drag (stable feel).
                self.drag = Some(Drag::Ring { ring, axis_world, last_angle: a, tangent, last_pos: p });
            }
            Drag::Ik { figure, kind, effector, start, start_target, plane_point, press, started } => {
                // Click selects; posing only starts once the pointer really moves.
                if !started && (p - press).length() < 5.0 {
                    return;
                }
                let fine = if shift { 0.3 } else { 1.0 };
                let dw = (view.drag_point(p, plane_point) - view.drag_point(press, plane_point)) * fine;
                let fig = &self.scene.figures[figure];
                let Some(c) = self.cache.figs.get(&fig.id) else { return };
                let target = start_target + c.world.inverse().transform_vector3(dw);
                self.ensure_undo("Pose");
                // Re-solve from the pose at the start of the drag (smooth + reversible).
                self.solve_drag(figure, &kind, effector, target, &start);
                self.scene.figures[figure].touch();
                self.stop_anim(figure);
                self.drag = Some(Drag::Ik { figure, kind, effector, start, start_target, plane_point, press, started: true });
            }
            Drag::MoveFigure { figure, grab_offset, plane_point } => {
                self.ensure_undo("Move");
                let target = if shift {
                    view.drag_point(p, plane_point)
                } else {
                    // Slide on the horizontal plane through the grab point.
                    let (o, d) = view.ray(p);
                    if d.y.abs() > 1e-4 {
                        let t = (plane_point.y - o.y) / d.y;
                        if t > 0.0 { o + d * t } else { view.drag_point(p, plane_point) }
                    } else {
                        view.drag_point(p, plane_point)
                    }
                };
                self.scene.figures[figure].transform.translation = target + grab_offset;
                self.needs_render = true;
            }
            Drag::Orbit { last, pan } => {
                let d = p - last;
                if pan {
                    self.cam.pan(d.x, d.y, view.size().y);
                } else {
                    self.cam.orbit(d.x * 0.0085, d.y * 0.0085);
                }
                self.cam_anim = None;
                self.drag = Some(Drag::Orbit { last: p, pan });
            }
            Drag::RotateFigure { figure } => {
                self.ensure_undo("Rotate figure");
                let dx = view.local(p).x;
                let prev = self.drag_prev_x.unwrap_or(dx);
                self.drag_prev_x = Some(dx);
                let t = &mut self.scene.figures[figure].transform;
                t.rotation = (Quat::from_rotation_y((dx - prev) * 0.012) * t.rotation).normalize();
                self.needs_render = true;
            }
        }
    }

    fn stop_anim(&mut self, f: usize) {
        if let Some(a) = self.scene.figures[f].anim.as_mut() {
            a.playing = false;
        }
    }

    fn apply_ik(&mut self, f: usize, kind: &IkKind, effector: usize, target: Vec3) {
        let fig = &mut self.scene.figures[f];
        let model = fig.model.clone();
        let shapes = fig.shapes_opt().map(|s| s.to_vec());
        let sk = &model.skeleton;
        let w = sk.evaluate(&fig.pose.locals, shapes.as_deref()).world;
        match kind {
            IkKind::Limb { root, mid, end, hint } => {
                let keep = mat_rotation(&w[*end]);
                let guided = self.opts.mode == crate::app::PoseMode::Guided && self.opts.limits;
                let is_arm = matches!(model.humanoid.human_of(*end), Some(H::LeftHand | H::RightHand));
                // Out of reach: lean the collarbone and spine towards the target (Design Doll style).
                let reach = (w[*mid].w_axis.truncate() - w[*root].w_axis.truncate()).length()
                    + (w[*end].w_axis.truncate() - w[*mid].w_axis.truncate()).length();
                let to_t = (target - w[*root].w_axis.truncate()).normalize_or_zero();
                // Only lean for reaches up / out / forward; low or behind-the-back targets just straighten the arm.
                if guided && is_arm && to_t.y > -0.25 && (target - w[*root].w_axis.truncate()).length() > reach * 0.96 {
                    let hm = &model.humanoid;
                    let sh = if matches!(hm.human_of(*end), Some(H::LeftHand)) { H::LeftShoulder } else { H::RightShoulder };
                    let chain: Vec<usize> = [H::Spine, H::Chest, H::UpperChest, sh].iter().filter_map(|h| hm.get(*h)).collect();
                    let limits: Vec<Option<poser_core::humanoid::JointLimit>> = chain.iter().map(|b| hm.human_of(*b).and_then(|h| h.default_limit())).collect();
                    let chain_c = chain.clone();
                    // Aim the shoulder joint (not the hand) a bit closer to the target.
                    let shoulder_goal = target - (target - w[*root].w_axis.truncate()).normalize_or_zero() * reach * 0.95;
                    // Small, evenly spread lean: the spine never bends more than ~25 degrees for a reach.
                    let caps: Vec<f32> = chain.iter().map(|b| if hm.human_of(*b) == Some(sh) { 0.35 } else { 0.14 }).collect();
                    ik::distribute_aim(sk, shapes.as_deref(), &mut fig.pose.locals, &chain, *root, shoulder_goal, &caps, |b, t| {
                        if let Some(Some(l)) = chain_c.iter().position(|x| *x == b).map(|i| limits[i]) {
                            let n = hm.to_normalized(sk, b, t.rotation);
                            t.rotation = hm.to_local(sk, b, l.clamp(n));
                        }
                    });
                }
                let anatomical = guided && self.guided_two_bone(f, *root, *mid, *end, target, *hint);
                let fig = &mut self.scene.figures[f];
                if !anatomical {
                    ik::two_bone_ik(sk, shapes.as_deref(), &mut fig.pose.locals, *root, *mid, *end, target, *hint);
                }
                let w2 = sk.evaluate(&fig.pose.locals, shapes.as_deref()).world;
                let pr = sk.bones[*end].parent.map(|p| mat_rotation(&w2[p])).unwrap_or(Quat::IDENTITY);
                let id = fig.id;
                // Guided: the hand rides with the forearm (no flopping wrist) unless pinned.
                let pinned = self.pins.contains_key(&(id, *end));
                if !(guided && is_arm && !pinned) {
                    fig.pose.locals[*end].rotation = (pr.inverse() * keep).normalize();
                }
                if let Some(pin) = self.pins.get_mut(&(id, *end)) {
                    pin.pos = target;
                    pin.rot = keep;
                }
                // Keep every joint of the limb anatomically possible.
                self.clamp_bone(f, *root);
                self.clamp_bone(f, *mid);
                self.clamp_bone(f, *end);
                if is_arm && guided {
                    // The shoulder follows the arm; then re-aim from the moved shoulder so the hand
                    // still lands on the target.
                    self.shoulder_rhythm(f);
                    if anatomical && self.guided_two_bone(f, *root, *mid, *end, target, *hint) {
                        self.clamp_bone(f, *root);
                        self.clamp_bone(f, *mid);
                        self.shoulder_rhythm(f);
                    }
                }
            }
            IkKind::Swing { root, end } => {
                let a = w[*root].w_axis.truncate();
                let e = w[*end].w_axis.truncate();
                let m = w[effector].w_axis.truncate();
                let axis = (e - a).normalize_or_zero();
                if axis != Vec3::ZERO {
                    let proj = |v: Vec3| {
                        let d = v - a;
                        d - axis * d.dot(axis)
                    };
                    let u = proj(m).normalize_or_zero();
                    let v = proj(target).normalize_or_zero();
                    if u != Vec3::ZERO && v != Vec3::ZERO {
                        let angle = u.cross(v).dot(axis).atan2(u.dot(v));
                        ik::swing_chain(sk, shapes.as_deref(), &mut fig.pose.locals, *root, *end, true, angle);
                    }
                }
                self.clamp_bone(f, *root);
                self.clamp_bone(f, effector);
                self.shoulder_rhythm(f);
            }
            IkKind::Hips => {
                let delta = target - w[effector].w_axis.truncate();
                // Reuse the grab logic (keeps feet planted + pins).
                let world_delta = fig.transform.rotation * (delta * fig.transform.scale.max_element());
                self.grab_bone(f, effector, world_delta);
                if self.opts.mode == crate::app::PoseMode::Guided {
                    self.clamp_humans(f, &[H::LeftUpperLeg, H::RightUpperLeg, H::LeftLowerLeg, H::RightLowerLeg, H::LeftFoot, H::RightFoot]);
                }
                return;
            }
            IkKind::Ccd { chain, effector: e } => {
                let limits: Vec<Option<poser_core::humanoid::JointLimit>> =
                    chain.iter().map(|b| if self.opts.limits { model.humanoid.human_of(*b).and_then(|h| h.default_limit()) } else { None }).collect();
                let hm = &model.humanoid;
                let chain_c = chain.clone();
                let is_finger = chain.iter().any(|b| hm.human_of(*b).is_some_and(|h| h.is_finger()));
                if !is_finger && hm.tpose_world.len() == sk.len() {
                    // Spine / neck: bend evenly along the chain (no kinks), capped per joint in Guided mode.
                    let cap = if self.opts.mode == crate::app::PoseMode::Guided && self.opts.limits { 0.45 } else { f32::MAX };
                    let caps = vec![cap; chain.len()];
                    let neck = hm.get(H::Neck).filter(|n| !chain.contains(n));
                    let head = hm.get(H::Head);
                    let keep_head = head.map(|h| mat_rotation(&w[h]));
                    ik::distribute_aim(sk, shapes.as_deref(), &mut fig.pose.locals, chain, *e, target, &caps, |b, t| {
                        if let Some(Some(l)) = chain_c.iter().position(|x| *x == b).map(|i| limits[i]) {
                            let n = hm.to_normalized(sk, b, t.rotation);
                            t.rotation = hm.to_local(sk, b, l.clamp(n));
                        }
                    });
                    // Bending the torso: the neck counter-balances so the head stays mostly upright.
                    if let (Some(nk), Some(hd), Some(keep)) = (neck, head, keep_head) {
                        let w2 = sk.evaluate(&fig.pose.locals, shapes.as_deref()).world;
                        let cur = mat_rotation(&w2[hd]);
                        let delta = (cur.slerp(keep, 0.6) * cur.inverse()).normalize();
                        ik::rotate_bone_world(sk, &w2, &mut fig.pose.locals, nk, delta);
                        if let Some(l) = H::Neck.default_limit().filter(|_| cap < f32::MAX) {
                            let n = hm.to_normalized(sk, nk, fig.pose.locals[nk].rotation);
                            fig.pose.locals[nk].rotation = hm.to_local(sk, nk, l.clamp(n));
                        }
                    }
                } else {
                ik::ccd(
                    sk,
                    shapes.as_deref(),
                    &mut fig.pose.locals,
                    chain,
                    *e,
                    target,
                    6,
                    |b, t| {
                        if let Some(Some(l)) = chain_c.iter().position(|x| *x == b).map(|i| limits[i]) {
                            if hm.tpose_world.len() == sk.len() {
                                let n = hm.to_normalized(sk, b, t.rotation);
                                t.rotation = hm.to_local(sk, b, l.clamp(n));
                            }
                        }
                    },
                    0.6,
                );
                }
            }
            IkKind::Aim { bone } => {
                ik::aim_bone(sk, shapes.as_deref(), &mut fig.pose.locals, *bone, target);
                self.clamp_bone(f, *bone);
                self.shoulder_rhythm(f);
            }
            IkKind::Translate { bone } => {
                let w = sk.evaluate(&fig.pose.locals, shapes.as_deref()).world;
                let pw = match sk.bones[*bone].parent {
                    Some(p) => w[p],
                    None => sk.root_transform,
                };
                let cur = w[*bone].w_axis.truncate();
                fig.pose.locals[*bone].translation += pw.inverse().transform_vector3(target - cur);
            }
        }
        self.solve_pins(f, Some(effector));
    }

    // ---- Overlays ---------------------------------------------------------------------------

    fn draw_overlays(&mut self, ui: &egui::Ui, view: &ViewCtx) {
        let painter = ui.painter_at(view.rect);
        let s = &self.scene.render;
        // Guides
        if s.thirds_grid {
            let r = view.rect;
            let c = Color32::from_white_alpha(60);
            for k in 1..3 {
                let x = r.left() + r.width() * k as f32 / 3.0;
                let y = r.top() + r.height() * k as f32 / 3.0;
                painter.line_segment([Pos2::new(x, r.top()), Pos2::new(x, r.bottom())], Stroke::new(1.0, c));
                painter.line_segment([Pos2::new(r.left(), y), Pos2::new(r.right(), y)], Stroke::new(1.0, c));
            }
        }
        if s.eye_level_line || s.perspective_guides {
            let f = view.cam.forward();
            let flat = Vec3::new(f.x, 0.0, f.z).normalize_or_zero();
            if flat != Vec3::ZERO && !view.cam.is_ortho() {
                let eye = view.cam.eye();
                if let Some(h) = view.project(eye + flat * 1000.0) {
                    let dir = egui::vec2(view.cam.roll.cos(), view.cam.roll.sin());
                    let a = h - dir * 4000.0;
                    let b = h + dir * 4000.0;
                    if s.eye_level_line {
                        painter.line_segment([a, b], Stroke::new(1.5, Color32::from_rgb(90, 200, 255)));
                        painter.text(Pos2::new(view.rect.left() + 8.0, h.y - 14.0), egui::Align2::LEFT_CENTER, "eye level", egui::FontId::proportional(11.0), Color32::from_rgb(90, 200, 255));
                    }
                    if s.perspective_guides {
                        // Vanishing points of the world X and Z axes on the horizon.
                        for dir3 in [Vec3::X, Vec3::NEG_X, Vec3::Z, Vec3::NEG_Z] {
                            if dir3.dot(f) <= 0.05 {
                                continue;
                            }
                            if let Some(vp) = view.project(eye + dir3 * 5000.0) {
                                painter.circle_stroke(vp, 5.0, Stroke::new(1.5, Color32::from_rgb(255, 120, 200)));
                                for k in 0..12 {
                                    let ang = k as f32 / 12.0 * std::f32::consts::TAU;
                                    let e = vp + egui::vec2(ang.cos(), ang.sin()) * 3000.0;
                                    painter.line_segment([vp, e], Stroke::new(0.7, Color32::from_rgba_unmultiplied(255, 120, 200, 70)));
                                }
                            }
                        }
                    }
                }
            }
        }
        // Figure overlays
        if let Some(f) = self.sel.figure.filter(|f| *f < self.scene.figures.len()) {
            let fig = &self.scene.figures[f];
            if let Some(c) = self.cache.figs.get(&fig.id) {
                let wp = |b: usize| c.world.transform_point3(c.pose.world[b].w_axis.truncate());
                let sk = &fig.model.skeleton;
                if s.show_bones {
                    for (i, bone) in sk.bones.iter().enumerate() {
                        if let Some(p) = bone.parent {
                            if let (Some(a), Some(b)) = (view.project(wp(p)), view.project(wp(i))) {
                                let col = if Some(p) == self.sel.bone { theme::ACTIVE } else { Color32::from_gray(200) };
                                gizmo::bone(&painter, a, b, col);
                            }
                        }
                    }
                }
                if s.show_joints && self.tool == Tool::Pose && fig.model.humanoid.is_valid() {
                    let main = [
                        H::Hips, H::Chest, H::Neck, H::Head, H::LeftUpperArm, H::RightUpperArm, H::LeftLowerArm, H::RightLowerArm,
                        H::LeftHand, H::RightHand, H::LeftUpperLeg, H::RightUpperLeg, H::LeftLowerLeg, H::RightLowerLeg, H::LeftFoot, H::RightFoot,
                    ];
                    for hb in main {
                        let Some(b) = fig.model.humanoid.get(hb) else { continue };
                        let Some(p) = view.project(wp(b)) else { continue };
                        let selected = self.sel.bone == Some(b);
                        let hovered = self.hover == Some((f, Some(b)));
                        let effector = matches!(hb, H::LeftHand | H::RightHand | H::LeftFoot | H::RightFoot | H::Head | H::Hips);
                        let r = if effector { 6.0 } else { 4.5 };
                        let stress = if s.joint_stress && self.opts.limits {
                            hb.default_limit().map(|l| {
                                let n = fig.model.humanoid.to_normalized(sk, b, fig.pose.locals[b].rotation);
                                l.stress(n)
                            })
                        } else {
                            None
                        };
                        let fill = if selected {
                            Some(theme::ACTIVE)
                        } else if let Some(st) = stress.filter(|s| *s > 0.75) {
                            Some(Color32::from_rgb(255, (220.0 * (1.5 - st).clamp(0.0, 1.0)) as u8, 40))
                        } else if hovered {
                            Some(Color32::from_white_alpha(140))
                        } else {
                            Some(Color32::from_black_alpha(60))
                        };
                        gizmo::joint(&painter, p, r, fill, if effector { Color32::from_rgb(255, 220, 120) } else { Color32::from_gray(230) });
                    }
                }
                for ((fid, b), _) in &self.pins {
                    if *fid == fig.id && *b < sk.len() {
                        if let Some(p) = view.project(wp(*b)) {
                            gizmo::pin(&painter, p);
                        }
                    }
                }
            }
        }
        if self.tool == Tool::Pose && self.modal.is_none() && self.gizmo != GizmoMode::Rotate {
            if let Some(hs) = self.handle_set(view) {
                let active = match &self.drag {
                    Some(Drag::Handle { handle, .. }) => Some(*handle),
                    _ => None,
                };
                hs.draw(&painter, view, self.gizmo == GizmoMode::Scale, self.hover_handle, active);
            }
        }
        if self.tool == Tool::Pose && self.opts.show_rings && self.modal.is_none() && self.gizmo == GizmoMode::Rotate {
            if let Some(rs) = self.ring_set(view) {
                let active = match &self.drag {
                    Some(Drag::Ring { ring, .. }) => Some(*ring),
                    _ => None,
                };
                rs.draw(&painter, view, self.hover_ring, active);
            }
        }
        if let Some(m) = &self.modal {
            let center = self.modal_center(m.figure, m.bone);
            if let (Some(c), Some(p)) = (view.project(center), ui.input(|i| i.pointer.latest_pos())) {
                painter.add(egui::Shape::dashed_line(&[c, p], Stroke::new(1.0, Color32::WHITE), 6.0, 4.0));
            }
        }
        self.draw_rig_markers(&painter, view);
        self.draw_armature(&painter, view);
        // Navigation gizmo + view label
        gizmo::axis_widget(&painter, self.axis_widget_center(view.rect), &self.cam);
        let proj = match self.cam.projection {
            poser_core::camera::Projection::Perspective => "Perspective",
            poser_core::camera::Projection::Orthographic => "Orthographic",
            poser_core::camera::Projection::Fake { .. } => "Dynamic perspective",
            poser_core::camera::Projection::Lens { .. } => "Lens",
        };
        let mut label = format!("User {proj}  |  {:.0}mm", self.cam.focal_length_mm());
        if let Some(f) = self.sel.figure.and_then(|f| self.scene.figures.get(f)) {
            label.push_str(&format!("\n{}", f.name));
            if let Some(b) = self.sel.bone {
                label.push_str(&format!(" : {}", f.model.skeleton.bones[b].name));
            }
        }
        painter.text(view.rect.left_top() + egui::vec2(10.0, 8.0), egui::Align2::LEFT_TOP, label, egui::FontId::proportional(13.0), Color32::from_gray(225));
        if !self.importing.is_empty() {
            painter.text(view.rect.center(), egui::Align2::CENTER_CENTER, "Importing...", egui::FontId::proportional(22.0), Color32::WHITE);
        }
        if let Some(Drag::RotateFigure { .. }) = self.drag {
        } else {
            self.drag_prev_x = None;
        }
    }
}

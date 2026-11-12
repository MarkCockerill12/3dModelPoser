//! Right sidebar: outliner + property tabs.
use crate::app::{PoserApp, PropsTab, Selection};
use crate::presets::hand_shape;
use egui::{Color32, Panel, RichText, Slider};
use poser_core::camera::Projection;
use poser_core::glam::{EulerRot, Quat, Vec3};
use poser_core::humanoid::{HumanBone, Side};
use poser_core::mannequin::MannequinParams;
use poser_core::scene::{LightKind, Shading};

fn rgba(ui: &mut egui::Ui, c: &mut poser_core::glam::Vec4) -> bool {
    let mut a = c.to_array();
    let r = ui.color_edit_button_rgba_unmultiplied(&mut a).changed();
    if r {
        *c = poser_core::glam::Vec4::from_array(a);
    }
    r
}

pub fn right_panel(app: &mut PoserApp, ui: &mut egui::Ui) {
    let before = app.scene.clone();
    let mut edited = false;
    Panel::right("sidebar").resizable(true).default_size(330.0).min_size(260.0).show(ui, |ui| {
        outliner(app, ui, &mut edited);
        ui.separator();
        let tabs = [
            (PropsTab::Pose, "Pose"),
            (PropsTab::Hands, "Hands"),
            (PropsTab::Body, "Body"),
            (PropsTab::Library, "Library"),
            (PropsTab::Camera, "Camera"),
            (PropsTab::Render, "Render"),
            (PropsTab::Rig, "Rig"),
        ];
        let w = (ui.available_width() - 3.0 * 4.0) / 4.0;
        egui::Grid::new("tabs").num_columns(4).spacing([4.0, 4.0]).show(ui, |ui| {
            for (i, (t, l)) in tabs.iter().enumerate() {
                let sel = app.props_tab == *t;
                if ui.add_sized([w, 22.0], egui::Button::selectable(sel, *l)).clicked() {
                    app.props_tab = *t;
                }
                if i % 4 == 3 {
                    ui.end_row();
                }
            }
        });
        ui.separator();
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| match app.props_tab {
            PropsTab::Rig => rig_tab(app, ui),
            PropsTab::Pose => pose_tab(app, ui, &mut edited),
            PropsTab::Hands => hands_tab(app, ui, &mut edited),
            PropsTab::Body => body_tab(app, ui, &mut edited),
            PropsTab::Library => library_tab(app, ui),
            PropsTab::Camera => camera_tab(app, ui),
            PropsTab::Render => render_tab(app, ui, &mut edited),
        });
    });
    let pointer_down = ui.input(|i| i.pointer.any_down());
    if edited {
        if !app.panel_editing {
            app.undo.push("Edit", &before);
            app.panel_editing = true;
        }
        app.needs_render = true;
    }
    if !pointer_down && !edited {
        app.panel_editing = false;
    }
}

fn outliner(app: &mut PoserApp, ui: &mut egui::Ui, edited: &mut bool) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("Scene").strong());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("+ Import").clicked() {
                app.import_dialog();
            }
            ui.menu_button("+ Human", |ui| {
                for (name, p) in poser_core::makehuman::HumanParams::presets() {
                    if ui.button(name).clicked() {
                        app.push_undo("Add human");
                        let i = app.add_human(p, name);
                        app.sel = Selection { figure: Some(i), bone: None };
                        ui.close();
                    }
                }
            });
        });
    });
    egui::ScrollArea::vertical().id_salt("outliner").max_height(150.0).show(ui, |ui| {
        let mut select = None;
        for (i, f) in app.scene.figures.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                if ui.checkbox(&mut f.visible, "").on_hover_text("Visible").changed() {
                    *edited = true;
                }
                let icon = if f.is_character() { if f.model.humanoid.is_valid() { "[H]" } else { "[R]" } } else { "[M]" };
                if ui.selectable_label(app.sel.figure == Some(i), format!("{icon} {}", f.name)).clicked() {
                    select = Some(i);
                }
            });
        }
        if let Some(i) = select {
            app.sel = Selection { figure: Some(i), bone: None };
            app.needs_render = true;
        }
    });
    if let Some(f) = app.sel.figure.filter(|f| *f < app.scene.figures.len()) {
        let fig = &app.scene.figures[f];
        if fig.model.has_skeleton() {
            egui::CollapsingHeader::new(format!("Bones ({})", fig.model.skeleton.len())).id_salt("bones").show(ui, |ui| {
                ui.add(egui::TextEdit::singleline(&mut app.bone_filter).hint_text("filter..."));
                egui::ScrollArea::vertical().id_salt("bonelist").max_height(180.0).show(ui, |ui| {
                    let filt = app.bone_filter.to_lowercase();
                    let mut pick = None;
                    for (i, b) in fig.model.skeleton.bones.iter().enumerate() {
                        if !filt.is_empty() && !b.name.to_lowercase().contains(&filt) {
                            continue;
                        }
                        let mut depth = 0;
                        let mut p = b.parent;
                        while let Some(x) = p {
                            depth += 1;
                            p = fig.model.skeleton.bones[x].parent;
                        }
                        let hb = fig.model.humanoid.human_of(i).map(|h| format!("  ({})", h.name())).unwrap_or_default();
                        let indent = if filt.is_empty() { "  ".repeat(depth.min(12)) } else { String::new() };
                        if ui.selectable_label(app.sel.bone == Some(i), RichText::new(format!("{indent}{}{hb}", b.name)).size(12.0)).clicked() {
                            pick = Some(i);
                        }
                    }
                    if let Some(b) = pick {
                        app.sel.bone = Some(b);
                        app.needs_render = true;
                    }
                });
            });
        }
    }
}

fn rig_tab(app: &mut PoserApp, ui: &mut egui::Ui) {
    if let Some(r) = app.rig.as_mut() {
        ui.label(RichText::new("Auto-rig: place the markers").strong());
        ui.label("Drag each coloured dot onto the matching spot. Check the front (1) and side (3) views: markers should sit inside the body at the joint centres.");
        ui.checkbox(&mut r.symmetric, "Symmetric (move left and right together)");
        egui::Grid::new("markers").num_columns(2).show(ui, |ui| {
            for i in 0..8 {
                ui.label(poser_rig::MARKER_NAMES[i]);
                let v = r.markers.get(i);
                ui.label(RichText::new(format!("{:.2} {:.2} {:.2}", v.x, v.y, v.z)).monospace().weak());
                ui.end_row();
            }
        });
        let busy = r.computing.is_some();
        let mut apply = false;
        let mut reset = false;
        let mut cancel = false;
        ui.horizontal(|ui| {
            apply = ui.add_enabled(!busy, egui::Button::new(RichText::new("Apply rig").strong())).clicked();
            if busy {
                ui.spinner();
            }
            reset = ui.add_enabled(!busy, egui::Button::new("Reset guesses")).clicked();
            cancel = ui.add_enabled(!busy, egui::Button::new("Cancel")).clicked();
        });
        if reset {
            if let Some(r) = app.rig.as_mut() {
                r.markers = poser_rig::guess_markers(&r.prepared);
            }
        }
        if apply {
            app.apply_rig();
        }
        if cancel {
            app.rig = None;
            app.undo();
            app.cam.projection = Projection::Perspective;
        }
        return;
    }
    let Some(f) = app.sel.figure.filter(|f| *f < app.scene.figures.len()) else {
        ui.label("Select a figure to rig.");
        return;
    };
    let fig = &app.scene.figures[f];
    let start;
    if fig.model.humanoid.is_valid() {
        ui.label(RichText::new(format!("Humanoid rig: {} bones ({} mapped to the standard skeleton).", fig.model.skeleton.len(), fig.model.humanoid.bones.len())).color(Color32::from_rgb(140, 220, 140)));
        ui.label(RichText::new("Presets, IK, mirroring, hands and retargeting all work on this figure.").weak());
        ui.add_space(6.0);
        start = ui.button("Re-rig with auto-rig (replaces skeleton and weights)").clicked();
    } else if fig.model.has_skeleton() {
        ui.label(RichText::new(format!("Custom rig with {} bones (not recognised as humanoid).", fig.model.skeleton.len())).color(Color32::from_rgb(255, 200, 120)));
        ui.label("You can still pose every bone with the rings and G/R/S. To get presets and IK, auto-rig it as a humanoid:");
        start = ui.button(RichText::new("Auto-rig as humanoid").strong()).clicked();
    } else {
        ui.label(RichText::new("No skeleton: this model is a static mesh.").color(Color32::from_rgb(255, 200, 120)));
        ui.label("Auto-rig builds a full humanoid skeleton and computes skin weights. Works best on a character standing upright in a T- or A-pose.");
        start = ui.button(RichText::new("Auto-rig as humanoid").strong()).clicked();
        ui.label(RichText::new("Props (swords, chairs...) do not need rigging: use Move/Turn to place them.").weak());
    }
    if start {
        app.start_rig(f);
    }
}


fn mode_switch(app: &mut PoserApp, ui: &mut egui::Ui) {
    use crate::app::PoseMode;
    ui.horizontal(|ui| {
        ui.label("Posing");
        let w = (ui.available_width() - 6.0) / 2.0;
        if ui.add_sized([w, 22.0], egui::Button::selectable(app.opts.mode == PoseMode::Guided, "Guided")).on_hover_text("Joints stay within natural human ranges, so poses always look plausible").clicked() {
            app.opts.mode = PoseMode::Guided;
        }
        if ui.add_sized([w, 22.0], egui::Button::selectable(app.opts.mode == PoseMode::Free, "Free")).on_hover_text("No limits: rotate any joint any amount; Ctrl+drag or G moves bones freely").clicked() {
            app.opts.mode = PoseMode::Free;
        }
    });
}

fn pose_tab(app: &mut PoserApp, ui: &mut egui::Ui, edited: &mut bool) {
    mode_switch(app, ui);
    ui.separator();
    let Some(f) = app.sel.figure.filter(|f| *f < app.scene.figures.len()) else {
        ui.label("Select a figure in the viewport or the list above.");
        return;
    };
    ui.horizontal(|ui| {
        ui.label("Name");
        if ui.text_edit_singleline(&mut app.scene.figures[f].name).changed() {
            *edited = true;
        }
    });
    let humanoid = app.scene.figures[f].model.humanoid.is_valid();
    if !app.scene.figures[f].model.has_skeleton() {
        ui.label(RichText::new("This model has no skeleton yet. Auto-rigging is coming in the Rig tab.").color(Color32::from_rgb(255, 200, 120)));
    } else if !humanoid {
        ui.label(RichText::new("Custom (non-humanoid) rig: FK posing only; presets and IK need a humanoid.").weak());
    }
    if let Some(b) = app.sel.bone {
        let fig = &app.scene.figures[f];
        let name = fig.model.skeleton.bones[b].name.clone();
        let hb = fig.model.humanoid.human_of(b);
        ui.label(RichText::new(format!("Joint: {name}{}", hb.map(|h| format!("  ({})", h.name())).unwrap_or_default())).strong());
        // Rotation sliders in the normalised (anatomical) frame for humanoids.
        let m = fig.model.clone();
        let use_norm = hb.is_some() && m.humanoid.tpose_world.len() == m.skeleton.len();
        let q = fig.pose.locals[b].rotation;
        let n = if use_norm { m.humanoid.to_normalized(&m.skeleton, b, q) } else { q };
        let (z, y, x) = n.to_euler(EulerRot::ZYX);
        let mut e = [x.to_degrees(), y.to_degrees(), z.to_degrees()];
        let lim = if app.opts.limits && app.opts.mode == crate::app::PoseMode::Guided { hb.and_then(|h| h.default_limit()) } else { None };
        let mut changed = false;
        for (i, (label, col)) in [("X", crate::theme::AXIS_X), ("Y", crate::theme::AXIS_Y), ("Z", crate::theme::AXIS_Z)].iter().enumerate() {
            let range = match (lim, i) {
                (Some(l), 0) => l.x.0..=l.x.1,
                (Some(l), 1) => l.y.0..=l.y.1,
                (Some(l), _) => l.z.0..=l.z.1,
                _ => -180.0..=180.0,
            };
            ui.horizontal(|ui| {
                ui.label(RichText::new(*label).color(*col).strong());
                changed |= ui.add(Slider::new(&mut e[i], range).suffix("°").fixed_decimals(1)).changed();
            });
        }
        if changed {
            let nq = Quat::from_euler(EulerRot::ZYX, e[2].to_radians(), e[1].to_radians(), e[0].to_radians());
            let fig = &mut app.scene.figures[f];
            fig.pose.locals[b].rotation = if use_norm { m.humanoid.to_local(&m.skeleton, b, nq) } else { nq };
            fig.touch();
            app.solve_pins(f, Some(b));
            *edited = true;
        }
        let fig = &mut app.scene.figures[f];
        ui.horizontal(|ui| {
            ui.label("Length");
            if ui.add(Slider::new(&mut fig.shapes[b].length, 0.2..=3.0)).changed() {
                fig.touch();
                *edited = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Girth  ");
            if ui.add(Slider::new(&mut fig.shapes[b].girth, 0.2..=3.0)).changed() {
                fig.touch();
                *edited = true;
            }
        });
        ui.horizontal(|ui| {
            if ui.button("Reset joint").clicked() {
                let r = m.skeleton.bones[b].rest;
                let fig = &mut app.scene.figures[f];
                fig.pose.locals[b] = if use_norm { poser_core::Transform { rotation: m.humanoid.tpose_local[b], ..r } } else { r };
                fig.shapes[b] = Default::default();
                fig.touch();
                *edited = true;
            }
            let pinned = app.pins.contains_key(&(app.scene.figures[f].id, b));
            if ui.selectable_label(pinned, "Pin").clicked() {
                app.toggle_pin(f, b);
            }
            if let Some(mb) = hb.map(|h| h.mirror()).filter(|m| Some(*m) != hb).and_then(|mh| m.humanoid.get(mh)) {
                if ui.button("Select mirror").clicked() {
                    app.sel.bone = Some(mb);
                    app.needs_render = true;
                }
            }
        });
        ui.separator();
    } else {
        ui.label(RichText::new("Click a body part to select a joint.").weak());
    }
    ui.label(RichText::new("Whole pose").strong());
    ui.horizontal_wrapped(|ui| {
        if ui.button("Rest").clicked() {
            app.reset_pose(f, false);
        }
        if humanoid && ui.button("T-pose").clicked() {
            app.reset_pose(f, true);
        }
        if humanoid && ui.button("Mirror L->R").clicked() {
            app.mirror_pose(f, true);
        }
        if humanoid && ui.button("Mirror R->L").clicked() {
            app.mirror_pose(f, false);
        }
        if humanoid && ui.button("Flip").clicked() {
            app.flip_pose(f);
        }
        if ui.button("Copy").clicked() {
            app.copy_pose();
        }
        if ui.add_enabled(app.clipboard.is_some(), egui::Button::new("Paste")).clicked() {
            app.paste_pose(false);
        }
    });
    ui.separator();
    ui.label(RichText::new("Posing options").strong());
    ui.add_enabled(app.opts.mode == crate::app::PoseMode::Guided, egui::Checkbox::new(&mut app.opts.limits, "Joint limits (Guided mode)"));
    ui.checkbox(&mut app.opts.local_axes, "Rings follow the bone (local axes)");
    ui.checkbox(&mut app.opts.show_rings, "Show rotation rings");
    ui.checkbox(&mut app.opts.spine_spread, "Spread spine rotation (smooth bend)");
    ui.checkbox(&mut app.opts.keep_feet_planted, "Keep feet planted when moving hips");
    if ui.checkbox(&mut app.scene.render.joint_stress, "Highlight stressed joints").changed() {
        *edited = true;
    }
    ui.separator();
    ui.label(RichText::new("Placement").strong());
    let t = &mut app.scene.figures[f].transform;
    let mut pos = t.translation.to_array();
    let mut yaw = t.rotation.to_euler(EulerRot::YXZ).0.to_degrees();
    let mut scale = t.scale.x;
    let mut ch = false;
    ui.horizontal(|ui| {
        ch |= ui.add(egui::DragValue::new(&mut pos[0]).speed(0.01).prefix("x ")).changed();
        ch |= ui.add(egui::DragValue::new(&mut pos[1]).speed(0.01).prefix("y ")).changed();
        ch |= ui.add(egui::DragValue::new(&mut pos[2]).speed(0.01).prefix("z ")).changed();
    });
    ui.horizontal(|ui| {
        ch |= ui.add(Slider::new(&mut yaw, -180.0..=180.0).text("turn").suffix("°")).changed();
    });
    ui.horizontal(|ui| {
        ch |= ui.add(Slider::new(&mut scale, 0.05..=5.0).logarithmic(true).text("scale")).changed();
    });
    if ch {
        t.translation = Vec3::from_array(pos);
        t.rotation = Quat::from_rotation_y(yaw.to_radians());
        t.scale = Vec3::splat(scale);
        *edited = true;
    }
}

fn hands_tab(app: &mut PoserApp, ui: &mut egui::Ui, edited: &mut bool) {
    let Some(f) = app.sel.figure.filter(|f| app.scene.figures.get(*f).is_some_and(|x| x.model.humanoid.is_valid())) else {
        ui.label("Select a humanoid figure.");
        return;
    };
    let id = app.scene.figures[f].id;
    for (side, left) in [(Side::Left, true), (Side::Right, false)] {
        ui.label(RichText::new(if left { "Left hand" } else { "Right hand" }).strong());
        let vals = app.hand_sliders.entry((id, left)).or_insert([0.2, 0.15, 0.22, 0.28, 0.36, 0.15, 0.2]);
        let mut ch = false;
        for (i, n) in ["Thumb", "Index", "Middle", "Ring", "Little", "Spread", "Thumb opp."].iter().enumerate() {
            let range = if i == 6 { -0.5..=1.0 } else { 0.0..=1.0 };
            ch |= ui.add(Slider::new(&mut vals[i], range).text(*n)).changed();
        }
        let v = *vals;
        ui.horizontal(|ui| {
            if ui.small_button("Grip all").clicked() {
                let vals = app.hand_sliders.get_mut(&(id, left)).unwrap();
                for x in vals.iter_mut().take(5) {
                    *x = (*x + 0.25).min(1.0);
                }
                ch = true;
            }
            if ui.small_button("Open all").clicked() {
                let vals = app.hand_sliders.get_mut(&(id, left)).unwrap();
                for x in vals.iter_mut().take(5) {
                    *x = 0.0;
                }
                ch = true;
            }
        });
        if ch {
            let v = app.hand_sliders[&(id, left)];
            let hp = hand_shape("custom", [v[0], v[1], v[2], v[3], v[4]], v[5], v[6]);
            let hp = if left { hp } else { hp.mirrored() };
            app.apply_pose_to(f, &hp.filtered(|b| b.is_finger() && b.side() == side), false);
            *edited = true;
        }
        let _ = v;
        ui.add_space(4.0);
    }
    ui.separator();
    ui.label(RichText::new("Hand presets").strong());
    let presets = app.hand_presets.clone();
    let ctx = ui.ctx().clone();
    app.ensure_thumbnails(&ctx);
    egui::Grid::new("handpresets").num_columns(4).show(ui, |ui| {
        for (i, p) in presets.iter().enumerate() {
            match app.thumbs.map.get(&format!("hand:{}", p.name)) {
                Some(t) => {
                    ui.add(egui::Image::new((t.id(), egui::vec2(48.0, 60.0)))).on_hover_text(&p.name);
                }
                None => {
                    ui.label(&p.name);
                }
            }
            for (lbl, sides) in [("L", vec![Side::Left]), ("R", vec![Side::Right]), ("Both", vec![Side::Left, Side::Right])] {
                if ui.small_button(lbl).clicked() {
                    app.push_undo("Hand preset");
                    for s in sides {
                        let hp = if s == Side::Left { p.clone() } else { p.mirrored() };
                        app.apply_pose_to(f, &hp.filtered(|b| b.is_finger() && b.side() == s), false);
                    }
                }
            }
            ui.end_row();
            let _ = i;
        }
    });
}

fn body_tab(app: &mut PoserApp, ui: &mut egui::Ui, edited: &mut bool) {
    let Some(f) = app.sel.figure.filter(|f| *f < app.scene.figures.len()) else {
        ui.label("Select a figure.");
        return;
    };
    let id = app.scene.figures[f].id;
    if let Some(mut p) = app.humans.get(&id).cloned() {
        use poser_core::makehuman::{HumanParams, age_to_years, years_to_age};
        ui.label(RichText::new("Body").strong());
        let mut ch = false;
        egui::ComboBox::from_id_salt("humanpreset").width(ui.available_width()).selected_text("Body type presets...").show_ui(ui, |ui| {
            for (name, preset) in HumanParams::presets() {
                if ui.selectable_label(false, name).clicked() {
                    p = preset;
                    ch = true;
                }
            }
        });
        let mut gender = p.gender;
        ch |= ui.add(Slider::new(&mut gender, 0.0..=1.0).text("Female - male")).changed();
        p.gender = gender;
        let mut years = age_to_years(p.age);
        if ui.add(Slider::new(&mut years, 1.0..=90.0).text("Age (years)").fixed_decimals(0)).changed() {
            p.age = years_to_age(years);
            ch = true;
        }
        ch |= ui.add(Slider::new(&mut p.muscle, 0.0..=1.0).text("Muscle")).changed();
        ch |= ui.add(Slider::new(&mut p.weight, 0.0..=1.0).text("Weight")).changed();
        let mut cm = p.height * 100.0;
        if ui.add(Slider::new(&mut cm, 50.0..=230.0).text("Height (cm)").fixed_decimals(0)).changed() {
            p.height = cm / 100.0;
            ch = true;
        }
        if p.gender < 0.95 {
            ch |= ui.add(Slider::new(&mut p.breast_size, 0.0..=1.0).text("Breast size")).changed();
            ch |= ui.add(Slider::new(&mut p.breast_firmness, 0.0..=1.0).text("Breast firmness")).changed();
        }
        egui::CollapsingHeader::new("Ethnicity").id_salt("eth").show(ui, |ui| {
            ch |= ui.add(Slider::new(&mut p.african, 0.0..=1.0).text("African")).changed();
            ch |= ui.add(Slider::new(&mut p.asian, 0.0..=1.0).text("Asian")).changed();
            ch |= ui.add(Slider::new(&mut p.caucasian, 0.0..=1.0).text("Caucasian")).changed();
        });
        egui::CollapsingHeader::new("Stylised proportions").id_salt("style").default_open(true).show(ui, |ui| {
            for (label, v, r) in [
                ("Head size", &mut p.head_size, 0.6..=2.0),
                ("Neck length", &mut p.neck_length, 0.4..=2.0),
                ("Torso length", &mut p.torso_length, 0.7..=1.4),
                ("Shoulder width", &mut p.shoulder_width, 0.6..=1.6),
                ("Arm length", &mut p.arm_length, 0.7..=1.4),
                ("Leg length", &mut p.leg_length, 0.7..=1.4),
                ("Hand size", &mut p.hand_size, 0.6..=1.6),
                ("Foot size", &mut p.foot_size, 0.6..=1.6),
            ] {
                ch |= ui.add(Slider::new(v, r).text(label)).changed();
            }
        });
        egui::CollapsingHeader::new("Clothes and hair").id_salt("outfit").show(ui, |ui| {
            let pick = |ui: &mut egui::Ui, label: &str, cat: &str, cur: &mut Option<String>| -> bool {
                let mut changed = false;
                egui::ComboBox::from_label(label).selected_text(cur.clone().unwrap_or_else(|| "None".into())).show_ui(ui, |ui| {
                    changed |= ui.selectable_value(cur, None, "None").changed();
                    for n in poser_core::makehuman::proxy_names(cat) {
                        changed |= ui.selectable_value(cur, Some(n.to_string()), n).changed();
                    }
                });
                changed
            };
            ch |= pick(ui, "Hair", "hair", &mut p.hair);
            ch |= pick(ui, "Eyebrows", "eyebrows", &mut p.eyebrows);
            for n in poser_core::makehuman::proxy_names("clothes") {
                let mut on = p.clothes.iter().any(|c| c == n);
                if ui.checkbox(&mut on, n.replace('_', " ")).changed() {
                    p.clothes.retain(|c| c != n);
                    if on {
                        p.clothes.push(n.to_string());
                    }
                    ch = true;
                }
            }
        });
        if ch {
            app.rebuild_human(f, p);
            *edited = true;
        }
        ui.separator();
    }
    if let Some(mut p) = app.mannequins.get(&id).cloned() {
        ui.label(RichText::new("Mannequin proportions").strong());
        egui::ComboBox::from_label("Preset").selected_text("Choose body type...").show_ui(ui, |ui| {
            for (name, preset) in MannequinParams::presets() {
                if ui.selectable_label(false, name).clicked() {
                    p = preset;
                    *edited = true;
                }
            }
        });
        let mut ch = false;
        let rows: [(&str, &mut f32, std::ops::RangeInclusive<f32>); 15] = [
            ("Height (m)", &mut p.height, 0.5..=2.4),
            ("Heads tall", &mut p.heads_tall, 2.0..=10.0),
            ("Male - female", &mut p.femininity, 0.0..=1.0),
            ("Muscle", &mut p.muscle, 0.0..=1.0),
            ("Weight", &mut p.weight, 0.0..=1.0),
            ("Shoulders", &mut p.shoulder_width, 0.6..=1.5),
            ("Hips", &mut p.hip_width, 0.6..=1.5),
            ("Neck length", &mut p.neck_length, 0.3..=2.5),
            ("Torso length", &mut p.torso_length, 0.6..=1.5),
            ("Arm length", &mut p.arm_length, 0.6..=1.5),
            ("Leg length", &mut p.leg_length, 0.6..=1.5),
            ("Hand size", &mut p.hand_size, 0.6..=1.6),
            ("Foot size", &mut p.foot_size, 0.6..=1.6),
            ("Head width", &mut p.head_width, 0.7..=1.4),
            ("Chest", &mut p.breast, 0.0..=1.0),
        ];
        for (label, v, r) in rows {
            ch |= ui.add(Slider::new(v, r).text(label)).changed();
        }
        ch |= ui.checkbox(&mut p.joint_balls, "Joint balls (wooden mannequin)").changed();
        if ch || *edited {
            app.rebuild_mannequin(f, p);
            *edited = true;
        }
        ui.separator();
    }
    let fig = &mut app.scene.figures[f];
    ui.label(RichText::new("Tint").strong());
    ui.horizontal(|ui| {
        let mut on = fig.tint.is_some();
        if ui.checkbox(&mut on, "Custom colour").changed() {
            fig.tint = if on { Some(poser_core::glam::Vec4::new(0.85, 0.7, 0.6, 1.0)) } else { None };
            *edited = true;
        }
        if let Some(t) = fig.tint.as_mut() {
            *edited |= rgba(ui, t);
        }
    });
    if fig.model.has_skeleton() && !app.mannequins.contains_key(&id) {
        ui.separator();
        ui.label(RichText::new("Proportions").strong());
        ui.label(RichText::new("Select a joint and use Length / Girth on the Pose tab, or scale it with S.").weak());
        let hm = fig.model.humanoid.clone();
        if hm.is_valid() {
            let groups: [(&str, &[HumanBone]); 6] = [
                ("Head size", &[HumanBone::Head]),
                ("Neck", &[HumanBone::Neck]),
                ("Torso", &[HumanBone::Spine, HumanBone::Chest, HumanBone::UpperChest]),
                ("Arms", &[HumanBone::LeftUpperArm, HumanBone::LeftLowerArm, HumanBone::RightUpperArm, HumanBone::RightLowerArm]),
                ("Legs", &[HumanBone::LeftUpperLeg, HumanBone::LeftLowerLeg, HumanBone::RightUpperLeg, HumanBone::RightLowerLeg]),
                ("Hands", &[HumanBone::LeftHand, HumanBone::RightHand]),
            ];
            for (label, bones) in groups {
                let Some(first) = bones.iter().find_map(|b| hm.get(*b)) else { continue };
                let mut len = fig.shapes[first].length;
                let mut gir = fig.shapes[first].girth;
                ui.horizontal(|ui| {
                    ui.label(label);
                    let a = ui.add(egui::DragValue::new(&mut len).speed(0.01).range(0.2..=3.0).prefix("len ")).changed();
                    let b = ui.add(egui::DragValue::new(&mut gir).speed(0.01).range(0.2..=3.0).prefix("girth ")).changed();
                    if a || b {
                        for hb in bones {
                            if let Some(i) = hm.get(*hb) {
                                if *hb == HumanBone::Head || *hb == HumanBone::LeftHand || *hb == HumanBone::RightHand {
                                    fig.shapes[i].length = len;
                                    fig.shapes[i].girth = len.max(gir).min(len.max(gir));
                                } else {
                                    fig.shapes[i].length = len;
                                    fig.shapes[i].girth = gir;
                                }
                            }
                        }
                        fig.touch();
                        *edited = true;
                    }
                });
            }
        }
    }
    if !fig.morph_names.is_empty() {
        ui.separator();
        ui.label(RichText::new(format!("Shape keys / morphs ({})", fig.morph_names.len())).strong());
        ui.add(egui::TextEdit::singleline(&mut app.morph_filter).hint_text("filter..."));
        let filt = app.morph_filter.to_lowercase();
        let mut ch = false;
        for (i, n) in fig.morph_names.iter().enumerate() {
            if !filt.is_empty() && !n.to_lowercase().contains(&filt) {
                continue;
            }
            ch |= ui.add(Slider::new(&mut fig.morph_weights[i], 0.0..=1.0).text(n)).changed();
        }
        if ui.button("Reset all morphs").clicked() {
            fig.morph_weights.iter_mut().for_each(|w| *w = 0.0);
            ch = true;
        }
        if ch {
            fig.touch();
            *edited = true;
        }
    }
}

fn library_tab(app: &mut PoserApp, ui: &mut egui::Ui) {
    let Some(f) = app.sel.figure.filter(|f| app.scene.figures.get(*f).is_some_and(|x| x.model.humanoid.is_valid())) else {
        ui.label("Select a humanoid figure to apply poses.");
        return;
    };
    let presets = app.body_presets.clone();
    let mut cats: Vec<String> = vec![];
    for p in &presets {
        if !cats.contains(&p.category) {
            cats.push(p.category.clone());
        }
    }
    let mut apply: Option<(poser_core::pose::HumanPose, bool)> = None;
    let ctx = ui.ctx().clone();
    app.ensure_thumbnails(&ctx);
    let tw = crate::thumbs::THUMB.0 as f32 * 0.62;
    let th = crate::thumbs::THUMB.1 as f32 * 0.62;
    let cols = ((ui.available_width() / (tw + 8.0)).floor() as usize).max(1);
    for c in cats {
        egui::CollapsingHeader::new(&c).id_salt(("libcat", c.clone())).default_open(true).show(ui, |ui| {
            egui::Grid::new(("lib", c.clone())).num_columns(cols).spacing([6.0, 6.0]).show(ui, |ui| {
                for (i, p) in presets.iter().filter(|p| p.category == c).enumerate() {
                    let r = ui.vertical(|ui| {
                        ui.set_width(tw);
                        let r = match app.thumbs.map.get(&format!("pose:{}", p.name)) {
                            Some(t) => ui.add(egui::Button::image(egui::Image::new((t.id(), egui::vec2(tw, th))))),
                            None => ui.add_sized([tw, th], egui::Button::new("...")),
                        };
                        ui.add(egui::Label::new(RichText::new(&p.name).size(11.0)).truncate());
                        r
                    }).inner;
                    let r = r.on_hover_text(format!("{}  (right-click: mirrored)", p.name));
                    if r.clicked() {
                        apply = Some((p.clone(), false));
                    }
                    if r.secondary_clicked() {
                        apply = Some((p.clone(), true));
                    }
                    if i % cols == cols - 1 {
                        ui.end_row();
                    }
                }
            });
        });
    }
    ui.label(RichText::new("Right-click a pose to apply it mirrored.").weak());
    ui.separator();
    ui.label(RichText::new("My poses").strong());
    ui.horizontal(|ui| {
        ui.text_edit_singleline(&mut app.new_pose_name);
        if ui.button("Save current").clicked() {
            let n = app.new_pose_name.clone();
            app.save_user_pose(n);
        }
    });
    let mut remove = None;
    for (i, p) in app.user_poses.iter().enumerate() {
        ui.horizontal(|ui| {
            if ui.button(&p.name).clicked() {
                apply = Some((p.clone(), false));
            }
            if ui.small_button("x").clicked() {
                remove = Some(i);
            }
        });
    }
    if let Some(i) = remove {
        app.user_poses.remove(i);
        crate::app::save_user_poses(&app.user_poses);
    }
    if let Some((p, mirrored)) = apply {
        app.push_undo("Apply pose");
        let p = if mirrored { p.mirrored() } else { p };
        // Keep the current hands if the preset has none.
        let keep_hands = !p.rotations.keys().any(|b| b.is_finger());
        let mut full = p;
        if keep_hands {
            if let Some(cur) = app.current_human_pose(f) {
                for (b, q) in cur.rotations {
                    if b.is_finger() {
                        full.rotations.insert(b, q);
                    }
                }
            }
        }
        app.apply_pose_to(f, &full, true);
        app.status = format!("Applied pose \"{}\"", full.name);
    }
}

fn camera_tab(app: &mut PoserApp, ui: &mut egui::Ui) {
    let cam = &mut app.cam;
    ui.label(RichText::new("Lens").strong());
    let mut mm = cam.focal_length_mm();
    if ui.add(Slider::new(&mut mm, 8.0..=300.0).logarithmic(true).text("focal length").suffix(" mm")).changed() {
        cam.set_focal_length_mm(mm);
    }
    ui.horizontal_wrapped(|ui| {
        for v in [14.0, 24.0, 35.0, 50.0, 85.0, 135.0] {
            if ui.small_button(format!("{v:.0}")).clicked() {
                cam.set_focal_length_mm(v);
            }
        }
    });
    let mut roll = cam.roll.to_degrees();
    if ui.add(Slider::new(&mut roll, -45.0..=45.0).text("dutch angle").suffix("°")).changed() {
        cam.roll = roll.to_radians();
    }
    ui.separator();
    ui.label(RichText::new("Projection").strong());
    let mode = match cam.projection {
        Projection::Perspective => 0,
        Projection::Orthographic => 1,
        Projection::Fake { .. } => 2,
        Projection::Lens { .. } => 3,
    };
    let mut m = mode;
    ui.horizontal_wrapped(|ui| {
        ui.selectable_value(&mut m, 0, "Perspective");
        ui.selectable_value(&mut m, 1, "Orthographic");
        ui.selectable_value(&mut m, 2, "Dynamic").on_hover_text("Design-Doll-style fake perspective: separate strength for near and far");
        ui.selectable_value(&mut m, 3, "Lens").on_hover_text("Fisheye-like barrel distortion");
    });
    if m != mode {
        cam.projection = match m {
            1 => Projection::Orthographic,
            2 => Projection::Fake {
                near_dist: (cam.distance - 1.0).max(0.1),
                far_dist: cam.distance + 1.0,
                fov_near: cam.fov_y * 1.6,
                fov_far: cam.fov_y * 0.7,
            },
            3 => Projection::Lens { k: 0.25 },
            _ => Projection::Perspective,
        };
    }
    match &mut cam.projection {
        Projection::Fake { near_dist, far_dist, fov_near, fov_far } => {
            ui.add(Slider::new(near_dist, 0.05..=20.0).logarithmic(true).text("near distance"));
            ui.add(Slider::new(far_dist, 0.1..=40.0).logarithmic(true).text("far distance"));
            let mut a = fov_near.to_degrees();
            if ui.add(Slider::new(&mut a, 5.0..=150.0).text("near FOV").suffix("°")).changed() {
                *fov_near = a.to_radians();
            }
            let mut b = fov_far.to_degrees();
            if ui.add(Slider::new(&mut b, 5.0..=150.0).text("far FOV").suffix("°")).changed() {
                *fov_far = b.to_radians();
            }
            ui.label(RichText::new("Wider near FOV = closer parts look bigger (dynamic foreshortening).").weak());
        }
        Projection::Lens { k } => {
            ui.add(Slider::new(k, -0.5..=1.5).text("distortion"));
        }
        _ => {}
    }
    ui.separator();
    ui.label(RichText::new("Reference image").strong());
    let ctx = ui.ctx().clone();
    ui.horizontal(|ui| {
        if ui.button("Load image...").clicked() {
            app.load_ref_image(&ctx);
        }
        if app.ref_image.is_some() && ui.button("Remove").clicked() {
            app.ref_image = None;
            app.needs_render = true;
        }
    });
    let mut rerender = false;
    if let Some(r) = app.ref_image.as_mut() {
        let a = ui.checkbox(&mut r.visible, "Show").changed();
        let b = ui.checkbox(&mut r.in_front, "In front (trace over the figure)").changed();
        rerender = a || b;
        ui.add(Slider::new(&mut r.opacity, 0.05..=1.0).text("Opacity"));
        ui.add(Slider::new(&mut r.scale, 0.2..=3.0).logarithmic(true).text("Size"));
        ui.add(Slider::new(&mut r.offset.x, -800.0..=800.0).text("Move X"));
        ui.add(Slider::new(&mut r.offset.y, -800.0..=800.0).text("Move Y"));
    }
    if rerender {
        app.needs_render = true;
    }
    ui.separator();
    ui.label(RichText::new("Saved views").strong());
    if ui.button("+ Save current view").clicked() {
        let n = app.saved_views.len() + 1;
        app.saved_views.push((format!("View {n}"), app.cam));
    }
    let mut go = None;
    let mut del = None;
    for (i, (n, _)) in app.saved_views.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            if ui.button("Go").clicked() {
                go = Some(i);
            }
            ui.text_edit_singleline(n);
            if ui.small_button("x").clicked() {
                del = Some(i);
            }
        });
    }
    if let Some(i) = go {
        let t = app.saved_views[i].1;
        let now = app.last_time;
        app.animate_camera_to(t, now);
    }
    if let Some(i) = del {
        app.saved_views.remove(i);
    }
    ui.separator();
    ui.label(RichText::new("Navigation").strong());
    use crate::app::ScrollMode;
    egui::ComboBox::from_label("Scroll / two-finger").selected_text(match app.opts.scroll {
        ScrollMode::Auto => "Auto (wheel zooms, touchpad orbits)",
        ScrollMode::AlwaysZoom => "Always zoom",
        ScrollMode::AlwaysOrbit => "Always orbit",
    }).show_ui(ui, |ui| {
        ui.selectable_value(&mut app.opts.scroll, ScrollMode::Auto, "Auto (wheel zooms, touchpad orbits)");
        ui.selectable_value(&mut app.opts.scroll, ScrollMode::AlwaysZoom, "Always zoom");
        ui.selectable_value(&mut app.opts.scroll, ScrollMode::AlwaysOrbit, "Always orbit");
    });
    ui.checkbox(&mut app.opts.emulate_3button, "Alt + left drag navigates (no middle button)");
    ui.checkbox(&mut app.opts.drag_empty_orbits, "Left-drag empty space orbits (Shift: pan)");
    ui.label(RichText::new("Right-drag also orbits (Shift pan, Ctrl zoom). Pinch zooms.").weak());
}

fn render_tab(app: &mut PoserApp, ui: &mut egui::Ui, edited: &mut bool) {
    let s = &mut app.scene.render;
    let before = s.clone();
    ui.label(RichText::new("Shading").strong());
    ui.horizontal_wrapped(|ui| {
        for (m, l) in [
            (Shading::Solid, "Clay"),
            (Shading::Textured, "Textured"),
            (Shading::Toon, "Toon"),
            (Shading::Matcap, "Form"),
            (Shading::Silhouette, "Silhouette"),
            (Shading::XRay, "X-ray"),
        ] {
            ui.selectable_value(&mut s.shading, m, l);
        }
    });
    ui.horizontal(|ui| {
        ui.label("Clay colour");
        rgba(ui, &mut s.clay_color);
    });
    if s.shading == Shading::Toon {
        ui.add(Slider::new(&mut s.toon_threshold, 0.0..=1.0).text("shadow threshold"));
        ui.horizontal(|ui| {
            ui.label("Shadow tone");
            rgba(ui, &mut s.toon_shadow);
        });
    }
    ui.add(Slider::new(&mut s.ambient, 0.0..=1.5).text("ambient"));
    ui.add(Slider::new(&mut s.rim, 0.0..=1.0).text("rim light"));
    ui.checkbox(&mut s.shadows, "Shadows");
    if ui.checkbox(&mut app.opts.dual_quaternion, "Volume-preserving skinning").on_hover_text("Dual-quaternion skinning keeps elbows, shoulders and wrists from collapsing at extreme angles").changed() {
        poser_core::skinning::DUAL_QUATERNION.store(app.opts.dual_quaternion, std::sync::atomic::Ordering::Relaxed);
        for f in &mut app.scene.figures {
            f.touch();
        }
    }
    ui.horizontal(|ui| {
        ui.checkbox(&mut s.outline, "Outline");
        ui.add(Slider::new(&mut s.outline_width, 0.5..=6.0).text("width"));
        rgba(ui, &mut s.outline_color);
    });
    ui.separator();
    ui.label(RichText::new("Viewport").strong());
    ui.horizontal(|ui| {
        ui.label("Background");
        rgba(ui, &mut s.background_top);
        rgba(ui, &mut s.background_bottom);
    });
    ui.checkbox(&mut s.show_grid, "Floor grid");
    ui.checkbox(&mut s.show_floor, "Floor (shadow catcher)");
    ui.checkbox(&mut s.show_joints, "Joint handles");
    ui.checkbox(&mut s.show_bones, "Bones");
    ui.checkbox(&mut s.eye_level_line, "Eye-level (horizon) line");
    ui.checkbox(&mut s.perspective_guides, "Vanishing point guides");
    ui.checkbox(&mut s.thirds_grid, "Rule-of-thirds grid");
    ui.add(Slider::new(&mut app.opts.render_scale, 0.4..=1.0).text("Viewport resolution")).on_hover_text("Lower = faster and less memory on integrated graphics");
    ui.horizontal(|ui| {
        ui.label("Anti-aliasing");
        ui.selectable_value(&mut s.msaa, 1, "Off");
        ui.selectable_value(&mut s.msaa, 4, "4×");
    });
    if *s != before {
        *edited = true;
    }
    ui.separator();
    ui.label(RichText::new("Lights").strong());
    let mut remove = None;
    for (i, l) in app.scene.lights.iter_mut().enumerate() {
        egui::CollapsingHeader::new(&l.name).id_salt(("light", i)).default_open(i == 0).show(ui, |ui| {
            let b = l.clone();
            ui.checkbox(&mut l.enabled, "On");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut l.kind, LightKind::Directional, "Sun");
                ui.selectable_value(&mut l.kind, LightKind::Point, "Point");
                ui.selectable_value(&mut l.kind, LightKind::Spot, "Spot");
            });
            let mut az = l.azimuth.to_degrees();
            let mut el = l.elevation.to_degrees();
            if ui.add(Slider::new(&mut az, -180.0..=180.0).text("direction").suffix("°")).changed() {
                l.azimuth = az.to_radians();
            }
            if ui.add(Slider::new(&mut el, -89.0..=89.0).text("height").suffix("°")).changed() {
                l.elevation = el.to_radians();
            }
            ui.add(Slider::new(&mut l.intensity, 0.0..=6.0).text("intensity"));
            let mut c = l.color.extend(1.0);
            if rgba(ui, &mut c) {
                l.color = c.truncate();
            }
            if l.kind == LightKind::Point {
                let mut p = l.position.to_array();
                ui.horizontal(|ui| {
                    for (k, v) in p.iter_mut().enumerate() {
                        ui.add(egui::DragValue::new(v).speed(0.02).prefix(["x ", "y ", "z "][k]));
                    }
                });
                l.position = Vec3::from_array(p);
            }
            if ui.small_button("Remove").clicked() {
                remove = Some(i);
            }
            if *l != b {
                *edited = true;
            }
        });
    }
    if let Some(i) = remove {
        app.scene.lights.remove(i);
        *edited = true;
    }
    if app.scene.lights.len() < 4 && ui.button("+ Add light").clicked() {
        let mut l = poser_core::scene::Light::key();
        l.name = format!("Light {}", app.scene.lights.len() + 1);
        l.intensity = 1.0;
        app.scene.lights.push(l);
        *edited = true;
    }
    ui.horizontal_wrapped(|ui| {
        ui.label("Time of day");
        for (n, el, col) in [("Noon", 70.0, [1.0, 0.98, 0.95]), ("Golden hour", 12.0, [1.0, 0.72, 0.45]), ("Overcast", 50.0, [0.85, 0.88, 0.95]), ("Dramatic", 25.0, [1.0, 0.95, 0.9])] {
            if ui.small_button(n).clicked() {
                if let Some(l) = app.scene.lights.first_mut() {
                    l.elevation = f32::to_radians(el);
                    l.color = Vec3::from_array(col);
                    l.intensity = if n == "Dramatic" { 3.6 } else if n == "Overcast" { 1.6 } else { 2.6 };
                }
                if let Some(l) = app.scene.lights.get_mut(1) {
                    l.intensity = if n == "Dramatic" { 0.15 } else if n == "Overcast" { 1.0 } else { 0.7 };
                }
                app.scene.render.ambient = if n == "Overcast" { 0.6 } else if n == "Dramatic" { 0.12 } else { 0.35 };
                *edited = true;
            }
        }
    });
}

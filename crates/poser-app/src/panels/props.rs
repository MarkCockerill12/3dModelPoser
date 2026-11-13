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
            ui.menu_button("+ Prop", |ui| {
                ui.set_min_width(170.0);
                for cat in ["Shapes", "Furniture", "Architecture", "Hand-held"] {
                    ui.label(RichText::new(cat).weak());
                    for k in poser_core::props::PropKind::ALL.into_iter().filter(|k| k.category() == cat) {
                        let tip = if k.handheld() { "Placed in the selected figure's right hand" } else { "Placed next to the selected figure" };
                        if ui.button(k.name()).on_hover_text(tip).clicked() {
                            app.add_prop(k);
                            ui.close();
                        }
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
            let shown = if !app.opts.all_bones && fig.model.humanoid.is_valid() { fig.model.humanoid.bones.len() } else { fig.model.skeleton.len() };
            egui::CollapsingHeader::new(format!("Joints ({shown})")).id_salt("bones").show(ui, |ui| {
                ui.add(egui::TextEdit::singleline(&mut app.bone_filter).hint_text("filter..."));
                egui::ScrollArea::vertical().id_salt("bonelist").max_height(180.0).show(ui, |ui| {
                    let filt = app.bone_filter.to_lowercase();
                    let mut pick = None;
                    let main_only = !app.opts.all_bones && fig.model.humanoid.is_valid();
                    for (i, b) in fig.model.skeleton.bones.iter().enumerate() {
                        if !filt.is_empty() && !b.name.to_lowercase().contains(&filt) {
                            continue;
                        }
                        if main_only && fig.model.humanoid.human_of(i).is_none() {
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
    if app.armature.is_some() {
        app.armature_panel(ui);
        return;
    }
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
    ui.separator();
    ui.label(RichText::new("Manual rigging").strong());
    ui.label(RichText::new("Build or fix the skeleton yourself (animals, creatures, props, odd proportions).").weak());
    let manual = ui.button("Edit skeleton manually...").clicked();
    if start {
        app.start_rig(f);
    } else if manual {
        app.start_armature_edit(f);
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
    if app.props.contains_key(&app.scene.figures[f].id) {
        ui.label(RichText::new("Prop: W move, E rotate, S scale. Attach it to a hand below.").weak());
    } else if !app.scene.figures[f].model.has_skeleton() {
        ui.label(RichText::new("This model has no skeleton yet. Auto-rig it in the Rig tab.").color(Color32::from_rgb(255, 200, 120)));
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
    if humanoid && ui.button("Pose from photo...").on_hover_text("Estimate the pose of the person in a photo (runs offline on your computer)").clicked() {
        app.photo_dialog();
    }
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
    if ui.checkbox(&mut app.opts.all_bones, "Select every bone (face, twist, toes)").on_hover_text("Off: clicking picks the main posing joint, like Design Doll").changed() {
        app.needs_render = true;
    }
    ui.checkbox(&mut app.opts.keep_feet_planted, "Keep feet planted when moving hips");
    ui.checkbox(&mut app.opts.symmetry, "Live symmetry (pose both sides at once)");
    ui.add_enabled(app.opts.mode == crate::app::PoseMode::Guided, egui::Checkbox::new(&mut app.opts.collisions, "Body collision (limbs can't pass through)"));
    ui.add_enabled(app.opts.mode == crate::app::PoseMode::Guided, egui::Checkbox::new(&mut app.opts.floor, "Floor collision"));
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
    if humanoid {
        let id = app.scene.figures[f].id;
        let mut on = app.look_at.contains(&id);
        if ui.checkbox(&mut on, "Look at camera (head and eyes follow the view)").changed() {
            if on {
                app.look_at.insert(id);
            } else {
                app.look_at.remove(&id);
            }
            app.needs_render = true;
        }
    }
    if app.scene.figures[f].attachment.is_none() && ui.button("Drop to floor").on_hover_text("Move the figure so its lowest point rests on the floor").clicked() {
        app.drop_to_floor(f);
    }
    attach_ui(app, ui, f);
}

/// "Attached to" picker: follow another figure's hand, head, hips...
fn attach_ui(app: &mut PoserApp, ui: &mut egui::Ui, f: usize) {
    use poser_core::humanoid::HumanBone as HB;
    const SPOTS: [(HB, &str); 7] = [
        (HB::RightHand, "right hand"),
        (HB::LeftHand, "left hand"),
        (HB::Head, "head"),
        (HB::Chest, "chest"),
        (HB::Hips, "hips"),
        (HB::RightFoot, "right foot"),
        (HB::LeftFoot, "left foot"),
    ];
    let fig = &app.scene.figures[f];
    let current = fig.attachment.and_then(|a| {
        let pi = app.scene.index_of(a.parent)?;
        let p = &app.scene.figures[pi];
        let spot = p.model.humanoid.human_of(a.bone).and_then(|h| SPOTS.iter().find(|s| s.0 == h)).map(|s| s.1.to_string());
        Some(format!("{}: {}", p.name, spot.unwrap_or_else(|| p.model.skeleton.bones[a.bone].name.clone())))
    });
    let mut choice: Option<Option<(usize, usize)>> = None;
    let mut hold: Option<(usize, bool)> = None;
    let holders: Vec<usize> = (0..app.scene.figures.len()).filter(|i| *i != f && app.scene.figures[*i].model.humanoid.is_valid()).collect();
    if holders.is_empty() {
        return;
    }
    ui.separator();
    ui.label(RichText::new("Attached to").strong());
    egui::ComboBox::from_id_salt("attach").width(200.0).truncate().selected_text(current.clone().unwrap_or_else(|| "Nothing".into())).show_ui(ui, |ui| {
        if ui.selectable_label(current.is_none(), "Nothing").clicked() {
            choice = Some(None);
        }
        for &h in &holders {
            let hf = &app.scene.figures[h];
            for (hb, label) in SPOTS {
                if let Some(b) = hf.model.humanoid.get(hb) {
                    if ui.selectable_label(false, format!("{}: {label}", hf.name)).clicked() {
                        choice = Some(Some((h, b)));
                    }
                }
            }
        }
    });
    if app.props.get(&app.scene.figures[f].id).is_some_and(|k| k.handheld()) {
        ui.horizontal(|ui| {
            let h = holders[0];
            if ui.button("Hold in right hand").clicked() {
                hold = Some((h, false));
            }
            if ui.button("Left hand").clicked() {
                hold = Some((h, true));
            }
        });
    }
    if let Some(c) = choice {
        app.set_attachment(f, c);
    }
    if let Some((h, left)) = hold {
        app.push_undo("Hold prop");
        app.put_in_hand(f, h, left);
    }
}

/// Per-material colours (and skin tones for the default body).
fn colours_ui(app: &mut PoserApp, ui: &mut egui::Ui, f: usize, edited: &mut bool) {
    use poser_core::glam::Vec4;
    egui::CollapsingHeader::new(RichText::new("Colours").strong()).id_salt("colours").show(ui, |ui| {
        let id = app.scene.figures[f].id;
        let mats: Vec<(String, Vec4)> = app.scene.figures[f].model.materials.iter().map(|m| (m.name.clone(), m.base_color)).collect();
        if app.humans.contains_key(&id) {
            if let Some(skin) = mats.iter().position(|m| m.0 == "Skin") {
                ui.label(RichText::new("Skin tone").weak());
                ui.horizontal(|ui| {
                    for (name, c) in [
                        ("Porcelain", [0.96, 0.85, 0.79]),
                        ("Light", [0.93, 0.78, 0.68]),
                        ("Medium", [0.84, 0.65, 0.52]),
                        ("Olive", [0.76, 0.60, 0.45]),
                        ("Tan", [0.67, 0.48, 0.34]),
                        ("Brown", [0.50, 0.34, 0.24]),
                        ("Dark", [0.33, 0.22, 0.16]),
                    ] {
                        let col = egui::Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8);
                        if ui.add(egui::Button::new("").fill(col).min_size(egui::vec2(20.0, 20.0))).on_hover_text(name).clicked() {
                            let mc = &mut app.scene.figures[f].material_colors;
                            if mc.len() <= skin {
                                mc.resize(skin + 1, None);
                            }
                            mc[skin] = Some(Vec4::new(c[0], c[1], c[2], 1.0));
                            *edited = true;
                        }
                    }
                });
            }
        }
        if app.scene.render.shading == poser_core::scene::Shading::Silhouette {
            ui.label(RichText::new("Colours don't show in Silhouette shading.").weak());
        }
        egui::ScrollArea::vertical().id_salt("matlist").max_height(200.0).show(ui, |ui| {
            for (mi, (name, base)) in mats.iter().enumerate() {
                ui.horizontal(|ui| {
                    let fig = &mut app.scene.figures[f];
                    if fig.material_colors.len() <= mi {
                        fig.material_colors.resize(mi + 1, None);
                    }
                    let mut c = fig.material_colors[mi].unwrap_or_else(|| {
                        // Show the model's own colour (stored linear) as the starting point.
                        let s = |x: f32| x.max(0.0).powf(1.0 / 2.2);
                        Vec4::new(s(base.x), s(base.y), s(base.z), 1.0)
                    });
                    if rgba(ui, &mut c) {
                        fig.material_colors[mi] = Some(c);
                        *edited = true;
                    }
                    let custom = fig.material_colors[mi].is_some();
                    if ui.add_enabled(custom, egui::Button::new("x").small()).on_hover_text("Back to the model's colour").clicked() {
                        fig.material_colors[mi] = None;
                        *edited = true;
                    }
                    ui.add(egui::Label::new(if name.is_empty() { format!("Material {mi}") } else { name.clone() }).truncate());
                });
            }
        });
    });
}

/// "Model parts": delete pieces of a model (briefcase, hat...) by clicking them.
fn parts_ui(app: &mut PoserApp, ui: &mut egui::Ui, f: usize) {
    let id = app.scene.figures[f].id;
    let active = app.parts.as_ref().is_some_and(|s| s.figure_id == id);
    egui::CollapsingHeader::new(RichText::new("Model parts").strong()).id_salt("parts").default_open(active).show(ui, |ui| {
        if !active {
            ui.label(RichText::new("Remove pieces you don't want, like a briefcase or hat.").weak());
            if ui.button("Edit parts...").on_hover_text("Click pieces in the viewport to select them, then Delete").clicked() {
                app.start_parts(f);
            }
        } else {
            let (by_mesh, nsel, pieces) = {
                let s = app.parts.as_ref().unwrap();
                (s.by_mesh, s.selected.len(), s.piece_count())
            };
            ui.horizontal(|ui| {
                ui.label("Click selects");
                let mut m = by_mesh;
                ui.selectable_value(&mut m, false, "Pieces").on_hover_text("Separate connected pieces");
                ui.selectable_value(&mut m, true, "Whole meshes").on_hover_text("Everything with the same mesh / material");
                if let Some(s) = app.parts.as_mut() {
                    s.by_mesh = m;
                }
            });
            ui.label(RichText::new(format!("{nsel} of {pieces} pieces selected. Shift+click adds.")).weak());
            ui.horizontal(|ui| {
                if ui.add_enabled(nsel > 0, egui::Button::new("Grow (G)")).on_hover_text("Also select every piece inside the selection's bounds (handles, latches...)").clicked() {
                    app.grow_parts();
                }
                if ui.add_enabled(nsel > 0, egui::Button::new("Delete (Del)")).clicked() {
                    app.delete_selected_parts();
                }
                if ui.button("Done").clicked() {
                    app.end_parts();
                }
            });
            ui.label(RichText::new("Meshes").weak());
            let names: Vec<(usize, String, usize)> = app.scene.figures[f].model.meshes.iter().enumerate().map(|(i, m)| (i, m.name.clone(), m.triangle_count())).collect();
            egui::ScrollArea::vertical().id_salt("partlist").max_height(160.0).show(ui, |ui| {
                for (i, name, tris) in names {
                    let label = if name.is_empty() { format!("Mesh {i}") } else { name };
                    let sel = app.mesh_selected(i);
                    if ui.add(egui::Button::selectable(sel, format!("{label}  ({tris} tris)")).truncate()).clicked() {
                        app.toggle_mesh_selection(i);
                    }
                }
            });
        }
        let n = app.scene.figures[f].part_edits.len();
        if n > 0 {
            ui.label(RichText::new(format!("{n} deletion{} on this model (Ctrl+Z undoes)", if n == 1 { "" } else { "s" })).weak());
        }
    });
    ui.separator();
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
    parts_ui(app, ui, f);
    colours_ui(app, ui, f, edited);
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
    ui.horizontal(|ui| {
        ui.label("Strength");
        ui.add(Slider::new(&mut app.pose_strength, 0.1..=1.0).fixed_decimals(2))
            .on_hover_text("Below 1, the pose is blended with the current one (e.g. 0.5 = halfway)");
    });
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
        if app.pose_strength < 0.999 {
            if let Some(cur) = app.current_human_pose(f) {
                let name = full.name.clone();
                full = cur.blend(&full, app.pose_strength);
                full.name = name;
            }
        }
        app.apply_library_pose(f, &full);
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
    ui.label(RichText::new("Second view (picture-in-picture)").strong());
    let views: Vec<(String, Option<poser_core::camera::Camera>)> = {
        let mut v: Vec<(String, Option<poser_core::camera::Camera>)> = vec![("Off".into(), None)];
        for (n, a) in [("Front", poser_core::camera::ViewAxis::Front), ("Side", poser_core::camera::ViewAxis::Right), ("Top", poser_core::camera::ViewAxis::Top), ("Back", poser_core::camera::ViewAxis::Back)] {
            let mut c = app.cam;
            c.set_view_axis(a);
            c.projection = Projection::Orthographic;
            let (cc, rr) = app.cache.scene_bounds(&app.scene);
            c.frame(cc, rr * 0.75);
            c.roll = 0.0;
            v.push((n.into(), Some(c)));
        }
        for (n, c) in &app.saved_views {
            v.push((n.clone(), Some(*c)));
        }
        v
    };
    ui.horizontal_wrapped(|ui| {
        for (n, c) in views {
            let on = match (app.inset, c) {
                (None, None) => true,
                (Some(a), Some(b)) => a.rotation.dot(b.rotation).abs() > 0.9999 && a.projection == b.projection,
                _ => false,
            };
            if ui.selectable_label(on, n).clicked() {
                app.inset = c;
                app.needs_render = true;
            }
        }
    });
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
    ui.checkbox(&mut app.opts.zoom_to_mouse, "Zoom towards the mouse pointer");
    ui.checkbox(&mut app.opts.orbit_selection, "Orbit around the selected joint / figure");
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
            (Shading::LineArt, "Line art"),
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

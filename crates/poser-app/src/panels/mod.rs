//! Menus, toolbar, status bar, timeline and dialogs.
mod props;

pub use props::right_panel;

use crate::app::{PoserApp, Tool};
use egui::{Panel, RichText};
use poser_core::camera::ViewAxis;
use poser_core::mannequin::MannequinParams;

pub fn top_bar(app: &mut PoserApp, ui: &mut egui::Ui) {
    Panel::top("menu").show(ui, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                ui.set_min_width(260.0);
                if ui.button("New scene  (Ctrl+N)").clicked() {
                    app.new_scene();
                    ui.close();
                }
                if ui.button("Open scene...  (Ctrl+O)").clicked() {
                    app.open_scene_dialog();
                    ui.close();
                }
                if ui.button("Save scene...  (Ctrl+S)").clicked() {
                    app.save_scene_dialog();
                    ui.close();
                }
                ui.separator();
                if ui.button("Pose from photo...").clicked() {
                    app.photo_dialog();
                    ui.close();
                }
                if ui.button("Import model...  (Ctrl+I)").clicked() {
                    app.import_dialog();
                    ui.close();
                }
                ui.menu_button("Add human", |ui| {
                    for (name, p) in poser_core::makehuman::HumanParams::presets() {
                        if ui.button(name).clicked() {
                            app.push_undo("Add human");
                            let i = app.add_human(p, name);
                            app.sel = crate::app::Selection { figure: Some(i), bone: None };
                            ui.close();
                        }
                    }
                });
                ui.menu_button("Add wooden mannequin", |ui| {
                    for (name, p) in MannequinParams::presets() {
                        if ui.button(name).clicked() {
                            app.push_undo("Add mannequin");
                            let i = app.add_mannequin(p, name);
                            app.sel = crate::app::Selection { figure: Some(i), bone: None };
                            ui.close();
                        }
                    }
                });
                ui.separator();
                if ui.button("Export posed model (GLB/OBJ)...").on_hover_text("Selected figure, baked in its current pose").clicked() {
                    app.export_model_dialog(true);
                    ui.close();
                }
                if ui.button("Export rigged model (GLB/FBX)...").on_hover_text("Selected figure with skeleton, skin weights and current pose").clicked() {
                    app.export_rigged_dialog();
                    ui.close();
                }
                if ui.button("Export whole scene (GLB/OBJ)...").clicked() {
                    app.export_model_dialog(false);
                    ui.close();
                }
                if ui.button("Export image...  (Ctrl+Shift+S)").clicked() {
                    app.export.open = true;
                    ui.close();
                }
                ui.separator();
                if ui.button("Quit").clicked() {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
            ui.menu_button("Edit", |ui| {
                if ui.add_enabled(app.undo.can_undo(), egui::Button::new("Undo  (Ctrl+Z)")).clicked() {
                    app.undo();
                }
                if ui.add_enabled(app.undo.can_redo(), egui::Button::new("Redo  (Ctrl+Shift+Z)")).clicked() {
                    app.redo();
                }
                ui.separator();
                if ui.button("Copy pose  (Ctrl+C)").clicked() {
                    app.copy_pose();
                }
                if ui.button("Paste pose  (Ctrl+V)").clicked() {
                    app.paste_pose(false);
                }
                if ui.button("Paste flipped  (Ctrl+Shift+V)").clicked() {
                    app.paste_pose(true);
                }
                ui.separator();
                if ui.button("Duplicate  (Shift+D)").clicked() {
                    app.duplicate_selected();
                }
                if ui.button("Delete  (X)").clicked() {
                    app.delete_selected();
                }
            });
            ui.menu_button("View", |ui| {
                let now = ui.input(|i| i.time);
                for (l, a) in [
                    ("Front  (1)", ViewAxis::Front),
                    ("Back  (Ctrl+1)", ViewAxis::Back),
                    ("Right  (3)", ViewAxis::Right),
                    ("Left  (Ctrl+3)", ViewAxis::Left),
                    ("Top  (7)", ViewAxis::Top),
                    ("Bottom  (Ctrl+7)", ViewAxis::Bottom),
                ] {
                    if ui.button(l).clicked() {
                        app.set_view(a, now);
                    }
                }
                ui.separator();
                if ui.button("Perspective / Ortho  (5)").clicked() {
                    app.toggle_ortho();
                }
                if ui.button("Frame all  (Home)").clicked() {
                    app.frame_all(true);
                }
                if ui.button("Frame selected  (.)").clicked() {
                    app.frame_selected();
                }
                ui.separator();
                ui.checkbox(&mut app.show_left, "Toolbar  (T)");
                ui.checkbox(&mut app.show_right, "Sidebar  (N)");
                ui.checkbox(&mut app.show_timeline, "Timeline");
            });
            ui.menu_button("Pose", |ui| {
                let Some(f) = app.sel.figure else {
                    ui.label("Select a figure first");
                    return;
                };
                if ui.button("Rest pose").clicked() {
                    app.reset_pose(f, false);
                }
                if ui.button("T-pose").clicked() {
                    app.reset_pose(f, true);
                }
                ui.separator();
                if ui.button("Mirror left -> right").clicked() {
                    app.mirror_pose(f, true);
                }
                if ui.button("Mirror right -> left").clicked() {
                    app.mirror_pose(f, false);
                }
                if ui.button("Flip pose  (M)").clicked() {
                    app.flip_pose(f);
                }
                ui.separator();
                if ui.button("Clear pins").clicked() {
                    let id = app.scene.figures[f].id;
                    app.pins.retain(|(fid, _), _| *fid != id);
                }
            });
            ui.menu_button("Help", |ui| {
                if ui.button("Controls  (F1)").clicked() {
                    app.help_open = true;
                }
                ui.label(format!("3D Model Poser v{}", env!("CARGO_PKG_VERSION")));
            });
            ui.separator();
            for (t, label, tip) in [
                (Tool::Pose, "Pose", "Click a body part to select it, drag to pose with IK; rings rotate the selected joint"),
                (Tool::Move, "Move", "Drag a figure across the floor (Shift: screen plane)"),
                (Tool::Rotate, "Turn", "Drag left/right to turn a figure"),
            ] {
                if ui.selectable_label(app.tool == t, label).on_hover_text(tip).clicked() {
                    app.tool = t;
                    app.needs_render = true;
                }
            }
        });
    });
}

pub fn status_bar(app: &mut PoserApp, ui: &mut egui::Ui) {
    Panel::bottom("status").exact_size(22.0).show(ui, |ui| {
        ui.horizontal_centered(|ui| {
            ui.label(RichText::new(&app.status).size(12.0));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let verts: usize = app.scene.figures.iter().map(|f| f.model.vertex_count()).sum();
                ui.label(RichText::new(format!("{} figures | {} verts | {:.1} ms", app.scene.figures.len(), verts, app.frame_ms)).size(12.0).weak());
            });
        });
    });
}

pub fn toolbar(app: &mut PoserApp, ui: &mut egui::Ui) {
    Panel::left("toolbar").exact_size(70.0).show(ui, |ui| {
        ui.vertical_centered(|ui| {
            ui.add_space(6.0);
            for (t, icon, tip) in [(Tool::Pose, "Pose", "Pose (rings + IK drag)"), (Tool::Move, "Move", "Move figure"), (Tool::Rotate, "Turn", "Turn figure")] {
                if ui.selectable_label(app.tool == t, RichText::new(icon).size(13.0)).on_hover_text(tip).clicked() {
                    app.tool = t;
                    app.needs_render = true;
                }
            }
            ui.separator();
            let now = ui.input(|i| i.time);
            for (icon, tip, a) in [("F", "Front view (1)", ViewAxis::Front), ("R", "Right view (3)", ViewAxis::Right), ("T", "Top view (7)", ViewAxis::Top)] {
                if ui.button(RichText::new(icon).size(14.0)).on_hover_text(tip).clicked() {
                    app.set_view(a, now);
                }
            }
            if ui.button(RichText::new("Ort").size(13.0)).on_hover_text("Perspective / ortho (5)").clicked() {
                app.toggle_ortho();
            }
            if ui.button(RichText::new("All").size(13.0)).on_hover_text("Frame all (Home)").clicked() {
                app.frame_all(true);
            }
            ui.separator();
            if ui.button(RichText::new("Pin").size(13.0)).on_hover_text("Pin / unpin selected joint (P)").clicked() {
                if let (Some(f), Some(b)) = (app.sel.figure, app.sel.bone) {
                    app.toggle_pin(f, b);
                }
            }
            if ui.button(RichText::new("Flip").size(13.0)).on_hover_text("Flip pose (M)").clicked() {
                if let Some(f) = app.sel.figure {
                    app.flip_pose(f);
                }
            }
            if ui.button(RichText::new("PNG").size(13.0)).on_hover_text("Export image (Ctrl+Shift+S)").clicked() {
                app.export.open = true;
            }
        });
    });
}

pub fn timeline(app: &mut PoserApp, ui: &mut egui::Ui) {
    let Some(fi) = app.sel.figure.filter(|f| *f < app.scene.figures.len()) else { return };
    let fid = app.scene.figures[fi].id;
    let has_clips = !app.scene.figures[fi].model.animations.is_empty();
    let has_rt = app.retargets.contains_key(&fid);
    if !app.scene.figures[fi].model.humanoid.is_valid() && !has_clips {
        return;
    }
    Panel::bottom("timeline").resizable(false).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new("Animation").strong());
            if has_clips {
                let fig = &mut app.scene.figures[fi];
                let names: Vec<String> = fig.model.animations.iter().map(|c| c.name.clone()).collect();
                let a = fig.anim.get_or_insert(poser_core::scene::AnimState { clip: 0, time: 0.0, playing: false, looping: true, speed: 1.0 });
                egui::ComboBox::from_id_salt("clip").selected_text(names.get(a.clip).cloned().unwrap_or_default()).show_ui(ui, |ui| {
                    for (i, n) in names.iter().enumerate() {
                        ui.selectable_value(&mut a.clip, i, n);
                    }
                });
                let clip = &fig.model.animations[a.clip.min(names.len() - 1)];
                let fps = clip.fps.max(1.0);
                let frames = clip.frame_count().max(1) - 1;
                if ui.button(if a.playing { "Pause" } else { "Play" }).clicked() {
                    a.playing = !a.playing;
                }
                let mut frame = (a.time * fps).round() as u32;
                let changed = ui.add(egui::Slider::new(&mut frame, 0..=frames).text("frame")).changed();
                ui.checkbox(&mut a.looping, "loop");
                ui.add(egui::DragValue::new(&mut a.speed).speed(0.05).range(0.05..=4.0).suffix("×"));
                if changed {
                    a.time = frame as f32 / fps;
                    a.playing = false;
                    app.sample_animation(fi);
                }
                if ui.button("Freeze pose").on_hover_text("Stop the animation and keep this frame as an editable pose").clicked() {
                    if let Some(a) = app.scene.figures[fi].anim.as_mut() {
                        a.playing = false;
                    }
                    app.status = "Frame frozen: pose it freely".into();
                }
                ui.separator();
            }
            if app.scene.figures[fi].model.humanoid.is_valid() {
                if ui.button("Retarget animation...").on_hover_text("Play a humanoid animation from another file (e.g. Mixamo FBX) on this character").clicked() {
                    app.retarget_dialog(fid);
                }
            }
            if has_rt {
                let mut remove = false;
                let mut sample = false;
                if let Some(rt) = app.retargets.get_mut(&fid) {
                    let names: Vec<String> = rt.source.animations.iter().map(|c| c.name.clone()).collect();
                    egui::ComboBox::from_id_salt("rt-clip").selected_text(names.get(rt.clip).cloned().unwrap_or_default()).show_ui(ui, |ui| {
                        for (i, n) in names.iter().enumerate() {
                            ui.selectable_value(&mut rt.clip, i, n);
                        }
                    });
                    if ui.button(if rt.playing { "Pause" } else { "Play" }).clicked() {
                        rt.playing = !rt.playing;
                    }
                    let clip = &rt.source.animations[rt.clip.min(names.len() - 1)];
                    let fps = clip.fps.max(1.0);
                    let mut frame = (rt.time * fps).round() as u32;
                    if ui.add(egui::Slider::new(&mut frame, 0..=clip.frame_count().max(1) - 1).text("frame")).changed() {
                        rt.time = frame as f32 / fps;
                        rt.playing = false;
                        sample = true;
                    }
                    if ui.button("Freeze").clicked() {
                        remove = true;
                    }
                }
                if sample {
                    app.sample_retarget(fi);
                }
                if remove {
                    app.retargets.remove(&fid);
                    app.status = "Retargeted frame frozen".into();
                }
            }
        });
    });
}

pub fn windows(app: &mut PoserApp, ctx: &egui::Context) {
    if app.export.open {
        let mut open = true;
        egui::Window::new("Export image").open(&mut open).resizable(false).collapsible(false).show(ctx, |ui| {
            egui::Grid::new("exp").num_columns(2).show(ui, |ui| {
                ui.label("Width");
                ui.add(egui::DragValue::new(&mut app.export.width).range(16..=8192));
                ui.end_row();
                ui.label("Height");
                ui.add(egui::DragValue::new(&mut app.export.height).range(16..=8192));
                ui.end_row();
                ui.label("Presets");
                ui.horizontal(|ui| {
                    for (l, w, h) in [("HD", 1920, 1080), ("4K", 3840, 2160), ("A4 300dpi", 2480, 3508), ("Square", 2048, 2048)] {
                        if ui.small_button(l).clicked() {
                            app.export.width = w;
                            app.export.height = h;
                        }
                    }
                });
                ui.end_row();
                ui.label("Background");
                ui.checkbox(&mut app.export.transparent, "Transparent");
                ui.end_row();
            });
            ui.label(RichText::new("Uses the current camera, shading and outline settings.").weak());
            if ui.button("Save PNG...").clicked() {
                if let Some(p) = rfd::FileDialog::new().add_filter("PNG", &["png"]).set_file_name("pose.png").save_file() {
                    app.export_image(p);
                    app.export.open = false;
                }
            }
        });
        if !open {
            app.export.open = false;
        }
    }
    if app.help_open {
        egui::Window::new("Controls").open(&mut app.help_open).default_width(420.0).show(ctx, |ui| {
            egui::Grid::new("help").striped(true).num_columns(2).show(ui, |ui| {
                for (k, v) in [
                    ("Middle mouse drag", "Orbit (Shift: pan, Ctrl: zoom)"),
                    ("Mouse wheel", "Zoom"),
                    ("Alt + left drag", "Orbit without a middle button (Shift pan, Ctrl zoom)"),
                    ("Touchpad 2-finger", "Orbit (Shift: pan), pinch to zoom"),
                    ("1 / 3 / 7", "Front / right / top view (Ctrl: opposite)"),
                    ("2 4 6 8", "Orbit 15° (Ctrl: pan, Shift+4/6: roll)"),
                    ("5", "Perspective / orthographic"),
                    ("Home  /  . or F", "Frame all / frame selected"),
                    ("Click body part", "Select joint (rings appear)"),
                    ("Drag body part", "IK pose: hands/feet reach, elbows/knees swing, hips move"),
                    ("Drag a ring", "Rotate the joint around that axis (white = view)"),
                    ("G / R / S", "Grab (IK) / rotate / scale bone, Blender style"),
                    ("  then X Y Z", "Constrain to axis (press twice: local)"),
                    ("  then type", "Exact value, Enter/LMB confirm, Esc/RMB cancel"),
                    ("R R", "Trackball rotate"),
                    ("Alt+R / Alt+G / Alt+S", "Clear rotation / location / scale"),
                    ("P", "Pin joint (stays put while posing others)"),
                    ("M", "Flip pose left/right"),
                    ("Ctrl+C / Ctrl+V", "Copy / paste pose (Ctrl+Shift+V flipped)"),
                    ("Ctrl+Z / Ctrl+Shift+Z", "Undo / redo"),
                    ("Shift+D / X", "Duplicate / delete figure"),
                    ("H / Alt+H", "Hide / unhide"),
                    ("N / T", "Toggle sidebar / toolbar"),
                    ("Drop files", "Import GLB, GLTF, VRM, FBX, OBJ"),
                ] {
                    ui.label(RichText::new(k).strong());
                    ui.label(v);
                    ui.end_row();
                }
            });
        });
    }
}

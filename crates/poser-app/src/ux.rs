//! Ease-of-use helpers: right-click joint menu, parent/child joint stepping, hover hints.
use crate::app::{PoserApp, Selection};
use crate::gizmo::ViewCtx;
use poser_core::humanoid::{HumanBone as H, Side};

impl PoserApp {
    /// Right-click (a click, not an orbit drag) on the figure: select what's under the pointer
    /// and offer the common actions for it.
    pub fn joint_context_menu(&mut self, resp: &egui::Response, view: &ViewCtx) {
        if resp.secondary_clicked() {
            if let Some(p) = resp.interact_pointer_pos() {
                if let Some((f, bone, _)) = self.pick(view, p) {
                    self.sel = Selection { figure: Some(f), bone };
                    self.needs_render = true;
                }
            }
        }
        resp.context_menu(|ui| {
            ui.set_min_width(210.0);
            self.joint_menu(ui);
        });
    }

    fn joint_menu(&mut self, ui: &mut egui::Ui) {
        let Some(f) = self.sel.figure.filter(|f| *f < self.scene.figures.len()) else {
            ui.label("Right-click a figure for options");
            return;
        };
        let humanoid = self.scene.figures[f].model.humanoid.is_valid();
        if let Some(b) = self.sel.bone {
            let fig = &self.scene.figures[f];
            let name = fig.model.humanoid.human_of(b).map(|h| h.name().to_string()).unwrap_or_else(|| fig.model.skeleton.bones[b].name.clone());
            ui.label(egui::RichText::new(name).strong());
            if ui.button("Reset this joint").clicked() {
                self.push_undo("Reset joint");
                let fig = &mut self.scene.figures[f];
                fig.pose.locals[b].rotation = fig.model.skeleton.bones[b].rest.rotation;
                fig.touch();
                ui.close();
            }
            if ui.button("Reset this limb").on_hover_text("This joint and everything below it").clicked() {
                self.push_undo("Reset limb");
                let fig = &mut self.scene.figures[f];
                let sk = fig.model.skeleton.clone();
                for i in 0..sk.len() {
                    if i == b || sk.is_ancestor(b, i) {
                        fig.pose.locals[i].rotation = sk.bones[i].rest.rotation;
                    }
                }
                fig.touch();
                ui.close();
            }
            let side = self.scene.figures[f].model.humanoid.human_of(b).map(|h| h.side());
            if let Some(s) = side.filter(|s| *s != Side::Center) {
                let label = if s == Side::Left { "Copy to the right side" } else { "Copy to the left side" };
                if ui.button(label).on_hover_text("Mirror this side's pose onto the other side").clicked() {
                    self.mirror_pose(f, s == Side::Left);
                    ui.close();
                }
            }
            let pinned = self.pins.contains_key(&(self.scene.figures[f].id, b));
            if ui.button(if pinned { "Unpin" } else { "Pin in place (P)" }).clicked() {
                self.toggle_pin(f, b);
                ui.close();
            }
            ui.horizontal(|ui| {
                if ui.button("Parent  [").clicked() {
                    self.step_joint(false);
                }
                if ui.button("Child  ]").clicked() {
                    self.step_joint(true);
                }
            });
            if ui.button("Frame (F)").clicked() {
                self.frame_selected();
                ui.close();
            }
            ui.separator();
        }
        if humanoid {
            if ui.button("Reset whole pose").clicked() {
                self.reset_pose(f, false);
                ui.close();
            }
            let id = self.scene.figures[f].id;
            let mut look = self.look_at.contains(&id);
            if ui.checkbox(&mut look, "Look at camera").changed() {
                if look {
                    self.look_at.insert(id);
                } else {
                    self.look_at.remove(&id);
                }
                self.needs_render = true;
            }
            ui.checkbox(&mut self.opts.symmetry, "Live symmetry");
        }
        if ui.button("Drop to floor").clicked() {
            self.drop_to_floor(f);
            ui.close();
        }
        if ui.button("Duplicate (Shift+D)").clicked() {
            self.duplicate_selected();
            ui.close();
        }
        if ui.button("Delete (X)").clicked() {
            self.delete_selected();
            ui.close();
        }
    }

    /// Select the parent (`child == false`) or child joint of the selection, skipping helpers.
    pub fn step_joint(&mut self, child: bool) {
        let (Some(f), Some(b)) = (self.sel.figure, self.sel.bone) else { return };
        let fig = &self.scene.figures[f];
        let sk = &fig.model.skeleton;
        let hm = &fig.model.humanoid;
        let next = if child {
            let via_map = hm.human_of(b).and_then(|h| h.tail_child()).and_then(|t| hm.get(t));
            via_map.or_else(|| sk.bones[b].children.first().copied())
        } else {
            let mut p = sk.bones[b].parent;
            while let Some(x) = p {
                if self.opts.all_bones || !hm.is_valid() || hm.human_of(x).is_some() {
                    break;
                }
                p = sk.bones[x].parent;
            }
            p
        };
        if let Some(n) = next {
            self.sel.bone = Some(n);
            self.needs_render = true;
        }
    }

    /// Short hint for what a drag on the hovered part will do (shown in the status bar).
    pub fn hover_hint(&self) -> Option<String> {
        let (f, b) = self.hover?;
        let fig = self.scene.figures.get(f)?;
        let Some(b) = b else {
            return Some(format!("{}: click to select, drag to move it (W/E/S for gizmos), right-click for options", fig.name));
        };
        let hm = &fig.model.humanoid;
        let what = match hm.human_of(b) {
            Some(H::LeftHand | H::RightHand) => "drag to reach with the whole arm (the hand stays on the forearm)",
            Some(H::LeftFoot | H::RightFoot) => "drag to move the foot with the whole leg",
            Some(H::LeftLowerArm | H::RightLowerArm | H::LeftLowerLeg | H::RightLowerLeg) => "drag to swing the elbow/knee round while the hand/foot stays",
            Some(H::Hips) => "drag to move the hips (feet stay planted)",
            Some(H::Head | H::Neck) => "drag to turn and tilt the head",
            Some(H::Spine | H::Chest | H::UpperChest) => "drag to bend the spine",
            Some(h) if h.is_finger() => "drag to curl the finger",
            _ => "drag to aim it",
        };
        let name = hm.human_of(b).map(|h| h.name().to_string()).unwrap_or_else(|| fig.model.skeleton.bones[b].name.clone());
        Some(format!("{name}: {what}. Rings rotate it (past its limit the next joint up follows). Right-click: options, [ ]: parent/child"))
    }
}

//! Pose / hand preset thumbnails rendered offscreen once and cached as egui textures.
use crate::app::PoserApp;
use crate::deform::DeformCache;
use poser_core::glam::{Quat, Vec3};
use poser_core::humanoid::HumanBone as H;
use poser_core::makehuman::{self, HumanParams};
use poser_core::pose::{HumanPose, apply_human_pose};
use poser_core::scene::Scene;
use std::collections::HashMap;
use std::sync::Arc;

pub const THUMB: (u32, u32) = (120, 150);

#[derive(Default)]
pub struct Thumbs {
    pub map: HashMap<String, egui::TextureHandle>,
}

impl PoserApp {
    /// Render thumbnails for every body, user and hand preset that doesn't have one yet.
    pub fn ensure_thumbnails(&mut self, ctx: &egui::Context) {
        let mut todo: Vec<(String, HumanPose, bool)> = vec![];
        for p in self.body_presets.iter().chain(self.user_poses.iter()) {
            let key = format!("pose:{}", p.name);
            if !self.thumbs.map.contains_key(&key) {
                todo.push((key, p.clone(), false));
            }
        }
        for p in &self.hand_presets {
            let key = format!("hand:{}", p.name);
            if !self.thumbs.map.contains_key(&key) {
                let mut hp = HumanPose { name: p.name.clone(), ..Default::default() };
                hp.rotations.insert(H::LeftUpperArm, Quat::from_rotation_arc(Vec3::X, Vec3::new(0.3, -0.2, 0.93).normalize()));
                hp.rotations.insert(H::LeftLowerArm, Quat::from_rotation_y(-100f32.to_radians()));
                hp.rotations.insert(H::RightUpperArm, Quat::from_rotation_z(80f32.to_radians()));
                for (b, q) in &p.rotations {
                    hp.rotations.insert(*b, *q);
                }
                todo.push((key, hp, true));
            }
        }
        if todo.is_empty() || self.renderer.is_none() {
            return;
        }
        let mut tmp = Scene::default();
        tmp.render = self.scene.render.clone();
        tmp.render.show_grid = false;
        tmp.render.show_floor = false;
        tmp.render.shadows = false;
        tmp.render.shading = poser_core::scene::Shading::Solid;
        tmp.render.background_top = poser_core::glam::Vec4::new(0.3, 0.3, 0.32, 1.0);
        tmp.render.background_bottom = poser_core::glam::Vec4::new(0.22, 0.22, 0.24, 1.0);
        let model = Arc::new(makehuman::build(&HumanParams::default()));
        tmp.add_model(model.clone());
        for (key, pose, hand) in todo {
            let fig = &mut tmp.figures[0];
            fig.pose = poser_core::pose::Pose::rest(&model.skeleton);
            apply_human_pose(&model.skeleton, &model.humanoid, &pose, &mut fig.pose.locals, None, true);
            fig.touch();
            let mut cache = DeformCache::default();
            cache.update(&tmp, None, None);
            let (c, r) = cache.scene_bounds(&tmp);
            let mut cam = poser_core::camera::Camera::default();
            if hand {
                cam.set_yaw_pitch(40f32.to_radians(), -15f32.to_radians());
                let cc = &cache.figs[&tmp.figures[0].id];
                let hb = model.humanoid.get(H::LeftHand).unwrap();
                let hp = cc.world.transform_point3(cc.pose.world[hb].w_axis.truncate());
                cam.frame(hp + Vec3::new(0.0, 0.03, 0.05), 0.12);
            } else {
                cam.set_yaw_pitch(30f32.to_radians(), -8f32.to_radians());
                cam.frame(c, r * 0.62);
            }
            let draws = crate::viewport::build_draws(&tmp, &cache, true);
            let desc = poser_render::FrameDesc {
                camera: &cam,
                settings: &tmp.render,
                lights: &tmp.lights,
                width: THUMB.0,
                height: THUMB.1,
                transparent: false,
                bounds_center: c,
                bounds_radius: r,
            };
            let rd = self.renderer.as_mut().unwrap();
            rd.use_slot(2);
            rd.render(&desc, &draws);
            if let Some((w, h, px)) = rd.read_output() {
                let img = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &px);
                self.thumbs.map.insert(key.clone(), ctx.load_texture(key, img, egui::TextureOptions::LINEAR));
            }
        }
        // The viewport's GPU buffers were evicted: re-upload everything next frame.
        for f in &mut self.scene.figures {
            f.touch();
        }
        for c in self.cache.figs.values_mut() {
            c.overlay_key.2 = !c.overlay_key.2;
        }
        self.needs_render = true;
    }
}

//! Shared post-processing for all importers.
use glam::{Mat4, Quat, Vec3};
use poser_core::humanoid::{HumanBone, detect_humanoid_weighted};
use poser_core::model::Model;

/// Downscale RGBA8 so neither side exceeds `max` (keeps VRAM low on integrated GPUs).
pub fn downscale_rgba(w: u32, h: u32, data: Vec<u8>, max: u32) -> (u32, u32, Vec<u8>) {
    if w <= max && h <= max {
        return (w, h, data);
    }
    let Some(img) = image::RgbaImage::from_raw(w, h, data) else {
        return (1, 1, vec![255; 4]);
    };
    let s = max as f32 / w.max(h) as f32;
    let nw = ((w as f32 * s) as u32).max(1);
    let nh = ((h as f32 * s) as u32).max(1);
    let r = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Triangle);
    (nw, nh, r.into_raw())
}

/// Rotate the whole model (rest pose) about the origin.
pub fn rotate_model(model: &mut Model, q: Quat) {
    let r = Mat4::from_quat(q);
    if model.has_skeleton() {
        model.skeleton.root_transform = r * model.skeleton.root_transform;
    }
    for m in &mut model.meshes {
        if !m.is_skinned() && m.rigid_bone.is_none() {
            m.rigid_transform = r * m.rigid_transform;
        }
    }
    if !model.has_skeleton() {
        return;
    }
    // Skinned meshes follow automatically (joint world includes the root transform).
}

/// Detect humanoid, fix facing so the character looks down +Z, compute bone axes and T-pose.
pub fn finalize(model: &mut Model) {
    if !model.has_skeleton() {
        return;
    }
    model.skeleton.rebuild_children();
    if !model.humanoid.is_valid() {
        let mut deforming = vec![false; model.skeleton.len()];
        for m in &model.meshes {
            if let Some(b) = m.rigid_bone {
                deforming[b] = true;
            }
            for (j, w) in m.joints.iter().zip(&m.weights) {
                for k in 0..4 {
                    if w[k] > 0.0 && (j[k] as usize) < deforming.len() {
                        deforming[j[k] as usize] = true;
                    }
                }
            }
        }
        let det = detect_humanoid_weighted(&model.skeleton, Some(&deforming));
        // Keep explicit (VRM) mappings, fill gaps with detection.
        for (h, b) in det.bones {
            model.humanoid.bones.entry(h).or_insert(b);
        }
    }
    // Facing: the character's left must point to +X.
    if model.humanoid.is_valid() {
        let w = model.skeleton.rest_world();
        let p = |h: HumanBone| model.humanoid.get(h).map(|b| w[b].w_axis.truncate());
        let left = match (p(HumanBone::LeftUpperArm), p(HumanBone::RightUpperArm)) {
            (Some(l), Some(r)) => l - r,
            _ => match (p(HumanBone::LeftUpperLeg), p(HumanBone::RightUpperLeg)) {
                (Some(l), Some(r)) => l - r,
                _ => Vec3::X,
            },
        };
        let flat = Vec3::new(left.x, 0.0, left.z).normalize_or(Vec3::X);
        let angle = flat.z.atan2(flat.x); // angle from +X towards +Z ... rotate back by it
        if angle.abs() > 0.5 {
            // Rotation about +Y by `angle` maps flat -> +X.
            let q = Quat::from_rotation_y(angle);
            rotate_model(model, q);
        }
    }
    let map = &model.humanoid;
    let rev = map.reverse(model.skeleton.len());
    let prefer: Vec<Option<usize>> = rev
        .iter()
        .map(|h| h.and_then(|h| h.tail_child()).and_then(|t| map.get(t)))
        .collect();
    model.skeleton.compute_axes(|i| prefer[i]);
    if model.humanoid.is_valid() {
        let mut m = std::mem::take(&mut model.humanoid);
        m.compute_tpose(&model.skeleton);
        model.humanoid = m;
    }
}

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
fn hm_pos(model: &Model, w: &[Mat4], h: HumanBone) -> Option<Vec3> {
    model.humanoid.get(h).map(|b| w[b].w_axis.truncate())
}

/// Game rips: an attachment bone (briefcase handle, weapon socket) parented to a hand but
/// exported sitting at the model origin, because the game snaps it onto the hand at runtime.
/// Put it where the game would: on its parent bone.
fn snap_detached_attachments(model: &mut Model) {
    let w = model.skeleton.rest_world();
    // Body scale: hips to feet (hips_height isn't computed yet at this point).
    let hm = &model.humanoid;
    let y = |h: HumanBone| hm.get(h).map(|b| w[b].w_axis.truncate());
    let hh = match (y(HumanBone::Hips), y(HumanBone::LeftFoot).or(y(HumanBone::RightFoot))) {
        (Some(h), Some(f)) => (h - f).length(),
        _ => return,
    }
    .max(1e-4);
    let deforming = {
        let mut d = vec![false; model.skeleton.len()];
        for m in &model.meshes {
            for (j, wt) in m.joints.iter().zip(&m.weights) {
                for k in 0..4 {
                    if wt[k] > 0.0 && (j[k] as usize) < d.len() {
                        d[j[k] as usize] = true;
                    }
                }
            }
        }
        d
    };
    let sk = &model.skeleton;
    let subtree_deforms = |b: usize| (0..sk.len()).any(|i| deforming[i] && (i == b || sk.is_ancestor(b, i)));
    let mut snap = vec![];
    for (i, bone) in sk.bones.iter().enumerate() {
        let Some(p) = bone.parent else { continue };
        if model.humanoid.human_of(i).is_some() || model.humanoid.human_of(p).is_none() {
            continue;
        }
        let pos = w[i].w_axis.truncate();
        let ppos = w[p].w_axis.truncate();
        // Body bones are never at the origin, so a child sitting there, far from its parent, is
        // a detached socket.
        if pos.length() < 0.05 * hh && (pos - ppos).length() > 0.4 * hh && subtree_deforms(i) {
            snap.push(i);
        }
    }
    for i in snap {
        log::info!("snapping detached attachment bone {} onto its parent", model.skeleton.bones[i].name);
        let p = model.skeleton.bones[i].parent.unwrap();
        // Grip point: in the palm when the parent is a hand, else the parent joint itself.
        let side = model.humanoid.human_of(p).map(|h| h.side());
        let knuckle = match (model.humanoid.human_of(p), side) {
            (Some(HumanBone::LeftHand), _) => hm_pos(model, &w, HumanBone::LeftMiddleProximal),
            (Some(HumanBone::RightHand), _) => hm_pos(model, &w, HumanBone::RightMiddleProximal),
            _ => None,
        };
        let pp = w[p].w_axis.truncate();
        let grip = knuckle.map(|k| pp.lerp(k, 0.6)).unwrap_or(pp);
        // Keep the orientation it was modelled in (a case hangs upright from its handle).
        // Bind pose in skeleton space: the parent's rest skin matrix carries any unit/axis
        // conversion the file applies to every bone.
        let c = w[p] * model.skeleton.bones[p].inverse_bind;
        let mut bind = c * model.skeleton.bones[i].inverse_bind.inverse();
        bind.w_axis = grip.extend(1.0);
        model.skeleton.bones[i].rest = poser_core::transform::Transform::from_mat4(&(w[p].inverse() * bind));
    }
}

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
    if model.humanoid.is_valid() {
        snap_detached_attachments(model);
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
        // Game rigs: skin driven by cloth-sim / helper bones that hang off the root would stay
        // behind when the body moves; let it follow the body underneath instead.
        poser_core::edit::adopt_floating_weights(model);
    }
}

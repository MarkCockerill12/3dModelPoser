//! Screen-space picking against deformed meshes (works with any projection mode).
use crate::camera::Camera;
use crate::model::Mesh;
use glam::{Mat4, Vec2, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub figure: usize,
    pub mesh: usize,
    pub triangle: usize,
    pub bary: Vec3,
    pub point: Vec3,
    pub depth: f32,
}

/// Test one deformed mesh (positions already in model space, `world` maps to world space).
#[allow(clippy::too_many_arguments)]
pub fn pick_mesh(
    cam: &Camera,
    size: Vec2,
    cursor: Vec2,
    world: Mat4,
    positions: &[Vec3],
    indices: &[u32],
    figure: usize,
    mesh: usize,
    best: &mut Option<Hit>,
) {
    let view = cam.view() * world;
    let aspect = size.x / size.y.max(1.0);
    let pp = cam.params(aspect);
    // Project all vertices once.
    let proj: Vec<Option<(Vec2, f32)>> = positions
        .iter()
        .map(|p| {
            let v = view.transform_point3(*p);
            let c = crate::camera::view_to_clip(&pp, v);
            if c.w <= 1e-6 {
                return None;
            }
            let ndc = c.truncate() / c.w;
            Some((Vec2::new((ndc.x * 0.5 + 0.5) * size.x, (0.5 - ndc.y * 0.5) * size.y), -v.z))
        })
        .collect();
    for (t, tri) in indices.chunks_exact(3).enumerate() {
        let (Some(a), Some(b), Some(c)) = (proj[tri[0] as usize], proj[tri[1] as usize], proj[tri[2] as usize]) else {
            continue;
        };
        let (pa, pb, pc) = (a.0, b.0, c.0);
        // Bounding box reject.
        let lo = pa.min(pb).min(pc);
        let hi = pa.max(pb).max(pc);
        if cursor.x < lo.x || cursor.y < lo.y || cursor.x > hi.x || cursor.y > hi.y {
            continue;
        }
        let v0 = pb - pa;
        let v1 = pc - pa;
        let v2 = cursor - pa;
        let den = v0.x * v1.y - v1.x * v0.y;
        if den.abs() < 1e-9 {
            continue;
        }
        let v = (v2.x * v1.y - v1.x * v2.y) / den;
        let w = (v0.x * v2.y - v2.x * v0.y) / den;
        let u = 1.0 - v - w;
        if u < -1e-4 || v < -1e-4 || w < -1e-4 {
            continue;
        }
        // Perspective-correct depth interpolation.
        let inv = u / a.1.max(1e-6) + v / b.1.max(1e-6) + w / c.1.max(1e-6);
        let depth = 1.0 / inv.max(1e-9);
        if best.is_none_or(|h| depth < h.depth) {
            let bary = Vec3::new(u / a.1, v / b.1, w / c.1) * depth;
            let p = positions[tri[0] as usize] * bary.x + positions[tri[1] as usize] * bary.y + positions[tri[2] as usize] * bary.z;
            *best = Some(Hit { figure, mesh, triangle: t, bary, point: world.transform_point3(p), depth });
        }
    }
}

/// Dominant bone at a hit on a skinned mesh.
pub fn hit_bone(mesh: &Mesh, hit: &Hit) -> Option<usize> {
    if !mesh.is_skinned() {
        return mesh.rigid_bone;
    }
    let tri = &mesh.indices[hit.triangle * 3..hit.triangle * 3 + 3];
    let mut acc: Vec<(usize, f32)> = vec![];
    for (k, &vi) in tri.iter().enumerate() {
        let b = hit.bary[k];
        let j = mesh.joints[vi as usize];
        let w = mesh.weights[vi as usize];
        for i in 0..4 {
            if w[i] <= 0.0 {
                continue;
            }
            let bone = j[i] as usize;
            match acc.iter_mut().find(|(x, _)| *x == bone) {
                Some(e) => e.1 += w[i] * b,
                None => acc.push((bone, w[i] * b)),
            }
        }
    }
    acc.into_iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|(b, _)| b)
}

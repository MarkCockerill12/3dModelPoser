//! Simple model edits: find separate pieces of a mesh ("islands") and delete them, e.g. a
//! briefcase or hat that came glued into an imported character.
use crate::glam::Vec3;
use crate::model::{Mesh, Model};
use std::collections::HashMap;

/// Connected pieces of a mesh: triangles that share a vertex *position* (so UV seams don't split
/// a piece). Returns the island id of every triangle and the number of islands.
pub fn mesh_islands(mesh: &Mesh) -> (Vec<u32>, usize) {
    let n = mesh.positions.len();
    // Weld by position first.
    let key = |p: Vec3| ((p.x * 1e5).round() as i64, (p.y * 1e5).round() as i64, (p.z * 1e5).round() as i64);
    let mut weld: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let canon: Vec<u32> = mesh.positions.iter().enumerate().map(|(i, p)| *weld.entry(key(*p)).or_insert(i as u32)).collect();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    for t in mesh.indices.chunks_exact(3) {
        let a = find(&mut parent, canon[t[0] as usize]);
        for &v in &t[1..] {
            let b = find(&mut parent, canon[v as usize]);
            if a != b {
                parent[b as usize] = a;
            }
        }
    }
    let mut ids: HashMap<u32, u32> = HashMap::new();
    let tri_island: Vec<u32> = mesh
        .indices
        .chunks_exact(3)
        .map(|t| {
            let r = find(&mut parent, canon[t[0] as usize]);
            let next = ids.len() as u32;
            *ids.entry(r).or_insert(next)
        })
        .collect();
    (tri_island, ids.len())
}

/// Vertices used by the triangles of `island` (for highlighting).
pub fn island_vertices(mesh: &Mesh, tri_island: &[u32], island: u32) -> Vec<u32> {
    let mut used = vec![false; mesh.positions.len()];
    for (t, tri) in mesh.indices.chunks_exact(3).enumerate() {
        if tri_island.get(t) == Some(&island) {
            for &v in tri {
                used[v as usize] = true;
            }
        }
    }
    (0..used.len() as u32).filter(|v| used[*v as usize]).collect()
}

/// Copy of `mesh` keeping only triangles for which `keep(triangle)` is true; unused vertices
/// (and their skin weights, colours, morph deltas) are dropped.
pub fn filter_triangles(mesh: &Mesh, keep: impl Fn(usize) -> bool) -> Mesh {
    let mut out = mesh.clone();
    let mut remap = vec![u32::MAX; mesh.positions.len()];
    let mut order: Vec<u32> = vec![];
    let mut indices = vec![];
    for (t, tri) in mesh.indices.chunks_exact(3).enumerate() {
        if !keep(t) {
            continue;
        }
        for &v in tri {
            if remap[v as usize] == u32::MAX {
                remap[v as usize] = order.len() as u32;
                order.push(v);
            }
            indices.push(remap[v as usize]);
        }
    }
    let pick = |len: usize| len == mesh.positions.len();
    let take = |v: &[crate::glam::Vec3]| order.iter().map(|i| v[*i as usize]).collect::<Vec<_>>();
    out.positions = take(&mesh.positions);
    if pick(mesh.normals.len()) {
        out.normals = take(&mesh.normals);
    }
    if pick(mesh.uvs.len()) {
        out.uvs = order.iter().map(|i| mesh.uvs[*i as usize]).collect();
    }
    if pick(mesh.colors.len()) {
        out.colors = order.iter().map(|i| mesh.colors[*i as usize]).collect();
    }
    if pick(mesh.joints.len()) && pick(mesh.weights.len()) {
        out.joints = order.iter().map(|i| mesh.joints[*i as usize]).collect();
        out.weights = order.iter().map(|i| mesh.weights[*i as usize]).collect();
    }
    if pick(mesh.part_ids.len()) {
        out.part_ids = order.iter().map(|i| mesh.part_ids[*i as usize]).collect();
    }
    out.indices = indices;
    for (mt, src) in out.morphs.iter_mut().zip(&mesh.morphs) {
        mt.indices.clear();
        mt.position_deltas.clear();
        mt.normal_deltas.clear();
        for (k, &vi) in src.indices.iter().enumerate() {
            let r = remap.get(vi as usize).copied().unwrap_or(u32::MAX);
            if r != u32::MAX {
                mt.indices.push(r);
                mt.position_deltas.push(src.position_deltas[k]);
                if let Some(n) = src.normal_deltas.get(k) {
                    mt.normal_deltas.push(*n);
                }
            }
        }
    }
    out
}

/// Remove whole islands: `remove` lists (mesh index, island id) as given by [`mesh_islands`].
/// Meshes left without triangles are dropped.
pub fn remove_islands(model: &Model, remove: &[(usize, u32)]) -> Model {
    let mut out = model.clone();
    let mut meshes = vec![];
    for (mi, m) in model.meshes.iter().enumerate() {
        let gone: Vec<u32> = remove.iter().filter(|r| r.0 == mi).map(|r| r.1).collect();
        if gone.is_empty() {
            meshes.push(m.clone());
            continue;
        }
        let (tri_island, _) = mesh_islands(m);
        let nm = filter_triangles(m, |t| !gone.contains(&tri_island[t]));
        if !nm.indices.is_empty() {
            meshes.push(nm);
        }
    }
    out.meshes = meshes;
    out
}

/// Remove whole meshes by index.
pub fn remove_meshes(model: &Model, remove: &[usize]) -> Model {
    let mut out = model.clone();
    out.meshes = model.meshes.iter().enumerate().filter(|(i, _)| !remove.contains(i)).map(|(_, m)| m.clone()).collect();
    out
}

/// Game rigs often skin clothing to cloth-simulation or helper bones that hang off the root
/// instead of the body (Source 2 `PHYS_cloth_*`). Posing the body would leave that skin behind
/// (a jacket panel floating where it was). Every weight on such a bone is handed to the body
/// skin nearest to the vertex, so the clothing follows the body underneath; a mesh that hangs
/// entirely off such bones (a loose prop) moves rigidly with the closest body bone.
/// Returns the number of vertices changed.
pub fn adopt_floating_weights(model: &mut Model) -> usize {
    use crate::humanoid::HumanBone;
    let sk = &model.skeleton;
    let Some(hips) = model.humanoid.get(HumanBone::Hips) else { return 0 };
    let n = sk.len();
    let on_body: Vec<bool> = (0..n).map(|b| b == hips || sk.is_ancestor(hips, b)).collect();
    let rest = model.rest_positions();
    // Floating share of every vertex.
    let floating = |m: &Mesh, v: usize| -> f32 {
        if !m.is_skinned() {
            return 0.0;
        }
        (0..4).filter(|k| m.weights[v][*k] > 0.0 && !on_body.get(m.joints[v][*k] as usize).copied().unwrap_or(true)).map(|k| m.weights[v][k]).sum()
    };
    let mut any = false;
    for m in &model.meshes {
        if (0..m.positions.len()).any(|v| floating(m, v) > 1e-4) {
            any = true;
            break;
        }
    }
    if !any {
        return 0;
    }
    // Donors: vertices driven only by the body. Uniform grid for nearest-neighbour lookup.
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for ps in &rest {
        for p in ps {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
    }
    let cell = ((hi - lo).max_element() / 48.0).max(1e-5);
    let key = |p: Vec3| {
        let c = ((p - lo) / cell).floor();
        (c.x as i32, c.y as i32, c.z as i32)
    };
    let mut grid: HashMap<(i32, i32, i32), Vec<(usize, usize)>> = HashMap::new();
    for (mi, m) in model.meshes.iter().enumerate() {
        if !m.is_skinned() {
            continue;
        }
        for v in 0..m.positions.len() {
            if floating(m, v) <= 1e-4 {
                grid.entry(key(rest[mi][v])).or_default().push((mi, v));
            }
        }
    }
    if grid.is_empty() {
        return 0;
    }
    let nearest = |p: Vec3| -> Option<(usize, usize, f32)> {
        let c = key(p);
        let mut best: Option<(usize, usize, f32)> = None;
        for r in 0..48i32 {
            // Once something is found, one more ring is enough to be sure it is the closest.
            if let Some(b) = best {
                if (r as f32 - 1.0) * cell > b.2 {
                    break;
                }
            }
            for x in -r..=r {
                for y in -r..=r {
                    for z in -r..=r {
                        if x.abs().max(y.abs()).max(z.abs()) != r {
                            continue;
                        }
                        if let Some(list) = grid.get(&(c.0 + x, c.1 + y, c.2 + z)) {
                            for &(mi, v) in list {
                                let d = (rest[mi][v] - p).length();
                                if best.is_none_or(|b| d < b.2) {
                                    best = Some((mi, v, d));
                                }
                            }
                        }
                    }
                }
            }
        }
        best
    };
    let donor = |mi: usize, v: usize, meshes: &[Mesh]| (meshes[mi].joints[v], meshes[mi].weights[v]);
    let mut changed = 0;
    let snapshot = model.meshes.clone();
    for (mi, m) in model.meshes.iter_mut().enumerate() {
        if !m.is_skinned() {
            continue;
        }
        let nv = m.positions.len();
        let fl: Vec<f32> = (0..nv).map(|v| floating(&snapshot[mi], v)).collect();
        if fl.iter().all(|f| *f <= 1e-4) {
            continue;
        }
        if fl.iter().all(|f| *f >= 0.999) {
            // A loose object: follow the closest body bone rigidly.
            let Some((dm, dv, _)) = (0..nv).step_by((nv / 2000).max(1)).filter_map(|v| nearest(rest[mi][v])).min_by(|a, b| a.2.total_cmp(&b.2)) else { continue };
            let (j, w) = donor(dm, dv, &snapshot);
            let k = (0..4).max_by(|a, b| w[*a].total_cmp(&w[*b])).unwrap_or(0);
            for v in 0..nv {
                m.joints[v] = [j[k], 0, 0, 0];
                m.weights[v] = [1.0, 0.0, 0.0, 0.0];
            }
            changed += nv;
            continue;
        }
        for v in 0..nv {
            if fl[v] <= 1e-4 {
                continue;
            }
            let Some((dm, dv, _)) = nearest(rest[mi][v]) else { continue };
            let (dj, dw) = donor(dm, dv, &snapshot);
            let mut acc: Vec<(u16, f32)> = vec![];
            let mut add = |j: u16, w: f32| {
                if w <= 0.0 {
                    return;
                }
                match acc.iter_mut().find(|x| x.0 == j) {
                    Some(x) => x.1 += w,
                    None => acc.push((j, w)),
                }
            };
            for k in 0..4 {
                let j = m.joints[v][k];
                if on_body.get(j as usize).copied().unwrap_or(false) {
                    add(j, m.weights[v][k]);
                }
            }
            for k in 0..4 {
                add(dj[k], dw[k] * fl[v]);
            }
            acc.sort_by(|a, b| b.1.total_cmp(&a.1));
            acc.truncate(4);
            let total: f32 = acc.iter().map(|x| x.1).sum::<f32>().max(1e-6);
            let mut j = [0u16; 4];
            let mut w = [0f32; 4];
            for (k, (bj, bw)) in acc.into_iter().enumerate() {
                j[k] = bj;
                w[k] = bw / total;
            }
            m.joints[v] = j;
            m.weights[v] = w;
            changed += 1;
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn islands_split_and_remove() {
        let a = crate::props::build(crate::props::PropKind::Chair);
        let m = &a.meshes[0];
        let (_, n) = mesh_islands(m);
        // 4 legs + seat + 2 back posts + backrest.
        assert_eq!(n, 8, "chair islands");
        let b = remove_islands(&a, &[(0, 0)]);
        assert!(b.meshes[0].triangle_count() < m.triangle_count());
        assert!(b.meshes[0].positions.len() < m.positions.len());
        assert!(b.meshes[0].indices.iter().all(|i| (*i as usize) < b.meshes[0].positions.len()));
    }
}

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

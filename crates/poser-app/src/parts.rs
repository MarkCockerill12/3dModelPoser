//! "Edit parts" tool: click separate pieces of a model (a briefcase, a hat, a stray prop) and
//! delete them. Works on pieces (connected islands) or whole meshes.
use crate::app::PoserApp;
use crate::gizmo::ViewCtx;
use egui::Pos2;
use poser_core::edit::{island_vertices, mesh_islands, remove_islands};
use poser_core::glam::{Vec3, Vec4};
use poser_core::picking::{Hit, pick_mesh};
use poser_core::scene::PartEdit;
use std::sync::Arc;

pub struct PartsSession {
    pub figure_id: u64,
    /// Click selects whole meshes instead of single pieces.
    pub by_mesh: bool,
    model_key: usize,
    /// Island id of every triangle, per mesh.
    islands: Vec<Vec<u32>>,
    counts: Vec<usize>,
    /// Rest-pose bounds of every island, per mesh.
    bounds: Vec<Vec<(Vec3, Vec3)>>,
    pub selected: Vec<(usize, u32)>,
    pub hover: Vec<(usize, u32)>,
    rev: u64,
}

impl PartsSession {
    fn units(&self, mesh: usize, island: u32) -> Vec<(usize, u32)> {
        if self.by_mesh {
            (0..self.counts[mesh] as u32).map(|i| (mesh, i)).collect()
        } else {
            vec![(mesh, island)]
        }
    }

    pub fn piece_count(&self) -> usize {
        self.counts.iter().sum()
    }
}

impl PoserApp {
    pub fn start_parts(&mut self, idx: usize) {
        let Some(fig) = self.scene.figures.get(idx) else { return };
        let id = fig.id;
        self.rig = None;
        self.armature = None;
        self.parts = Some(PartsSession {
            figure_id: id,
            by_mesh: false,
            model_key: 0,
            islands: vec![],
            counts: vec![],
            bounds: vec![],
            selected: vec![],
            hover: vec![],
            rev: 0,
        });
        self.refresh_parts();
        self.status = "Edit parts: click pieces to select (Shift adds), Grow selects everything inside, Delete removes, Esc finishes".into();
    }

    pub fn end_parts(&mut self) {
        if let Some(s) = self.parts.take() {
            self.cache.highlight.remove(&s.figure_id);
            self.cache.highlight_rev += 1;
        }
        self.needs_render = true;
    }

    /// Recompute islands when the figure's model changed (after a delete or undo).
    fn refresh_parts(&mut self) {
        let Some(s) = self.parts.as_mut() else { return };
        let Some(fig) = self.scene.figures.iter().find(|f| f.id == s.figure_id) else {
            self.parts = None;
            return;
        };
        let key = Arc::as_ptr(&fig.model) as usize;
        if key == s.model_key {
            return;
        }
        s.model_key = key;
        let rest = fig.model.rest_positions();
        s.islands.clear();
        s.counts.clear();
        s.bounds.clear();
        for (mi, m) in fig.model.meshes.iter().enumerate() {
            let (ti, n) = mesh_islands(m);
            let mut b = vec![(Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)); n];
            for (t, tri) in m.indices.chunks_exact(3).enumerate() {
                for &v in tri {
                    let p = rest[mi][v as usize];
                    let e = &mut b[ti[t] as usize];
                    e.0 = e.0.min(p);
                    e.1 = e.1.max(p);
                }
            }
            s.islands.push(ti);
            s.counts.push(n);
            s.bounds.push(b);
        }
        s.selected.clear();
        s.hover.clear();
        s.rev += 1;
    }

    fn parts_figure(&self) -> Option<usize> {
        self.scene.index_of(self.parts.as_ref()?.figure_id)
    }

    fn pick_part(&self, view: &ViewCtx, p: Pos2) -> Option<(usize, u32)> {
        let f = self.parts_figure()?;
        let fig = &self.scene.figures[f];
        let c = self.cache.figs.get(&fig.id)?;
        let mut best: Option<Hit> = None;
        for (mi, m) in c.meshes.iter().enumerate() {
            pick_mesh(&view.cam, view.size(), view.local(p), c.world, &m.positions, &fig.model.meshes[mi].indices, f, mi, &mut best);
        }
        let h = best?;
        let s = self.parts.as_ref()?;
        Some((h.mesh, *s.islands.get(h.mesh)?.get(h.triangle)?))
    }

    pub fn parts_pointer(&mut self, view: &ViewCtx, pointer: Option<Pos2>, clicked: bool, shift: bool) {
        self.refresh_parts();
        let hit = pointer.and_then(|p| self.pick_part(view, p));
        let Some(s) = self.parts.as_mut() else { return };
        let hover = hit.map(|(m, i)| s.units(m, i)).unwrap_or_default();
        if hover != s.hover {
            s.hover = hover;
            s.rev += 1;
        }
        if clicked {
            match hit {
                Some((m, i)) => {
                    let units = s.units(m, i);
                    let all_in = units.iter().all(|u| s.selected.contains(u));
                    if !shift {
                        s.selected.clear();
                    }
                    if shift && all_in {
                        s.selected.retain(|u| !units.contains(u));
                    } else {
                        for u in units {
                            if !s.selected.contains(&u) {
                                s.selected.push(u);
                            }
                        }
                    }
                }
                None if !shift => s.selected.clear(),
                None => {}
            }
            s.rev += 1;
        }
        self.update_parts_highlight();
    }

    /// Add every piece (in any mesh) whose bounds sit inside the selection's bounds, e.g. the
    /// handle and latches of a selected briefcase.
    pub fn grow_parts(&mut self) {
        let Some(s) = self.parts.as_mut() else { return };
        if s.selected.is_empty() {
            return;
        }
        let mut lo = Vec3::splat(f32::MAX);
        let mut hi = Vec3::splat(f32::MIN);
        for &(m, i) in &s.selected {
            let b = s.bounds[m][i as usize];
            lo = lo.min(b.0);
            hi = hi.max(b.1);
        }
        let pad = (hi - lo) * 0.03;
        let (lo, hi) = (lo - pad, hi + pad);
        for (m, bs) in s.bounds.iter().enumerate() {
            for (i, b) in bs.iter().enumerate() {
                let u = (m, i as u32);
                if !s.selected.contains(&u) && b.0.cmpge(lo).all() && b.1.cmple(hi).all() {
                    s.selected.push(u);
                }
            }
        }
        s.rev += 1;
        self.update_parts_highlight();
    }

    /// Select / deselect a whole mesh from the list.
    pub fn toggle_mesh_selection(&mut self, mesh: usize) {
        let Some(s) = self.parts.as_mut() else { return };
        let units: Vec<(usize, u32)> = (0..s.counts[mesh] as u32).map(|i| (mesh, i)).collect();
        if units.iter().all(|u| s.selected.contains(u)) {
            s.selected.retain(|u| u.0 != mesh);
        } else {
            for u in units {
                if !s.selected.contains(&u) {
                    s.selected.push(u);
                }
            }
        }
        s.rev += 1;
        self.update_parts_highlight();
    }

    pub fn mesh_selected(&self, mesh: usize) -> bool {
        self.parts.as_ref().is_some_and(|s| s.selected.iter().any(|u| u.0 == mesh))
    }

    pub fn delete_selected_parts(&mut self) {
        let Some(f) = self.parts_figure() else { return };
        let Some(s) = self.parts.as_ref() else { return };
        if s.selected.is_empty() {
            self.status = "Nothing selected: click a piece first".into();
            return;
        }
        let sel = s.selected.clone();
        let n = sel.len();
        self.push_undo("Delete parts");
        let fig = &self.scene.figures[f];
        let after_rig = self.rig_markers.contains_key(&fig.id);
        let model = Arc::new(remove_islands(&fig.model, &sel));
        let fig = &mut self.scene.figures[f];
        fig.model = model;
        fig.part_edits.push(PartEdit { islands: sel, after_rig });
        fig.touch();
        self.refresh_parts();
        self.update_parts_highlight();
        self.needs_render = true;
        self.status = format!("Deleted {n} piece{} (Ctrl+Z to undo)", if n == 1 { "" } else { "s" });
    }

    fn update_parts_highlight(&mut self) {
        let Some(f) = self.parts_figure() else { return };
        let Some(s) = self.parts.as_ref() else { return };
        let fig = &self.scene.figures[f];
        let mut out = vec![];
        let mut add = |units: &[(usize, u32)], col: Vec4| {
            for &(m, i) in units {
                if let (Some(mesh), Some(ti)) = (fig.model.meshes.get(m), s.islands.get(m)) {
                    out.push((m, island_vertices(mesh, ti, i), col));
                }
            }
        };
        let hover: Vec<(usize, u32)> = s.hover.iter().filter(|u| !s.selected.contains(u)).copied().collect();
        add(&hover, Vec4::new(1.0, 1.0, 1.0, 0.3));
        add(&s.selected, Vec4::new(1.0, 0.25, 0.2, 0.55));
        self.cache.highlight.insert(fig.id, out);
        self.cache.highlight_rev += 1;
        self.needs_render = true;
    }
}

/// Apply recorded part deletions to a freshly loaded / rebuilt model.
pub fn apply_part_edits(model: poser_core::model::Model, edits: &[PartEdit], after_rig: bool) -> poser_core::model::Model {
    let mut m = model;
    for e in edits.iter().filter(|e| e.after_rig == after_rig) {
        m = remove_islands(&m, &e.islands);
    }
    m
}

//! MikuMikuDance PMX (2.0/2.1) import. PMD files are reported as unsupported for now.
use crate::util::{downscale_rgba, finalize};
use anyhow::{Result, bail};
use glam::{Mat4, Vec2, Vec3, Vec4};
use poser_core::model::{AlphaMode, Material, Mesh, Model, MorphTarget, SourceFormat, TextureData};
use poser_core::skeleton::{Bone, Skeleton};
use poser_core::transform::Transform;
use std::path::Path;
use std::sync::Arc;

/// MMD units are roughly 8 cm.
const SCALE: f32 = 0.08;

struct R<'a> {
    b: &'a [u8],
    p: usize,
    utf8: bool,
}

impl<'a> R<'a> {
    fn need(&self, n: usize) -> Result<()> {
        if self.p + n > self.b.len() {
            bail!("unexpected end of file at {}", self.p);
        }
        Ok(())
    }
    fn u8(&mut self) -> Result<u8> {
        self.need(1)?;
        self.p += 1;
        Ok(self.b[self.p - 1])
    }
    fn i32(&mut self) -> Result<i32> {
        self.need(4)?;
        let v = i32::from_le_bytes(self.b[self.p..self.p + 4].try_into()?);
        self.p += 4;
        Ok(v)
    }
    fn f32(&mut self) -> Result<f32> {
        self.need(4)?;
        let v = f32::from_le_bytes(self.b[self.p..self.p + 4].try_into()?);
        self.p += 4;
        Ok(v)
    }
    fn v2(&mut self) -> Result<Vec2> {
        Ok(Vec2::new(self.f32()?, self.f32()?))
    }
    fn v3(&mut self) -> Result<Vec3> {
        Ok(Vec3::new(self.f32()?, self.f32()?, self.f32()?))
    }
    fn v4(&mut self) -> Result<Vec4> {
        Ok(Vec4::new(self.f32()?, self.f32()?, self.f32()?, self.f32()?))
    }
    fn skip(&mut self, n: usize) -> Result<()> {
        self.need(n)?;
        self.p += n;
        Ok(())
    }
    fn text(&mut self) -> Result<String> {
        let n = self.i32()?.max(0) as usize;
        self.need(n)?;
        let bytes = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(if self.utf8 {
            String::from_utf8_lossy(bytes).to_string()
        } else {
            let u: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            String::from_utf16_lossy(&u)
        })
    }
    /// Signed index of `size` bytes (-1 = none).
    fn idx(&mut self, size: u8) -> Result<i32> {
        Ok(match size {
            1 => self.u8()? as i8 as i32,
            2 => {
                self.need(2)?;
                let v = i16::from_le_bytes([self.b[self.p], self.b[self.p + 1]]);
                self.p += 2;
                v as i32
            }
            _ => self.i32()?,
        })
    }
    /// Unsigned vertex index.
    fn vidx(&mut self, size: u8) -> Result<u32> {
        Ok(match size {
            1 => self.u8()? as u32,
            2 => {
                self.need(2)?;
                let v = u16::from_le_bytes([self.b[self.p], self.b[self.p + 1]]);
                self.p += 2;
                v as u32
            }
            _ => self.i32()? as u32,
        })
    }
}

/// MMD space (left-handed, Y up, model faces -Z) -> ours (right-handed, faces +Z), metres.
fn cv(v: Vec3) -> Vec3 {
    Vec3::new(v.x, v.y, -v.z) * SCALE
}

fn cn(v: Vec3) -> Vec3 {
    Vec3::new(v.x, v.y, -v.z).normalize_or(Vec3::Y)
}

pub fn load(path: &Path) -> Result<Model> {
    let data = std::fs::read(path)?;
    if data.len() < 4 || &data[0..4] != b"PMX " {
        if data.len() > 3 && &data[0..3] == b"Pmd" {
            bail!("PMD (old MikuMikuDance format) is not supported yet. Convert it to PMX with PMXEditor.");
        }
        bail!("not a PMX file");
    }
    let mut r = R { b: &data, p: 4, utf8: false };
    let _version = r.f32()?;
    let nglob = r.u8()? as usize;
    let mut g = [0u8; 8];
    for (i, x) in g.iter_mut().enumerate().take(nglob.min(8)) {
        *x = r.u8()?;
        let _ = i;
    }
    if nglob > 8 {
        r.skip(nglob - 8)?;
    }
    r.utf8 = g[0] == 1;
    let (add_uv, vsz, tsz, msz, bsz, morph_sz) = (g[1] as usize, g[2], g[3], g[4], g[5], g[6]);
    let name_local = r.text()?;
    let name_en = r.text()?;
    let _c1 = r.text()?;
    let _c2 = r.text()?;

    // Vertices
    let nv = r.i32()?.max(0) as usize;
    let mut pos = Vec::with_capacity(nv);
    let mut nrm = Vec::with_capacity(nv);
    let mut uvs = Vec::with_capacity(nv);
    let mut joints = Vec::with_capacity(nv);
    let mut weights = Vec::with_capacity(nv);
    for _ in 0..nv {
        pos.push(cv(r.v3()?));
        nrm.push(cn(r.v3()?));
        uvs.push(r.v2()?);
        r.skip(16 * add_uv)?;
        let kind = r.u8()?;
        let (j, w): ([i32; 4], [f32; 4]) = match kind {
            0 => ([r.idx(bsz)?, -1, -1, -1], [1.0, 0.0, 0.0, 0.0]),
            1 => {
                let a = r.idx(bsz)?;
                let b = r.idx(bsz)?;
                let wa = r.f32()?;
                ([a, b, -1, -1], [wa, 1.0 - wa, 0.0, 0.0])
            }
            2 | 4 => {
                let j = [r.idx(bsz)?, r.idx(bsz)?, r.idx(bsz)?, r.idx(bsz)?];
                let w = [r.f32()?, r.f32()?, r.f32()?, r.f32()?];
                (j, w)
            }
            3 => {
                let a = r.idx(bsz)?;
                let b = r.idx(bsz)?;
                let wa = r.f32()?;
                r.skip(36)?; // SDEF C, R0, R1 (treated as linear blend)
                ([a, b, -1, -1], [wa, 1.0 - wa, 0.0, 0.0])
            }
            k => bail!("unknown weight type {k}"),
        };
        let _edge = r.f32()?;
        let mut jj = [0u16; 4];
        let mut ww = [0f32; 4];
        for k in 0..4 {
            if j[k] >= 0 && w[k] > 0.0 {
                jj[k] = j[k] as u16;
                ww[k] = w[k];
            }
        }
        joints.push(jj);
        weights.push(ww);
    }
    // Faces (flip winding for the handedness change).
    let ni = r.i32()?.max(0) as usize;
    let mut indices = Vec::with_capacity(ni);
    for _ in 0..ni {
        indices.push(r.vidx(vsz)?);
    }
    for t in indices.chunks_exact_mut(3) {
        t.swap(1, 2);
    }
    // Textures
    let base = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let nt = r.i32()?.max(0) as usize;
    let mut tex_paths = vec![];
    for _ in 0..nt {
        tex_paths.push(r.text()?.replace('\\', "/"));
    }
    let mut model = Model::empty(if name_en.trim().is_empty() { name_local.clone() } else { name_en }, SourceFormat::Pmx);
    model.path = Some(path.to_path_buf());
    let mut tex_map: Vec<Option<usize>> = vec![None; nt];
    for (i, tp) in tex_paths.iter().enumerate() {
        let p = base.join(tp);
        if let Ok(img) = image::open(&p) {
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            let (w, h, d) = downscale_rgba(w, h, rgba.into_raw(), 2048);
            model.textures.push(TextureData { name: tp.clone(), width: w, height: h, rgba: Arc::new(d) });
            tex_map[i] = Some(model.textures.len() - 1);
        }
    }
    // Materials: each covers a run of indices -> one mesh per material.
    let nm = r.i32()?.max(0) as usize;
    model.materials.clear();
    let mut runs = vec![];
    for _ in 0..nm {
        let name = r.text()?;
        let _en = r.text()?;
        let diffuse = r.v4()?;
        r.skip(12 + 4 + 12)?; // specular, strength, ambient
        let flags = r.u8()?;
        r.skip(16 + 4)?; // edge colour, edge size
        let tex = r.idx(tsz)?;
        let _env = r.idx(tsz)?;
        let _env_mode = r.u8()?;
        let toon_shared = r.u8()?;
        if toon_shared == 0 {
            r.idx(tsz)?;
        } else {
            r.u8()?;
        }
        let _memo = r.text()?;
        let count = r.i32()?.max(0) as usize;
        let tex_idx = if tex >= 0 { tex_map.get(tex as usize).copied().flatten() } else { None };
        model.materials.push(Material {
            name,
            base_color: Vec4::new(diffuse.x, diffuse.y, diffuse.z, diffuse.w.clamp(0.0, 1.0)),
            base_color_texture: tex_idx,
            alpha_mode: if diffuse.w < 0.99 { AlphaMode::Blend } else { AlphaMode::Mask },
            alpha_cutoff: 0.3,
            double_sided: flags & 1 != 0,
        });
        runs.push(count);
    }
    // Bones
    let nb = r.i32()?.max(0) as usize;
    let mut raw_bones: Vec<(String, i32, Vec3, Option<(i32, f32)>)> = Vec::with_capacity(nb);
    for _ in 0..nb {
        let name = r.text()?;
        let _en = r.text()?;
        let p = cv(r.v3()?);
        let parent = r.idx(bsz)?;
        let _layer = r.i32()?;
        let flags = {
            r.need(2)?;
            let f = u16::from_le_bytes([r.b[r.p], r.b[r.p + 1]]);
            r.p += 2;
            f
        };
        if flags & 0x0001 != 0 {
            r.idx(bsz)?;
        } else {
            r.v3()?;
        }
        let mut inherit = None;
        if flags & 0x0100 != 0 || flags & 0x0200 != 0 {
            let src = r.idx(bsz)?;
            let w = r.f32()?;
            if flags & 0x0100 != 0 {
                inherit = Some((src, w));
            }
        }
        if flags & 0x0400 != 0 {
            r.v3()?;
        }
        if flags & 0x0800 != 0 {
            r.v3()?;
            r.v3()?;
        }
        if flags & 0x2000 != 0 {
            r.i32()?;
        }
        if flags & 0x0020 != 0 {
            r.idx(bsz)?;
            r.i32()?;
            r.f32()?;
            let links = r.i32()?.max(0);
            for _ in 0..links {
                r.idx(bsz)?;
                if r.u8()? == 1 {
                    r.v3()?;
                    r.v3()?;
                }
            }
        }
        raw_bones.push((name, parent, p, inherit));
    }
    // Morphs (vertex morphs only).
    let mut vmorphs: Vec<(String, Vec<(u32, Vec3)>)> = vec![];
    if let Ok(nmorph) = r.i32() {
        for _ in 0..nmorph.max(0) {
            let name = r.text()?;
            let _en = r.text()?;
            let _panel = r.u8()?;
            let kind = r.u8()?;
            let n = r.i32()?.max(0);
            let mut offs = vec![];
            for _ in 0..n {
                match kind {
                    0 | 9 => {
                        r.idx(morph_sz)?;
                        r.f32()?;
                    }
                    1 => {
                        let v = r.vidx(vsz)?;
                        let d = r.v3()?;
                        offs.push((v, Vec3::new(d.x, d.y, -d.z) * SCALE));
                    }
                    2 => {
                        r.idx(bsz)?;
                        r.skip(28)?;
                    }
                    3..=7 => {
                        r.vidx(vsz)?;
                        r.skip(16)?;
                    }
                    8 => {
                        r.idx(msz)?;
                        r.skip(1 + 16 + 12 + 4 + 12 + 16 + 4 + 16 + 16 + 16)?;
                    }
                    10 => {
                        r.idx(g[7])?;
                        r.skip(1 + 24)?;
                    }
                    _ => bail!("unknown morph type {kind}"),
                }
            }
            if kind == 1 && !offs.is_empty() {
                vmorphs.push((name, offs));
            }
        }
    }

    // Order bones parents-first.
    let mut order: Vec<usize> = vec![];
    let mut placed = vec![false; nb];
    let mut guard = 0;
    while order.len() < nb && guard < nb + 2 {
        guard += 1;
        for i in 0..nb {
            if placed[i] {
                continue;
            }
            let p = raw_bones[i].1;
            if p < 0 || p as usize >= nb || placed[p as usize] || p as usize == i {
                placed[i] = true;
                order.push(i);
            }
        }
    }
    let mut new_of = vec![0usize; nb];
    for (n, o) in order.iter().enumerate() {
        new_of[*o] = n;
    }
    let mut skel = Skeleton::default();
    for &o in &order {
        let (name, parent, p, inherit) = &raw_bones[o];
        let par = (*parent >= 0 && (*parent as usize) < nb && *parent as usize != o).then(|| new_of[*parent as usize]);
        let pp = par.map(|q| raw_bones[order[q]].2).unwrap_or(Vec3::ZERO);
        let mut b = Bone::new(name.clone(), par, Transform::from_translation(*p - pp));
        b.inherit = inherit.and_then(|(s, w)| (s >= 0 && (s as usize) < nb).then(|| (new_of[s as usize], w)));
        skel.bones.push(b);
    }
    skel.recompute_inverse_bind();
    for j in &mut joints {
        for x in j.iter_mut() {
            *x = new_of.get(*x as usize).copied().unwrap_or(0) as u16;
        }
    }
    // Split per material into meshes, keeping a compact vertex set per mesh.
    let mut start = 0usize;
    for (mi, count) in runs.iter().enumerate() {
        let end = (start + count).min(indices.len());
        let slice = &indices[start..end];
        start = end;
        if slice.is_empty() {
            continue;
        }
        let mut remap = std::collections::HashMap::new();
        let mut m = Mesh { name: model.materials[mi].name.clone(), material: mi, rigid_transform: Mat4::IDENTITY, ..Default::default() };
        for &v in slice {
            let id = *remap.entry(v).or_insert_with(|| {
                let v = v as usize;
                m.positions.push(pos[v]);
                m.normals.push(nrm[v]);
                m.uvs.push(uvs[v]);
                m.joints.push(joints[v]);
                m.weights.push(weights[v]);
                (m.positions.len() - 1) as u32
            });
            m.indices.push(id);
        }
        for (name, offs) in &vmorphs {
            let mut t = MorphTarget { name: name.clone(), ..Default::default() };
            for (v, d) in offs {
                if let Some(&local) = remap.get(v) {
                    t.indices.push(local);
                    t.position_deltas.push(*d);
                }
            }
            if !t.indices.is_empty() {
                m.morphs.push(t);
            }
        }
        m.sanitize();
        model.meshes.push(m);
    }
    if model.meshes.is_empty() {
        bail!("no geometry");
    }
    model.skeleton = skel;
    finalize(&mut model);
    Ok(model)
}

//! MikuMikuDance PMD (legacy) import.
use crate::util::{downscale_rgba, finalize};
use anyhow::{Result, bail};
use glam::{Mat4, Vec2, Vec3, Vec4};
use poser_core::model::{AlphaMode, Material, Mesh, Model, MorphTarget, SourceFormat, TextureData};
use poser_core::skeleton::{Bone, Skeleton};
use poser_core::transform::Transform;
use std::path::Path;
use std::sync::Arc;

const SCALE: f32 = 0.08;

struct R<'a> {
    b: &'a [u8],
    p: usize,
}

impl R<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        if self.p + n > self.b.len() {
            bail!("unexpected end of PMD at {}", self.p);
        }
        self.p += n;
        Ok(&self.b[self.p - n..self.p])
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn v3(&mut self) -> Result<Vec3> {
        Ok(Vec3::new(self.f32()?, self.f32()?, self.f32()?))
    }
    /// Fixed-size Shift-JIS string.
    fn sjis(&mut self, n: usize) -> Result<String> {
        let raw = self.take(n)?;
        let end = raw.iter().position(|c| *c == 0).unwrap_or(n);
        let (s, _, _) = encoding_rs::SHIFT_JIS.decode(&raw[..end]);
        Ok(s.trim().to_string())
    }
}

fn cv(v: Vec3) -> Vec3 {
    Vec3::new(v.x, v.y, -v.z) * SCALE
}

pub fn load(path: &Path) -> Result<Model> {
    let data = std::fs::read(path)?;
    if data.len() < 7 || &data[0..3] != b"Pmd" {
        bail!("not a PMD file");
    }
    let mut r = R { b: &data, p: 3 };
    let _ver = r.f32()?;
    let name = r.sjis(20)?;
    let _comment = r.sjis(256)?;
    let nv = r.u32()? as usize;
    let (mut pos, mut nrm, mut uvs, mut joints, mut weights) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..nv {
        pos.push(cv(r.v3()?));
        let n = r.v3()?;
        nrm.push(Vec3::new(n.x, n.y, -n.z).normalize_or(Vec3::Y));
        uvs.push(Vec2::new(r.f32()?, r.f32()?));
        let (a, b) = (r.u16()?, r.u16()?);
        let w = r.u8()? as f32 / 100.0;
        let _edge = r.u8()?;
        joints.push([a, b, 0, 0]);
        weights.push([w, 1.0 - w, 0.0, 0.0]);
    }
    let ni = r.u32()? as usize;
    let mut idx: Vec<u32> = (0..ni).map(|_| r.u16().map(|x| x as u32)).collect::<Result<_>>()?;
    for t in idx.chunks_exact_mut(3) {
        t.swap(1, 2);
    }
    let nm = r.u32()? as usize;
    let base = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let mut model = Model::empty(if name.is_empty() { "PMD".to_string() } else { name }, SourceFormat::Pmx);
    model.path = Some(path.to_path_buf());
    model.materials.clear();
    let mut runs = vec![];
    for _ in 0..nm {
        let diffuse = Vec4::new(r.f32()?, r.f32()?, r.f32()?, r.f32()?);
        r.take(4 + 12 + 12)?;
        let _toon = r.u8()?;
        let _edge = r.u8()?;
        let count = r.u32()? as usize;
        let tex = r.sjis(20)?;
        let tex = tex.split('*').next().unwrap_or("").replace('\\', "/");
        let mut tex_idx = None;
        if !tex.is_empty() {
            if let Ok(img) = image::open(base.join(&tex)) {
                let rgba = img.to_rgba8();
                let (w, h) = rgba.dimensions();
                let (w, h, d) = downscale_rgba(w, h, rgba.into_raw(), 2048);
                model.textures.push(TextureData { name: tex.clone(), width: w, height: h, rgba: Arc::new(d) });
                tex_idx = Some(model.textures.len() - 1);
            }
        }
        model.materials.push(Material {
            name: format!("material{}", model.materials.len()),
            base_color: Vec4::new(diffuse.x, diffuse.y, diffuse.z, diffuse.w.clamp(0.0, 1.0)),
            base_color_texture: tex_idx,
            alpha_mode: if diffuse.w < 0.99 { AlphaMode::Blend } else { AlphaMode::Mask },
            alpha_cutoff: 0.3,
            double_sided: true,
            clay_shade: 1.0,
        });
        runs.push(count);
    }
    let nb = r.u16()? as usize;
    let mut raw = vec![];
    for _ in 0..nb {
        let bname = r.sjis(20)?;
        let parent = r.u16()?;
        let _tail = r.u16()?;
        let _ty = r.u8()?;
        let _ik = r.u16()?;
        let p = cv(r.v3()?);
        raw.push((bname, parent, p));
    }
    // IK chains (skipped).
    let nik = r.u16()? as usize;
    for _ in 0..nik {
        r.take(4)?;
        let links = r.u8()? as usize;
        r.take(2 + 4)?;
        r.take(links * 2)?;
    }
    // Skin morphs: index 0 is the base (absolute positions), others are offsets into it.
    let ns = r.u16()? as usize;
    let mut base_idx: Vec<u32> = vec![];
    let mut morphs: Vec<(String, Vec<(u32, Vec3)>)> = vec![];
    for k in 0..ns {
        let mname = r.sjis(20)?;
        let n = r.u32()? as usize;
        let _ty = r.u8()?;
        let mut offs = vec![];
        for _ in 0..n {
            let i = r.u32()?;
            let d = r.v3()?;
            if k == 0 {
                base_idx.push(i);
            } else if let Some(&v) = base_idx.get(i as usize) {
                offs.push((v, Vec3::new(d.x, d.y, -d.z) * SCALE));
            }
        }
        if k > 0 && !offs.is_empty() {
            morphs.push((mname, offs));
        }
    }
    // Bones parents-first.
    let mut order = vec![];
    let mut placed = vec![false; nb];
    for _ in 0..nb + 1 {
        for i in 0..nb {
            if placed[i] {
                continue;
            }
            let p = raw[i].1 as usize;
            if raw[i].1 == 0xFFFF || p >= nb || placed[p] || p == i {
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
        let (bname, parent, p) = &raw[o];
        let par = (*parent != 0xFFFF && (*parent as usize) < nb && *parent as usize != o).then(|| new_of[*parent as usize]);
        let pp = par.map(|q| raw[order[q]].2).unwrap_or(Vec3::ZERO);
        skel.bones.push(Bone::new(bname.clone(), par, Transform::from_translation(*p - pp)));
    }
    skel.recompute_inverse_bind();
    for j in &mut joints {
        for x in j.iter_mut().take(2) {
            *x = new_of.get(*x as usize).copied().unwrap_or(0) as u16;
        }
    }
    let mut start = 0;
    for (mi, count) in runs.iter().enumerate() {
        let end = (start + count).min(idx.len());
        let slice = &idx[start..end];
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
        for (mname, offs) in &morphs {
            let mut t = MorphTarget { name: mname.clone(), ..Default::default() };
            for (v, d) in offs {
                if let Some(&l) = remap.get(v) {
                    t.indices.push(l);
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

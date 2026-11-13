//! Built-in props (procedural, so no asset licences): furniture, blocks and hand-held objects.
//! Sizes are in metres; floor props stand on y = 0, hand-held props have their grip at the
//! origin with the object pointing along +Y.
use crate::glam::{Mat4, Quat, Vec3};
use crate::model::{Mesh, Model, SourceFormat};
use serde::{Deserialize, Serialize};
use std::f32::consts::{PI, TAU};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropKind {
    Cube,
    Block,
    Ball,
    Cylinder,
    Cone,
    Chair,
    Stool,
    Table,
    Bench,
    Stairs,
    Wall,
    Ramp,
    Sword,
    Staff,
    Bat,
    Book,
    Phone,
    Cup,
}

impl PropKind {
    pub const ALL: [PropKind; 18] = [
        PropKind::Cube,
        PropKind::Block,
        PropKind::Ball,
        PropKind::Cylinder,
        PropKind::Cone,
        PropKind::Chair,
        PropKind::Stool,
        PropKind::Table,
        PropKind::Bench,
        PropKind::Stairs,
        PropKind::Wall,
        PropKind::Ramp,
        PropKind::Sword,
        PropKind::Staff,
        PropKind::Bat,
        PropKind::Book,
        PropKind::Phone,
        PropKind::Cup,
    ];

    pub fn name(self) -> &'static str {
        match self {
            PropKind::Cube => "Cube",
            PropKind::Block => "Block (seat height)",
            PropKind::Ball => "Ball",
            PropKind::Cylinder => "Pillar",
            PropKind::Cone => "Cone",
            PropKind::Chair => "Chair",
            PropKind::Stool => "Stool",
            PropKind::Table => "Table",
            PropKind::Bench => "Bench",
            PropKind::Stairs => "Stairs",
            PropKind::Wall => "Wall",
            PropKind::Ramp => "Ramp",
            PropKind::Sword => "Sword",
            PropKind::Staff => "Staff / pole",
            PropKind::Bat => "Bat",
            PropKind::Book => "Book",
            PropKind::Phone => "Phone",
            PropKind::Cup => "Cup",
        }
    }

    /// Hand-held props (attach to a hand by default).
    pub fn handheld(self) -> bool {
        matches!(self, PropKind::Sword | PropKind::Staff | PropKind::Bat | PropKind::Book | PropKind::Phone | PropKind::Cup)
    }

    pub fn category(self) -> &'static str {
        match self {
            PropKind::Cube | PropKind::Block | PropKind::Ball | PropKind::Cylinder | PropKind::Cone => "Shapes",
            PropKind::Chair | PropKind::Stool | PropKind::Table | PropKind::Bench => "Furniture",
            PropKind::Stairs | PropKind::Wall | PropKind::Ramp => "Architecture",
            _ => "Hand-held",
        }
    }
}

#[derive(Default)]
struct Builder {
    pos: Vec<Vec3>,
    idx: Vec<u32>,
}

impl Builder {
    fn quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3) {
        let i = self.pos.len() as u32;
        self.pos.extend([a, b, c, d]);
        self.idx.extend([i, i + 1, i + 2, i, i + 2, i + 3]);
    }

    fn tri(&mut self, a: Vec3, b: Vec3, c: Vec3) {
        let i = self.pos.len() as u32;
        self.pos.extend([a, b, c]);
        self.idx.extend([i, i + 1, i + 2]);
    }

    /// Axis-aligned box from min corner to max corner, transformed by `m`.
    fn cuboid(&mut self, lo: Vec3, hi: Vec3, m: Mat4) {
        let p = |x: f32, y: f32, z: f32| m.transform_point3(Vec3::new(x, y, z));
        let (a, b) = (lo, hi);
        self.quad(p(a.x, a.y, b.z), p(b.x, a.y, b.z), p(b.x, b.y, b.z), p(a.x, b.y, b.z)); // +z
        self.quad(p(b.x, a.y, a.z), p(a.x, a.y, a.z), p(a.x, b.y, a.z), p(b.x, b.y, a.z)); // -z
        self.quad(p(b.x, a.y, b.z), p(b.x, a.y, a.z), p(b.x, b.y, a.z), p(b.x, b.y, b.z)); // +x
        self.quad(p(a.x, a.y, a.z), p(a.x, a.y, b.z), p(a.x, b.y, b.z), p(a.x, b.y, a.z)); // -x
        self.quad(p(a.x, b.y, b.z), p(b.x, b.y, b.z), p(b.x, b.y, a.z), p(a.x, b.y, a.z)); // +y
        self.quad(p(a.x, a.y, a.z), p(b.x, a.y, a.z), p(b.x, a.y, b.z), p(a.x, a.y, b.z)); // -y
    }

    fn bx(&mut self, center: Vec3, size: Vec3) {
        self.cuboid(center - size * 0.5, center + size * 0.5, Mat4::IDENTITY);
    }

    /// Cylinder / cone frustum along +Y from `base`, radii r0 (bottom) and r1 (top).
    fn frustum(&mut self, base: Vec3, h: f32, r0: f32, r1: f32, n: usize, caps: bool) {
        let ring = |r: f32, y: f32, i: usize| {
            let a = i as f32 / n as f32 * TAU;
            base + Vec3::new(a.cos() * r, y, -a.sin() * r)
        };
        for i in 0..n {
            let (a0, a1) = (ring(r0, 0.0, i), ring(r0, 0.0, i + 1));
            let (b0, b1) = (ring(r1, h, i), ring(r1, h, i + 1));
            if r1 > 1e-5 {
                self.quad(a0, a1, b1, b0);
            } else {
                self.tri(a0, a1, b0);
            }
            if caps {
                self.tri(base, a1, a0);
                if r1 > 1e-5 {
                    self.tri(base + Vec3::Y * h, b0, b1);
                }
            }
        }
    }

    fn sphere(&mut self, c: Vec3, r: f32, n: usize) {
        let p = |i: usize, j: usize| {
            let th = i as f32 / n as f32 * PI;
            let ph = j as f32 / (2 * n) as f32 * TAU;
            c + Vec3::new(th.sin() * ph.cos(), th.cos(), -th.sin() * ph.sin()) * r
        };
        for i in 0..n {
            for j in 0..2 * n {
                self.quad(p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1));
            }
        }
    }

    fn finish(self, name: &str, smooth: bool, color: Vec3) -> Model {
        let mut mesh = Mesh {
            name: name.into(),
            positions: self.pos,
            indices: self.idx,
            rigid_transform: Mat4::IDENTITY,
            ..Default::default()
        };
        if smooth {
            weld(&mut mesh);
        }
        mesh.compute_normals();
        let mut m = Model::empty(name, SourceFormat::Procedural);
        m.meshes.push(mesh);
        m.materials[0].name = name.into();
        m.materials[0].base_color = color.extend(1.0);
        m
    }
}

/// Merge coincident vertices (smooth shading for round props).
fn weld(m: &mut Mesh) {
    let mut map = std::collections::HashMap::new();
    let mut pos = vec![];
    let key = |p: Vec3| ((p.x * 1e4).round() as i64, (p.y * 1e4).round() as i64, (p.z * 1e4).round() as i64);
    let remap: Vec<u32> = m
        .positions
        .iter()
        .map(|p| {
            *map.entry(key(*p)).or_insert_with(|| {
                pos.push(*p);
                (pos.len() - 1) as u32
            })
        })
        .collect();
    m.indices = m.indices.iter().map(|i| remap[*i as usize]).collect();
    m.positions = pos;
}

pub fn build(kind: PropKind) -> Model {
    let mut b = Builder::default();
    let mut smooth = false;
    let v = Vec3::new;
    match kind {
        PropKind::Cube => b.bx(v(0.0, 0.25, 0.0), Vec3::splat(0.5)),
        PropKind::Block => b.bx(v(0.0, 0.225, 0.0), v(0.6, 0.45, 0.45)),
        PropKind::Ball => {
            b.sphere(v(0.0, 0.11, 0.0), 0.11, 16);
            smooth = true;
        }
        PropKind::Cylinder => {
            b.frustum(Vec3::ZERO, 2.4, 0.18, 0.18, 32, true);
        }
        PropKind::Cone => b.frustum(Vec3::ZERO, 0.5, 0.25, 0.0, 32, true),
        PropKind::Chair => {
            for (x, z) in [(-0.2, -0.2), (0.2, -0.2), (-0.2, 0.2), (0.2, 0.2)] {
                b.bx(v(x, 0.215, z), v(0.04, 0.43, 0.04));
            }
            b.bx(v(0.0, 0.45, 0.0), v(0.46, 0.04, 0.46));
            for x in [-0.2f32, 0.2] {
                b.bx(v(x, 0.68, -0.21), v(0.04, 0.44, 0.04));
            }
            b.bx(v(0.0, 0.82, -0.21), v(0.44, 0.16, 0.03));
        }
        PropKind::Stool => {
            b.frustum(v(0.0, 0.42, 0.0), 0.04, 0.18, 0.18, 24, true);
            for i in 0..3 {
                let a = i as f32 / 3.0 * TAU;
                let foot = v(a.cos() * 0.17, 0.0, a.sin() * 0.17);
                let top = v(a.cos() * 0.1, 0.43, a.sin() * 0.1);
                let dir = (top - foot).normalize();
                let m = Mat4::from_rotation_translation(Quat::from_rotation_arc(Vec3::Y, dir), foot);
                b.cuboid(v(-0.015, 0.0, -0.015), v(0.015, (top - foot).length(), 0.015), m);
            }
        }
        PropKind::Table => {
            b.bx(v(0.0, 0.73, 0.0), v(1.2, 0.04, 0.7));
            for (x, z) in [(-0.55, -0.3), (0.55, -0.3), (-0.55, 0.3), (0.55, 0.3)] {
                b.bx(v(x, 0.355, z), v(0.05, 0.71, 0.05));
            }
        }
        PropKind::Bench => {
            b.bx(v(0.0, 0.44, 0.0), v(1.5, 0.05, 0.38));
            for x in [-0.65f32, 0.65] {
                b.bx(v(x, 0.21, 0.0), v(0.05, 0.42, 0.34));
            }
        }
        PropKind::Stairs => {
            for i in 0..5 {
                let h = 0.17 * (i + 1) as f32;
                b.bx(v(0.0, h * 0.5, -0.28 * i as f32), v(1.0, h, 0.28));
            }
        }
        PropKind::Wall => b.bx(v(0.0, 1.25, 0.0), v(3.0, 2.5, 0.15)),
        PropKind::Ramp => {
            let (w, l, h) = (0.5f32, 1.2f32, 0.4f32);
            let p = [v(-w, 0.0, l * 0.5), v(w, 0.0, l * 0.5), v(w, 0.0, -l * 0.5), v(-w, 0.0, -l * 0.5), v(-w, h, -l * 0.5), v(w, h, -l * 0.5)];
            b.quad(p[0], p[1], p[5], p[4]); // slope
            b.quad(p[3], p[2], p[1], p[0]); // bottom
            b.quad(p[2], p[3], p[4], p[5]); // back
            b.tri(p[1], p[2], p[5]);
            b.tri(p[0], p[4], p[3]);
        }
        PropKind::Sword => {
            b.frustum(v(0.0, -0.1, 0.0), 0.2, 0.016, 0.016, 12, true); // grip
            b.sphere(v(0.0, -0.11, 0.0), 0.025, 6); // pommel
            b.bx(v(0.0, 0.11, 0.0), v(0.2, 0.025, 0.035)); // guard
            let m = Mat4::IDENTITY;
            b.cuboid(v(-0.022, 0.12, -0.005), v(0.022, 0.86, 0.005), m); // blade
            b.tri(v(-0.022, 0.86, 0.005), v(0.022, 0.86, 0.005), v(0.0, 0.94, 0.0));
            b.tri(v(0.022, 0.86, -0.005), v(-0.022, 0.86, -0.005), v(0.0, 0.94, 0.0));
        }
        PropKind::Staff => {
            b.frustum(v(0.0, -0.8, 0.0), 1.8, 0.017, 0.017, 16, true);
            smooth = false;
        }
        PropKind::Bat => {
            b.frustum(v(0.0, -0.12, 0.0), 0.25, 0.014, 0.016, 16, true);
            b.frustum(v(0.0, 0.13, 0.0), 0.3, 0.016, 0.033, 16, false);
            b.frustum(v(0.0, 0.43, 0.0), 0.28, 0.033, 0.033, 16, true);
        }
        PropKind::Book => b.bx(v(0.0, 0.0, 0.0), v(0.16, 0.23, 0.03)),
        PropKind::Phone => b.bx(v(0.0, 0.0, 0.0), v(0.072, 0.15, 0.008)),
        PropKind::Cup => {
            b.frustum(v(0.0, -0.05, 0.0), 0.1, 0.035, 0.042, 24, false);
            b.frustum(v(0.0, -0.05, 0.0), 0.0, 0.035, 0.0, 24, true);
        }
    }
    // Sensible starting colours (linear) for textured shading; editable per prop.
    let color = match kind {
        PropKind::Chair | PropKind::Stool | PropKind::Table | PropKind::Bench | PropKind::Staff | PropKind::Bat => v(0.33, 0.18, 0.08),
        PropKind::Sword => v(0.6, 0.62, 0.66),
        PropKind::Ball => v(0.75, 0.12, 0.08),
        PropKind::Book => v(0.12, 0.2, 0.45),
        PropKind::Phone => v(0.03, 0.03, 0.035),
        PropKind::Cup => v(0.85, 0.85, 0.82),
        PropKind::Wall | PropKind::Stairs | PropKind::Ramp | PropKind::Cylinder => v(0.6, 0.58, 0.55),
        _ => v(0.7, 0.7, 0.7),
    };
    b.finish(kind.name(), smooth, color)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_props_build() {
        for k in PropKind::ALL {
            let m = build(k);
            assert!(m.triangle_count() > 0, "{k:?}");
            assert!(m.meshes[0].positions.iter().all(|p| p.is_finite()));
        }
    }
}

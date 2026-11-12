//! Screen-space overlays: rotation rings, bones, joints, guides.
use crate::theme;
use egui::{Color32, Painter, Pos2, Rect, Stroke};
use poser_core::camera::Camera;
use poser_core::glam::{Mat4, Quat, Vec2, Vec3};

/// Camera + viewport rectangle (egui points).
#[derive(Clone, Copy)]
pub struct ViewCtx {
    pub cam: Camera,
    pub rect: Rect,
}

impl ViewCtx {
    pub fn size(&self) -> Vec2 {
        Vec2::new(self.rect.width().max(1.0), self.rect.height().max(1.0))
    }

    pub fn project(&self, p: Vec3) -> Option<Pos2> {
        let (s, _) = self.cam.project(p, self.size())?;
        Some(Pos2::new(self.rect.min.x + s.x, self.rect.min.y + s.y))
    }

    pub fn project_depth(&self, p: Vec3) -> Option<(Pos2, f32)> {
        let (s, d) = self.cam.project(p, self.size())?;
        Some((Pos2::new(self.rect.min.x + s.x, self.rect.min.y + s.y), d))
    }

    pub fn local(&self, p: Pos2) -> Vec2 {
        Vec2::new(p.x - self.rect.min.x, p.y - self.rect.min.y)
    }

    pub fn ray(&self, p: Pos2) -> (Vec3, Vec3) {
        self.cam.ray(self.local(p), self.size())
    }

    /// World units per screen point at a world position.
    pub fn world_per_point(&self, at: Vec3) -> f32 {
        let h = self.size().y;
        if self.cam.is_ortho() {
            2.0 * self.cam.ortho_half_height() / h
        } else {
            let d = (at - self.cam.eye()).dot(self.cam.forward()).max(0.01);
            2.0 * d * (self.cam.fov_y * 0.5).tan() / h
        }
    }

    /// Intersect a pointer ray with the plane through `p` facing the camera.
    pub fn drag_point(&self, pointer: Pos2, p: Vec3) -> Vec3 {
        let (o, d) = self.ray(pointer);
        let n = -self.cam.forward();
        let denom = d.dot(n);
        if denom.abs() < 1e-6 {
            return p;
        }
        let t = (p - o).dot(n) / denom;
        o + d * t
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ring {
    X,
    Y,
    Z,
    View,
    /// Elbow/knee swing around the shoulder-wrist (hip-ankle) line.
    Swing,
}

pub struct RingSet {
    pub center: Vec3,
    /// World-space axes: [X (side), Y (along bone / twist), Z].
    pub axes: [Vec3; 3],
    pub radius: f32,
    /// (centre, axis, radius) of the swing donut when a limb is selected.
    pub swing: Option<(Vec3, Vec3, f32)>,
}

pub const RING_PX: f32 = 70.0;

impl RingSet {
    /// Frame for a bone: Y = bone direction, X = bone local X projected, Z = X x Y.
    pub fn for_bone(center: Vec3, rot: Quat, bone_axis_local: Vec3, local_axes: bool, view: &ViewCtx) -> Self {
        let axes = if local_axes {
            let y = (rot * bone_axis_local).normalize_or(Vec3::Y);
            let mut x = rot * Vec3::X;
            x = (x - y * x.dot(y)).normalize_or_zero();
            if x == Vec3::ZERO {
                x = y.any_orthonormal_vector();
            }
            let z = x.cross(y).normalize();
            [x, y, z]
        } else {
            [Vec3::X, Vec3::Y, Vec3::Z]
        };
        Self { center, axes, radius: RING_PX * view.world_per_point(center), swing: None }
    }

    pub fn axis(&self, r: Ring, view: &ViewCtx) -> Vec3 {
        match r {
            Ring::X => self.axes[0],
            Ring::Y => self.axes[1],
            Ring::Z => self.axes[2],
            Ring::View => -view.cam.forward(),
            Ring::Swing => self.swing.map(|s| s.1).unwrap_or(Vec3::Y),
        }
    }

    fn ring_center(&self, r: Ring) -> Vec3 {
        match (r, self.swing) {
            (Ring::Swing, Some(s)) => s.0,
            _ => self.center,
        }
    }

    #[allow(dead_code)]
    fn circle(&self, axis: Vec3, radius: f32, n: usize) -> Vec<Vec3> {
        self.circle_at(self.center, axis, radius, n)
    }

    pub fn circle_at(&self, center: Vec3, axis: Vec3, radius: f32, n: usize) -> Vec<Vec3> {
        let u = axis.any_orthonormal_vector();
        let v = axis.cross(u);
        (0..=n)
            .map(|i| {
                let a = i as f32 / n as f32 * std::f32::consts::TAU;
                center + (u * a.cos() + v * a.sin()) * radius
            })
            .collect()
    }

    fn ring_radius(&self, r: Ring) -> f32 {
        if let (Ring::Swing, Some(s)) = (r, self.swing) {
            return s.2;
        }
        if r == Ring::View { self.radius * 1.18 } else { self.radius }
    }

    pub fn hit(&self, view: &ViewCtx, p: Pos2) -> Option<Ring> {
        let mut best: Option<(Ring, f32)> = None;
        let cd = view.project_depth(self.center).map(|x| x.1).unwrap_or(0.0);
        for r in [Ring::View, Ring::X, Ring::Y, Ring::Z, Ring::Swing] {
            if r == Ring::Swing && self.swing.is_none() {
                continue;
            }
            let pts = self.circle_at(self.ring_center(r), self.axis(r, view), self.ring_radius(r), 64);
            let proj: Vec<Option<(Pos2, f32)>> = pts.iter().map(|q| view.project_depth(*q)).collect();
            for w in proj.windows(2) {
                if let (Some(a), Some(b)) = (w[0], w[1]) {
                    // Prefer the front half.
                    let back = r != Ring::View && (a.1 + b.1) * 0.5 > cd + self.radius * 0.05;
                    let d = seg_dist(p, a.0, b.0) + if back { 4.0 } else { 0.0 };
                    if d < 8.0 && best.is_none_or(|(_, bd)| d < bd) {
                        best = Some((r, d));
                    }
                }
            }
        }
        best.map(|b| b.0)
    }

    pub fn draw(&self, painter: &Painter, view: &ViewCtx, hovered: Option<Ring>, active: Option<Ring>) {
        let cd = view.project_depth(self.center).map(|x| x.1).unwrap_or(0.0);
        for (r, col) in [(Ring::View, Color32::from_gray(220)), (Ring::X, theme::AXIS_X), (Ring::Y, theme::AXIS_Y), (Ring::Z, theme::AXIS_Z), (Ring::Swing, Color32::from_rgb(255, 170, 60))] {
            if r == Ring::Swing && self.swing.is_none() {
                continue;
            }
            if active.is_some_and(|a| a != r) {
                continue;
            }
            let hot = hovered == Some(r) || active == Some(r);
            let width = if hot { 3.5 } else { 2.0 };
            let pts = self.circle_at(self.ring_center(r), self.axis(r, view), self.ring_radius(r), 72);
            let proj: Vec<Option<(Pos2, f32)>> = pts.iter().map(|q| view.project_depth(*q)).collect();
            for w in proj.windows(2) {
                if let (Some(a), Some(b)) = (w[0], w[1]) {
                    let back = r != Ring::View && (a.1 + b.1) * 0.5 > cd;
                    let c = if hot {
                        Color32::from_rgb(255, 230, 120)
                    } else if back {
                        col.gamma_multiply(0.28)
                    } else {
                        col
                    };
                    painter.line_segment([a.0, b.0], Stroke::new(width, c));
                }
            }
        }
        if let Some(c) = view.project(self.center) {
            painter.circle_filled(c, 3.0, Color32::WHITE);
        }
    }
}

pub fn seg_dist(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_sq().max(1e-6)).clamp(0.0, 1.0);
    (a + ab * t - p).length()
}

/// Screen angle of `p` around `c` (y-down screen coordinates).
pub fn screen_angle(c: Pos2, p: Pos2) -> f32 {
    (p.y - c.y).atan2(p.x - c.x)
}

pub fn wrap_angle(a: f32) -> f32 {
    let mut a = a;
    while a > std::f32::consts::PI {
        a -= std::f32::consts::TAU;
    }
    while a < -std::f32::consts::PI {
        a += std::f32::consts::TAU;
    }
    a
}

/// Draw a joint marker (DD-style hollow circle) with optional fill colour.
pub fn joint(painter: &Painter, p: Pos2, r: f32, fill: Option<Color32>, stroke: Color32) {
    if let Some(f) = fill {
        painter.circle_filled(p, r, f);
    }
    painter.circle_stroke(p, r, Stroke::new(1.5, stroke));
}

pub fn pin(painter: &Painter, p: Pos2) {
    painter.circle_filled(p + egui::vec2(9.0, -9.0), 4.5, theme::PIN);
    painter.line_segment([p + egui::vec2(9.0, -9.0), p + egui::vec2(4.0, -4.0)], Stroke::new(2.0, theme::PIN));
}

/// Simple octahedral-ish bone drawn as a tapered outline.
pub fn bone(painter: &Painter, a: Pos2, b: Pos2, color: Color32) {
    let d = b - a;
    let len = d.length();
    if len < 2.0 {
        return;
    }
    let n = egui::vec2(-d.y, d.x) / len * (len * 0.12).clamp(2.0, 8.0);
    let m = a + d * 0.2;
    let pts = vec![a, m + n, b, m - n, a];
    painter.add(egui::Shape::line(pts, Stroke::new(1.2, color)));
}

type AxisItem = (f32, Pos2, Color32, Option<&'static str>, poser_core::camera::ViewAxis);

/// Positions of the navigation gizmo bubbles, back to front.
pub fn axis_points(center: Pos2, cam: &Camera) -> Vec<AxisItem> {
    use poser_core::camera::ViewAxis as VA;
    let view = cam.view();
    let r = 34.0;
    let axes = [
        (Vec3::X, theme::AXIS_X, "X", VA::Right),
        (Vec3::Y, theme::AXIS_Y, "Y", VA::Top),
        (Vec3::Z, theme::AXIS_Z, "Z", VA::Front),
    ];
    let mut items: Vec<AxisItem> = vec![];
    for (a, c, l, va) in axes {
        for sign in [1.0f32, -1.0] {
            let v = Mat4::from_quat(Quat::from_mat4(&view)).transform_vector3(a * sign);
            let p = center + egui::vec2(v.x, -v.y) * r;
            let opp = match (va, sign > 0.0) {
                (VA::Right, false) => VA::Left,
                (VA::Top, false) => VA::Bottom,
                (VA::Front, false) => VA::Back,
                (x, _) => x,
            };
            items.push((v.z, p, if sign > 0.0 { c } else { c.gamma_multiply(0.55) }, (sign > 0.0).then_some(l), opp));
        }
    }
    items.sort_by(|a, b| a.0.total_cmp(&b.0));
    items
}

/// World-space axis gizmo in the corner (Blender navigation gizmo).
pub fn axis_widget(painter: &Painter, center: Pos2, cam: &Camera) -> Vec<(Pos2, poser_core::camera::ViewAxis)> {
    painter.circle_filled(center, 46.0, Color32::from_black_alpha(40));
    let mut hits = vec![];
    for (_, p, c, label, va) in axis_points(center, cam) {
        if let Some(l) = label {
            painter.line_segment([center, p], Stroke::new(2.0, c));
            painter.circle_filled(p, 9.0, c);
            painter.text(p, egui::Align2::CENTER_CENTER, l, egui::FontId::proportional(11.0), Color32::BLACK);
        } else {
            painter.circle_filled(p, 7.0, c);
        }
        hits.push((p, va));
    }
    hits
}

//! Orbit camera with Blender-style navigation and artist projection modes
//! (perspective, orthographic, Design-Doll-like depth-varying "fake" perspective, lens).
use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Projection {
    Perspective,
    Orthographic,
    /// Field of view varies with depth: `fov_near` at `near_dist`, `fov_far` at `far_dist`
    /// (distances measured from the camera). Exaggerates foreshortening for dynamic shots.
    Fake { near_dist: f32, far_dist: f32, fov_near: f32, fov_far: f32 },
    /// Barrel (fisheye-like) distortion on top of perspective. `k` > 0 bulges.
    Lens { k: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Camera {
    pub pivot: Vec3,
    pub distance: f32,
    /// Camera orientation (camera space -> world). Camera looks down its local -Z.
    pub rotation: Quat,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    pub projection: Projection,
    pub near: f32,
    pub far: f32,
    /// Screen-space roll (Dutch angle) in radians, applied around the view axis.
    pub roll: f32,
}

impl Default for Camera {
    fn default() -> Self {
        let mut c = Self {
            pivot: Vec3::new(0.0, 0.9, 0.0),
            distance: 3.6,
            rotation: Quat::IDENTITY,
            fov_y: 39.6f32.to_radians(), // 50mm on 36mm film (Blender default)
            projection: Projection::Perspective,
            near: 0.01,
            far: 500.0,
            roll: 0.0,
        };
        c.set_yaw_pitch(20f32.to_radians(), -8f32.to_radians());
        c
    }
}

/// GPU-facing projection parameters (mirrors the WGSL struct).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ProjParams {
    /// x: mode (0 persp, 1 ortho, 2 fake, 3 lens), y: aspect, z: near, w: ortho half-height
    pub a: [f32; 4],
    /// x: focal (1/tan(fov/2)), y: near_dist, z: far_dist, w: lens k
    pub b: [f32; 4],
    /// x: focal near, y: focal far, z: roll cos, w: roll sin
    pub c: [f32; 4],
}

pub fn focal(fov: f32) -> f32 {
    1.0 / (fov.clamp(0.01, 3.1) * 0.5).tan()
}

impl Camera {
    pub fn set_yaw_pitch(&mut self, yaw: f32, pitch: f32) {
        self.rotation = Quat::from_rotation_y(yaw) * Quat::from_rotation_x(pitch);
    }

    pub fn eye(&self) -> Vec3 {
        self.pivot + self.rotation * Vec3::new(0.0, 0.0, self.distance)
    }

    pub fn forward(&self) -> Vec3 {
        self.rotation * -Vec3::Z
    }

    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }

    pub fn view(&self) -> Mat4 {
        Mat4::from_rotation_translation(self.rotation, self.eye()).inverse()
    }

    pub fn is_ortho(&self) -> bool {
        matches!(self.projection, Projection::Orthographic)
    }

    /// Orthographic half-height matching the perspective framing at the pivot.
    pub fn ortho_half_height(&self) -> f32 {
        self.distance * (self.fov_y * 0.5).tan()
    }

    pub fn params(&self, aspect: f32) -> ProjParams {
        let (mode, nd, fd, fn_, ff, k) = match self.projection {
            Projection::Perspective => (0.0, 0.0, 1.0, focal(self.fov_y), focal(self.fov_y), 0.0),
            Projection::Orthographic => (1.0, 0.0, 1.0, 1.0, 1.0, 0.0),
            Projection::Fake { near_dist, far_dist, fov_near, fov_far } => {
                (2.0, near_dist, far_dist.max(near_dist + 1e-3), focal(fov_near), focal(fov_far), 0.0)
            }
            Projection::Lens { k } => (3.0, 0.0, 1.0, focal(self.fov_y), focal(self.fov_y), k),
        };
        ProjParams {
            a: [mode, aspect, self.near, self.ortho_half_height()],
            b: [focal(self.fov_y), nd, fd, k],
            c: [fn_, ff, self.roll.cos(), self.roll.sin()],
        }
    }

    /// View-space point -> clip space (same maths as the vertex shader).
    pub fn view_to_clip(&self, p: Vec3, aspect: f32) -> Vec4 {
        let pp = self.params(aspect);
        view_to_clip(&pp, p)
    }

    /// World point -> (pixel position, view depth). None when behind the camera.
    pub fn project(&self, world: Vec3, size: Vec2) -> Option<(Vec2, f32)> {
        let v = self.view().transform_point3(world);
        let aspect = size.x / size.y.max(1.0);
        let c = self.view_to_clip(v, aspect);
        if c.w <= 1e-6 {
            return None;
        }
        let ndc = c.truncate() / c.w;
        Some((Vec2::new((ndc.x * 0.5 + 0.5) * size.x, (0.5 - ndc.y * 0.5) * size.y), -v.z))
    }

    /// Pixel -> world ray (origin, direction). Exact for perspective / ortho.
    pub fn ray(&self, px: Vec2, size: Vec2) -> (Vec3, Vec3) {
        let aspect = size.x / size.y.max(1.0);
        let mut ndc = Vec2::new(px.x / size.x * 2.0 - 1.0, 1.0 - px.y / size.y * 2.0);
        // Undo roll.
        let (c, s) = (self.roll.cos(), self.roll.sin());
        ndc = Vec2::new(c * ndc.x + s * ndc.y, -s * ndc.x + c * ndc.y);
        if self.is_ortho() {
            let h = self.ortho_half_height();
            let o = self.eye() + self.right() * ndc.x * h * aspect + self.up() * ndc.y * h;
            (o, self.forward())
        } else {
            let f = match self.projection {
                Projection::Fake { near_dist, far_dist, fov_near, fov_far } => {
                    let t = ((self.distance - near_dist) / (far_dist - near_dist).max(1e-3)).clamp(0.0, 1.0);
                    focal(fov_near) + (focal(fov_far) - focal(fov_near)) * t
                }
                _ => focal(self.fov_y),
            };
            let d = Vec3::new(ndc.x * aspect / f, ndc.y / f, -1.0);
            (self.eye(), (self.rotation * d).normalize())
        }
    }

    // ---- Blender-style navigation -------------------------------------------------------

    /// Turntable orbit: yaw around world up, pitch around the view right axis.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        let yaw = Quat::from_rotation_y(-dx);
        let pitch = Quat::from_axis_angle(Vec3::X, -dy);
        self.rotation = (yaw * self.rotation * pitch).normalize();
    }

    /// Trackball-free pan in screen pixels.
    pub fn pan(&mut self, dx_px: f32, dy_px: f32, viewport_h: f32) {
        let scale = 2.0 * self.ortho_half_height() / viewport_h.max(1.0);
        self.pivot += (-self.right() * dx_px + self.up() * dy_px) * scale;
    }

    pub fn zoom(&mut self, factor: f32) {
        self.distance = (self.distance * factor).clamp(0.02, 1000.0);
    }

    /// Move pivot and eye together along the view direction (Blender dolly).
    pub fn dolly(&mut self, amount: f32) {
        self.pivot += self.forward() * amount * self.distance;
    }

    pub fn set_view_axis(&mut self, axis: ViewAxis) {
        use std::f32::consts::FRAC_PI_2;
        self.rotation = match axis {
            ViewAxis::Front => Quat::IDENTITY,
            ViewAxis::Back => Quat::from_rotation_y(std::f32::consts::PI),
            ViewAxis::Right => Quat::from_rotation_y(FRAC_PI_2),
            ViewAxis::Left => Quat::from_rotation_y(-FRAC_PI_2),
            ViewAxis::Top => Quat::from_rotation_x(-FRAC_PI_2),
            ViewAxis::Bottom => Quat::from_rotation_x(FRAC_PI_2),
        };
    }

    /// Frame a bounding sphere.
    pub fn frame(&mut self, center: Vec3, radius: f32) {
        self.pivot = center;
        let r = radius.max(0.05);
        self.distance = r / (self.fov_y * 0.5).sin() * 1.1;
    }

    pub fn lerp(&self, o: &Camera, t: f32) -> Camera {
        let mut c = *o;
        c.pivot = self.pivot.lerp(o.pivot, t);
        c.distance = self.distance + (o.distance - self.distance) * t;
        c.rotation = self.rotation.slerp(o.rotation, t);
        c.fov_y = self.fov_y + (o.fov_y - self.fov_y) * t;
        c.roll = self.roll + (o.roll - self.roll) * t;
        c
    }

    /// Focal length in mm for a 36mm sensor (Blender convention, vertical fit approximated).
    pub fn focal_length_mm(&self) -> f32 {
        18.0 / (self.fov_y * 0.5).tan()
    }

    pub fn set_focal_length_mm(&mut self, mm: f32) {
        self.fov_y = 2.0 * (18.0 / mm.max(1.0)).atan();
    }
}

/// Shared projection maths (CPU copy of the WGSL `project` function).
pub fn view_to_clip(pp: &ProjParams, p: Vec3) -> Vec4 {
    let mode = pp.a[0] as i32;
    let aspect = pp.a[1];
    let near = pp.a[2];
    let d = -p.z;
    let (x, y, w) = match mode {
        1 => {
            let h = pp.a[3];
            (p.x / (h * aspect), p.y / h, 1.0)
        }
        2 => {
            let t = ((d - pp.b[1]) / (pp.b[2] - pp.b[1])).clamp(0.0, 1.0);
            let f = pp.c[0] + (pp.c[1] - pp.c[0]) * t;
            (p.x * f / aspect, p.y * f, d)
        }
        _ => {
            let f = pp.b[0];
            (p.x * f / aspect, p.y * f, d)
        }
    };
    let (mut x, mut y) = (x, y);
    if mode == 3 && w > 1e-6 {
        let nx = x / w;
        let ny = y / w;
        let r2 = nx * nx + ny * ny;
        let s = 1.0 / (1.0 + pp.b[3] * r2);
        x *= s;
        y *= s;
    }
    // Roll
    let (c, s) = (pp.c[2], pp.c[3]);
    let rx = c * x - s * y;
    let ry = s * x + c * y;
    // Reverse-Z infinite far for perspective, linear for ortho.
    let z = if mode == 1 { (1.0 - (d + 1000.0) / 2000.0).clamp(0.0, 1.0) } else { near };
    Vec4::new(rx, ry, z, w)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewAxis {
    Front,
    Back,
    Right,
    Left,
    Top,
    Bottom,
}

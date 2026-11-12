use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Translation / rotation / scale, applied as `T * R * S`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    pub fn from_translation(t: Vec3) -> Self {
        Self { translation: t, ..Self::IDENTITY }
    }

    pub fn from_rotation(r: Quat) -> Self {
        Self { rotation: r, ..Self::IDENTITY }
    }

    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    pub fn from_mat4(m: &Mat4) -> Self {
        let (scale, rotation, translation) = m.to_scale_rotation_translation();
        Self { translation, rotation: rotation.normalize(), scale }
    }

    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            translation: self.translation.lerp(other.translation, t),
            rotation: self.rotation.slerp(other.rotation, t),
            scale: self.scale.lerp(other.scale, t),
        }
    }
}

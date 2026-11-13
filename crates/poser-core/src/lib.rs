//! Core data model and algorithms for 3D Model Poser (no GPU dependencies).
pub mod animation;
pub mod camera;
pub mod collision;
pub mod edit;
pub mod humanoid;
pub mod ik;
pub mod makehuman;
pub mod mannequin;
pub mod model;
pub mod picking;
pub mod pose;
pub mod props;
pub mod scene;
pub mod skeleton;
pub mod skinning;
pub mod transform;

pub use glam;
pub use transform::Transform;

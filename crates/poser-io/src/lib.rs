//! Importers / exporters for 3D Model Poser.
pub mod export;
pub mod export_fbx;
pub mod export_rigged;
pub mod fbx_import;
pub mod gltf_import;
pub mod pmx_import;
pub mod util;

use anyhow::{Result, bail};
use poser_core::model::Model;
use std::path::Path;

/// File extensions the importer understands.
pub const IMPORT_EXTENSIONS: &[&str] = &["glb", "gltf", "vrm", "fbx", "obj", "pmx", "pmd"];

/// Load any supported model file.
pub fn load_model(path: &Path) -> Result<Model> {
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    match ext.as_str() {
        "glb" | "gltf" | "vrm" => gltf_import::load(path),
        "fbx" | "obj" => fbx_import::load(path),
        "pmx" | "pmd" => pmx_import::load(path),
        _ => bail!("unsupported file type: .{ext}"),
    }
}

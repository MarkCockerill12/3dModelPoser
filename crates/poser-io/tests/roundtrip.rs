use poser_core::mannequin::{MannequinParams, build};
use poser_io::export::{ExportMesh, write_glb, write_obj};

#[test]
fn glb_and_obj_roundtrip() {
    let m = build(&MannequinParams::default());
    let rest = m.rest_positions();
    let meshes: Vec<ExportMesh> = m
        .meshes
        .iter()
        .zip(&rest)
        .map(|(mesh, pos)| ExportMesh {
            name: "body".into(),
            positions: pos.clone(),
            normals: mesh.normals.clone(),
            uvs: &mesh.uvs,
            indices: &mesh.indices,
            material: &m.materials[0],
            texture: None,
        })
        .collect();
    let dir = std::env::temp_dir().join("poser_roundtrip");
    std::fs::create_dir_all(&dir).unwrap();
    let glb = dir.join("t.glb");
    write_glb(&glb, &meshes).unwrap();
    let back = poser_io::load_model(&glb).unwrap();
    assert_eq!(back.vertex_count(), m.vertex_count());
    assert_eq!(back.triangle_count(), m.triangle_count());
    let obj = dir.join("t.obj");
    write_obj(&obj, &meshes).unwrap();
    let back = poser_io::load_model(&obj).unwrap();
    assert_eq!(back.triangle_count(), m.triangle_count());
}

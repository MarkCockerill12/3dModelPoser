//! Binary FBX 7.4 export: mesh + skeleton + skin clusters + bind pose, posed as in the app.
use crate::export_rigged::bake_shapes;
use anyhow::Result;
use glam::{EulerRot, Mat4, Vec3};
use poser_core::model::Model;
use poser_core::skeleton::BoneShape;
use poser_core::transform::Transform;
use std::path::Path;

enum P {
    I32(i32),
    I64(i64),
    F64(f64),
    Str(String),
    ArrF64(Vec<f64>),
    ArrI32(Vec<i32>),
}

struct Node {
    name: &'static str,
    props: Vec<P>,
    children: Vec<Node>,
}

fn n(name: &'static str, props: Vec<P>) -> Node {
    Node { name, props, children: vec![] }
}

fn nc(name: &'static str, props: Vec<P>, children: Vec<Node>) -> Node {
    Node { name, props, children }
}

fn s(x: &str) -> P {
    P::Str(x.to_string())
}

fn write_props(out: &mut Vec<u8>, props: &[P]) {
    for p in props {
        match p {
            P::I32(v) => {
                out.push(b'I');
                out.extend_from_slice(&v.to_le_bytes());
            }
            P::I64(v) => {
                out.push(b'L');
                out.extend_from_slice(&v.to_le_bytes());
            }
            P::F64(v) => {
                out.push(b'D');
                out.extend_from_slice(&v.to_le_bytes());
            }
            P::Str(v) => {
                out.push(b'S');
                out.extend_from_slice(&(v.len() as u32).to_le_bytes());
                out.extend_from_slice(v.as_bytes());
            }
            P::ArrF64(v) => {
                out.push(b'd');
                out.extend_from_slice(&(v.len() as u32).to_le_bytes());
                out.extend_from_slice(&0u32.to_le_bytes());
                out.extend_from_slice(&((v.len() * 8) as u32).to_le_bytes());
                for x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
            P::ArrI32(v) => {
                out.push(b'i');
                out.extend_from_slice(&(v.len() as u32).to_le_bytes());
                out.extend_from_slice(&0u32.to_le_bytes());
                out.extend_from_slice(&((v.len() * 4) as u32).to_le_bytes());
                for x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
    }
}

fn write_node(out: &mut Vec<u8>, node: &Node) {
    let start = out.len();
    out.extend_from_slice(&[0u8; 12]); // end offset, prop count, prop list len (patched)
    out.push(node.name.len() as u8);
    out.extend_from_slice(node.name.as_bytes());
    let ps = out.len();
    write_props(out, &node.props);
    let plen = out.len() - ps;
    for c in &node.children {
        write_node(out, c);
    }
    if !node.children.is_empty() || node.props.is_empty() {
        out.extend_from_slice(&[0u8; 13]);
    }
    let end = out.len() as u32;
    out[start..start + 4].copy_from_slice(&end.to_le_bytes());
    out[start + 4..start + 8].copy_from_slice(&(node.props.len() as u32).to_le_bytes());
    out[start + 8..start + 12].copy_from_slice(&(plen as u32).to_le_bytes());
}

fn p70(name: &str, ty: &str, ty2: &str, flags: &str, vals: Vec<P>) -> Node {
    let mut props = vec![s(name), s(ty), s(ty2), s(flags)];
    props.extend(vals);
    n("P", props)
}

fn vec3p(name: &str, v: Vec3) -> Node {
    let ty = if name == "Lcl Translation" || name == "Lcl Rotation" || name == "Lcl Scaling" { name } else { "Vector3D" };
    p70(name, ty, "", "A", vec![P::F64(v.x as f64), P::F64(v.y as f64), P::F64(v.z as f64)])
}

fn mat_f64(m: &Mat4) -> Vec<f64> {
    m.to_cols_array().iter().map(|x| *x as f64).collect()
}

fn lcl(t: &Transform) -> Vec<Node> {
    // FBX eEulerXYZ: R = Rz * Ry * Rx (X applied first).
    let m = glam::Mat3::from_quat(t.rotation.normalize());
    let r = |row: usize, col: usize| m.col(col)[row];
    let y = (-r(2, 0)).clamp(-1.0, 1.0).asin();
    let (x, z) = if r(2, 0).abs() < 0.99999 { (r(2, 1).atan2(r(2, 2)), r(1, 0).atan2(r(0, 0))) } else { ((-r(1, 2)).atan2(r(1, 1)), 0.0) };
    let _ = EulerRot::XYZ;
    vec![
        vec3p("Lcl Translation", t.translation),
        vec3p("Lcl Rotation", Vec3::new(x.to_degrees(), y.to_degrees(), z.to_degrees())),
        vec3p("Lcl Scaling", t.scale),
    ]
}

pub fn write_fbx(path: &Path, model: &Model, shapes: &[BoneShape], locals: &[Transform], transform: &Transform) -> Result<()> {
    let (model, locals) = bake_shapes(model, shapes, locals);
    let sk = &model.skeleton;
    let mut id: i64 = 1_000_000;
    let mut next = || {
        id += 1;
        id
    };
    let mut objects: Vec<Node> = vec![];
    let mut conns: Vec<Node> = vec![];
    let oo = |c: i64, p: i64| n("C", vec![s("OO"), P::I64(c), P::I64(p)]);

    // Armature root (placement * armature transform).
    let root_id = next();
    let place = transform.to_mat4();
    let mesh_global = place * sk.root_transform;
    let root_t = Transform::from_mat4(&mesh_global);
    objects.push(nc("Model", vec![P::I64(root_id), s("Armature\0\x01Model"), s("Null")], vec![n("Version", vec![P::I32(232)]), nc("Properties70", vec![], lcl(&root_t))]));
    conns.push(oo(root_id, 0));

    // Bones
    let bind_world = sk.rest_world();
    let mut bone_ids = vec![];
    for (i, b) in sk.bones.iter().enumerate() {
        let mid = next();
        let aid = next();
        bone_ids.push(mid);
        let t = locals.get(i).copied().unwrap_or(b.rest);
        objects.push(nc("NodeAttribute", vec![P::I64(aid), s("\0\x01NodeAttribute"), s("LimbNode")], vec![n("TypeFlags", vec![s("Skeleton")])]));
        objects.push(nc(
            "Model",
            vec![P::I64(mid), P::Str(format!("{}\0\x01Model", b.name)), s("LimbNode")],
            vec![n("Version", vec![P::I32(232)]), nc("Properties70", vec![], lcl(&t))],
        ));
        conns.push(oo(aid, mid));
        conns.push(oo(mid, b.parent.map(|p| bone_ids[p]).unwrap_or(root_id)));
    }

    // Meshes
    let mut pose_nodes = vec![];
    for (bi, _) in sk.bones.iter().enumerate() {
        pose_nodes.push(nc("PoseNode", vec![], vec![n("Node", vec![P::I64(bone_ids[bi])]), n("Matrix", vec![P::ArrF64(mat_f64(&(place * bind_world[bi])))])]));
    }
    for (mi, m) in model.meshes.iter().enumerate() {
        if m.positions.is_empty() || m.indices.is_empty() {
            continue;
        }
        let gid = next();
        let mid = next();
        let matid = next();
        // Unskinned parts are baked into model space.
        let rigid = if m.is_skinned() {
            Mat4::IDENTITY
        } else {
            match m.rigid_bone {
                Some(b) => bind_world[b] * m.rigid_transform,
                None => m.rigid_transform,
            }
        };
        let mesh_root = sk.root_transform.inverse() * rigid;
        let verts: Vec<f64> = m.positions.iter().flat_map(|p| {
            let q = mesh_root.transform_point3(*p);
            [q.x as f64, q.y as f64, q.z as f64]
        }).collect();
        let mut pvi = Vec::with_capacity(m.indices.len());
        for t in m.indices.chunks_exact(3) {
            pvi.push(t[0] as i32);
            pvi.push(t[1] as i32);
            pvi.push(-(t[2] as i32) - 1);
        }
        let nm = glam::Mat3::from_mat4(mesh_root).inverse().transpose();
        let normals: Vec<f64> = m.indices.iter().flat_map(|i| {
            let v = (nm * m.normals[*i as usize]).normalize_or(Vec3::Y);
            [v.x as f64, v.y as f64, v.z as f64]
        }).collect();
        let mut geo_children = vec![
            n("Vertices", vec![P::ArrF64(verts)]),
            n("PolygonVertexIndex", vec![P::ArrI32(pvi)]),
            n("GeometryVersion", vec![P::I32(124)]),
            nc("LayerElementNormal", vec![P::I32(0)], vec![
                n("Version", vec![P::I32(101)]),
                n("Name", vec![s("")]),
                n("MappingInformationType", vec![s("ByPolygonVertex")]),
                n("ReferenceInformationType", vec![s("Direct")]),
                n("Normals", vec![P::ArrF64(normals)]),
            ]),
        ];
        let has_uv = m.uvs.len() == m.positions.len();
        if has_uv {
            let uv: Vec<f64> = m.uvs.iter().flat_map(|u| [u.x as f64, 1.0 - u.y as f64]).collect();
            let uvi: Vec<i32> = m.indices.iter().map(|i| *i as i32).collect();
            geo_children.push(nc("LayerElementUV", vec![P::I32(0)], vec![
                n("Version", vec![P::I32(101)]),
                n("Name", vec![s("UVMap")]),
                n("MappingInformationType", vec![s("ByPolygonVertex")]),
                n("ReferenceInformationType", vec![s("IndexToDirect")]),
                n("UV", vec![P::ArrF64(uv)]),
                n("UVIndex", vec![P::ArrI32(uvi)]),
            ]));
        }
        geo_children.push(nc("LayerElementMaterial", vec![P::I32(0)], vec![
            n("Version", vec![P::I32(101)]),
            n("Name", vec![s("")]),
            n("MappingInformationType", vec![s("AllSame")]),
            n("ReferenceInformationType", vec![s("IndexToDirect")]),
            n("Materials", vec![P::ArrI32(vec![0])]),
        ]));
        let mut layer = vec![
            n("Version", vec![P::I32(100)]),
            nc("LayerElement", vec![], vec![n("Type", vec![s("LayerElementNormal")]), n("TypedIndex", vec![P::I32(0)])]),
            nc("LayerElement", vec![], vec![n("Type", vec![s("LayerElementMaterial")]), n("TypedIndex", vec![P::I32(0)])]),
        ];
        if has_uv {
            layer.push(nc("LayerElement", vec![], vec![n("Type", vec![s("LayerElementUV")]), n("TypedIndex", vec![P::I32(0)])]));
        }
        geo_children.push(nc("Layer", vec![P::I32(0)], layer));
        let name = if m.name.is_empty() { format!("mesh{mi}") } else { m.name.clone() };
        objects.push(nc("Geometry", vec![P::I64(gid), P::Str(format!("{name}\0\x01Geometry")), s("Mesh")], geo_children));
        objects.push(nc("Model", vec![P::I64(mid), P::Str(format!("{name}\0\x01Model")), s("Mesh")], vec![n("Version", vec![P::I32(232)]), nc("Properties70", vec![], lcl(&Transform::IDENTITY))]));
        let mat = &model.materials[m.material.min(model.materials.len() - 1)];
        let c = mat.base_color;
        objects.push(nc("Material", vec![P::I64(matid), P::Str(format!("{}\0\x01Material", mat.name)), s("")], vec![
            n("Version", vec![P::I32(102)]),
            n("ShadingModel", vec![s("lambert")]),
            nc("Properties70", vec![], vec![p70("DiffuseColor", "Color", "", "A", vec![P::F64(c.x as f64), P::F64(c.y as f64), P::F64(c.z as f64)])]),
        ]));
        conns.push(oo(gid, mid));
        conns.push(oo(matid, mid));
        conns.push(oo(mid, root_id));
        pose_nodes.push(nc("PoseNode", vec![], vec![n("Node", vec![P::I64(mid)]), n("Matrix", vec![P::ArrF64(mat_f64(&mesh_global))])]));
        if m.is_skinned() {
            let skin_id = next();
            objects.push(nc("Deformer", vec![P::I64(skin_id), s("\0\x01Deformer"), s("Skin")], vec![n("Version", vec![P::I32(101)]), n("Link_DeformAcuracy", vec![P::F64(50.0)])]));
            conns.push(oo(skin_id, gid));
            for (bi, _) in sk.bones.iter().enumerate() {
                let mut idx = vec![];
                let mut wts = vec![];
                for (v, (j, w)) in m.joints.iter().zip(&m.weights).enumerate() {
                    for k in 0..4 {
                        if j[k] as usize == bi && w[k] > 0.0 {
                            idx.push(v as i32);
                            wts.push(w[k] as f64);
                        }
                    }
                }
                if idx.is_empty() {
                    continue;
                }
                let cid = next();
                // Transform: mesh (bind) -> bone at bind; TransformLink: bone bind world.
                let link = place * bind_world[bi];
                objects.push(nc("Deformer", vec![P::I64(cid), s("\0\x01SubDeformer"), s("Cluster")], vec![
                    n("Version", vec![P::I32(100)]),
                    n("Indexes", vec![P::ArrI32(idx)]),
                    n("Weights", vec![P::ArrF64(wts)]),
                    n("Transform", vec![P::ArrF64(mat_f64(&(link.inverse() * mesh_global)))]),
                    n("TransformLink", vec![P::ArrF64(mat_f64(&link))]),
                ]));
                conns.push(oo(cid, skin_id));
                conns.push(oo(bone_ids[bi], cid));
            }
        }
    }
    let pose_id = next();
    let nposes = pose_nodes.len() as i32;
    let mut pose_children = vec![n("Type", vec![s("BindPose")]), n("Version", vec![P::I32(100)]), n("NbPoseNodes", vec![P::I32(nposes)])];
    pose_children.extend(pose_nodes);
    objects.push(nc("Pose", vec![P::I64(pose_id), s("BindPose\0\x01Pose"), s("BindPose")], pose_children));

    let doc = vec![
        nc("FBXHeaderExtension", vec![], vec![
            n("FBXHeaderVersion", vec![P::I32(1003)]),
            n("FBXVersion", vec![P::I32(7400)]),
            n("Creator", vec![s("3D Model Poser")]),
        ]),
        nc("GlobalSettings", vec![], vec![
            n("Version", vec![P::I32(1000)]),
            nc("Properties70", vec![], vec![
                p70("UpAxis", "int", "Integer", "", vec![P::I32(1)]),
                p70("UpAxisSign", "int", "Integer", "", vec![P::I32(1)]),
                p70("FrontAxis", "int", "Integer", "", vec![P::I32(2)]),
                p70("FrontAxisSign", "int", "Integer", "", vec![P::I32(1)]),
                p70("CoordAxis", "int", "Integer", "", vec![P::I32(0)]),
                p70("CoordAxisSign", "int", "Integer", "", vec![P::I32(1)]),
                p70("UnitScaleFactor", "double", "Number", "", vec![P::F64(100.0)]),
            ]),
        ]),
        nc("Objects", vec![], objects),
        nc("Connections", vec![], conns),
    ];
    let mut out = b"Kaydara FBX Binary  \x00\x1a\x00".to_vec();
    out.extend_from_slice(&7400u32.to_le_bytes());
    for node in &doc {
        write_node(&mut out, node);
    }
    out.extend_from_slice(&[0u8; 13]);
    // Standard footer.
    out.extend_from_slice(&[0xfa, 0xbc, 0xab, 0x09, 0xd0, 0xc8, 0xd4, 0x66, 0xb1, 0x76, 0xfb, 0x83, 0x1c, 0xf7, 0x26, 0x7e]);
    out.extend_from_slice(&[0u8; 4]);
    while out.len() % 16 != 0 {
        out.push(0);
    }
    out.extend_from_slice(&7400u32.to_le_bytes());
    out.extend_from_slice(&[0u8; 120]);
    out.extend_from_slice(&[0xf8, 0x5a, 0x8c, 0x6a, 0xde, 0xf5, 0xd9, 0x7e, 0xec, 0xe9, 0x0c, 0xe3, 0x75, 0x8f, 0x29, 0x0b]);
    std::fs::write(path, out)?;
    Ok(())
}

//! Parse a tiny hand-built PMX 2.0 file.
fn text(b: &mut Vec<u8>, s: &str) {
    let u: Vec<u8> = s.encode_utf16().flat_map(|c| c.to_le_bytes()).collect();
    b.extend_from_slice(&(u.len() as i32).to_le_bytes());
    b.extend_from_slice(&u);
}
fn f(b: &mut Vec<u8>, v: &[f32]) {
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
}

#[test]
fn parses_minimal_pmx() {
    let mut b = b"PMX ".to_vec();
    f(&mut b, &[2.0]);
    b.push(8);
    b.extend_from_slice(&[0, 0, 4, 1, 1, 2, 1, 1]); // utf16, 0 add uv, vtx idx 4, tex 1, mat 1, bone 2, morph 1, rigid 1
    for s in ["model", "model", "", ""] {
        text(&mut b, s);
    }
    b.extend_from_slice(&3i32.to_le_bytes());
    for (i, p) in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 10.0, 0.0]].iter().enumerate() {
        f(&mut b, p);
        f(&mut b, &[0.0, 0.0, -1.0, 0.0, 0.0]);
        b.push(0); // BDEF1
        b.extend_from_slice(&(if i == 2 { 1i16 } else { 0i16 }).to_le_bytes());
        f(&mut b, &[1.0]);
    }
    b.extend_from_slice(&3i32.to_le_bytes());
    for i in [0i32, 1, 2] {
        b.extend_from_slice(&i.to_le_bytes());
    }
    b.extend_from_slice(&0i32.to_le_bytes()); // textures
    b.extend_from_slice(&1i32.to_le_bytes()); // materials
    text(&mut b, "skin");
    text(&mut b, "skin");
    f(&mut b, &[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5]);
    b.push(1);
    f(&mut b, &[0.0, 0.0, 0.0, 1.0, 1.0]);
    b.push(0xff); // tex -1
    b.push(0xff); // env -1
    b.push(0);
    b.push(1);
    b.push(0);
    text(&mut b, "");
    b.extend_from_slice(&3i32.to_le_bytes());
    b.extend_from_slice(&2i32.to_le_bytes()); // bones
    for (name, pos, parent) in [("センター", [0.0f32, 0.0, 0.0], -1i16), ("頭", [0.0, 10.0, 0.0], 0i16)] {
        text(&mut b, name);
        text(&mut b, name);
        f(&mut b, &pos);
        b.extend_from_slice(&parent.to_le_bytes());
        b.extend_from_slice(&0i32.to_le_bytes());
        b.extend_from_slice(&0u16.to_le_bytes());
        f(&mut b, &[0.0, 1.0, 0.0]);
    }
    b.extend_from_slice(&0i32.to_le_bytes()); // morphs
    let p = std::env::temp_dir().join("poser_test.pmx");
    std::fs::write(&p, &b).unwrap();
    let m = poser_io::load_model(&p).unwrap();
    assert_eq!(m.vertex_count(), 3);
    assert_eq!(m.skeleton.len(), 2);
    assert_eq!(m.skeleton.bones[1].name, "頭");
    let (lo, hi) = m.rest_bounds();
    assert!((hi.y - lo.y - 0.8).abs() < 1e-4, "MMD units -> metres");
}

fn sj(b: &mut Vec<u8>, s: &str, n: usize) {
    let (e, _, _) = encoding_rs::SHIFT_JIS.encode(s);
    let mut v = e.to_vec();
    v.resize(n, 0);
    b.extend_from_slice(&v);
}
fn f(b: &mut Vec<u8>, v: &[f32]) {
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
}
#[test]
fn parses_minimal_pmd() {
    let mut b = b"Pmd".to_vec();
    f(&mut b, &[1.0]);
    sj(&mut b, "テスト", 20);
    sj(&mut b, "", 256);
    b.extend_from_slice(&3u32.to_le_bytes());
    for (i, p) in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 10.0, 0.0]].iter().enumerate() {
        f(&mut b, p);
        f(&mut b, &[0.0, 0.0, -1.0, 0.0, 0.0]);
        let bone = if i == 2 { 1u16 } else { 0 };
        b.extend_from_slice(&bone.to_le_bytes());
        b.extend_from_slice(&bone.to_le_bytes());
        b.push(100);
        b.push(0);
    }
    b.extend_from_slice(&3u32.to_le_bytes());
    for i in [0u16, 1, 2] {
        b.extend_from_slice(&i.to_le_bytes());
    }
    b.extend_from_slice(&1u32.to_le_bytes());
    f(&mut b, &[1.0, 1.0, 1.0, 1.0, 5.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]);
    b.push(0);
    b.push(0);
    b.extend_from_slice(&3u32.to_le_bytes());
    sj(&mut b, "", 20);
    b.extend_from_slice(&2u16.to_le_bytes());
    for (name, parent, y) in [("センター", 0xFFFFu16, 0.0f32), ("頭", 0, 10.0)] {
        sj(&mut b, name, 20);
        b.extend_from_slice(&parent.to_le_bytes());
        b.extend_from_slice(&0u16.to_le_bytes());
        b.push(0);
        b.extend_from_slice(&0u16.to_le_bytes());
        f(&mut b, &[0.0, y, 0.0]);
    }
    b.extend_from_slice(&0u16.to_le_bytes()); // ik
    b.extend_from_slice(&0u16.to_le_bytes()); // skins
    let p = std::env::temp_dir().join("poser_test.pmd");
    std::fs::write(&p, &b).unwrap();
    let m = poser_io::load_model(&p).unwrap();
    assert_eq!(m.vertex_count(), 3);
    assert_eq!(m.skeleton.bones[1].name, "頭");
    assert_eq!(m.name, "テスト");
}

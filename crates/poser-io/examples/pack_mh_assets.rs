//! Pack MakeHuman CC0 clothes/hair/eyebrows (mhclo proxies) into assets/makehuman/proxies.bin.gz.
//! usage: cargo run -p poser-io --example pack_mh_assets -- <system_assets dir> <out file>
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

fn s(buf: &mut Vec<u8>, t: &str) {
    buf.extend_from_slice(&(t.len() as u16).to_le_bytes());
    buf.extend_from_slice(t.as_bytes());
}

fn pack_item(dir: &Path, category: &str, out: &mut Vec<u8>) -> Option<()> {
    let name = dir.file_name()?.to_string_lossy().to_string();
    let mhclo = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).find(|p| p.extension().is_some_and(|e| e == "mhclo"))?;
    let text = std::fs::read_to_string(&mhclo).ok()?;
    let mut refs: Vec<([u32; 3], [f32; 3], [f32; 3])> = vec![];
    let mut scales = [(0u32, 0u32, 1.0f32); 3];
    let mut deletes: Vec<u32> = vec![];
    let mut obj_file = String::new();
    let mut mat_file = String::new();
    let mut section = "";
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.is_empty() || line.starts_with('#') {
            continue;
        }
        match t[0] {
            "verts" => section = "verts",
            "delete_verts" => section = "delete",
            "obj_file" => obj_file = t[1].to_string(),
            "material" => mat_file = t[1].to_string(),
            "x_scale" | "y_scale" | "z_scale" => {
                let i = (t[0].as_bytes()[0] - b'x') as usize;
                scales[i] = (t[1].parse().ok()?, t[2].parse().ok()?, t[3].parse().ok()?);
            }
            _ if section == "verts" && t[0].chars().next()?.is_ascii_digit() => {
                if t.len() >= 9 {
                    let v: Vec<f32> = t[..9].iter().map(|x| x.parse().unwrap_or(0.0)).collect();
                    refs.push(([v[0] as u32, v[1] as u32, v[2] as u32], [v[3], v[4], v[5]], [v[6], v[7], v[8]]));
                } else {
                    let a: u32 = t[0].parse().ok()?;
                    refs.push(([a, a, a], [1.0, 0.0, 0.0], [0.0; 3]));
                }
            }
            _ if section == "delete" => {
                let mut i = 0;
                while i < t.len() {
                    if i + 2 < t.len() && t[i + 1] == "-" {
                        let (a, b): (u32, u32) = (t[i].parse().ok()?, t[i + 2].parse().ok()?);
                        deletes.extend(a..=b);
                        i += 3;
                    } else {
                        if let Ok(a) = t[i].parse() {
                            deletes.push(a);
                        }
                        i += 1;
                    }
                }
            }
            _ => section = "",
        }
    }
    // OBJ: positions only matter for count; faces + uvs (split at uv seams).
    let obj = std::fs::read_to_string(dir.join(&obj_file)).ok()?;
    let mut vts: Vec<[f32; 2]> = vec![];
    let mut nv = 0usize;
    let mut verts: Vec<(u32, [f32; 2])> = vec![];
    let mut dedup: HashMap<(u32, u32), u32> = HashMap::new();
    let mut tris: Vec<u32> = vec![];
    for line in obj.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.first() {
            Some(&"v") => nv += 1,
            Some(&"vt") => vts.push([t[1].parse().ok()?, 1.0 - t[2].parse::<f32>().ok()?]),
            Some(&"f") => {
                let c: Vec<u32> = t[1..]
                    .iter()
                    .map(|x| {
                        let mut it = x.split('/');
                        let v: u32 = it.next().unwrap().parse::<u32>().unwrap() - 1;
                        let vt: u32 = it.next().and_then(|y| y.parse::<u32>().ok()).map(|y| y - 1).unwrap_or(u32::MAX);
                        *dedup.entry((v, vt)).or_insert_with(|| {
                            verts.push((v, vts.get(vt as usize).copied().unwrap_or([0.0, 0.0])));
                            (verts.len() - 1) as u32
                        })
                    })
                    .collect();
                for k in 1..c.len() - 1 {
                    tris.extend_from_slice(&[c[0], c[k], c[k + 1]]);
                }
            }
            _ => {}
        }
    }
    if nv != refs.len() {
        eprintln!("skip {name}: obj {nv} verts vs mhclo {}", refs.len());
        return None;
    }
    // Material
    let mat = std::fs::read_to_string(dir.join(&mat_file)).unwrap_or_default();
    let mut color = [0.8f32, 0.8, 0.8];
    let mut tex = None;
    for line in mat.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.first() == Some(&"diffuseColor") && t.len() >= 4 {
            color = [t[1].parse().unwrap_or(0.8), t[2].parse().unwrap_or(0.8), t[3].parse().unwrap_or(0.8)];
        }
        if t.first() == Some(&"diffuseTexture") && t.len() >= 2 {
            tex = Some(t[1].to_string());
        }
    }
    let alpha = category != "clothes";
    let tex_bytes = tex
        .and_then(|f| image::open(dir.join(Path::new(&f).file_name()?)).ok())
        .map(|img| {
            let img = img.resize(512, 512, image::imageops::FilterType::Triangle);
            let mut c = std::io::Cursor::new(vec![]);
            if alpha {
                img.to_rgba8().write_to(&mut c, image::ImageFormat::Png).unwrap();
            } else {
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut c, 82).encode_image(&img.to_rgb8()).unwrap();
            }
            c.into_inner()
        })
        .unwrap_or_default();
    s(out, &name);
    s(out, category);
    out.extend_from_slice(&(refs.len() as u32).to_le_bytes());
    for (r, w, o) in &refs {
        for x in r {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for x in w.iter().chain(o.iter()) {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    for (a, b, r) in scales {
        out.extend_from_slice(&a.to_le_bytes());
        out.extend_from_slice(&b.to_le_bytes());
        out.extend_from_slice(&r.to_le_bytes());
    }
    out.extend_from_slice(&(verts.len() as u32).to_le_bytes());
    for (v, uv) in &verts {
        out.extend_from_slice(&v.to_le_bytes());
        out.extend_from_slice(&uv[0].to_le_bytes());
        out.extend_from_slice(&uv[1].to_le_bytes());
    }
    out.extend_from_slice(&(tris.len() as u32).to_le_bytes());
    for t in &tris {
        out.extend_from_slice(&t.to_le_bytes());
    }
    out.extend_from_slice(&(deletes.len() as u32).to_le_bytes());
    for d in &deletes {
        out.extend_from_slice(&d.to_le_bytes());
    }
    for c in color {
        out.extend_from_slice(&c.to_le_bytes());
    }
    out.push(alpha as u8);
    out.extend_from_slice(&(tex_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&tex_bytes);
    println!("{category}/{name}: {} verts, {} tris, tex {} KB", verts.len(), tris.len() / 3, tex_bytes.len() / 1024);
    Some(())
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let root = Path::new(&a[0]);
    let mut body = vec![];
    let mut count = 0u32;
    for cat in ["clothes", "hair", "eyebrows"] {
        let mut dirs: Vec<_> = std::fs::read_dir(root.join(cat)).unwrap().flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
        dirs.sort();
        for d in dirs {
            if pack_item(&d, cat, &mut body).is_some() {
                count += 1;
            }
        }
    }
    let mut buf = b"MHP1".to_vec();
    buf.extend_from_slice(&count.to_le_bytes());
    buf.extend_from_slice(&body);
    let f = std::fs::File::create(&a[1]).unwrap();
    let mut gz = flate2::write::GzEncoder::new(f, flate2::Compression::best());
    gz.write_all(&buf).unwrap();
    gz.finish().unwrap();
    println!("{count} items, {:.1} MB", std::fs::metadata(&a[1]).unwrap().len() as f64 / 1e6);
}

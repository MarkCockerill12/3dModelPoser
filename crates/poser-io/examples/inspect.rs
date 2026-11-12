//! Print a summary of what the importer makes of a file: `cargo run -p poser-io --example inspect -- file.glb`
use poser_core::humanoid::HumanBone;

fn main() {
    for arg in std::env::args().skip(1) {
        let t = std::time::Instant::now();
        match poser_io::load_model(std::path::Path::new(&arg)) {
            Ok(m) => {
                let (lo, hi) = m.rest_bounds();
                println!("== {arg}  ({:.0} ms)", t.elapsed().as_secs_f32() * 1000.0);
                println!("   meshes {} verts {} tris {} materials {} textures {}", m.meshes.len(), m.vertex_count(), m.triangle_count(), m.materials.len(), m.textures.len());
                println!("   bounds {:.3?} .. {:.3?}  (height {:.3})", lo, hi, hi.y - lo.y);
                println!("   bones {}  humanoid valid {}  mapped {}", m.skeleton.len(), m.humanoid.is_valid(), m.humanoid.bones.len());
                let mut lo = poser_core::glam::Vec3::splat(f32::MAX);
                let mut hi = poser_core::glam::Vec3::splat(f32::MIN);
                for me in &m.meshes { for p in &me.positions { lo = lo.min(*p); hi = hi.max(*p); } }
                println!("   raw mesh-space bounds {:.3?} .. {:.3?}", lo, hi);
                if m.has_skeleton() {
                    let w = m.skeleton.rest_world();
                    for h in [HumanBone::Hips, HumanBone::Head, HumanBone::LeftHand, HumanBone::LeftFoot] {
                        if let Some(b) = m.humanoid.get(h) { println!("     rest {} at {:.3?}", h.name(), w[b].w_axis.truncate()); }
                    }
                    println!("     root_transform {:?}", m.skeleton.root_transform.to_scale_rotation_translation());
                }
                let mut missing = vec![];
                for h in HumanBone::ALL {
                    match m.humanoid.get(*h) {
                        Some(b) => {
                            if matches!(h, HumanBone::Hips | HumanBone::Spine | HumanBone::Chest | HumanBone::UpperChest | HumanBone::Neck | HumanBone::Head | HumanBone::LeftUpperArm | HumanBone::LeftLowerArm | HumanBone::LeftHand | HumanBone::LeftUpperLeg | HumanBone::LeftLowerLeg | HumanBone::LeftFoot | HumanBone::LeftIndexProximal | HumanBone::LeftThumbMetacarpal) {
                                println!("     {:>22} -> {}", h.name(), m.skeleton.bones[b].name);
                            }
                        }
                        None => missing.push(h.name()),
                    }
                }
                println!("   missing: {}", missing.join(", "));
                println!("   morphs {}  animations {}", m.morph_names().len(), m.animations.len());
                for a in m.animations.iter().take(5) {
                    println!("     anim '{}' {:.2}s {} channels", a.name, a.duration, a.channels.len());
                }
            }
            Err(e) => println!("== {arg}: ERROR {e:#}"),
        }
    }
}

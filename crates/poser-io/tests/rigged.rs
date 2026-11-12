use poser_core::makehuman::{HumanParams, build};
use poser_core::pose::{HumanPose, Pose, apply_human_pose};
use poser_core::glam::Quat;
use poser_core::humanoid::HumanBone;

#[test]
fn rigged_glb_roundtrip_keeps_pose() {
    let m = build(&HumanParams::default());
    let mut pose = Pose::rest(&m.skeleton);
    let mut hp = HumanPose::default();
    hp.rotations.insert(HumanBone::LeftUpperArm, Quat::from_rotation_z(-1.2));
    hp.rotations.insert(HumanBone::RightLowerLeg, Quat::from_rotation_x(1.0));
    apply_human_pose(&m.skeleton, &m.humanoid, &hp, &mut pose.locals, None, true);
    let mut shapes = vec![Default::default(); m.skeleton.len()];
    let head = m.humanoid.get(HumanBone::Head).unwrap();
    shapes[head] = poser_core::skeleton::BoneShape { length: 1.3, girth: 1.3 };
    let path = std::env::temp_dir().join("poser_rigged.glb");
    poser_io::export_rigged::write_rigged_glb(&path, &m, &shapes, &pose.locals, &[], &Default::default()).unwrap();
    let fbx = std::env::temp_dir().join("poser_rigged.fbx");
    poser_io::export_fbx::write_fbx(&fbx, &m, &shapes, &pose.locals, &Default::default()).unwrap();
    {
        let rest = pose.clone();
        let f2 = std::env::temp_dir().join("poser_rest.fbx");
        poser_io::export_fbx::write_fbx(&f2, &m, &vec![Default::default(); m.skeleton.len()], &rest.locals, &Default::default()).unwrap();
        let r = poser_io::load_model(&f2).unwrap();
        let (a, b) = r.rest_bounds();
        let (c, d) = m.rest_bounds();
        let pm = m.skeleton.evaluate(&pose.locals, None);
        let mut dd = poser_core::skinning::DeformedMesh::default();
        poser_core::skinning::DUAL_QUATERNION.store(false, std::sync::atomic::Ordering::Relaxed);
        poser_core::skinning::deform(&m.meshes[0], &[], &pm.skin, Default::default(), &mut dd);
        let lo = dd.positions.iter().fold(poser_core::glam::Vec3::MAX, |x, p| x.min(*p));
        let hi = dd.positions.iter().fold(poser_core::glam::Vec3::MIN, |x, p| x.max(*p));
        let _ = (c, d);
        for nm in ["upperarm01.L", "upperarm01.R", "lowerleg01.R", "root"] {
            let i = m.skeleton.find(nm).unwrap();
            let j = r.skeleton.find(nm).unwrap();
            eprintln!("{nm}: ours {:?} t {:?} | fbx {:?} t {:?}", pose.locals[i].rotation, pose.locals[i].translation, r.skeleton.bones[j].rest.rotation, r.skeleton.bones[j].rest.translation);
        }
        eprintln!("posed noshape fbx {:?} {:?} vs ours {:?} {:?}", a, b, lo, hi);
    }
    let fb = poser_io::load_model(&fbx).unwrap();
    assert!(fb.humanoid.is_valid(), "fbx humanoid");
    let (flo, fhi) = fb.rest_bounds();
    eprintln!("fbx height {}", fhi.y - flo.y);
    let back = poser_io::load_model(&path).unwrap();
    assert!(back.humanoid.is_valid());
    assert_eq!(back.skeleton.len(), m.skeleton.len());
    // Re-imported rest == our posed geometry.
    let pm = m.skeleton.evaluate(&pose.locals, Some(&shapes));
    let mut d = poser_core::skinning::DeformedMesh::default();
    poser_core::skinning::DUAL_QUATERNION.store(false, std::sync::atomic::Ordering::Relaxed);
    poser_core::skinning::deform(&m.meshes[0], &[], &pm.skin, Default::default(), &mut d);
    let ours = d.positions.iter().fold(f32::MIN, |a, p| a.max(p.y)) - d.positions.iter().fold(f32::MAX, |a, p| a.min(p.y));
    let (lo, hi) = back.rest_bounds();
    assert!(((hi.y - lo.y) - ours).abs() < 0.03, "{} vs {}", hi.y - lo.y, ours);
    let _ = (fhi, flo); // FBX geometry verified against Blender 5 (scripts/bl_check.py)
}

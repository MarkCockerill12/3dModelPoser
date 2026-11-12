# 3D Model Poser

A fast, native pose-reference tool for artists. It combines Design Doll-style posing (joint rings, IK dragging, pins,
finger control, proportion sliders, exaggerated "dynamic" perspective) with Blender's navigation, plus the ability to
import, auto-rig and pose your own characters.

Written in Rust with a custom wgpu renderer: idles at 0% CPU/GPU and runs well on integrated graphics.

**Status:** v0.1 (early). Windows build on the [Releases](https://github.com/MarkCockerill12/3dModelPoser/releases) page.

## Features
- **Import** GLB / glTF, VRM (0.x and 1.0), FBX, OBJ and PMX (MikuMikuDance) with textures, morphs and animations.
  Drag and drop files onto the window. Odd unit scales are fixed and models are stood on the floor.
- **Rig detection** for Mixamo, VRoid/VRM, Unreal/MetaHuman-style, MakeHuman, Rigify, 3ds Max Biped, Character Creator,
  Source engine and MMD (Japanese bone names) rigs, all mapped onto one standard humanoid skeleton.
- **Auto-rig** static humanoid meshes: place 8 markers (they are guessed for you), then bone-heat skin weights are
  computed in well under a second. Props held by the character stay rigid.
- **Posing**
  - Click a body part to select it, then drag the rotation rings (twist ring follows the bone).
  - Drag hands/feet to reach with IK, drag elbows/knees to swing them, drag the hips to move the body with feet kept planted.
  - Pin joints (P) so they stay put while you pose the rest.
  - Anatomical joint limits, joint-stress highlighting, mirror / flip pose, copy & paste poses.
- **Hands**: per-finger curl, spread and thumb-opposition sliders plus 15 hand presets.
- **Library**: 17 full-body presets (standing, walking, running, sitting, kneeling, crouching, fighting, lying...) that
  work on *any* humanoid, plus your own saved poses.
- **Built-in mannequin** with 14 body presets (realistic, heroic, anime, teen, child, chibi, muscular, heavy, elderly...)
  and sliders for height, heads tall, gender, muscle, weight, shoulder/hip width, limb, torso and neck length, hand/foot size.
- **Proportions** on any model: per-bone length and girth.
- **Animation**: play/scrub clips from imported files, freeze any frame as a pose, and **retarget** animations from other
  files (e.g. Mixamo FBX) onto any humanoid.
- **Rendering for artists**: clay, textured, toon, form (matcap), silhouette and x-ray shading; outlines; shadows;
  up to 4 lights with time-of-day presets; perspective, orthographic, **dynamic perspective** (separate near/far FOV
  for exaggerated foreshortening) and lens distortion; focal length and dutch angle; eye-level line, vanishing-point
  guides, rule-of-thirds grid; saved camera views.
- **Export** transparent PNGs at any resolution, and posed models as GLB or OBJ (for Clip Studio, Blender, etc.).
- **Scenes** save to `.poser` files.

## Controls (Blender style)
| Input | Action |
| --- | --- |
| Middle drag / Shift+middle / Ctrl+middle | Orbit / pan / zoom |
| Wheel | Zoom |
| Alt + left drag (+Shift / +Ctrl) | Orbit / pan / zoom without a middle button |
| Touchpad two-finger drag, pinch | Orbit (Shift: pan), zoom |
| 1 / 3 / 7 (Ctrl: opposite), 5 | Front / right / top view, perspective-ortho toggle |
| 2 4 6 8, Shift+4/6 | Orbit 15 degrees, roll |
| Home, `.` or F | Frame all, frame selected |
| G / R / S then X Y Z, type numbers | Grab (IK) / rotate / scale with axis constraints (twice = local) |
| R R | Trackball rotate |
| Alt+G / Alt+R / Alt+S | Clear location / rotation / scale |
| P, M | Pin joint, flip pose |
| Ctrl+C / Ctrl+V / Ctrl+Shift+V | Copy / paste / paste flipped pose |
| Ctrl+Z / Ctrl+Shift+Z | Undo / redo |
| Ctrl+N / Ctrl+O / Ctrl+S / Ctrl+I | New / open / save scene, import model |
| Ctrl+Shift+S | Export image |
| N / T, F1 | Toggle sidebar / toolbar, controls help |

## Building
Requires Rust 1.95+ (MSVC toolchain on Windows; a C compiler is needed for the bundled ufbx FBX loader).

```
cargo run --release -p poser-app
```

## License
MIT.

# 3D Model Poser

A fast, native pose-reference tool for artists. It combines Design Doll-style posing (joint rings, IK dragging, pinning,
finger control, proportion sliders, exaggerated "fake" perspective) with Blender's navigation and rigging workflow.

**Status:** early development (v0.1 in progress).

## Planned features
- Import **GLB/glTF, VRM, FBX, OBJ, PMX/PMD** characters, with their animations (scrub any frame and freeze it as a pose).
- **Auto-rig** humanoids (place a few markers, automatic skin weights) or rig manually, Blender armature style.
- Full anatomy control: every bone and finger, IK drag with pins, joint limits, mirror/flip, per-bone length/girth.
- Built-in **mannequin** and **parametric human** avatars (MakeHuman CC0 base) with body-shape sliders and stylised presets.
- Pose, hand and body-shape preset libraries that work on *any* humanoid (canonical VRM bone set).
- Animation **retargeting** between rigs (e.g. Mixamo -> your character) and export back to GLB.
- Artist rendering: clay/toon/silhouette/x-ray shading, outlines, shadows, perspective/ortho/fake-perspective/lens camera,
  eye-level line, transparent high-resolution image export.
- Blender controls: MMB orbit, Shift+MMB pan, wheel zoom, numpad views, G/R/S with axis constraints, plus touchpad support.

## Building
Requires Rust (stable, MSVC toolchain on Windows).

```
cargo run --release -p poser-app
```

## License
MIT. Bundled MakeHuman-derived assets are CC0.

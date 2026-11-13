# 3D Model Poser

A fast, native pose-reference tool for artists. It combines Design Doll-style posing (joint rings, IK dragging, pins,
finger control, proportion sliders, exaggerated "dynamic" perspective) with Blender's navigation, plus the ability to
import, auto-rig and pose your own characters.

Written in Rust with a custom wgpu renderer: idles at 0% CPU/GPU and runs well on integrated graphics.

**Status:** v0.6 (preview). Windows build on the [Releases](https://github.com/MarkCockerill12/3dModelPoser/releases) page.

## Features
- **Realistic default body** built from MakeHuman's CC0 assets: one continuous anatomical body (not segments) with
  sliders for gender, age (1-90), muscle, weight, height, ethnicity and breast size/firmness, stylised proportions
  (head size, neck, torso, shoulders, arms, legs, hands, feet) and 21 body presets (athletic, bodybuilder, curvy,
  elderly, teen, child, toddler, heroic comic, anime, chibi, fashion...). Optional CC0 clothes, shoes, hats,
  10 hairstyles and 12 eyebrow styles that fit and follow every pose. The wooden art mannequin is still available.
- **Import** GLB / glTF, VRM (0.x and 1.0), FBX, OBJ, PMX and PMD (MikuMikuDance) with textures, morphs and animations.
  Drag and drop files onto the window. Odd unit scales are fixed and models are stood on the floor. Game rips are
  cleaned up automatically: clothing driven by cloth-simulation bones follows the body, and attachment bones exported
  at the origin (a briefcase handle) are put back in the hand.
- **Rig detection** for Mixamo, VRoid/VRM, Unreal, MakeHuman, Rigify, 3ds Max Biped, Character Creator, Source engine
  and MMD (Japanese bone names) rigs, all mapped onto one standard humanoid skeleton.
- **Auto-rig** static humanoid meshes (8 draggable markers, finger bones fitted per hand, bone-heat skin weights in
  well under a second; props held by the character stay rigid) or **rig manually** Blender-style: add / extrude / move
  / delete bones, set parents and humanoid roles, symmetrize, then bind with automatic weights.
- **Posing**
  - **Guided mode** (Design Doll feel: anatomical limits, natural IK) or **Free mode** (Blender feel: no limits,
    Ctrl+drag or G moves any bone).
  - Click a body part to select its joint (only the main posing joints: face/twist/toe helper bones follow
    automatically; tick "Select every bone" for full rig access), then use the move / rotate / scale gizmos; the orange "donut" swings the elbow/knee while the
    hand/foot stays put; rotating one spine joint bends the whole spine smoothly.
  - Drag hands/feet to reach with IK, drag fingers to curl them, drag the hips with feet kept planted.
  - **Body and floor collision** (Guided mode): hands and feet slide over the body instead of passing through it,
    and nothing goes through the floor. Arms and legs use an anatomical IK that picks a natural elbow/knee direction.
  - **Linked joints** (Guided mode): rotating a joint past its range carries on into the next joint up, like real
    anatomy: twisting the hand rolls the forearm, bending it further lifts the forearm and then the upper arm; the
    same for foot, shin and thigh, and head and neck.
  - **Right-click** a body part for a menu (reset joint or limb, copy to the other side, pin, look at camera, drop to
    floor...); **[ / ]** select the parent / child joint; the status bar explains what dragging the hovered part does.
  - **Live symmetry**: pose one side and the other mirrors it. **Look at camera**: head and eyes follow the view.
  - Pins, joint-stress highlighting, mirror / flip, copy & paste poses, drop to floor.
  - Volume-preserving (dual-quaternion) skinning keeps shoulders, elbows and wrists from collapsing.
- **Edit parts**: delete pieces of an imported model (a briefcase, hat or stray prop) by clicking them, by whole
  mesh, or with *Grow* to grab everything inside the selection (handles, latches). Undoable and saved in scenes.
- **Props**: 18 built-in props (cube, block, ball, pillar, cone, chair, stool, table, bench, stairs, wall, ramp,
  sword, staff, bat, book, phone, cup). Hand-held props snap into a hand with the fingers curled round them, and any
  object can be attached to a figure's hand, head, chest, hips or feet so it follows the pose.
- **Pose from photo**: load a photo and the pose of the person is estimated offline (RTMPose, nothing is uploaded)
  and applied to your figure, with the photo shown behind it. **Hands are detected too** (21 points per hand), so the
  finger curl, spread and thumb follow the photo.
- **Hands**: per-finger curl, spread and thumb-opposition sliders plus 15 hand presets.
- **Library** with thumbnails: 17 full-body presets that work on *any* humanoid, plus your own saved poses. Poses are
  checked against the joint limits and the figure is grounded automatically; a *Strength* slider blends a pose with
  the current one (kneeling, sitting and lying poses rest
  on whatever the figure stood on; jumps hover).
- **Animation**: play/scrub clips from imported files, freeze any frame, and **retarget** animations from other files
  (e.g. Mixamo FBX) onto any humanoid.
- **Colours**: skin-tone swatches for the default body and a colour per material for any model or prop (shown in
  textured, clay and toon shading).
- **Rendering for artists**: clay, textured, toon, form, **line art**, silhouette and x-ray shading; outlines; shadows; up to 4 lights
  with time-of-day presets; an **FOV slider** in the top bar (dolly zoom: the subject keeps its size while the
  perspective gets more or less dramatic); perspective, orthographic, **dynamic perspective** (separate near/far FOV) and lens
  distortion; focal length and dutch angle; eye-level line, vanishing-point guides, rule-of-thirds grid; saved views;
  a **picture-in-picture** second view; a **reference image** behind or traced over the figure.
- **Export** transparent PNGs at any resolution, **turntable sheets** (4-12 angles around the scene, optionally one file
  each), posed models as GLB or OBJ, and **rigged models as GLB or FBX**
  (skeleton, skin weights and current pose; tested in Blender 5).
- **Scenes** save to `.poser` files.

## Controls (Blender style, mouse and touchpad)
| Input | Action |
| --- | --- |
| Middle drag / Shift+middle / Ctrl+middle | Orbit / pan / zoom |
| Right drag (Shift / Ctrl) | Orbit (pan / zoom) |
| Left drag on empty space (Shift) | Orbit (pan) |
| Wheel | Zoom (towards the mouse pointer; option to orbit around the selection) |
| Alt + left drag (+Shift / +Ctrl) | Orbit / pan / zoom without a middle button |
| Touchpad two-finger drag, pinch | Orbit (Shift: pan), zoom. Camera tab: "always zoom" for smooth-scroll mice |
| 1 / 3 / 7 (Ctrl: opposite), 5 | Front / right / top view, perspective-ortho toggle |
| 2 4 6 8, Shift+4/6 | Orbit 15 degrees, roll |
| Home, `.` or F | Frame all, frame selected |
| W / E or R / S | Move arrows / rotation rings / scale handles (joint, or whole figure if no joint is selected) |
| G, or R / S pressed again | Blender-style free grab / rotate / scale; then X Y Z to constrain (twice = local), type numbers |
| Drag the corner axis widget | Orbit to any angle (click a bubble to snap to that view) |
| R R | Trackball rotate |
| Alt+G / Alt+R / Alt+S | Clear location / rotation / scale |
| P, M | Pin joint, flip pose |
| Ctrl+C / Ctrl+V / Ctrl+Shift+V | Copy / paste / paste flipped pose |
| Ctrl+Z / Ctrl+Shift+Z | Undo / redo |
| Ctrl+N / Ctrl+O / Ctrl+S / Ctrl+I | New / open / save scene, import model |
| Ctrl+Shift+S | Export image |
| Skeleton edit: E / Ctrl+click / X | Extrude child bone / add bone / delete bone |
| N / T, F1 | Toggle sidebar / toolbar, controls help |

## Building
Requires Rust 1.95+ (MSVC toolchain on Windows; a C compiler is needed for the bundled ufbx FBX loader).

```
cargo run --release -p poser-app
```

## License
MIT. Bundled MakeHuman body, clothes and hair assets are CC0 (see `assets/makehuman`). The RTMPose model is
Apache-2.0 (see `assets/ai`).

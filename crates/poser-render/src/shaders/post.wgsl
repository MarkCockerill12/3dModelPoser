// 3D Model Poser - outline post-process.

struct Globals {
    view: mat4x4<f32>,
    proj_a: vec4<f32>,      // mode, aspect, near, ortho half height
    proj_b: vec4<f32>,      // focal, fake near dist, fake far dist, lens k
    proj_c: vec4<f32>,      // focal near, focal far, roll cos, roll sin
    cam_pos: vec4<f32>,
    light_dir: array<vec4<f32>, 4>,   // xyz direction light travels, w enabled
    light_col: array<vec4<f32>, 4>,   // rgb * intensity
    shadow_mat: mat4x4<f32>,
    settings0: vec4<f32>,   // shading mode, ambient, toon threshold, rim
    settings1: vec4<f32>,   // shadows on, outline width, shadow bias, transparent bg
    clay: vec4<f32>,
    toon_shadow: vec4<f32>,
    outline_color: vec4<f32>,
    bg_top: vec4<f32>,
    bg_bottom: vec4<f32>,
    viewport: vec4<f32>,    // w, h, 1/w, 1/h
};

struct BgOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> BgOut {
    var o: BgOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    o.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    o.uv = vec2<f32>(x, y);
    return o;
}

// ---------------------------------------------------------------------------------------
// Post: outline compositing

@group(0) @binding(0) var<uniform> pg: Globals;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var info_tex: texture_2d<f32>;
@group(0) @binding(3) var id_tex: texture_2d<u32>;

@fragment
fn fs_post(i: BgOut) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(color_tex));
    let p = vec2<i32>(i.pos.xy);
    var c = textureLoad(color_tex, p, 0);
    let width = pg.settings1.y;
    if (width <= 0.01) {
        return c;
    }
    let nd = textureLoad(info_tex, p, 0);
    let id = textureLoad(id_tex, p, 0).r;
    var edge = 0.0;
    let r = i32(ceil(width));
    for (var y = -r; y <= r; y = y + 1) {
        for (var x = -r; x <= r; x = x + 1) {
            if (x == 0 && y == 0) {
                continue;
            }
            let fd = length(vec2<f32>(f32(x), f32(y)));
            if (fd > width + 0.5) {
                continue;
            }
            let q = clamp(p + vec2<i32>(x, y), vec2<i32>(0), dims - vec2<i32>(1));
            let nd2 = textureLoad(info_tex, q, 0);
            let id2 = textureLoad(id_tex, q, 0).r;
            var e = 0.0;
            if (id2 != id) {
                e = 1.0;
            } else if (id != 0u) {
                let dz = abs(nd.w - nd2.w) / max(min(nd.w, nd2.w), 1e-3);
                if (dz > 0.08 + 0.02 * fd) {
                    e = 1.0;
                }
                if (dot(nd.xyz, nd2.xyz) < 0.55) {
                    e = max(e, 0.7);
                }
            }
            let soft = clamp(width + 0.5 - fd, 0.0, 1.0);
            edge = max(edge, e * soft);
        }
    }
    let oc = pg.outline_color;
    let rgb = mix(c.rgb, oc.rgb, edge * oc.a);
    let a = max(c.a, edge * oc.a);
    return vec4<f32>(rgb, a);
}

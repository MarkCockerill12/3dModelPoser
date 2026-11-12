// 3D Model Poser - main shaders.

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

struct Object {
    model: mat4x4<f32>,
    normal_mat: mat4x4<f32>,
    color: vec4<f32>,
    params: vec4<f32>,      // has texture, unused, object id, alpha cutoff (<0 = none)
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var shadow_tex: texture_depth_2d;
@group(0) @binding(2) var shadow_samp: sampler_comparison;
@group(1) @binding(0) var<uniform> obj: Object;
@group(2) @binding(0) var base_tex: texture_2d<f32>;
@group(2) @binding(1) var base_samp: sampler;

fn project(p: vec3<f32>) -> vec4<f32> {
    let mode = i32(g.proj_a.x + 0.5);
    let aspect = g.proj_a.y;
    let near = g.proj_a.z;
    let d = -p.z;
    var x: f32;
    var y: f32;
    var w: f32;
    if (mode == 1) {
        let h = g.proj_a.w;
        x = p.x / (h * aspect);
        y = p.y / h;
        w = 1.0;
    } else if (mode == 2) {
        let t = clamp((d - g.proj_b.y) / (g.proj_b.z - g.proj_b.y), 0.0, 1.0);
        let f = mix(g.proj_c.x, g.proj_c.y, t);
        x = p.x * f / aspect;
        y = p.y * f;
        w = d;
    } else {
        let f = g.proj_b.x;
        x = p.x * f / aspect;
        y = p.y * f;
        w = d;
    }
    if (mode == 3 && w > 1e-6) {
        let nx = x / w;
        let ny = y / w;
        let s = 1.0 / (1.0 + g.proj_b.w * (nx * nx + ny * ny));
        x = x * s;
        y = y * s;
    }
    let c = g.proj_c.z;
    let s = g.proj_c.w;
    let rx = c * x - s * y;
    let ry = s * x + c * y;
    var z: f32;
    if (mode == 1) {
        z = clamp(1.0 - (d + 1000.0) / 2000.0, 0.0, 1.0);
    } else {
        z = near;
    }
    return vec4<f32>(rx, ry, z, w);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(c, vec3<f32>(2.2));
}

// ---------------------------------------------------------------------------------------
// Meshes

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) overlay: vec4<f32>,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) overlay: vec4<f32>,
    @location(4) view_pos: vec3<f32>,
};

@vertex
fn vs_mesh(v: VIn) -> VOut {
    var o: VOut;
    let w = obj.model * vec4<f32>(v.pos, 1.0);
    let vp = (g.view * w).xyz;
    o.clip = project(vp);
    o.world = w.xyz;
    o.normal = normalize((obj.normal_mat * vec4<f32>(v.normal, 0.0)).xyz);
    o.uv = v.uv;
    o.overlay = v.overlay;
    o.view_pos = vp;
    return o;
}

fn shadow_factor(world: vec3<f32>, n: vec3<f32>) -> f32 {
    if (g.settings1.x < 0.5) {
        return 1.0;
    }
    let lp = g.shadow_mat * vec4<f32>(world + n * 0.01, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let texel = 1.0 / vec2<f32>(textureDimensions(shadow_tex));
    var sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            sum = sum + textureSampleCompareLevel(shadow_tex, shadow_samp, uv + vec2<f32>(f32(x), f32(y)) * texel * 1.5, ndc.z - g.settings1.z);
        }
    }
    return sum / 9.0;
}

fn shade(base: vec3<f32>, n: vec3<f32>, world: vec3<f32>, view_pos: vec3<f32>) -> vec3<f32> {
    let mode = i32(g.settings0.x + 0.5);
    let v = normalize(g.cam_pos.xyz - world);
    let ambient = g.settings0.y;
    let rim_k = g.settings0.w;
    if (mode == 3) {
        // Silhouette
        return base;
    }
    if (mode == 5) {
        // Matcap-like: view-space normal gradient, no lights.
        let vn = normalize((g.view * vec4<f32>(n, 0.0)).xyz);
        let k = 0.35 + 0.55 * (vn.y * 0.5 + 0.5) + 0.15 * vn.z;
        let rim = pow(1.0 - clamp(vn.z, 0.0, 1.0), 3.0);
        return base * k + vec3<f32>(rim * 0.25);
    }
    let sh = shadow_factor(world, n);
    if (mode == 2) {
        // Toon: two tones from the key light.
        let l = -g.light_dir[0].xyz;
        let ndl = dot(n, l) * sh;
        var c = base;
        if (ndl < g.settings0.z * 2.0 - 1.0) {
            c = base * g.toon_shadow.rgb;
        }
        let rim = smoothstep(0.62, 0.68, 1.0 - max(dot(n, v), 0.0)) * rim_k;
        return c + vec3<f32>(rim) * 0.6;
    }
    // Solid / textured / x-ray: hemisphere ambient + wrapped lambert + soft spec + rim.
    let hemi = mix(vec3<f32>(0.42, 0.40, 0.38), vec3<f32>(1.0, 1.0, 1.02), n.y * 0.5 + 0.5);
    var col = base * hemi * ambient;
    for (var i = 0; i < 4; i = i + 1) {
        if (g.light_dir[i].w < 0.5) {
            continue;
        }
        let l = -g.light_dir[i].xyz;
        var ndl = max(dot(n, l), 0.0);
        if (i == 0) {
            ndl = ndl * sh;
        }
        let h = normalize(l + v);
        let spec = pow(max(dot(n, h), 0.0), 48.0) * 0.12 * ndl;
        col = col + (base * ndl + vec3<f32>(spec)) * g.light_col[i].rgb * 0.5;
    }
    let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * rim_k;
    return col + vec3<f32>(rim) * 0.5;
}

@fragment
fn fs_mesh(i: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    var n = normalize(i.normal);
    if (!front) {
        n = -n;
    }
    var base = obj.color;
    if (obj.params.x > 0.5) {
        base = base * textureSample(base_tex, base_samp, i.uv);
    } else if (obj.params.y > 0.5) {
        base.a = base.a * textureSample(base_tex, base_samp, i.uv).a;
    }
    if (obj.params.w >= 0.0 && base.a < obj.params.w) {
        discard;
    }
    var c = shade(base.rgb, n, i.world, i.view_pos);
    c = mix(c, i.overlay.rgb, i.overlay.a);
    let mode = i32(g.settings0.x + 0.5);
    var a = base.a;
    if (mode == 4) {
        a = 0.32;
    }
    return vec4<f32>(c, a);
}

// Info pass for outlines: view-space normal + linear depth, object id.
struct InfoOut {
    @location(0) nd: vec4<f32>,
    @location(1) id: u32,
};

@fragment
fn fs_info(i: VOut, @builtin(front_facing) front: bool) -> InfoOut {
    var n = normalize(i.normal);
    if (!front) {
        n = -n;
    }
    if ((obj.params.x > 0.5 || obj.params.y > 0.5) && obj.params.w >= 0.0) {
        let a = obj.color.a * textureSample(base_tex, base_samp, i.uv).a;
        if (a < obj.params.w) {
            discard;
        }
    }
    let vn = normalize((g.view * vec4<f32>(n, 0.0)).xyz);
    var o: InfoOut;
    o.nd = vec4<f32>(vn, -i.view_pos.z);
    o.id = u32(obj.params.z + 0.5);
    return o;
}

// Shadow map (standard orthographic light projection).
@vertex
fn vs_shadow(v: VIn) -> @builtin(position) vec4<f32> {
    return g.shadow_mat * obj.model * vec4<f32>(v.pos, 1.0);
}

// ---------------------------------------------------------------------------------------
// Background gradient

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

@fragment
fn fs_background(i: BgOut) -> @location(0) vec4<f32> {
    let t = clamp(i.uv.y, 0.0, 1.0);
    return vec4<f32>(mix(g.bg_top.rgb, g.bg_bottom.rgb, t), g.bg_top.a);
}

// ---------------------------------------------------------------------------------------
// Floor grid + shadow catcher

struct GridOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
};

@vertex
fn vs_grid(@location(0) pos: vec3<f32>) -> GridOut {
    var o: GridOut;
    let vp = (g.view * vec4<f32>(pos, 1.0)).xyz;
    o.clip = project(vp);
    o.world = pos;
    return o;
}

fn grid_line(coord: vec2<f32>, scale: f32) -> f32 {
    let c = coord / scale;
    let d = fwidth(c);
    let gl = abs(fract(c - 0.5) - 0.5) / max(d, vec2<f32>(1e-5));
    return 1.0 - min(min(gl.x, gl.y), 1.0);
}

@fragment
fn fs_grid(i: GridOut) -> @location(0) vec4<f32> {
    let p = i.world.xz;
    let dist = length(i.world - g.cam_pos.xyz);
    let fade = 1.0 - smoothstep(6.0, 30.0, length(p));
    var a = 0.0;
    var col = vec3<f32>(0.0);
    let show_grid = g.clay.a > 0.5;
    if (show_grid) {
        let minor = grid_line(p, 0.1) * (1.0 - smoothstep(2.0, 8.0, dist));
        let major = grid_line(p, 1.0);
        a = max(minor * 0.18, major * 0.42) * fade;
        col = vec3<f32>(0.62);
        let ax = 1.0 - min(abs(p.y) / max(fwidth(p.y), 1e-5), 1.0);
        let az = 1.0 - min(abs(p.x) / max(fwidth(p.x), 1e-5), 1.0);
        if (ax > 0.01) {
            col = mix(col, vec3<f32>(0.85, 0.22, 0.25), ax);
            a = max(a, ax * 0.85 * fade);
        }
        if (az > 0.01) {
            col = mix(col, vec3<f32>(0.40, 0.70, 0.18), az);
            a = max(a, az * 0.85 * fade);
        }
    }
    // Shadow catcher
    let sh = shadow_factor(i.world, vec3<f32>(0.0, 1.0, 0.0));
    let shadow_a = (1.0 - sh) * 0.45 * (1.0 - smoothstep(4.0, 12.0, length(p)));
    let out_col = mix(col, vec3<f32>(0.0), shadow_a / max(a + shadow_a, 1e-4));
    return vec4<f32>(out_col, clamp(a + shadow_a, 0.0, 1.0));
}

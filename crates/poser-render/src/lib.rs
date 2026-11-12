//! wgpu renderer for 3D Model Poser: shadowed clay/textured/toon shading, floor grid,
//! screen-space outlines, and offscreen rendering for the viewport and image export.
use egui_wgpu::wgpu;
use glam::{Mat4, Vec3, Vec4};
use poser_core::camera::Camera;
use poser_core::model::{AlphaMode, Material, TextureData};
use poser_core::scene::{Light, LightKind, RenderSettings, Shading};
use std::collections::{HashMap, HashSet};
use wgpu::util::DeviceExt;

const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const SHADOW_SIZE: u32 = 2048;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
    view: [[f32; 4]; 4],
    proj_a: [f32; 4],
    proj_b: [f32; 4],
    proj_c: [f32; 4],
    cam_pos: [f32; 4],
    light_dir: [[f32; 4]; 4],
    light_col: [[f32; 4]; 4],
    shadow_mat: [[f32; 4]; 4],
    settings0: [f32; 4],
    settings1: [f32; 4],
    clay: [f32; 4],
    toon_shadow: [f32; 4],
    outline_color: [f32; 4],
    bg_top: [f32; 4],
    bg_bottom: [f32; 4],
    viewport: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ObjectUniform {
    model: [[f32; 4]; 4],
    normal_mat: [[f32; 4]; 4],
    color: [f32; 4],
    params: [f32; 4],
}

/// sRGB colour (as edited in the UI) -> linear.
pub fn to_linear(c: Vec4) -> Vec4 {
    Vec4::new(c.x.powf(2.2), c.y.powf(2.2), c.z.powf(2.2), c.w)
}

/// One mesh to draw this frame.
pub struct MeshDraw<'a> {
    /// Stable key (figure id, mesh index).
    pub key: (u64, usize),
    /// Identifies the owning model (for texture sharing). Usually `Arc::as_ptr as usize`.
    pub model_key: usize,
    pub indices: &'a [u32],
    pub uvs: &'a [glam::Vec2],
    pub positions: &'a [Vec3],
    pub normals: &'a [Vec3],
    /// Re-upload positions/normals.
    pub deform_dirty: bool,
    /// Per-vertex overlay colour (rgb, amount), linear. `None` keeps the previous upload.
    pub overlay: Option<&'a [Vec4]>,
    pub material: &'a Material,
    pub textures: &'a [TextureData],
    pub world: Mat4,
    /// Linear colour multiplier.
    pub color: Vec4,
    pub use_texture: bool,
    pub object_id: u32,
    pub cast_shadow: bool,
}

pub struct FrameDesc<'a> {
    pub camera: &'a Camera,
    pub settings: &'a RenderSettings,
    pub lights: &'a [Light],
    pub width: u32,
    pub height: u32,
    pub transparent: bool,
    pub bounds_center: Vec3,
    pub bounds_radius: f32,
}

struct GpuMesh {
    pos: wgpu::Buffer,
    nrm: wgpu::Buffer,
    uv: wgpu::Buffer,
    overlay: wgpu::Buffer,
    idx: wgpu::Buffer,
    count: u32,
    verts: usize,
    obj_buf: wgpu::Buffer,
    obj_bg: wgpu::BindGroup,
}

struct Targets {
    width: u32,
    height: u32,
    msaa: u32,
    color_msaa: Option<wgpu::TextureView>,
    color_resolved: wgpu::TextureView,
    depth: wgpu::TextureView,
    info_nd: wgpu::TextureView,
    info_id: wgpu::TextureView,
    info_depth: wgpu::TextureView,
    output: wgpu::Texture,
    output_view: wgpu::TextureView,
    output_view_srgb: wgpu::TextureView,
    post_bg: wgpu::BindGroup,
}

struct Pipelines {
    msaa: u32,
    mesh_opaque: wgpu::RenderPipeline,
    mesh_blend: wgpu::RenderPipeline,
    background: wgpu::RenderPipeline,
    grid: wgpu::RenderPipeline,
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    main_module: wgpu::ShaderModule,
    globals_buf: wgpu::Buffer,
    object_bgl: wgpu::BindGroupLayout,
    material_bgl: wgpu::BindGroupLayout,
    post_bgl: wgpu::BindGroupLayout,
    globals_bg: wgpu::BindGroup,
    globals_bg_shadowpass: wgpu::BindGroup,
    main_layout: wgpu::PipelineLayout,
    pipelines: Pipelines,
    info_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    post_pipeline: wgpu::RenderPipeline,
    shadow_view: wgpu::TextureView,
    linear_sampler: wgpu::Sampler,
    white_bg: wgpu::BindGroup,
    dummy_obj_bg: wgpu::BindGroup,
    textures: HashMap<(usize, usize), wgpu::BindGroup>,
    meshes: HashMap<(u64, usize), GpuMesh>,
    grid_vb: wgpu::Buffer,
    grid_ib: wgpu::Buffer,
    grid_count: u32,
    targets: Option<Targets>,
}

fn bgl_entry(binding: u32, vis: wgpu::ShaderStages, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: vis, ty, count: None }
}

fn uniform_ty() -> wgpu::BindingType {
    wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }
}

fn vertex_layouts() -> [Option<wgpu::VertexBufferLayout<'static>>; 4] {
    const POS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];
    const NRM: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![1 => Float32x3];
    const UV: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![2 => Float32x2];
    const OV: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![3 => Float32x4];
    [
        Some(wgpu::VertexBufferLayout { array_stride: 12, step_mode: wgpu::VertexStepMode::Vertex, attributes: &POS }),
        Some(wgpu::VertexBufferLayout { array_stride: 12, step_mode: wgpu::VertexStepMode::Vertex, attributes: &NRM }),
        Some(wgpu::VertexBufferLayout { array_stride: 8, step_mode: wgpu::VertexStepMode::Vertex, attributes: &UV }),
        Some(wgpu::VertexBufferLayout { array_stride: 16, step_mode: wgpu::VertexStepMode::Vertex, attributes: &OV }),
    ]
}

fn depth_state(write: bool, compare: wgpu::CompareFunction) -> Option<wgpu::DepthStencilState> {
    Some(wgpu::DepthStencilState {
        format: DEPTH,
        depth_write_enabled: Some(write),
        depth_compare: Some(compare),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    })
}

impl Renderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let device = device.clone();
        let queue = queue.clone();
        let main_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("main.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/main.wgsl").into()),
        });
        let post_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/post.wgsl").into()),
        });
        let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let globals_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[
                bgl_entry(0, vf, uniform_ty()),
                bgl_entry(
                    1,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                bgl_entry(2, wgpu::ShaderStages::FRAGMENT, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison)),
            ],
        });
        let object_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("object"),
            entries: &[bgl_entry(0, vf, uniform_ty())],
        });
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &[
                bgl_entry(
                    0,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                bgl_entry(1, wgpu::ShaderStages::FRAGMENT, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
            ],
        });
        let tex_entry = |b: u32, st: wgpu::TextureSampleType| {
            bgl_entry(
                b,
                wgpu::ShaderStages::FRAGMENT,
                wgpu::BindingType::Texture { sample_type: st, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            )
        };
        let post_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post"),
            entries: &[
                bgl_entry(0, vf, uniform_ty()),
                tex_entry(1, wgpu::TextureSampleType::Float { filterable: false }),
                tex_entry(2, wgpu::TextureSampleType::Float { filterable: false }),
                tex_entry(3, wgpu::TextureSampleType::Uint),
            ],
        });

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shadow_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow"),
            size: wgpu::Extent3d { width: SHADOW_SIZE, height: SHADOW_SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_tex.create_view(&Default::default());
        let dummy_shadow = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow-dummy"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_shadow_view = dummy_shadow.create_view(&Default::default());
        let cmp_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow-cmp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let make_globals_bg = |shadow: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("globals"),
                layout: &globals_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: globals_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(shadow) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&cmp_sampler) },
                ],
            })
        };
        let globals_bg = make_globals_bg(&shadow_view);
        let globals_bg_shadowpass = make_globals_bg(&dummy_shadow_view);

        let main_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("main"),
            bind_group_layouts: &[Some(&globals_bgl), Some(&object_bgl), Some(&material_bgl)],
            immediate_size: 0,
        });
        let post_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post"),
            bind_group_layouts: &[Some(&post_bgl)],
            immediate_size: 0,
        });

        let layouts = vertex_layouts();
        let info_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("info"),
            layout: Some(&main_layout),
            vertex: wgpu::VertexState {
                module: &main_module,
                entry_point: Some("vs_mesh"),
                compilation_options: Default::default(),
                buffers: &layouts,
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: depth_state(true, wgpu::CompareFunction::GreaterEqual),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &main_module,
                entry_point: Some("fs_info"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Uint,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow"),
            layout: Some(&main_layout),
            vertex: wgpu::VertexState {
                module: &main_module,
                entry_point: Some("vs_shadow"),
                compilation_options: Default::default(),
                buffers: &layouts,
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });
        let post_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("post"),
            layout: Some(&post_layout),
            vertex: wgpu::VertexState {
                module: &post_module,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &post_module,
                entry_point: Some("fs_post"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: COLOR, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let pipelines = Self::build_pipelines(&device, &main_module, &main_layout, 4);

        // White 1x1 texture for untextured materials.
        let white = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("white"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: COLOR,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[255, 255, 255, 255],
        );
        let white_view = white.create_view(&Default::default());
        let white_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("white"),
            layout: &material_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&white_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&linear_sampler) },
            ],
        });

        let dummy_obj = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dummy-object"),
            contents: bytemuck::bytes_of(&ObjectUniform {
                model: Mat4::IDENTITY.to_cols_array_2d(),
                normal_mat: Mat4::IDENTITY.to_cols_array_2d(),
                color: [1.0; 4],
                params: [0.0, 0.0, 0.0, -1.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let dummy_obj_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dummy-object"),
            layout: &object_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: dummy_obj.as_entire_binding() }],
        });

        // Floor grid: subdivided plane so non-linear projections stay correct.
        let n = 96usize;
        let half = 24.0f32;
        let mut verts = Vec::with_capacity((n + 1) * (n + 1));
        for z in 0..=n {
            for x in 0..=n {
                let fx = (x as f32 / n as f32) * 2.0 - 1.0;
                let fz = (z as f32 / n as f32) * 2.0 - 1.0;
                // Denser near the origin.
                let px = fx * fx.abs() * half;
                let pz = fz * fz.abs() * half;
                verts.push([px, 0.0, pz]);
            }
        }
        let mut idx: Vec<u32> = Vec::with_capacity(n * n * 6);
        for z in 0..n {
            for x in 0..n {
                let i = (z * (n + 1) + x) as u32;
                let r = (n + 1) as u32;
                idx.extend_from_slice(&[i, i + r, i + 1, i + 1, i + r, i + r + 1]);
            }
        }
        let grid_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid-vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let grid_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid-ib"),
            contents: bytemuck::cast_slice(&idx),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            device,
            queue,
            main_module,
            globals_buf,
            object_bgl,
            material_bgl,
            post_bgl,
            globals_bg,
            globals_bg_shadowpass,
            main_layout,
            pipelines,
            info_pipeline,
            shadow_pipeline,
            post_pipeline,
            shadow_view,
            linear_sampler,
            white_bg,
            dummy_obj_bg,
            textures: HashMap::new(),
            meshes: HashMap::new(),
            grid_vb,
            grid_ib,
            grid_count: idx.len() as u32,
            targets: None,
        }
    }

    fn build_pipelines(device: &wgpu::Device, module: &wgpu::ShaderModule, layout: &wgpu::PipelineLayout, msaa: u32) -> Pipelines {
        let layouts = vertex_layouts();
        let ms = wgpu::MultisampleState { count: msaa, mask: !0, alpha_to_coverage_enabled: false };
        let mesh = |blend: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(if blend { "mesh-blend" } else { "mesh-opaque" }),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_mesh"),
                    compilation_options: Default::default(),
                    buffers: &layouts,
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: depth_state(!blend, wgpu::CompareFunction::GreaterEqual),
                multisample: ms,
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_mesh"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: COLOR,
                        blend: if blend { Some(wgpu::BlendState::ALPHA_BLENDING) } else { None },
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let mesh_opaque = mesh(false);
        let mesh_blend = mesh(true);
        let background = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("background"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: depth_state(false, wgpu::CompareFunction::Always),
            multisample: ms,
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some("fs_background"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: COLOR, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        const GRID: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];
        let grid = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("grid"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_grid"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &GRID,
                })],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: depth_state(false, wgpu::CompareFunction::GreaterEqual),
            multisample: ms,
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some("fs_grid"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: COLOR,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Pipelines { msaa, mesh_opaque, mesh_blend, background, grid }
    }

    fn texture(&self, w: u32, h: u32, format: wgpu::TextureFormat, samples: u32, usage: wgpu::TextureUsages, label: &str) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    fn ensure_targets(&mut self, w: u32, h: u32, msaa: u32) {
        let msaa = if msaa >= 4 { 4 } else { 1 };
        if self.pipelines.msaa != msaa {
            self.pipelines = Self::build_pipelines(&self.device, &self.main_module, &self.main_layout, msaa);
        }
        if let Some(t) = &self.targets {
            if t.width == w && t.height == h && t.msaa == msaa {
                return;
            }
        }
        let ra = wgpu::TextureUsages::RENDER_ATTACHMENT;
        let tb = wgpu::TextureUsages::TEXTURE_BINDING;
        let color_msaa = (msaa > 1).then(|| self.texture(w, h, COLOR, msaa, ra, "color-msaa").create_view(&Default::default()));
        let color_resolved = self.texture(w, h, COLOR, 1, ra | tb, "color").create_view(&Default::default());
        let depth = self.texture(w, h, DEPTH, msaa, ra, "depth").create_view(&Default::default());
        let info_nd = self.texture(w, h, wgpu::TextureFormat::Rgba16Float, 1, ra | tb, "info-nd").create_view(&Default::default());
        let info_id = self.texture(w, h, wgpu::TextureFormat::R32Uint, 1, ra | tb, "info-id").create_view(&Default::default());
        let info_depth = self.texture(w, h, DEPTH, 1, ra, "info-depth").create_view(&Default::default());
        // egui samples raw (gamma-encoded) bytes: store as Rgba8Unorm, render through an sRGB view.
        let output = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("output"),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: ra | tb | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[COLOR],
        });
        let output_view = output.create_view(&Default::default());
        let output_view_srgb = output.create_view(&wgpu::TextureViewDescriptor { format: Some(COLOR), ..Default::default() });
        let post_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post"),
            layout: &self.post_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.globals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&color_resolved) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&info_nd) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&info_id) },
            ],
        });
        self.targets = Some(Targets {
            width: w,
            height: h,
            msaa,
            color_msaa,
            color_resolved,
            depth,
            info_nd,
            info_id,
            info_depth,
            output,
            output_view,
            output_view_srgb,
            post_bg,
        });
    }

    fn texture_bg(&mut self, model_key: usize, tex_index: usize, data: &TextureData) -> wgpu::BindGroup {
        if let Some(bg) = self.textures.get(&(model_key, tex_index)) {
            return bg.clone();
        }
        let tex = self.device.create_texture_with_data(
            &self.queue,
            &wgpu::TextureDescriptor {
                label: Some(&data.name),
                size: wgpu::Extent3d { width: data.width.max(1), height: data.height.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: COLOR,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &data.rgba,
        );
        let view = tex.create_view(&Default::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material"),
            layout: &self.material_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.linear_sampler) },
            ],
        });
        self.textures.insert((model_key, tex_index), bg.clone());
        bg
    }

    /// Drop textures for a model that was removed.
    pub fn forget_model(&mut self, model_key: usize) {
        self.textures.retain(|(m, _), _| *m != model_key);
    }

    fn upload_mesh(&mut self, d: &MeshDraw) {
        let needs_new = match self.meshes.get(&d.key) {
            Some(m) => m.verts != d.positions.len() || m.count as usize != d.indices.len(),
            None => true,
        };
        let usage_dyn = wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;
        if needs_new {
            let dev = &self.device;
            let n = d.positions.len().max(1);
            let pos = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pos"),
                contents: bytemuck::cast_slice(if d.positions.is_empty() { &[Vec3::ZERO] } else { d.positions }),
                usage: usage_dyn,
            });
            let nrm = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("nrm"),
                contents: bytemuck::cast_slice(if d.normals.len() == d.positions.len() && !d.normals.is_empty() {
                    d.normals.to_vec()
                } else {
                    vec![Vec3::Y; n]
                }
                .as_slice()),
                usage: usage_dyn,
            });
            let uvs: Vec<glam::Vec2> = if d.uvs.len() == d.positions.len() { d.uvs.to_vec() } else { vec![glam::Vec2::ZERO; n] };
            let uv = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uv"),
                contents: bytemuck::cast_slice(&uvs),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ov: Vec<Vec4> = match d.overlay {
                Some(o) if o.len() == d.positions.len() => o.to_vec(),
                _ => vec![Vec4::ZERO; n],
            };
            let overlay = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("overlay"),
                contents: bytemuck::cast_slice(&ov),
                usage: usage_dyn,
            });
            let idx_data: &[u32] = if d.indices.is_empty() { &[0, 0, 0] } else { d.indices };
            let idx = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("idx"),
                contents: bytemuck::cast_slice(idx_data),
                usage: wgpu::BufferUsages::INDEX,
            });
            let obj_buf = dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("object"),
                size: std::mem::size_of::<ObjectUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let obj_bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("object"),
                layout: &self.object_bgl,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: obj_buf.as_entire_binding() }],
            });
            self.meshes.insert(
                d.key,
                GpuMesh { pos, nrm, uv, overlay, idx, count: d.indices.len() as u32, verts: d.positions.len(), obj_buf, obj_bg },
            );
        } else {
            let m = &self.meshes[&d.key];
            if d.deform_dirty && !d.positions.is_empty() {
                self.queue.write_buffer(&m.pos, 0, bytemuck::cast_slice(d.positions));
                if d.normals.len() == d.positions.len() {
                    self.queue.write_buffer(&m.nrm, 0, bytemuck::cast_slice(d.normals));
                }
            }
            if let Some(o) = d.overlay {
                if o.len() == d.positions.len() && !o.is_empty() {
                    self.queue.write_buffer(&m.overlay, 0, bytemuck::cast_slice(o));
                }
            }
        }
        let m = &self.meshes[&d.key];
        let nm = d.world.inverse().transpose();
        let has_tex = d.use_texture && d.material.base_color_texture.is_some_and(|t| t < d.textures.len());
        let cutoff = match d.material.alpha_mode {
            AlphaMode::Mask => d.material.alpha_cutoff,
            AlphaMode::Blend => 0.02,
            AlphaMode::Opaque => -1.0,
        };
        let u = ObjectUniform {
            model: d.world.to_cols_array_2d(),
            normal_mat: nm.to_cols_array_2d(),
            color: d.color.to_array(),
            params: [if has_tex { 1.0 } else { 0.0 }, 0.0, d.object_id as f32, cutoff],
        };
        self.queue.write_buffer(&m.obj_buf, 0, bytemuck::bytes_of(&u));
    }

    fn globals(&self, f: &FrameDesc) -> (Globals, Mat4) {
        let s = f.settings;
        let cam = f.camera;
        let aspect = f.width as f32 / f.height.max(1) as f32;
        let pp = cam.params(aspect);
        let mut light_dir = [[0.0f32; 4]; 4];
        let mut light_col = [[0.0f32; 4]; 4];
        let mut key_dir = Vec3::new(-0.4, -0.8, -0.45).normalize();
        let mut first = true;
        for (i, l) in f.lights.iter().filter(|l| l.enabled).take(4).enumerate() {
            let dir = match l.kind {
                LightKind::Directional | LightKind::Spot => l.direction(),
                LightKind::Point => (f.bounds_center - l.position).normalize_or(Vec3::NEG_Y),
            };
            if first {
                key_dir = dir;
                first = false;
            }
            light_dir[i] = [dir.x, dir.y, dir.z, 1.0];
            let c = to_linear(l.color.extend(1.0)).truncate() * l.intensity;
            light_col[i] = [c.x, c.y, c.z, 1.0];
        }
        let r = f.bounds_radius.max(0.5) * 1.3;
        let center = f.bounds_center;
        let eye = center - key_dir * r * 2.5;
        let up = if key_dir.y.abs() > 0.95 { Vec3::Z } else { Vec3::Y };
        let lview = glam::camera::rh::view::look_at_mat4(eye, center, up);
        let lproj = glam::camera::rh::proj::directx::orthographic(-r, r, -r, r, 0.01, r * 5.0);
        let shadow_mat = lproj * lview;
        let shading = match s.shading {
            Shading::Solid => 0.0,
            Shading::Textured => 1.0,
            Shading::Toon => 2.0,
            Shading::Silhouette => 3.0,
            Shading::XRay => 4.0,
            Shading::Matcap => 5.0,
        };
        let eye_pos = cam.eye();
        let clay = to_linear(s.clay_color);
        let g = Globals {
            view: cam.view().to_cols_array_2d(),
            proj_a: pp.a,
            proj_b: pp.b,
            proj_c: pp.c,
            cam_pos: [eye_pos.x, eye_pos.y, eye_pos.z, 1.0],
            light_dir,
            light_col,
            shadow_mat: shadow_mat.to_cols_array_2d(),
            settings0: [shading, s.ambient, s.toon_threshold, s.rim],
            settings1: [
                if s.shadows { 1.0 } else { 0.0 },
                if s.outline { s.outline_width } else { 0.0 },
                0.0015,
                if f.transparent { 1.0 } else { 0.0 },
            ],
            clay: [clay.x, clay.y, clay.z, if s.show_grid { 1.0 } else { 0.0 }],
            toon_shadow: to_linear(s.toon_shadow).to_array(),
            outline_color: to_linear(s.outline_color).to_array(),
            bg_top: to_linear(s.background_top).to_array(),
            bg_bottom: to_linear(s.background_bottom).to_array(),
            viewport: [f.width as f32, f.height as f32, 1.0 / f.width.max(1) as f32, 1.0 / f.height.max(1) as f32],
        };
        (g, shadow_mat)
    }

    /// Render a frame into the internal output texture and return its view.
    pub fn render(&mut self, f: &FrameDesc, draws: &[MeshDraw]) -> &wgpu::TextureView {
        let w = f.width.max(1);
        let h = f.height.max(1);
        self.ensure_targets(w, h, f.settings.msaa);
        let mut seen = HashSet::new();
        for d in draws {
            self.upload_mesh(d);
            seen.insert(d.key);
        }
        self.meshes.retain(|k, _| seen.contains(k));
        let mats: Vec<wgpu::BindGroup> = draws
            .iter()
            .map(|d| match d.material.base_color_texture {
                Some(t) if d.use_texture && t < d.textures.len() => self.texture_bg(d.model_key, t, &d.textures[t]),
                _ => self.white_bg.clone(),
            })
            .collect();
        let (g, _) = self.globals(f);
        self.queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&g));

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        let t = self.targets.as_ref().unwrap();
        let xray = f.settings.shading == Shading::XRay;

        // Shadow pass
        if f.settings.shadows {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.shadow_pipeline);
            pass.set_bind_group(0, &self.globals_bg_shadowpass, &[]);
            for (d, mat) in draws.iter().zip(&mats) {
                if !d.cast_shadow {
                    continue;
                }
                let m = &self.meshes[&d.key];
                Self::draw_mesh(&mut pass, m, mat);
            }
        }

        // Main pass
        {
            let (view, resolve) = match &t.color_msaa {
                Some(ms) => (ms, Some(&t.color_resolved)),
                None => (&t.color_resolved, None),
            };
            let clear = if f.transparent { wgpu::Color::TRANSPARENT } else { wgpu::Color::BLACK };
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: resolve,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.globals_bg, &[]);
            pass.set_bind_group(2, &self.white_bg, &[]);
            if !f.transparent {
                pass.set_pipeline(&self.pipelines.background);
                pass.set_bind_group(1, &self.dummy_obj_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            // Opaque meshes
            if !xray {
                pass.set_pipeline(&self.pipelines.mesh_opaque);
                for (d, mat) in draws.iter().zip(&mats) {
                    if d.material.alpha_mode != AlphaMode::Blend {
                        Self::draw_mesh(&mut pass, &self.meshes[&d.key], mat);
                    }
                }
            }
            // Floor
            if (f.settings.show_grid || f.settings.shadows) && f.settings.show_floor {
                pass.set_bind_group(1, &self.dummy_obj_bg, &[]);
                pass.set_bind_group(2, &self.white_bg, &[]);
                pass.set_pipeline(&self.pipelines.grid);
                pass.set_vertex_buffer(0, self.grid_vb.slice(..));
                pass.set_index_buffer(self.grid_ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.grid_count, 0, 0..1);
            }
            // Transparent meshes
            pass.set_pipeline(&self.pipelines.mesh_blend);
            for (d, mat) in draws.iter().zip(&mats) {
                if xray || d.material.alpha_mode == AlphaMode::Blend {
                    Self::draw_mesh(&mut pass, &self.meshes[&d.key], mat);
                }
            }
        }

        // Info pass (outlines)
        if f.settings.outline {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("info"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &t.info_nd,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &t.info_id,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.info_depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.info_pipeline);
            pass.set_bind_group(0, &self.globals_bg, &[]);
            for (d, mat) in draws.iter().zip(&mats) {
                if d.material.alpha_mode != AlphaMode::Blend || xray {
                    Self::draw_mesh(&mut pass, &self.meshes[&d.key], mat);
                }
            }
        }

        // Post (outline composite) -> output
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("post"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &t.output_view_srgb,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.post_pipeline);
            pass.set_bind_group(0, &t.post_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([enc.finish()]);
        &self.targets.as_ref().unwrap().output_view
    }

    fn draw_mesh(pass: &mut wgpu::RenderPass<'_>, m: &GpuMesh, mat: &wgpu::BindGroup) {
        if m.count == 0 {
            return;
        }
        pass.set_bind_group(1, &m.obj_bg, &[]);
        pass.set_bind_group(2, mat, &[]);
        pass.set_vertex_buffer(0, m.pos.slice(..));
        pass.set_vertex_buffer(1, m.nrm.slice(..));
        pass.set_vertex_buffer(2, m.uv.slice(..));
        pass.set_vertex_buffer(3, m.overlay.slice(..));
        pass.set_index_buffer(m.idx.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..m.count, 0, 0..1);
    }

    pub fn output_view(&self) -> Option<&wgpu::TextureView> {
        self.targets.as_ref().map(|t| &t.output_view)
    }

    /// Read back the current output texture as RGBA8 (sRGB) pixels.
    pub fn read_output(&self) -> Option<(u32, u32, Vec<u8>)> {
        let t = self.targets.as_ref()?;
        let (w, h) = (t.width, t.height);
        let bpr = (w * 4).div_ceil(256) * 256;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (bpr * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &t.output, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit([enc.finish()]);
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        rx.recv().ok()?.ok()?;
        let data = slice.get_mapped_range().ok()?;
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for row in 0..h {
            let s = (row * bpr) as usize;
            out.extend_from_slice(&data[s..s + (w * 4) as usize]);
        }
        drop(data);
        buf.unmap();
        Some((w, h, out))
    }
}

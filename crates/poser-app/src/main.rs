#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod armature;
mod body;
mod deform;
mod gizmo;
mod input;
mod panels;
mod parts;
mod photo;
mod presets;
mod scenefile;
mod theme;
mod thumbs;
mod ux;
mod viewport;

/// Debug / test commands (--preset-sheet, --prop-test, --photo-test) run without any window,
/// on an offscreen GPU device, so they never pop up over other programs.
fn headless() -> Option<()> {
    let args: Vec<String> = std::env::args().collect();
    if !args.iter().any(|a| a == "--preset-sheet" || a == "--prop-test" || a == "--photo-test") {
        return None;
    }
    use egui_wgpu::wgpu;
    fn block_on<F: std::future::Future>(f: F) -> F::Output {
        let waker = std::task::Waker::noop();
        let mut cx = std::task::Context::from_waker(waker);
        let mut f = std::pin::pin!(f);
        loop {
            if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
                return v;
            }
            std::thread::yield_now();
        }
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::LowPower, ..Default::default() })).ok()?;
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    let ctx = egui::Context::default();
    let mut app = app::PoserApp::new_with(&ctx, Some(poser_render::Renderer::new(&device, &queue)));
    app.run_headless(&ctx);
    Some(())
}

fn main() -> eframe::Result {
    if headless().is_some() {
        return Ok(());
    }
    let icon = theme::app_icon();
    let mut wgpu_options = egui_wgpu::WgpuConfiguration::default();
    if let egui_wgpu::WgpuSetup::CreateNew(c) = &mut wgpu_options.wgpu_setup {
        // Laptop friendly: prefer the power-saving GPU and a memory-lean allocator.
        c.power_preference = egui_wgpu::wgpu::PowerPreference::LowPower;
        let base = c.device_descriptor.clone();
        c.device_descriptor = std::sync::Arc::new(move |a| {
            let mut d = base(a);
            d.memory_hints = egui_wgpu::wgpu::MemoryHints::MemoryUsage;
            d
        });
    }
    let options = eframe::NativeOptions {
        wgpu_options,
        viewport: egui::ViewportBuilder::default()
            .with_title("3D Model Poser")
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([800.0, 500.0])
            .with_drag_and_drop(true)
            .with_icon(icon),
        renderer: eframe::Renderer::Wgpu,
        multisampling: 0,
        depth_buffer: 0,
        persist_window: true,
        ..Default::default()
    };
    eframe::run_native("3D Model Poser", options, Box::new(|cc| Ok(Box::new(app::PoserApp::new(cc)))))
}

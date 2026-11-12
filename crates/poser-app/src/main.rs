#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod deform;
mod gizmo;
mod input;
mod panels;
mod presets;
mod scenefile;
mod theme;
mod viewport;

fn main() -> eframe::Result {
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

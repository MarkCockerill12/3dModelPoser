//! Blender-like dark theme.
use egui::{Color32, CornerRadius, Stroke, Visuals};

pub const ACCENT: Color32 = Color32::from_rgb(71, 114, 179); // Blender selection blue
pub const ACTIVE: Color32 = Color32::from_rgb(255, 160, 40); // active bone orange
pub const AXIS_X: Color32 = Color32::from_rgb(237, 57, 82);
pub const AXIS_Y: Color32 = Color32::from_rgb(130, 204, 26);
pub const AXIS_Z: Color32 = Color32::from_rgb(47, 132, 227);
pub const PIN: Color32 = Color32::from_rgb(230, 60, 60);

pub fn apply(ctx: &egui::Context) {
    let mut v = Visuals::dark();
    let bg = Color32::from_rgb(48, 48, 48);
    let panel = Color32::from_rgb(40, 40, 40);
    v.panel_fill = panel;
    v.window_fill = Color32::from_rgb(36, 36, 36);
    v.extreme_bg_color = Color32::from_rgb(29, 29, 29);
    v.faint_bg_color = Color32::from_rgb(44, 44, 44);
    v.selection.bg_fill = ACCENT;
    v.selection.stroke = Stroke::new(1.0, Color32::WHITE);
    v.hyperlink_color = ACCENT;
    let r = CornerRadius::same(4);
    v.widgets.noninteractive.bg_fill = panel;
    v.widgets.noninteractive.corner_radius = r;
    v.widgets.inactive.bg_fill = Color32::from_rgb(84, 84, 84);
    v.widgets.inactive.weak_bg_fill = Color32::from_rgb(70, 70, 70);
    v.widgets.inactive.corner_radius = r;
    v.widgets.hovered.bg_fill = Color32::from_rgb(100, 100, 100);
    v.widgets.hovered.weak_bg_fill = Color32::from_rgb(90, 90, 90);
    v.widgets.hovered.corner_radius = r;
    v.widgets.active.bg_fill = ACCENT;
    v.widgets.active.weak_bg_fill = ACCENT;
    v.widgets.active.corner_radius = r;
    v.widgets.open.bg_fill = bg;
    v.window_corner_radius = CornerRadius::same(6);
    ctx.set_visuals(v);
    ctx.global_style_mut(|s| {
        s.spacing.item_spacing = egui::vec2(6.0, 4.0);
        s.spacing.slider_width = 150.0;
        s.spacing.interact_size.y = 20.0;
    });
}

/// Simple generated window icon (mannequin silhouette on blue).
pub fn app_icon() -> egui::IconData {
    let n = 64usize;
    let mut rgba = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let fx = x as f32 / n as f32 - 0.5;
            let fy = y as f32 / n as f32 - 0.5;
            let i = (y * n + x) * 4;
            let r = (fx * fx + fy * fy).sqrt();
            let mut c = if r < 0.48 { [71, 114, 179, 255] } else { [0, 0, 0, 0] };
            // head
            let head = ((fx) * (fx) + (fy + 0.26) * (fy + 0.26)).sqrt() < 0.08;
            // body
            let body = fx.abs() < 0.06 && fy > -0.17 && fy < 0.1;
            let arms = fy > -0.14 && fy < -0.09 && fx.abs() < 0.24;
            let legs = fy >= 0.1 && fy < 0.36 && (fx.abs() - 0.05 - (fy - 0.1) * 0.25).abs() < 0.035;
            if head || body || arms || legs {
                c = [240, 236, 228, 255];
            }
            rgba[i..i + 4].copy_from_slice(&c);
        }
    }
    egui::IconData { rgba, width: n as u32, height: n as u32 }
}

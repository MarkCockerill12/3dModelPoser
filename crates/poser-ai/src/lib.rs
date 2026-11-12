//! Offline 2D human pose estimation (RTMPose-s, COCO-17 keypoints) using the pure-Rust `tract`
//! ONNX runtime. No network access.
use anyhow::{Context, Result};
use glam::Vec2;
use std::sync::OnceLock;
use tract_onnx::prelude::*;

static MODEL: &[u8] = include_bytes!("../../../assets/ai/rtmpose-s-body7.onnx");
const W: usize = 192;
const H: usize = 256;

/// COCO keypoint order.
pub const NOSE: usize = 0;
pub const L_EYE: usize = 1;
pub const R_EYE: usize = 2;
pub const L_EAR: usize = 3;
pub const R_EAR: usize = 4;
pub const L_SHOULDER: usize = 5;
pub const R_SHOULDER: usize = 6;
pub const L_ELBOW: usize = 7;
pub const R_ELBOW: usize = 8;
pub const L_WRIST: usize = 9;
pub const R_WRIST: usize = 10;
pub const L_HIP: usize = 11;
pub const R_HIP: usize = 12;
pub const L_KNEE: usize = 13;
pub const R_KNEE: usize = 14;
pub const L_ANKLE: usize = 15;
pub const R_ANKLE: usize = 16;

type Plan = std::sync::Arc<TypedRunnableModel>;

fn plan() -> Result<&'static Plan> {
    static P: OnceLock<Result<Plan, String>> = OnceLock::new();
    P.get_or_init(|| {
        tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(MODEL))
            .and_then(|m| {
                let batch = m.sym("batch");
                let shape: TVec<TDim> = tvec!(batch.into(), 3.into(), (H as i64).into(), (W as i64).into());
                m.with_input_fact(0, f32::fact(shape).into())
            })
            .and_then(|m| m.into_optimized())
            .and_then(|m| m.into_runnable())
            .map_err(|e| format!("{e:#}"))
    })
    .as_ref()
    .map_err(|e| anyhow::anyhow!("{e}"))
}

/// One keypoint in image pixels with a confidence score.
#[derive(Clone, Copy, Debug)]
pub struct Keypoint {
    pub pos: Vec2,
    pub score: f32,
}

/// Run the network on a crop box (x0, y0, w, h) of the image (box gets padded to 3:4).
fn run_crop(img: &image::RgbImage, bx: (f32, f32, f32, f32)) -> Result<Vec<Keypoint>> {
    let (cx, cy) = (bx.0 + bx.2 * 0.5, bx.1 + bx.3 * 0.5);
    let mut bw = bx.2 * 1.25;
    let mut bh = bx.3 * 1.25;
    // Match the network aspect ratio (w/h = 0.75).
    if bw / bh > W as f32 / H as f32 {
        bh = bw * H as f32 / W as f32;
    } else {
        bw = bh * W as f32 / H as f32;
    }
    let (x0, y0) = (cx - bw * 0.5, cy - bh * 0.5);
    let (iw, ih) = (img.width() as i32, img.height() as i32);
    let mean = [123.675f32, 116.28, 103.53];
    let std = [58.395f32, 57.12, 57.375];
    let input: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, H, W), |(_, c, y, x)| {
        let sx = (x0 + (x as f32 + 0.5) * bw / W as f32) as i32;
        let sy = (y0 + (y as f32 + 0.5) * bh / H as f32) as i32;
        let v = if sx >= 0 && sy >= 0 && sx < iw && sy < ih { img.get_pixel(sx as u32, sy as u32)[c] as f32 } else { 0.0 };
        (v - mean[c]) / std[c]
    })
    .into();
    let out = plan()?.run(tvec!(input.into()))?;
    let sx = out[0].to_plain_array_view::<f32>().context("simcc_x")?;
    let sy = out[1].to_plain_array_view::<f32>().context("simcc_y")?;
    let (nk, nx) = (sx.shape()[1], sx.shape()[2]);
    let ny = sy.shape()[2];
    let mut kps = Vec::with_capacity(nk);
    for k in 0..nk {
        let (mut bx_i, mut bvx) = (0usize, f32::MIN);
        for i in 0..nx {
            let v = sx[[0, k, i]];
            if v > bvx {
                bvx = v;
                bx_i = i;
            }
        }
        let (mut by_i, mut bvy) = (0usize, f32::MIN);
        for i in 0..ny {
            let v = sy[[0, k, i]];
            if v > bvy {
                bvy = v;
                by_i = i;
            }
        }
        // SimCC split ratio 2.
        let px = bx_i as f32 / 2.0;
        let py = by_i as f32 / 2.0;
        kps.push(Keypoint {
            pos: Vec2::new(x0 + px * bw / W as f32, y0 + py * bh / H as f32),
            score: bvx.min(bvy),
        });
    }
    Ok(kps)
}

/// Estimate the 17 COCO keypoints of the main person in the image.
pub fn estimate(img: &image::RgbImage) -> Result<Vec<Keypoint>> {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let first = run_crop(img, (0.0, 0.0, w, h))?;
    // Refine on a crop around the detected person.
    let good: Vec<&Keypoint> = first.iter().filter(|k| k.score > 0.3).collect();
    if good.len() < 5 {
        return Ok(first);
    }
    let (lo, hi) = good.iter().fold((Vec2::splat(f32::MAX), Vec2::splat(f32::MIN)), |(a, b), k| (a.min(k.pos), b.max(k.pos)));
    let size = hi - lo;
    let bx = (lo.x - size.x * 0.1, lo.y - size.y * 0.12, size.x * 1.2, size.y * 1.24);
    run_crop(img, bx)
}

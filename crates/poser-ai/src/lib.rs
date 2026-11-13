//! Offline 2D human pose estimation (RTMPose-s, COCO-17 keypoints) using the pure-Rust `tract`
//! ONNX runtime. No network access.
use anyhow::{Context, Result};
use glam::Vec2;
use std::sync::OnceLock;
use tract_onnx::prelude::*;

static MODEL: &[u8] = include_bytes!("../../../assets/ai/rtmpose-s-body7.onnx");
static HAND_MODEL: &[u8] = include_bytes!("../../../assets/ai/rtmpose-m-hand.onnx");
const W: usize = 192;
const H: usize = 256;
const HAND: usize = 256;

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

fn load(bytes: &[u8], w: usize, h: usize) -> Result<Plan, String> {
    tract_onnx::onnx()
        .model_for_read(&mut std::io::Cursor::new(bytes))
        .and_then(|m| {
            let batch = m.sym("batch");
            let shape: TVec<TDim> = tvec!(batch.into(), 3.into(), (h as i64).into(), (w as i64).into());
            m.with_input_fact(0, f32::fact(shape).into())
        })
        .and_then(|m| m.into_optimized())
        .and_then(|m| m.into_runnable())
        .map_err(|e| format!("{e:#}"))
}

fn plan() -> Result<&'static Plan> {
    static P: OnceLock<Result<Plan, String>> = OnceLock::new();
    P.get_or_init(|| load(MODEL, W, H)).as_ref().map_err(|e| anyhow::anyhow!("{e}"))
}

fn hand_plan() -> Result<&'static Plan> {
    static P: OnceLock<Result<Plan, String>> = OnceLock::new();
    P.get_or_init(|| load(HAND_MODEL, HAND, HAND)).as_ref().map_err(|e| anyhow::anyhow!("{e}"))
}

/// One keypoint in image pixels with a confidence score.
#[derive(Clone, Copy, Debug)]
pub struct Keypoint {
    pub pos: Vec2,
    pub score: f32,
}

/// Run the network on a crop box (x0, y0, w, h) of the image (box gets padded to 3:4).
fn run_crop(img: &image::RgbImage, bx: (f32, f32, f32, f32)) -> Result<Vec<Keypoint>> {
    run_net(plan()?, W, H, img, bx)
}

fn run_net(net: &Plan, nw: usize, nh: usize, img: &image::RgbImage, bx: (f32, f32, f32, f32)) -> Result<Vec<Keypoint>> {
    let (cx, cy) = (bx.0 + bx.2 * 0.5, bx.1 + bx.3 * 0.5);
    let mut bw = bx.2 * 1.25;
    let mut bh = bx.3 * 1.25;
    // Match the network aspect ratio.
    if bw / bh > nw as f32 / nh as f32 {
        bh = bw * nh as f32 / nw as f32;
    } else {
        bw = bh * nw as f32 / nh as f32;
    }
    let (x0, y0) = (cx - bw * 0.5, cy - bh * 0.5);
    let (iw, ih) = (img.width() as i32, img.height() as i32);
    let mean = [123.675f32, 116.28, 103.53];
    let std = [58.395f32, 57.12, 57.375];
    let input: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, nh, nw), |(_, c, y, x)| {
        let sx = (x0 + (x as f32 + 0.5) * bw / nw as f32) as i32;
        let sy = (y0 + (y as f32 + 0.5) * bh / nh as f32) as i32;
        let v = if sx >= 0 && sy >= 0 && sx < iw && sy < ih { img.get_pixel(sx as u32, sy as u32)[c] as f32 } else { 0.0 };
        (v - mean[c]) / std[c]
    })
    .into();
    let out = net.run(tvec!(input.into()))?;
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
            pos: Vec2::new(x0 + px * bw / nw as f32, y0 + py * bh / nh as f32),
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

/// Finger shape estimated from a photo: curl 0..1 per finger (thumb, index, middle, ring,
/// little), spread 0..1, thumb opposition 0..1.
#[derive(Clone, Copy, Debug)]
pub struct HandShape {
    pub curl: [f32; 5],
    pub spread: f32,
    pub opposition: f32,
    pub score: f32,
}

/// Estimate both hands (index 0 = the person's left) from crops around the body's wrists.
pub fn estimate_hands(img: &image::RgbImage, body: &[Keypoint]) -> [Option<HandShape>; 2] {
    let mut out = [None, None];
    for (i, (wi, ei)) in [(L_WRIST, L_ELBOW), (R_WRIST, R_ELBOW)].into_iter().enumerate() {
        let (Some(w), Some(e)) = (body.get(wi), body.get(ei)) else { continue };
        if w.score < 0.3 || e.score < 0.2 {
            continue;
        }
        let fore = w.pos - e.pos;
        let len = fore.length();
        if len < 4.0 {
            continue;
        }
        // The hand sits beyond the wrist, about 0.8 forearm-lengths across.
        let c = w.pos + fore * 0.35;
        let size = len * 0.9;
        let Ok(net) = hand_plan() else { return out };
        let bx = (c.x - size * 0.5, c.y - size * 0.5, size, size);
        let Ok(k) = run_net(net, HAND, HAND, img, bx) else { continue };
        if let Ok(prefix) = std::env::var("POSER_HANDDBG") {
            // Debug: the crop (scaled up) with the detected points.
            let s = 8.0;
            let (x0, y0) = (bx.0 - bx.2 * 0.125, bx.1 - bx.3 * 0.125);
            let side = (bx.2 * 1.25 * s) as u32;
            let mut dbg = image::RgbImage::from_fn(side, side, |x, y| {
                let sx = (x0 + x as f32 / s) as i32;
                let sy = (y0 + y as f32 / s) as i32;
                if sx >= 0 && sy >= 0 && (sx as u32) < img.width() && (sy as u32) < img.height() { *img.get_pixel(sx as u32, sy as u32) } else { image::Rgb([0, 0, 0]) }
            });
            for (n, kp) in k.iter().enumerate() {
                let (px, py) = (((kp.pos.x - x0) * s) as i32, ((kp.pos.y - y0) * s) as i32);
                let col = [[255, 0, 0], [255, 128, 0], [0, 200, 0], [0, 128, 255], [200, 0, 255]][(n.max(1) - 1) / 4];
                for dy in -3..=3 {
                    for dx in -3..=3 {
                        let (x, y) = (px + dx, py + dy);
                        if x >= 0 && y >= 0 && (x as u32) < side && (y as u32) < side {
                            dbg.put_pixel(x as u32, y as u32, image::Rgb(col));
                        }
                    }
                }
            }
            let _ = dbg.save(format!("{prefix}_{i}.png"));
            println!("hand {i}: box {bx:?} scores {:?}", k.iter().map(|p| (p.score * 100.0) as i32).collect::<Vec<_>>());
        }
        out[i] = hand_shape(&k);
    }
    out
}

fn hand_shape(k: &[Keypoint]) -> Option<HandShape> {
    if k.len() < 21 {
        return None;
    }
    let score = k.iter().map(|p| p.score).sum::<f32>() / 21.0;
    if score < 0.25 {
        return None;
    }
    let p = |i: usize| k[i].pos;
    let palm = (p(9) - p(0)).length().max(1.0);
    let ang = |a: Vec2, b: Vec2| {
        let (a, b) = (a.normalize_or_zero(), b.normalize_or_zero());
        a.dot(b).clamp(-1.0, 1.0).acos()
    };
    // Straight finger length relative to the palm (thumb from its base joint).
    let expected = [0.75f32, 0.85, 0.95, 0.88, 0.7];
    let ranges = [150f32, 252.0, 252.0, 252.0, 252.0];
    let mut curl = [0.25f32; 5];
    for (f, c) in curl.iter_mut().enumerate() {
        let b = 1 + 4 * f;
        let segs = [p(b) - p(0), p(b + 1) - p(b), p(b + 2) - p(b + 1), p(b + 3) - p(b + 2)];
        let bend = (ang(segs[0], segs[1]) + ang(segs[1], segs[2]) + ang(segs[2], segs[3])).to_degrees();
        let from_angles = (bend / ranges[f]).clamp(0.0, 1.0);
        // A curled finger also looks short (its tip comes back towards the knuckle).
        let chord = (p(b + 3) - p(b + if f == 0 { 1 } else { 0 })).length() / palm;
        let from_length = (1.0 - chord / expected[f]).clamp(0.0, 1.0);
        let fs = (k[b].score + k[b + 1].score + k[b + 2].score + k[b + 3].score) / 4.0;
        if fs > 0.2 {
            *c = from_angles.max(from_length * 0.9).clamp(0.0, 1.0);
        }
    }
    let spread_deg = ang(p(8) - p(5), p(20) - p(17)).to_degrees();
    let spread = ((spread_deg - 10.0) / 30.0).clamp(0.0, 1.0);
    let opposition = ((1.0 - (p(4) - p(5)).length() / palm) * 0.9).clamp(0.0, 1.0);
    Some(HandShape { curl, spread, opposition, score })
}


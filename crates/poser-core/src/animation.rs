use crate::transform::Transform;
use glam::{Quat, Vec3};

#[derive(Clone, Debug)]
pub struct Track<T> {
    pub times: Vec<f32>,
    pub values: Vec<T>,
    pub step: bool,
}

impl<T> Track<T> {
    fn locate(&self, t: f32) -> (usize, usize, f32) {
        let n = self.times.len();
        if n == 0 {
            return (0, 0, 0.0);
        }
        if t <= self.times[0] {
            return (0, 0, 0.0);
        }
        if t >= self.times[n - 1] {
            return (n - 1, n - 1, 0.0);
        }
        let i = self.times.partition_point(|x| *x <= t).saturating_sub(1);
        let j = (i + 1).min(n - 1);
        let span = self.times[j] - self.times[i];
        let f = if span > 1e-9 && !self.step { (t - self.times[i]) / span } else { 0.0 };
        (i, j, f)
    }
}

impl Track<Vec3> {
    pub fn sample(&self, t: f32) -> Option<Vec3> {
        if self.values.is_empty() {
            return None;
        }
        let (i, j, f) = self.locate(t);
        Some(self.values[i].lerp(self.values[j], f))
    }
}

impl Track<Quat> {
    pub fn sample(&self, t: f32) -> Option<Quat> {
        if self.values.is_empty() {
            return None;
        }
        let (i, j, f) = self.locate(t);
        let a = self.values[i];
        let mut b = self.values[j];
        if a.dot(b) < 0.0 {
            b = -b;
        }
        Some(a.slerp(b, f).normalize())
    }
}

impl Track<Vec<f32>> {
    pub fn sample_weights(&self, t: f32) -> Option<Vec<f32>> {
        if self.values.is_empty() {
            return None;
        }
        let (i, j, f) = self.locate(t);
        Some(self.values[i].iter().zip(&self.values[j]).map(|(a, b)| a + (b - a) * f).collect())
    }
}

#[derive(Clone, Debug, Default)]
pub struct BoneChannel {
    pub bone: usize,
    pub translation: Option<Track<Vec3>>,
    pub rotation: Option<Track<Quat>>,
    pub scale: Option<Track<Vec3>>,
}

#[derive(Clone, Debug)]
pub struct MorphChannel {
    /// Morph names in the order of the weight vectors.
    pub names: Vec<String>,
    pub weights: Track<Vec<f32>>,
}

#[derive(Clone, Debug, Default)]
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    /// Nominal frame rate for the frame scrubber.
    pub fps: f32,
    pub channels: Vec<BoneChannel>,
    pub morph_channels: Vec<MorphChannel>,
}

impl AnimationClip {
    pub fn frame_count(&self) -> u32 {
        (self.duration * self.fps).round() as u32 + 1
    }

    /// Sample bone locals into `out` (which should start as the rest pose).
    pub fn sample_into(&self, t: f32, out: &mut [Transform]) {
        for ch in &self.channels {
            let Some(tr) = out.get_mut(ch.bone) else { continue };
            if let Some(v) = ch.translation.as_ref().and_then(|k| k.sample(t)) {
                tr.translation = v;
            }
            if let Some(v) = ch.rotation.as_ref().and_then(|k| k.sample(t)) {
                tr.rotation = v;
            }
            if let Some(v) = ch.scale.as_ref().and_then(|k| k.sample(t)) {
                tr.scale = v;
            }
        }
    }

    pub fn recompute_duration(&mut self) {
        let mut d: f32 = 0.0;
        for c in &self.channels {
            for last in [
                c.translation.as_ref().and_then(|k| k.times.last()),
                c.rotation.as_ref().and_then(|k| k.times.last()),
                c.scale.as_ref().and_then(|k| k.times.last()),
            ]
            .into_iter()
            .flatten()
            {
                d = d.max(*last);
            }
        }
        for m in &self.morph_channels {
            if let Some(l) = m.weights.times.last() {
                d = d.max(*l);
            }
        }
        self.duration = d;
        if self.fps <= 0.0 {
            self.fps = 30.0;
        }
    }
}

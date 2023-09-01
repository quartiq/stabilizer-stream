use crate::{Filter, HbfDecCascade};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

/// Window kernel
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Window<const N: usize> {
    pub win: [f32; N],
    pub power: f32,
    /// Normalized effective noise bandwidth (in bins)
    pub nenbw: f32,
}

/// Hann window
impl<const N: usize> Window<N> {
    pub fn hann() -> Self {
        assert!(N > 0);
        let df = core::f32::consts::PI / N as f32;
        let mut win = [0.0; N];
        for (i, w) in win.iter_mut().enumerate() {
            *w = (df * i as f32).sin().powi(2);
        }
        Self {
            win,
            power: 4.0,
            nenbw: 1.5,
        }
    }
}

/// Detrend method
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Detrend {
    /// No detrending
    None,
    /// Remove the mean of first and last item per segment
    Mean,
    /// Remove linear interpolation between first and last item for each segment
    Linear,
}

/// Power spectral density accumulator and decimator
///
/// One stage in [PsdCascade].
#[derive(Clone)]
pub struct Psd<const N: usize> {
    hbf: HbfDecCascade,
    buf: [f32; N],
    idx: usize,
    spectrum: [f32; N], // using only the positive half
    count: usize,
    fft: Arc<dyn Fft<f32>>,
    win: Arc<Window<N>>,
    detrend: Detrend,
    drain: usize,
}

impl<const N: usize> Psd<N> {
    pub fn new(fft: Arc<dyn Fft<f32>>, win: Arc<Window<N>>) -> Self {
        let hbf = HbfDecCascade::default();
        assert_eq!(N & 1, 0);
        assert_eq!(N, fft.len());
        // check fft and decimation block size compatibility
        assert!(hbf.block_size().0 <= N / 2); // needed for processing and dropping blocks
        assert!(hbf.block_size().1 >= N / 2); // needed for processing and dropping blocks
        assert!(N >= 2); // Nyquist and DC distinction
        Self {
            hbf,
            buf: [0.0; N],
            idx: 0,
            spectrum: [0.0; N],
            count: 0,
            fft,
            win,
            detrend: Detrend::None,
            drain: 0,
        }
    }

    pub fn set_detrend(&mut self, d: Detrend) {
        self.detrend = d;
    }

    pub fn set_stage_length(&mut self, n: usize) {
        self.hbf.set_n(n);
        self.drain = self.hbf.response_length();
    }
}

pub trait Stage {
    /// Process items
    ///
    /// Unused items are buffered.
    /// Full FFT blocks are processed.
    /// Overlap is kept.
    /// Decimation is performed on fully processed input items.
    ///
    /// # Args
    /// * `x`: input items
    /// * `y`: output items
    ///
    /// # Returns
    /// number if items written to `y`
    fn process(&mut self, x: &[f32], y: &mut [f32]) -> usize;
    /// Return the positive frequency half of the spectrum
    fn spectrum(&self) -> &[f32];
    /// PSD normalization factor
    ///
    /// one-sided
    fn gain(&self) -> f32;
    fn count(&self) -> usize;
}

impl<const N: usize> Stage for Psd<N> {
    fn process(&mut self, mut x: &[f32], y: &mut [f32]) -> usize {
        let mut c = [Complex::default(); N];
        let mut n = 0;
        while !x.is_empty() {
            // load
            let take = x.len().min(self.buf.len() - self.idx);
            let (chunk, rest) = x.split_at(take);
            x = rest;
            self.buf[self.idx..][..take].copy_from_slice(chunk);
            self.idx += take;
            if self.idx < N {
                break;
            }
            // compute detrend
            let (mean, slope) = match self.detrend {
                Detrend::None => (0.0, 0.0),
                Detrend::Mean => (0.5 * (self.buf[0] + self.buf[N - 1]), 0.0),
                Detrend::Linear => (
                    self.buf[0],
                    (self.buf[N - 1] - self.buf[0]) / (N - 1) as f32,
                ),
            };
            // apply detrending, window, make complex
            for (i, (c, (x, w))) in c
                .iter_mut()
                .zip(self.buf.iter().zip(self.win.win.iter()))
                .enumerate()
            {
                c.re = (x - mean - i as f32 * slope) * w;
                c.im = 0.0;
            }
            // fft in-place
            self.fft.process(&mut c);
            // convert positive frequency spectrum to power
            // and accumulate
            for (c, p) in c[..N / 2].iter().zip(self.spectrum.iter_mut()) {
                *p += c.norm_sqr();
            }
            self.count += 1;

            // decimate left half
            let (left, right) = self.buf.split_at_mut(N / 2);
            let mut k = self.hbf.process_block(None, left);
            // drain decimator impulse response to initial state (zeros)
            (k, self.drain) = (k.saturating_sub(self.drain), self.drain.saturating_sub(k));
            y[n..][..k].copy_from_slice(&left[..k]);
            n += k;
            // drop the left keep the right as overlap
            left.copy_from_slice(right);
            self.idx = N / 2;
        }
        n
    }

    fn spectrum(&self) -> &[f32] {
        &self.spectrum[..N / 2]
    }

    fn count(&self) -> usize {
        self.count
    }

    fn gain(&self) -> f32 {
        // 2 for one-sided
        // overlap compensated by counting
        self.win.power / ((self.count * N / 2) as f32 * self.win.nenbw)
    }
}

/// Online power spectral density calculation
///
/// This performs efficient long term power spectral density monitoring in real time.
/// The idea is to make short FFTs and decimate in stages, then
/// stitch together the FFT bins from the different stages.
/// This allows arbitrarily large effective FFTs sizes in practice with only
/// logarithmically increasing memory consumption. And it gets rid of the delay in
/// recording and computing the large FFTs. The effective full FFT size grows in real-time
/// and does not need to be fixed before recording and computing.
/// This is well defined with the caveat that spur power depends on the changing bin width.
/// It's also typically what some modern signal analyzers or noise metrology instruments do.
///
/// See also [`csdl`](https://github.com/jordens/csdl) or
/// [LPSD](https://doi.org/10.1016/j.measurement.2005.10.010).
///
/// Infinite averaging
/// Incremental updates
/// Automatic FFT stage extension
#[derive(Clone)]
pub struct PsdCascade<const N: usize> {
    stages: Vec<Psd<N>>,
    fft: Arc<dyn Fft<f32>>,
    stage_length: usize,
    detrend: Detrend,
    win: Arc<Window<N>>,
}

impl<const N: usize> Default for PsdCascade<N> {
    /// Create a new Psd instance
    ///
    /// fft_size: size of the FFT blocks and the window
    /// stage_length: number of decimation stages. rate change per stage is 1 << stage_length
    /// detrend: [Detrend] method
    fn default() -> Self {
        let fft = FftPlanner::<f32>::new().plan_fft_forward(N);
        let win = Arc::new(Window::hann());
        Self {
            stages: Vec::with_capacity(4),
            fft,
            stage_length: 4,
            detrend: Detrend::None,
            win,
        }
    }
}

impl<const N: usize> PsdCascade<N> {
    pub fn set_stage_length(&mut self, n: usize) {
        self.stage_length = n;
        for stage in self.stages.iter_mut() {
            stage.set_stage_length(n);
        }
    }

    pub fn set_detrend(&mut self, d: Detrend) {
        self.detrend = d;
    }

    /// Process input items
    pub fn process(&mut self, x: &[f32]) {
        let mut a = ([0f32; N], [0f32; N]);
        let (mut y, mut z) = (&mut a.0[..], &mut a.1[..]);
        for mut x in x.chunks(N << self.stage_length) {
            let mut i = 0;
            while !x.is_empty() {
                while i >= self.stages.len() {
                    let mut stage = Psd::new(self.fft.clone(), self.win.clone());
                    stage.set_stage_length(self.stage_length);
                    stage.set_detrend(self.detrend);
                    self.stages.push(stage);
                }
                let n = self.stages[i].process(x, &mut y[..]);
                core::mem::swap(&mut z, &mut y);
                x = &z[..n];
                i += 1;
            }
        }
    }

    /// Return the PSD and a Vec of segement break information
    ///
    /// # Args
    /// * `min_count`: minimum number of averages to include in output
    ///
    /// # Returns
    /// * `psd`: `Vec` normalized reversed (Nyquist first, DC last)
    /// * `breaks`: `Vec` of stage breaks `[start index in psd, average count, highest bin index, effective fft size]`
    pub fn get(&self, min_count: usize) -> (Vec<f32>, Vec<[usize; 4]>) {
        let mut p = Vec::with_capacity(self.stages.len() * N / 2);
        let mut b = Vec::with_capacity(self.stages.len());
        let mut n = 0;
        for stage in self.stages.iter().take_while(|s| s.count >= min_count) {
            let mut pi = stage.spectrum();
            let f = stage.fft.len();
            // a stage yields frequency bins 0..N/2 ty its nyquist
            // 0..floor(0.4*N) is its passband if it was preceeded by a decimator
            // 0..floor(0.4*N/R) is next lower stage
            // hence take bins ceil(0.4*N/R)..floor(0.4*N) from a stage
            if !p.is_empty() {
                // not the first stage
                // remove transition band of previous stage's decimator, floor
                let f_pass = 4 * f / 10;
                pi = &pi[..f_pass];
                // remove low f bins from previous stage, ceil
                let f_low = (4 * f + (10 << stage.hbf.n()) - 1) / (10 << stage.hbf.n());
                p.truncate(p.len() - f_low);
            }
            let g = stage.gain() * (1 << n) as f32;
            b.push([p.len(), stage.count(), pi.len(), f << n]);
            p.extend(pi.iter().rev().map(|pi| pi * g));
            n += stage.hbf.n();
        }
        if !p.is_empty() {
            // correct DC and Nyquist bins as both only contribute once to the one-sided spectrum
            // this matches matplotlib and matlab but is certainly a questionable step
            // need special care when interpreting and integrating the PSD
            p[0] *= 0.5;
            let n = p.len();
            p[n - 1] *= 0.5;
        }
        (p, b)
    }

    /// Compute PSD bin center frequencies from stage breaks.
    pub fn f(&self, b: &[[usize; 4]]) -> Vec<f32> {
        let Some(bi) = b.last() else { return vec![] };
        let mut f = Vec::with_capacity(bi[0] + bi[2]);
        for bi in b.iter() {
            f.truncate(bi[0]);
            let df = 1.0 / bi[3] as f32;
            f.extend((0..bi[2]).rev().map(|f| f as f32 * df));
        }
        assert_eq!(f.len(), bi[0] + bi[2]);
        f
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test() {
        assert_eq!(crate::HBF_PASSBAND, 0.4);

        // make uniform noise [-1, 1), ignore the epsilon.
        let x: Vec<_> = (0..1 << 20)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();
        let xm = x.iter().map(|x| *x as f64).sum::<f64>() as f32 / x.len() as f32;
        // mean is 0, take 10 sigma here and elsewhere
        assert!(xm.abs() < 10.0 / (x.len() as f32).sqrt());
        let xv = x.iter().map(|x| (x * x) as f64).sum::<f64>() as f32 / x.len() as f32;
        // variance is 1/3
        assert!((xv * 3.0 - 1.0).abs() < 10.0 / (x.len() as f32).sqrt());

        const F: usize = 1 << 9;
        let n = 3;
        let mut s = Psd::<F>::new(FftPlanner::new().plan_fft_forward(F), Window::hann());
        s.set_stage_length(n);
        let mut y = vec![];
        for x in x.chunks(F << n) {
            let mut yi = [0.0; F];
            let k = s.process(x, &mut yi[..]);
            y.extend_from_slice(&yi[..k]);
        }
        let mut hbf = HbfDecCascade::default();
        hbf.set_n(n);
        assert_eq!(y.len(), ((x.len() - F / 2) >> n) - hbf.response_length());
        let p: Vec<_> = s.spectrum().iter().map(|p| p * s.gain()).collect();
        // psd of a stage
        assert!(
            p.iter()
                // 0.5 for one-sided spectrum
                .all(|p| (p * 0.5 * 3.0 - 1.0).abs() < 10.0 / (s.count() as f32).sqrt()),
            "{:?}",
            &p[..]
        );

        let mut d = PsdCascade::<F>::default();
        d.set_stage_length(n);
        d.set_detrend(Detrend::None);
        d.process(&x);
        let (mut p, b) = d.get(1);
        // tweak DC and Nyquist to make checks less code
        let n = p.len();
        p[0] *= 2.0;
        p[n - 1] *= 2.0;
        for (i, bi) in b.iter().enumerate() {
            // let (start, count, high, size) = bi.into();
            let end = b.get(i + 1).map(|bi| bi[0]).unwrap_or(n);
            let pi = &p[bi[0]..end];
            // psd of the cascade
            assert!(pi
                .iter()
                // 0.5 for one-sided spectrum
                .all(|p| (p * 0.5 * 3.0 - 1.0).abs() < 10.0 / (bi[1] as f32).sqrt()));
        }
    }
}

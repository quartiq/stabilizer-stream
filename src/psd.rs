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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
pub struct Psd<const N: usize> {
    hbf: HbfDecCascade,
    buf: heapless::Vec<f32, N>,
    out: heapless::Vec<f32, N>,
    spectrum: [f32; N], // using only the positive half
    count: usize,
    fft: Arc<dyn Fft<f32>>,
    win: Window<N>,
    detrend: Detrend,
    drain: usize,
}

impl<const N: usize> Psd<N> {
    pub fn new(fft: Arc<dyn Fft<f32>>, win: Window<N>) -> Self {
        let hbf = HbfDecCascade::default();
        assert_eq!(N & 1, 0);
        assert_eq!(N, fft.len());
        // check fft and decimation block size compatibility
        assert!(hbf.block_size().0 <= N / 2); // needed for processing and dropping blocks
        assert!(hbf.block_size().1 >= N / 2); // needed for processing and dropping blocks
        assert!(N >= 2); // Nyquist and DC distinction
        Self {
            hbf,
            buf: heapless::Vec::new(),
            out: heapless::Vec::new(),
            spectrum: [0.0; N],
            count: 0,
            fft,
            win,
            detrend: Detrend::None,
            drain: 0,
        }
    }

    pub fn detrend(mut self, d: Detrend) -> Self {
        self.detrend = d;
        self
    }

    pub fn stage_length(mut self, n: usize) -> Self {
        self.hbf.set_n(n);
        self.drain = self.hbf.response_length();
        self
    }
}

pub trait Stage {
    /// Process items
    ///
    /// Unusde items are buffered.
    /// Full FFT blocks are processed.
    /// Overlap is kept.
    /// Decimation is performed on fully processed input items.
    ///
    /// # Args
    /// * `x`: input items
    /// * `detrend`: [Detrend] method
    ///
    /// # Returns
    /// decimated output
    fn process(&mut self, x: &[f32]) -> &[f32];
    /// Return the positive frequency half of the spectrum
    fn spectrum(&self) -> &[f32];
    /// PSD normalization factor
    ///
    /// one-sided
    fn gain(&self) -> f32;
    fn count(&self) -> usize;
}

impl<const N: usize> Stage for Psd<N> {
    fn process(&mut self, mut x: &[f32]) -> &[f32] {
        self.out.clear();
        let mut c = [Complex::default(); N];
        while !x.is_empty() {
            // load
            let take = x.len().min(self.buf.capacity() - self.buf.len());
            let (chunk, rest) = x.split_at(take);
            x = rest;
            self.buf.extend_from_slice(chunk).unwrap();
            if self.buf.len() < N {
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

            // decimate non-overlapping chunks
            let (left, right) = self.buf.split_at_mut(N / 2);
            let mut k = self.hbf.process_block(None, left);
            // drain decimator impulse response to initial state (zeros)
            (k, self.drain) = (k.saturating_sub(self.drain), self.drain.saturating_sub(k));
            self.out.extend_from_slice(&left[..k]).unwrap();
            // drop the overlapped and processed chunks
            left.copy_from_slice(right);
            self.buf.truncate(N / 2);
        }
        &self.out
    }

    fn spectrum(&self) -> &[f32] {
        &self.spectrum[..N / 2]
    }

    fn count(&self) -> usize {
        self.count
    }

    fn gain(&self) -> f32 {
        // 2 for one-sided
        // overlap compensated by count
        2.0 * self.win.power / ((self.count * N) as f32 * self.win.nenbw)
    }
}

/// Online PSD calculator
///
/// Infinite averaging
/// Incremental updates
/// Automatic FFT stage extension
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
            stages: vec![],
            fft,
            stage_length: 4,
            detrend: Detrend::None,
            win,
        }
    }
}

impl<const N: usize> PsdCascade<N> {
    pub fn stage_length(mut self, n: usize) -> Self {
        self.stage_length = n;
        self
    }

    pub fn detrend(mut self, d: Detrend) -> Self {
        self.detrend = d;
        self
    }

    /// Process input items
    pub fn process(&mut self, x: &[f32]) {
        let mut x = x;
        x = self.stages.iter_mut().fold(x, |x, stage| stage.process(x));
        if !x.is_empty() {
            let mut stage = Psd::new(self.fft.clone(), *self.win)
                .stage_length(self.stage_length)
                .detrend(self.detrend);
            stage.process(x);
            self.stages.push(stage);
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
        let mut p = vec![];
        let mut b = vec![];
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
        // correct DC and Nyquist bins as both only contribute once to the one-sided spectrum
        // this matches matplotlib and matlab but is certainly a questionable step
        // need special care when interpreting and integrating the PSD
        p[0] *= 0.5;
        let n = p.len();
        p[n - 1] *= 0.5;
        (p, b)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test() {
        assert_eq!(crate::HBF_PASSBAND, 0.4);

        // make uniform noise [-1, 1), ignore the epsilon.
        let x: Vec<f32> = (0..1 << 20)
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
        let mut s = Psd::<{ 1 << 9 }>::new(FftPlanner::new().plan_fft_forward(F), Window::hann())
            .stage_length(n);
        let mut y = vec![];
        for x in x.chunks(F << n) {
            y.extend_from_slice(s.process(x));
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

        let mut d = PsdCascade::<F>::default()
            .stage_length(n)
            .detrend(Detrend::None);
        for x in x.chunks(F << n) {
            d.process(x);
        }
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

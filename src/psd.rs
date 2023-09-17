use idsp::hbf::{Filter, HbfDecCascade};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

/// Window kernel
///
/// <https://holometer.fnal.gov/GH_FFT.pdf>
/// <https://gist.github.com/endolith/c4b8e1e3c630a260424123b4e9d964c4>
/// <https://docs.google.com/spreadsheets/d/1glvo-y1tqCiYwK0QQWhB4AAcDFiK_C_0M4SeA0Uyqdc/edit>
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Window<const N: usize> {
    pub win: [f32; N],
    /// Mean squared
    pub power: f32,
    /// Normalized effective noise bandwidth (in bins)
    pub nenbw: f32,
}

impl<const N: usize> Window<N> {
    /// Rectangular window
    pub fn rectangular() -> Self {
        assert!(N > 0);
        Self {
            win: [1.0; N],
            power: 1.0,
            nenbw: 1.0,
        }
    }

    /// Hann window
    ///
    /// This is the "numerical" version of the window with period `N`, `win[0] = win[N]`
    /// (conceptually), specifically `win[0] != win[win.len() - 1]`.
    /// Matplotlib's `matplotlib.mlab.window_hanning()` (but not scipy.signal.get_window())
    /// uses the symetric one of period `N-1`, with `win[0] = win[N - 1] = 0`
    /// which looses a lot of useful properties (exact nenbw() and power() independent of `N`,
    /// exact optimal overlap etc)
    pub fn hann() -> Self {
        assert!(N > 0);
        let df = core::f32::consts::PI / N as f32;
        let mut win = [0.0; N];
        for (i, w) in win.iter_mut().enumerate() {
            *w = (df * i as f32).sin().powi(2);
        }
        Self {
            win,
            power: 0.25,
            nenbw: 1.5,
        }
    }
}

/// Detrend method
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Detrend {
    /// No detrending
    None,
    /// Subtract the midpoint of each segment
    Mid,
    /// Remove linear interpolation between first and last item for each segment
    Span,
    // TODO: mean
    // TODO: linear
}

/// Power spectral density accumulator and decimator
///
/// Note: Don't feed more than N*1e7 items without expecting loss of accuracy
///
/// One stage in [PsdCascade].
#[derive(Clone)]
pub struct Psd<const N: usize> {
    hbf: HbfDecCascade,
    buf: [f32; N],
    idx: usize,
    spectrum: [f32; N], // using only the positive half N/2 + 1
    count: usize,
    fft: Arc<dyn Fft<f32>>,
    win: Arc<Window<N>>,
    overlap: usize,
    detrend: Detrend,
    drain: usize,
}

impl<const N: usize> Psd<N> {
    pub fn new(fft: Arc<dyn Fft<f32>>, win: Arc<Window<N>>) -> Self {
        let hbf = HbfDecCascade::default();
        assert_eq!(N, fft.len());
        // check fft and decimation block size compatibility
        assert!(N >= 2); // Nyquist and DC distinction
        let mut s = Self {
            hbf,
            buf: [0.0; N],
            idx: 0,
            spectrum: [0.0; N],
            count: 0,
            fft,
            win,
            overlap: 0,
            detrend: Detrend::None,
            drain: 0,
        };
        s.set_overlap(N / 2);
        s.set_stage_depth(0);
        s
    }

    pub fn set_overlap(&mut self, o: usize) {
        assert_eq!(o % self.hbf.block_size().0, 0);
        assert!(self.hbf.block_size().1 >= o);
        assert!(o < N);
        self.overlap = o;
    }

    pub fn set_detrend(&mut self, d: Detrend) {
        self.detrend = d;
    }

    pub fn set_stage_depth(&mut self, n: usize) {
        self.hbf.set_depth(n);
        self.drain = self.hbf.response_length();
    }
}

pub trait PsdStage {
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
    fn process<'a>(&mut self, x: &[f32], y: &'a mut [f32]) -> &'a mut [f32];
    /// Return the positive frequency half of the spectrum
    fn spectrum(&self) -> &[f32];
    /// PSD normalization factor
    ///
    /// one-sided
    fn gain(&self) -> f32;
    /// Number of averages
    fn count(&self) -> usize;
    /// Currently buffered items
    fn buf(&self) -> &[f32];
}

impl<const N: usize> PsdStage for Psd<N> {
    fn process<'a>(&mut self, mut x: &[f32], y: &'a mut [f32]) -> &'a mut [f32] {
        let mut n = 0;
        let mut chunk;
        while !x.is_empty() {
            // load
            let take = x.len().min(self.buf.len() - self.idx);
            (chunk, x) = x.split_at(take);
            self.buf[self.idx..][..take].copy_from_slice(chunk);
            self.idx += take;
            if self.idx < N {
                break;
            }
            // compute detrend
            let (mut mean, slope) = match self.detrend {
                Detrend::None => (0.0, 0.0),
                Detrend::Mid => (self.buf[N / 2], 0.0),
                Detrend::Span => (
                    self.buf[0],
                    (self.buf[N - 1] - self.buf[0]) / (N - 1) as f32,
                ),
            };
            // apply detrending, window, make complex
            let mut c = [Complex::default(); N];
            for (c, (x, w)) in c.iter_mut().zip(self.buf.iter().zip(self.win.win.iter())) {
                c.re = (x - mean) * w;
                mean += slope;
            }
            // fft in-place
            self.fft.process(&mut c);
            // convert positive frequency spectrum to power
            // and accumulate
            // TODO: accuracy for large counts
            for (c, p) in c[..N / 2 + 1]
                .iter()
                .zip(self.spectrum[..N / 2 + 1].iter_mut())
            {
                *p += c.norm_sqr();
            }
            self.count += 1;

            // decimate overlap
            let mut yi = self.hbf.process_block(None, &mut self.buf[..self.overlap]);
            // drain decimator impulse response to initial state (zeros)
            let skip = self.drain.min(yi.len());
            self.drain -= skip;
            yi = &mut yi[skip..];
            // yield l
            y[n..][..yi.len()].copy_from_slice(yi);
            n += yi.len();
            // drop the left keep the right as overlap
            self.buf.copy_within(self.overlap..N, 0);
            self.idx = N - self.overlap;
        }
        &mut y[..n]
    }

    fn spectrum(&self) -> &[f32] {
        &self.spectrum[..N / 2 + 1]
    }

    fn count(&self) -> usize {
        self.count
    }

    fn gain(&self) -> f32 {
        // 2 for one-sided
        // overlap is compensated by counting
        1.0 / ((self.count * N / 2) as f32 * self.win.nenbw * self.win.power)
    }

    fn buf(&self) -> &[f32] {
        &self.buf[..self.idx]
    }
}

/// Stage break information
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Break {
    /// Start index in PSD and frequencies
    pub start: usize,
    /// Number of averages
    pub count: usize,
    /// Highes FFT bin (at `start`)
    pub highest_bin: usize,
    /// The effective FFT size
    pub effective_fft_size: usize,
}

impl Break {
    /// Compute PSD bin center frequencies from stage breaks.
    pub fn frequencies(b: &[Break]) -> Vec<f32> {
        let Some(bi) = b.last() else { return vec![] };
        let mut f = Vec::with_capacity(bi.start + bi.highest_bin);
        for bi in b.iter() {
            f.truncate(bi.start);
            let df = 1.0 / bi.effective_fft_size as f32;
            f.extend((0..bi.highest_bin).rev().map(|f| f as f32 * df));
        }
        assert_eq!(f.len(), bi.start + bi.highest_bin);
        debug_assert_eq!(f.first(), Some(&0.5));
        debug_assert_eq!(f.last(), Some(&0.0));
        f
    }
}

/// Online power spectral density estimation
///
/// This performs efficient long term power spectral density monitoring in real time.
/// The idea is to perform FFTs over relatively short windows and simultaneously decimate
/// the time domain data, everything in multiple stages, then
/// stitch together the FFT bins from the different stages.
/// This allows arbitrarily large effective FFTs sizes in practice with only
/// logarithmically increasing memory and cpu consumption. And it makes available PSD data
/// from higher frequency stages early to get rid of the delay in
/// recording and computing the large FFTs. The effective full FFT size grows in real-time
/// and does not need to be fixed.
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
    overlap: usize,
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
            stage_length: 1,
            detrend: Detrend::None,
            overlap: N / 2,
            win,
        }
    }
}

impl<const N: usize> PsdCascade<N> {
    pub fn set_window(&mut self, win: Window<N>) {
        self.win = Arc::new(win);
    }

    pub fn set_stage_depth(&mut self, n: usize) {
        assert!(n > 0);
        self.stage_length = n;
        for stage in self.stages.iter_mut() {
            stage.set_stage_depth(n);
        }
    }

    pub fn set_detrend(&mut self, d: Detrend) {
        self.detrend = d;
    }

    fn get_or_add(&mut self, i: usize) -> &mut Psd<N> {
        while i >= self.stages.len() {
            let mut stage = Psd::new(self.fft.clone(), self.win.clone());
            stage.set_stage_depth(self.stage_length);
            stage.set_detrend(self.detrend);
            stage.set_overlap(self.overlap);
            self.stages.push(stage);
        }
        &mut self.stages[i]
    }

    /// Process input items
    pub fn process(&mut self, x: &[f32]) {
        let mut a = ([0f32; N], [0f32; N]);
        let (mut y, mut z) = (&mut a.0[..], &mut a.1[..]);
        for mut x in x.chunks(N << self.stage_length) {
            let mut i = 0;
            while !x.is_empty() {
                let n = self.get_or_add(i).process(x, y).len();
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
    /// * `breaks`: `Vec` of stage breaks
    pub fn psd(&self, min_count: usize) -> (Vec<f32>, Vec<Break>) {
        let mut p = Vec::with_capacity(self.stages.len() * (N / 2 + 1));
        let mut b = Vec::with_capacity(self.stages.len());
        let mut n = 0;
        for stage in self.stages.iter().take_while(|s| s.count >= min_count) {
            let mut pi = stage.spectrum();
            // a stage yields frequency bins 0..N/2 ty its nyquist
            // 0..floor(0.4*N) is its passband if it was preceeded by a decimator
            // 0..floor(0.4*N/R) is next lower stage
            // hence take bins ceil(0.4*N/R)..floor(0.4*N) from a stage
            if !p.is_empty() {
                // not the first stage
                // remove transition band of previous stage's decimator, floor
                let f_pass = 4 * N / 10;
                pi = &pi[..f_pass];
                // remove low f bins from previous stage, ceil
                let f_low = (4 * N + (10 << stage.hbf.depth()) - 1) / (10 << stage.hbf.depth());
                p.truncate(p.len() - f_low);
            }
            let g = stage.gain() * (1 << n) as f32;
            b.push(Break {
                start: p.len(),
                count: stage.count(),
                highest_bin: pi.len(),
                effective_fft_size: N << n,
            });
            p.extend(pi.iter().rev().map(|pi| pi * g));
            n += stage.hbf.depth();
        }
        // correct DC and Nyquist bins as both only contribute once to the one-sided spectrum
        // this matches matplotlib and matlab but is certainly a questionable step
        // need special care when interpreting and integrating the PSD: DC and nyquist bins
        // must be counted as only half the width as the "usual" bins 0 < i < N/2
        if let Some(p) = p.first_mut() {
            *p *= 0.5;
        }
        if let Some(p) = p.last_mut() {
            *p *= 0.5;
        }
        (p, b)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// 44 insns per input sample: > 100 MS/s per core
    #[test]
    #[ignore]
    fn insn() {
        let mut s = PsdCascade::<{ 1 << 9 }>::default();
        s.set_stage_depth(3);
        s.set_detrend(Detrend::Mid);
        let x: Vec<_> = (0..1 << 16)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();
        for _ in 0..1 << 11 {
            s.process(&x);
        }
    }

    /// full accuracy tests
    #[test]
    fn exact() {
        const N: usize = 4;
        let mut s = Psd::<N>::new(
            FftPlanner::new().plan_fft_forward(N),
            Arc::new(Window::rectangular()),
        );
        let x = vec![1.0; N];
        let mut y = vec![0.0; N];
        let y = s.process(&x, &mut y[..]);
        assert_eq!(y, &x[..N / 2]);
        println!("{:?}, {}", s.spectrum(), s.gain());

        let mut s = PsdCascade::<N>::default();
        s.set_window(Window::hann());
        s.process(&x);
        let (p, b) = s.psd(0);
        let f = Break::frequencies(&b);
        println!("{:?}, {:?}", p, f);
        assert!(p
            .iter()
            .zip([0.0, 4.0 / 3.0, 8.0 / 3.0].iter())
            .all(|(p, p0)| (p - p0).abs() < 1e-7));
        assert!(f
            .iter()
            .zip([0.5, 0.25, 0.0].iter())
            .all(|(p, p0)| (p - p0).abs() < 1e-7));
    }

    #[test]
    fn test() {
        assert_eq!(idsp::hbf::HBF_PASSBAND, 0.4);

        // make uniform noise [-1, 1), ignore the epsilon.
        let x: Vec<_> = (0..1 << 16)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();
        let xm = x.iter().map(|x| *x as f64).sum::<f64>() as f32 / x.len() as f32;
        // mean is 0, take 10 sigma here and elsewhere
        assert!(xm.abs() < 10.0 / (x.len() as f32).sqrt());
        let xv = x.iter().map(|x| (x * x) as f64).sum::<f64>() as f32 / x.len() as f32;
        // variance is 1/3
        assert!((xv * 3.0 - 1.0).abs() < 10.0 / (x.len() as f32).sqrt());

        const N: usize = 1 << 9;
        let n = 3;
        let mut s = Psd::<N>::new(
            FftPlanner::new().plan_fft_forward(N),
            Arc::new(Window::hann()),
        );
        s.set_stage_depth(n);
        let mut y = vec![0.0; x.len() >> n];
        let y = s.process(&x, &mut y[..]);

        let mut hbf = HbfDecCascade::default();
        hbf.set_depth(n);
        assert_eq!(y.len(), ((x.len() - N / 2) >> n) - hbf.response_length());
        let p: Vec<_> = s.spectrum().iter().map(|p| p * s.gain()).collect();
        // psd of a stage
        assert!(
            p.iter()
                // 0.5 for one-sided spectrum
                .all(|p| (p * 0.5 * 3.0 - 1.0).abs() < 10.0 / (s.count() as f32).sqrt()),
            "{:?}",
            &p[..]
        );

        let mut d = PsdCascade::<N>::default();
        d.set_stage_depth(n);
        d.set_detrend(Detrend::None);
        d.process(&x);
        let (mut p, b) = d.psd(1);
        // tweak DC and Nyquist to make checks less code
        let n = p.len();
        p[0] *= 2.0;
        p[n - 1] *= 2.0;
        for (i, bi) in b.iter().enumerate() {
            // let (start, count, high, size) = bi.into();
            let end = b.get(i + 1).map(|bi| bi.start).unwrap_or(n);
            let pi = &p[bi.start..end];
            // psd of the cascade
            assert!(pi
                .iter()
                // 0.5 for one-sided spectrum
                .all(|p| (p * 0.5 * 3.0 - 1.0).abs() < 10.0 / (bi.count as f32).sqrt()));
        }
    }
}

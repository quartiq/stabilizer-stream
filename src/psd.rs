use idsp::hbf::{Filter, HbfDecCascade, HBF_PASSBAND};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::{ops::Range, sync::Arc};

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
    /// Optimal overlap
    pub overlap: usize,
}

impl<const N: usize> Window<N> {
    /// Rectangular window
    pub fn rectangular() -> Self {
        assert!(N > 0);
        Self {
            win: [1.0; N],
            power: 1.0,
            nenbw: 1.0,
            overlap: 0,
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
            overlap: N / 2,
        }
    }
}

/// Detrend method
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum)]
pub enum Detrend {
    /// No detrending
    #[default]
    None,
    /// Subtract the midpoint of each segment
    Midpoint,
    /// Remove linear interpolation between first and last item for each segment
    Span,
    /// Remove the mean of the segment
    Mean,
    /// Remove the linear regression of each segment
    Linear,
}

impl core::fmt::Display for Detrend {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        core::fmt::Debug::fmt(self, f)
    }
}

impl Detrend {
    pub fn apply<const N: usize>(&self, x: &[f32; N], win: &Window<N>) -> [Complex<f32>; N] {
        // apply detrending, window, make complex
        let mut c = [Complex::default(); N];

        match self {
            Detrend::None => {
                for ((c, x), w) in c.iter_mut().zip(x.iter()).zip(win.win.iter()) {
                    c.re = x * w;
                    c.im = 0.0;
                }
            }
            Detrend::Midpoint => {
                let offset = x[N / 2];
                for ((c, x), w) in c.iter_mut().zip(x.iter()).zip(win.win.iter()) {
                    c.re = (x - offset) * w;
                    c.im = 0.0;
                }
            }
            Detrend::Span => {
                let mut offset = x[0];
                let slope = (x[N - 1] - x[0]) / (N - 1) as f32;
                for ((c, x), w) in c.iter_mut().zip(x.iter()).zip(win.win.iter()) {
                    c.re = (x - offset) * w;
                    c.im = 0.0;
                    offset += slope;
                }
            }
            Detrend::Mean => {
                let offset = x.iter().sum::<f32>() / N as f32;
                for ((c, x), w) in c.iter_mut().zip(x.iter()).zip(win.win.iter()) {
                    c.re = (x - offset) * w;
                    c.im = 0.0;
                }
            }
            Detrend::Linear => unimplemented!(),
        };
        c
    }
}

/// Power spectral density accumulator and decimator
///
/// One stage in [PsdCascade].
#[derive(Clone)]
pub struct Psd<const N: usize> {
    hbf: HbfDecCascade,
    buf: [f32; N],
    idx: usize,
    spectrum: [f32; N], // using only the positive half N/2 + 1
    count: u32,
    drain: usize,
    fft: Arc<dyn Fft<f32>>,
    win: Arc<Window<N>>,
    detrend: Detrend,
    avg: u32,
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
            detrend: Detrend::default(),
            drain: 0,
            avg: u32::MAX,
        };
        s.set_stage_depth(0);
        s
    }

    pub fn set_avg(&mut self, avg: u32) {
        self.avg = avg;
    }

    pub fn set_detrend(&mut self, d: Detrend) {
        self.detrend = d;
    }

    pub fn set_stage_depth(&mut self, n: usize) {
        self.hbf.set_depth(n);
        self.drain = self.hbf.response_length() as _;
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
    /// Note: When feeding more than ~N*1e6 items expect loss of accuracy
    /// due to rounding errors on accumulation.
    ///
    /// Note: Also be aware of the usual accuracy limitation of the item data type
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
    fn count(&self) -> u32;
    /// Currently buffered input items
    fn buf(&self) -> &[f32];
}

impl<const N: usize> PsdStage for Psd<N> {
    fn process<'a>(&mut self, mut x: &[f32], y: &'a mut [f32]) -> &'a mut [f32] {
        let mut n = 0;
        // TODO: this could be made faster with less copying for internal segments of x
        while !x.is_empty() {
            // load
            let take = x.len().min(self.buf.len() - self.idx);
            let chunk;
            (chunk, x) = x.split_at(take);
            self.buf[self.idx..][..take].copy_from_slice(chunk);
            self.idx += take;
            if self.idx < N {
                break;
            }

            // detrend and window
            let mut c = self.detrend.apply(&self.buf, &self.win);
            // fft in-place
            self.fft.process(&mut c);

            let is_first = self.count == 0;

            // normalize and keep for EWMA
            let g = if self.count > self.avg {
                let g = self.avg as f32 / self.count as f32;
                self.count = self.avg;
                g
            } else {
                1.0
            };
            self.count += 1;

            // convert positive frequency spectrum to power and accumulate
            for (c, p) in c[..N / 2 + 1]
                .iter()
                .zip(self.spectrum[..N / 2 + 1].iter_mut())
            {
                *p = g * *p + c.norm_sqr();
            }

            let start = if is_first {
                // decimate entire segment into lower half, keep overlap later
                0
            } else {
                // keep overlap
                self.buf.copy_within(N - self.win.overlap..N, 0);
                // decimate only new items into third quarter
                self.win.overlap
            };

            // decimate
            let mut yi = self.hbf.process_block(None, &mut self.buf[start..]);
            // drain decimator impulse response to initial state (zeros)
            let skip = self.drain.min(yi.len());
            self.drain -= skip;
            yi = &mut yi[skip..];
            // yield l
            y[n..][..yi.len()].copy_from_slice(yi);
            n += yi.len();

            if is_first {
                // keep overlap after decimating entire segment
                self.buf.copy_within(N - self.win.overlap..N, 0);
            }
            self.idx = self.win.overlap;
        }
        &mut y[..n]
    }

    fn spectrum(&self) -> &[f32] {
        &self.spectrum[..N / 2 + 1]
    }

    fn count(&self) -> u32 {
        self.count
    }

    fn gain(&self) -> f32 {
        // 2 for one-sided
        // overlap is compensated by counting
        (N as u32 / 2 * self.count) as f32 * self.win.nenbw * self.win.power
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
    /// Was included in output
    pub include: bool,
    /// Number of averages
    pub count: u32,
    /// Averaging limit
    pub avg: u32,
    /// FFT bin
    pub bins: (usize, usize),
    /// FFT size
    pub fft_size: usize,
    /// The decimation power of two
    pub decimation: usize,
    /// Unprocessed number of input samples (includes overlap)
    pub pending: usize,
    /// Total number of samples processed (excluding overlap, ignoring averaging)
    pub processed: usize,
}

impl Break {
    /// Compute PSD bin center frequencies from stage breaks.
    pub fn frequencies(b: &[Self]) -> Vec<f32> {
        let f: Vec<_> = b
            .iter()
            .filter(|bi| bi.include)
            .flat_map(|bi| {
                let rbw = bi.rbw();
                bi.bins().map(move |f| f as f32 * rbw)
            })
            .collect();
        debug_assert_eq!(f.first(), Some(&0.0));
        debug_assert_eq!(f.last(), Some(&0.5));
        f
    }

    pub fn effective_fft_size(&self) -> usize {
        self.fft_size << self.decimation
    }

    /// Absolute resolution bandwidth
    pub fn rbw(&self) -> f32 {
        1.0 / self.effective_fft_size() as f32
    }

    pub fn bins(&self) -> Range<usize> {
        self.bins.0..self.bins.1
    }
}

/// PSD segment merge options
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MergeOpts {
    /// Remove low resolution bins
    pub remove_overlap: bool,
    /// Minimum averaging level
    pub min_count: u32,
    /// Remove decimation filter transition bands
    pub remove_transition_band: bool,
}

impl Default for MergeOpts {
    fn default() -> Self {
        Self {
            remove_overlap: true,
            min_count: 1,
            remove_transition_band: true,
        }
    }
}

/// Averaging options
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AvgOpts {
    /// Averaring limit
    pub limit: u32,
    /// Averaging
    pub count: u32,
}

impl Default for AvgOpts {
    fn default() -> Self {
        Self {
            limit: u32::MAX,
            count: u32::MAX,
        }
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
/// from higher frequency stages immediately to get rid of the delay in
/// recording and computing large FFTs. The effective full FFT size grows in real-time,
/// is unlimited, and does not need to be fixed.
/// This is well defined with the caveat that spur power (bin power not dominated by noise)
/// depends on the stage-dependent bin width.
/// This also typically what some modern signal analyzers or noise metrology instruments do.
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
    stage_depth: usize,
    detrend: Detrend,
    win: Arc<Window<N>>,
    avg: AvgOpts,
}

impl<const N: usize> PsdCascade<N> {
    /// Create a new Psd instance
    ///
    /// fft_size: size of the FFT blocks and the window
    /// stage_length: number of decimation stages. rate change per stage is 1 << stage_length
    /// detrend: [Detrend] method
    pub fn new(stage_depth: usize) -> Self {
        assert!(stage_depth > 0);
        Self {
            stages: Vec::with_capacity(4),
            fft: FftPlanner::new().plan_fft_forward(N),
            stage_depth,
            detrend: Detrend::None,
            win: Arc::new(Window::hann()),
            avg: AvgOpts::default(),
        }
    }

    /// Resolution bandwidth (relative)
    pub fn rbw(&self) -> f32 {
        (1 << self.stage_depth) as f32 / (N as f32 * HBF_PASSBAND)
    }

    pub fn set_avg(&mut self, avg: AvgOpts) {
        self.avg = avg;
        for (i, stage) in self.stages.iter_mut().enumerate() {
            stage.set_avg((self.avg.count >> (self.stage_depth * i)).min(self.avg.limit));
        }
    }

    pub fn set_detrend(&mut self, d: Detrend) {
        self.detrend = d;
        for stage in self.stages.iter_mut() {
            stage.set_detrend(self.detrend);
        }
    }

    fn get_or_add(&mut self, i: usize) -> &mut Psd<N> {
        while i >= self.stages.len() {
            let mut stage = Psd::new(self.fft.clone(), self.win.clone());
            stage.set_stage_depth(self.stage_depth);
            stage.set_detrend(self.detrend);
            stage.set_avg((self.avg.count >> (self.stage_depth * i)).min(self.avg.limit));
            self.stages.push(stage);
        }
        &mut self.stages[i]
    }

    /// Process input items
    pub fn process(&mut self, x: &[f32]) {
        let mut a = ([0f32; N], [0f32; N]);
        let (mut y, mut z) = (&mut a.0, &mut a.1);
        for mut x in x.chunks(N << self.stage_depth) {
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
    /// * `min_count`: minimum number of averages to include in output, if zero, also return
    ///   bins that would otherwise be masked by lower stage bins.
    ///
    /// # Returns
    /// * `psd`: `Vec` normalized reversed (DC first, Nyquist last)
    /// * `breaks`: `Vec` of stage breaks
    pub fn psd(&self, opts: &MergeOpts) -> (Vec<f32>, Vec<Break>) {
        let mut p = Vec::with_capacity(self.stages.len() * (N / 2 + 1));
        let mut b = Vec::with_capacity(self.stages.len());
        let mut decimation = self
            .stages
            .iter()
            .rev()
            .map(|stage| stage.hbf.depth())
            .sum();
        let mut end = 0;
        for stage in self.stages.iter().rev() {
            decimation -= stage.hbf.depth();
            // a stage yields frequency bins 0..N/2 from DC up to its nyquist
            // 0..floor(0.4*N) is its passband if it was preceeded by a decimator
            // 0..floor(0.4*N)/R is the passband of the next lower stage
            // hence take bins ceil(floor(0.4*N)/R)..floor(0.4*N) from a non-edge stage
            let start = if opts.remove_overlap {
                // remove low f bins, ceil
                (end + (1 << stage.hbf.depth()) - 1) >> stage.hbf.depth()
            } else {
                0
            };
            end = if decimation > 0 && opts.remove_transition_band {
                // remove transition band of higher stage's decimator
                2 * N / 5 // 0.4, floor
            } else {
                N / 2 + 1
            };
            let include = stage.count() >= opts.min_count;
            b.push(Break {
                start: p.len(),
                count: stage.count(),
                include,
                avg: stage.avg,
                bins: (start, end),
                fft_size: N,
                decimation,
                processed: N * stage.count() as usize
                    - stage.win.overlap * stage.count().saturating_sub(1) as usize,
                pending: stage.buf().len(),
            });
            if include {
                let g = (1 << decimation) as f32 / stage.gain();
                p.extend(stage.spectrum()[start..end].iter().map(|pi| pi * g));
            } else {
                end = start;
            }
        }
        // Do not "correct" DC and Nyquist bins.
        // Common psd algorithms argue that as both only contribute once to the one-sided
        // spectrum, they should be scaled by 0.5.
        // This would match matplotlib and matlab but is a highly questionable step usually done to
        // satisfy a oversimplified Parseval check.
        // The DC and Nyquist bins must not be scaled by 0.5, simply because modulation with
        // a frequency that is not exactly DC or Nyquist
        // but still contributes to those bins would be counted wrong. This is always the case
        // for noise (not spurs). In turn take care when doing Parseval checks or simply
        // use trapezoidal integration.
        // See also Heinzel, Rüdiger, Shilling:
        // "Spectrum and spectral density estimation by the Discrete Fourier transform (DFT),
        // including a comprehensive list of window functions and some new flat-top windows.";
        // 2002
        // if let Some(p) = p.first_mut() {
        //     *p *= 0.5;
        // }
        // if let Some(p) = p.last_mut() {
        //    *p *= 0.5;
        // }
        (p, b)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// 36 insns per item: > 200 MS/s per skylake core
    #[test]
    #[ignore]
    fn insn() {
        let mut s = PsdCascade::<{ 1 << 9 }>::new(3);
        let x: Vec<_> = (0..1 << 16).map(|_| rand::random::<f32>() - 0.5).collect();
        for _ in 0..(1 << 12) {
            // + 293
            s.process(&x);
        }
    }

    /// 36 insns per item: > 200 MS/s per skylake core
    #[test]
    #[ignore]
    fn puff() {
        let server_addr = format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
        let _puffin_server = puffin_http::Server::new(&server_addr).unwrap();
        eprintln!("Run this to view profiling data:  puffin_viewer {server_addr}");
        puffin::set_scopes_on(true);

        let mut s = PsdCascade::<{ 1 << 9 }>::new(3);
        let x: Vec<_> = (0..1 << 16).map(|_| rand::random::<f32>() - 0.5).collect();
        let mut f = || {
            puffin::profile_function!();
            // for _ in 0..(1 << 12) {
            // + 293
            s.process(&x);
        };
        f();
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
        let y = s.process(&x, &mut y);
        assert_eq!(y, &x[..N]);
        println!("{:?}, {}", s.spectrum(), s.gain());

        let mut s = PsdCascade::<N>::new(1);
        s.process(&x);
        let merge_opts = MergeOpts {
            remove_overlap: false,
            min_count: 0,
            remove_transition_band: true,
        };
        let (p, b) = s.psd(&merge_opts);
        let f = Break::frequencies(&b);
        println!("{:?}, {:?}", p, f);
        assert!(p
            .iter()
            .zip([16.0 / 3.0, 4.0 / 3.0, 0.0].iter())
            .all(|(p, p0)| (p - p0).abs() < 1e-7));
        assert!(f
            .iter()
            .zip([0.0, 0.25, 0.5].iter())
            .all(|(p, p0)| (p - p0).abs() < 1e-7));
    }

    #[test]
    fn test() {
        assert_eq!(idsp::hbf::HBF_PASSBAND, 0.4);

        // make uniform noise, with zero mean and rms = 1 ignore the epsilon.
        let x: Vec<_> = (0..1 << 16)
            .map(|_| (rand::random::<f32>() - 0.5) * 12f32.sqrt())
            .collect();
        let xm = x.iter().map(|x| *x as f64).sum::<f64>() as f32 / x.len() as f32;
        // mean is 0, take 10 sigma here and elsewhere
        assert!(xm.abs() < 10.0 / (x.len() as f32).sqrt());
        let xv = x.iter().map(|x| (x * x) as f64).sum::<f64>() as f32 / x.len() as f32;
        // variance is 1
        assert!((xv - 1.0).abs() < 10.0 / (x.len() as f32).sqrt());

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
        assert_eq!(y.len(), (x.len() >> n) - hbf.response_length());
        let g = 1.0 / s.gain();
        let p: Vec<_> = s.spectrum().iter().map(|p| p * g).collect();
        // psd of a stage
        assert!(
            p.iter()
                // 0.5 for one-sided spectrum
                .all(|p| (p * 0.5 - 1.0).abs() < 10.0 / (s.count() as f32).sqrt()),
            "{:?}",
            &p[..]
        );

        let mut d = PsdCascade::<N>::new(n);
        d.process(&x);
        let (p, b) = d.psd(&MergeOpts::default());
        for b in b.iter() {
            // psd of the stage
            assert!(p[b.start..b.start + b.bins.1 - b.bins.0]
                .iter()
                // 0.5 for one-sided spectrum
                .all(|p| (p * 0.5 - 1.0).abs() < 10.0 / (b.count as f32).sqrt()));
        }
    }
}

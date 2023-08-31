use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

use crate::{Filter, HbfDecCascade};

/// Window kernel
#[allow(clippy::len_without_is_empty)]
pub trait Window {
    fn len(&self) -> usize;
    fn get(&self) -> &[f32];
    /// Normalized effective noise bandwidth (in bins)
    fn nenbw(&self) -> f32;
    fn power(&self) -> f32;
}

/// Hann window
pub struct Hann {
    win: Vec<f32>,
}

impl Hann {
    pub fn new(len: usize) -> Self {
        assert!(len > 0);
        let df = core::f32::consts::PI / len as f32;
        Self {
            win: Vec::from_iter((0..len).map(|i| (df * i as f32).sin().powi(2))),
        }
    }
}

impl Window for Hann {
    fn get(&self) -> &[f32] {
        &self.win
    }
    fn power(&self) -> f32 {
        4.0
    }
    fn nenbw(&self) -> f32 {
        1.5
    }
    fn len(&self) -> usize {
        self.win.len()
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
pub struct Psd {
    hbf: HbfDecCascade,
    buf: Vec<f32>,
    psd: Vec<f32>,
    count: usize,
    fft: Arc<dyn Fft<f32>>,
    win: Arc<dyn Window>,
}

impl Psd {
    pub fn new(fft: Arc<dyn Fft<f32>>, win: Arc<dyn Window>, stage_length: usize) -> Self {
        let mut hbf = HbfDecCascade::default();
        hbf.set_n(stage_length);
        // check fft and decimation block size compatibility
        assert!(hbf.block_size().0 <= fft.len() / 2);
        assert!(hbf.block_size().1 >= fft.len() / 2);
        Self {
            hbf,
            buf: vec![],
            psd: vec![],
            count: 0,
            fft,
            win,
        }
    }

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
    pub fn process(&mut self, x: &[f32], detrend: Detrend) -> Vec<f32> {
        self.buf.extend_from_slice(x);
        let n = self.fft.len() / 2;

        // fft with n/2 overlap
        let m = self
            .buf
            .windows(2 * n)
            .step_by(n)
            .map(|block| {
                let (mean, slope) = match detrend {
                    Detrend::None => (0.0, 0.0),
                    Detrend::Mean => (0.5 * (block[0] + block[2 * n - 1]), 0.0),
                    Detrend::Linear => {
                        (block[0], (block[2 * n - 1] - block[0]) / (2 * n - 1) as f32)
                    }
                };
                // apply detrending, window, make complex
                let mut p: Vec<_> = block
                    .iter()
                    .zip(self.win.get().iter())
                    .enumerate()
                    .map(|(i, (x, w))| Complex::new((x - mean - i as f32 * slope) * w, 0.0))
                    .collect();
                // fft in-place
                self.fft.process(&mut p);
                // convert positive frequency spectrum to power
                let p = p[..n].iter().map(|y| y.norm_sqr());
                // accumulate
                if self.psd.is_empty() {
                    self.psd.extend(p);
                } else {
                    // TODO note that this looses accuracy for very large averages
                    for (psd, p) in self.psd.iter_mut().zip(p) {
                        *psd += p;
                    }
                }
            })
            .count();
        self.count += m;

        let mut y = vec![];
        // decimate chunks
        for block in self.buf[..m * n].chunks_mut(self.hbf.block_size().1) {
            assert!(block.len() >= self.hbf.block_size().0);
            let k = self.hbf.process_block(None, block);
            y.extend_from_slice(&block[..k]);
        }
        // drop the overlapped and processed chunks
        self.buf.drain(..m * n);
        y
    }

    /// PSD normalization factor
    pub fn gain(&self) -> f32 {
        // 2 for one-sided
        // 0.5 for overlap
        self.win.power() / ((self.count * self.fft.len()) as f32 * self.win.nenbw())
    }
}

/// Online PSD calculator
///
/// Infinite averaging
/// Incremental updates
/// Automatic FFT stage extension
pub struct PsdCascade {
    stages: Vec<Psd>,
    fft: Arc<dyn Fft<f32>>,
    win: Arc<dyn Window>,
    stage_length: usize,
    detrend: Detrend,
}

impl PsdCascade {
    /// Create a new Psd instance
    ///
    /// fft_size: size of the FFT blocks and the window
    /// stage_length: number of decimation stages. rate change per stage is 1 << stage_length
    /// detrend: [Detrend] method
    pub fn new(fft_size: usize, stage_length: usize, detrend: Detrend) -> Self {
        let fft = FftPlanner::<f32>::new().plan_fft_forward(fft_size);
        let win = Arc::new(Hann::new(fft_size));
        Self {
            stages: vec![],
            fft,
            win,
            stage_length,
            detrend,
        }
    }

    /// Process input items
    pub fn process(&mut self, x: &[f32]) {
        let mut x = x;
        let mut y: Vec<_>;
        let mut i = 0;
        while !x.is_empty() {
            if i + 1 > self.stages.len() {
                self.stages.push(Psd::new(
                    self.fft.clone(),
                    self.win.clone(),
                    self.stage_length,
                ));
            }
            y = self.stages[i].process(x, self.detrend);
            x = &y;
            i += 1;
        }
    }

    /// Return the PSD and a Vec of segement information
    ///
    /// # Args
    /// * `min_count`: minimum number of averages to include in output
    ///
    /// # Returns
    /// * `Vec` of `[end index, average count, highest bin, effective fft size]`
    /// * PSD `Vec` normalized
    pub fn get(&self, min_count: usize) -> (Vec<f32>, Vec<[usize; 4]>) {
        let mut p = vec![];
        let mut b = vec![];
        let mut n = 0;
        for stage in self.stages.iter().take_while(|s| s.count >= min_count) {
            let mut pi = &stage.psd[..];
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
            // stage start index, number of averages, highest bin freq, 1/bin width (effective fft size)
            let g = stage.gain() * (1 << n) as f32;
            b.push([p.len(), stage.count, pi.len(), f << n]);
            p.extend(pi.iter().rev().map(|pi| pi * g));
            n += stage.hbf.n();
        }
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
        let mut x: Vec<f32> = (0..1 << 20)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();
        let xm = x.iter().map(|x| *x as f64).sum::<f64>() as f32 / x.len() as f32;
        // mean is 0, take 10 sigma here and elsewhere
        assert!(xm.abs() < 10.0 / (x.len() as f32).sqrt());
        let xv = x.iter().map(|x| (x * x) as f64).sum::<f64>() as f32 / x.len() as f32;
        // variance is 1/3
        assert!((xv * 3.0 - 1.0).abs() < 10.0 / (x.len() as f32).sqrt());

        let f = 1 << 9;
        let n = 3;
        let mut s = Psd::new(
            FftPlanner::new().plan_fft_forward(f),
            Arc::new(Hann::new(f)),
            n,
        );
        let y = s.process(&mut x, Detrend::None);
        assert_eq!(y.len(), (x.len() - f / 2) >> n);
        let p: Vec<_> = s.psd.iter().map(|p| p * s.gain()).collect();
        // psd of a stage
        assert!(p
            .iter()
            .all(|p| (p * 3.0 - 1.0).abs() < 10.0 * (f as f32 / x.len() as f32).sqrt()));

        let mut d = PsdCascade::new(f, n, Detrend::None);
        d.process(&x);
        let (y, b) = d.get(1);
        for (i, bi) in b.iter().enumerate() {
            // let (start, count, high, size) = bi.into();
            let end = b.get(i + 1).map(|bi| bi[0]).unwrap_or(y.len());
            let yi = &y[bi[0]..end];
            // psd of the cascade
            assert!(yi
                .iter()
                .all(|yi| (yi * 3.0 - 1.0).abs() < 10.0 / (bi[1] as f32).sqrt()));
        }
    }
}

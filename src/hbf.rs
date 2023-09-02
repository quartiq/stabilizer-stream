/// Filter input items into output items.
pub trait Filter {
    /// Input/output item type.
    type Item;

    /// Process a block of items.
    ///
    /// Input items can be either in `x` or in `y`.
    /// In the latter case the filtering operation is done in-place.
    /// Output is always written into `y`.
    /// The number of items written into `y` is returned.
    /// Input and output size relations must match the filter requirements
    /// (decimation/interpolation and maximum block size).
    /// When using in-place operation, `y` needs to contain the input items
    /// (fewer than `y.len()` in the case of interpolation) and must be able to
    /// contain the output items.
    fn process_block(&mut self, x: Option<&[Self::Item]>, y: &mut [Self::Item]) -> usize;

    /// Return the block size granularity and the maximum block size.
    ///
    /// For in-place processing, this refers to constraints on `y`.
    /// Otherwise this refers to the larger of `x` and `y` (`x` for decimation and `y` for interpolation).
    /// The granularity is also the rate change in the case of interpolation/decimation filters.
    fn block_size(&self) -> (usize, usize);

    /// Finite impulse response length in numer of output items minus one
    /// Get this many to drain all previous memory
    fn response_length(&self) -> usize;

    // TODO: process items with automatic blocks
    // fn process(&mut self, x: Option<&[Self::Item]>, y: &mut [Self::Item]) -> usize {}
}

/// Symmetric FIR filter prototype.
///
/// DSP taps 2*M
///
/// M: number of taps
/// N: state size: N = 2*M - 1 + {input/output}.len()
///
/// Decimation/interpolation filters
///
/// These focus on half-band filters (rate change of 2) and cascades of HBF.
/// The half-band filter has unique properties that make it preferrable in many cases:
///
/// * only needs N multiplications (fused multiply accumulate) for 4*N taps
/// * stores less state compared with with a straight FIR
/// * as a FIR filter has linear phase/flat group delay
/// * very small passband ripple and excellent stopband attenuation
/// * as a cascade of decimation/interpolation filters, the higher-rate filters
///   need successively fewer taps, allowing the filtering to be dominated by
///   only the highest rate filter with the fewest taps
/// * high dynamic range (compared with a biquad IIR)
/// * can be combined with a CIC filter for non-power-of-two or even higher rate changes
///
/// The implementations here are all `no_std` and `no-alloc`.
/// They support (but don't require) in-place filtering to reduce memory usage.
/// They unroll and optimmize extremely well targetting current architectures,
/// e.g. requiring less than 4 instructions per input item for the full `HbfDecCascade` on Skylake.
/// The filters are optimized for decent block sizes and perform best (i.e. with negligible
/// overhead) for blocks of 32 high-rate items or more, depending very much on architecture.

#[derive(Clone, Debug, Copy)]
pub struct SymFir<'a, const M: usize, const N: usize> {
    x: [f32; N],
    taps: &'a [f32; M],
}

impl<'a, const M: usize, const N: usize> SymFir<'a, M, N> {
    /// taps: one-sided, expluding center tap, oldest to one-before-center
    pub fn new(taps: &'a [f32; M]) -> Self {
        debug_assert!(N >= M * 2);
        Self { x: [0.0; N], taps }
    }

    /// Perform the FIR convolution and yield results iteratively.
    #[inline]
    pub fn get(&self) -> impl Iterator<Item = f32> + '_ {
        self.x.windows(2 * M).map(|x| {
            let (old, new) = x.split_at(M);
            old.iter()
                .zip(new.iter().rev())
                .zip(self.taps.iter())
                .map(|((xo, xn), tap)| (xo + xn) * tap)
                .sum()
        })
    }
}

// TODO: pub struct SymFirInt<R>, SymFirDec<R>

/// Half band decimator (decimate by two)
///
/// The effective number of DSP taps is 4*M - 1.
///
/// M: number of taps
/// N: state size: N = 2*M - 1 + output.len()
#[derive(Clone, Debug, Copy)]
pub struct HbfDec<'a, const M: usize, const N: usize> {
    even: [f32; N], // This is an upper bound to N - M (unstable const expr)
    odd: SymFir<'a, M, N>,
}

impl<'a, const M: usize, const N: usize> HbfDec<'a, M, N> {
    /// Non-zero (odd) taps from oldest to one-before-center.
    pub fn new(taps: &'a [f32; M]) -> Self {
        Self {
            even: [0.0; N],
            odd: SymFir::new(taps),
        }
    }
}

impl<'a, const M: usize, const N: usize> Filter for HbfDec<'a, M, N> {
    type Item = f32;

    #[inline]
    fn block_size(&self) -> (usize, usize) {
        (2, 2 * (N - (2 * M - 1)))
    }

    #[inline]
    fn response_length(&self) -> usize {
        2 * M - 1
    }

    fn process_block(&mut self, x: Option<&[Self::Item]>, y: &mut [Self::Item]) -> usize {
        let x = x.unwrap_or(y);
        debug_assert_eq!(x.len() & 1, 0);
        let k = x.len() / 2;
        // load input
        for (xi, (even, odd)) in x.chunks_exact(2).zip(
            self.even[M - 1..][..k]
                .iter_mut()
                .zip(self.odd.x[2 * M - 1..][..k].iter_mut()),
        ) {
            *even = xi[0];
            *odd = xi[1];
        }
        // compute output
        for (yi, (even, odd)) in y[..k]
            .iter_mut()
            .zip(self.even[..k].iter().zip(self.odd.get()))
        {
            *yi = 0.5 * even + odd;
        }
        // keep state
        self.even.copy_within(k..k + M - 1, 0);
        self.odd.x.copy_within(k..k + 2 * M - 1, 0);
        k
    }
}

/// Half band interpolator (interpolation rate 2)
///
/// The effective number of DSP taps is 4*M - 1.
///
/// M: number of taps
/// N: state size: N = 2*M - 1 + input.len()
#[derive(Clone, Debug, Copy)]
pub struct HbfInt<'a, const M: usize, const N: usize> {
    fir: SymFir<'a, M, N>,
}

impl<'a, const M: usize, const N: usize> HbfInt<'a, M, N> {
    /// Non-zero (odd) taps from oldest to one-before-center.
    pub fn new(taps: &'a [f32; M]) -> Self {
        Self {
            fir: SymFir::new(taps),
        }
    }
}

impl<'a, const M: usize, const N: usize> Filter for HbfInt<'a, M, N> {
    type Item = f32;

    #[inline]
    fn block_size(&self) -> (usize, usize) {
        (2, 2 * (N - (2 * M - 1)))
    }

    #[inline]
    fn response_length(&self) -> usize {
        4 * M - 2
    }

    fn process_block(&mut self, x: Option<&[Self::Item]>, y: &mut [Self::Item]) -> usize {
        debug_assert_eq!(y.len() & 1, 0);
        let k = y.len() / 2;
        let x = x.unwrap_or(&y[..k]);
        // load input
        self.fir.x[2 * M - 1..][..k].copy_from_slice(x);
        // compute output
        for (yi, (even, odd)) in y
            .chunks_exact_mut(2)
            .zip(self.fir.get().zip(self.fir.x[M..][..k].iter()))
        {
            yi[0] = 2.0 * even;
            yi[1] = *odd;
        }
        // keep state
        self.fir.x.copy_within(k..k + 2 * M - 1, 0);
        y.len()
    }
}

/// Standard/optimal half-band filter cascade taps
///
/// * more than 98 dB stop band attenuation
/// * 0.4 pass band (relative to lowest sample rate)
/// * less than 0.001 dB ripple
/// * linear phase/flat group delay
/// * rate change up to 2**5 = 32
/// * lowest rate filter is at 0 index
/// * use taps 0..n for 2**n interpolation/decimation
#[allow(clippy::excessive_precision, clippy::type_complexity)]
pub const HBF_TAPS: ([f32; 15], [f32; 6], [f32; 3], [f32; 3], [f32; 2]) = (
    // 15 coefficients (effective number of DSP taps 4*15-1 = 59), transition band width .2 fs
    [
        3.51072006e-05,
        -1.21639791e-04,
        3.17513468e-04,
        -6.98912706e-04,
        1.37306791e-03,
        -2.48201920e-03,
        4.20903456e-03,
        -6.79138003e-03,
        1.05502027e-02,
        -1.59633823e-02,
        2.38512144e-02,
        -3.59007172e-02,
        5.64710020e-02,
        -1.01639797e-01,
        3.16796462e-01,
    ],
    // 6, .47
    [
        -0.00043471,
        0.00288919,
        -0.01100837,
        0.03178935,
        -0.08313839,
        0.30989656,
    ],
    // 3, .754
    [0.00707325, -0.0521982, 0.29513371],
    // 3, .877
    [0.00613987, -0.04965391, 0.29351417],
    // 2, .94
    [-0.03145898, 0.28145805],
);

/// Passband width in units of lowest sample rate
pub const HBF_PASSBAND: f32 = 0.4;

/// Max low-rate block size (HbfIntCascade input, HbfDecCascade output)
pub const HBF_CASCADE_BLOCK: usize = 1 << 6;

/// Half-band decimation filter cascade with optimal taps
///
/// See [HBF_TAPS].
/// Only in-place processing is implemented.
/// Supports rate changes of 1, 2, 4, 8, and 16.
#[derive(Copy, Clone, Debug)]
pub struct HbfDecCascade {
    n: usize,
    stages: (
        HbfDec<'static, { HBF_TAPS.0.len() }, { 2 * HBF_TAPS.0.len() - 1 + HBF_CASCADE_BLOCK }>,
        HbfDec<'static, { HBF_TAPS.1.len() }, { 2 * HBF_TAPS.1.len() - 1 + HBF_CASCADE_BLOCK * 2 }>,
        HbfDec<'static, { HBF_TAPS.2.len() }, { 2 * HBF_TAPS.2.len() - 1 + HBF_CASCADE_BLOCK * 4 }>,
        HbfDec<'static, { HBF_TAPS.3.len() }, { 2 * HBF_TAPS.3.len() - 1 + HBF_CASCADE_BLOCK * 8 }>,
    ),
}

impl Default for HbfDecCascade {
    fn default() -> Self {
        Self {
            n: 0,
            stages: (
                HbfDec::new(&HBF_TAPS.0),
                HbfDec::new(&HBF_TAPS.1),
                HbfDec::new(&HBF_TAPS.2),
                HbfDec::new(&HBF_TAPS.3),
            ),
        }
    }
}

impl HbfDecCascade {
    #[inline]
    pub fn set_n(&mut self, n: usize) {
        assert!(n <= 4);
        self.n = n;
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }
}

impl Filter for HbfDecCascade {
    type Item = f32;

    #[inline]
    fn block_size(&self) -> (usize, usize) {
        (
            1 << self.n,
            match self.n {
                0 => usize::MAX,
                1 => self.stages.0.block_size().1,
                2 => self.stages.1.block_size().1,
                3 => self.stages.2.block_size().1,
                _ => self.stages.3.block_size().1,
            },
        )
    }

    #[inline]
    fn response_length(&self) -> usize {
        let mut n = 0;
        if self.n > 3 {
            n = n / 2 + self.stages.3.response_length();
        }
        if self.n > 2 {
            n = n / 2 + self.stages.2.response_length();
        }
        if self.n > 1 {
            n = n / 2 + self.stages.1.response_length();
        }
        if self.n > 0 {
            n = n / 2 + self.stages.0.response_length();
        }
        n
    }

    fn process_block(&mut self, x: Option<&[f32]>, y: &mut [f32]) -> usize {
        if x.is_some() {
            unimplemented!(); // TODO: pair of intermediate buffers
        }

        let mut n = y.len();
        if self.n > 3 {
            n = self.stages.3.process_block(None, &mut y[..n]);
        }
        if self.n > 2 {
            n = self.stages.2.process_block(None, &mut y[..n]);
        }
        if self.n > 1 {
            n = self.stages.1.process_block(None, &mut y[..n]);
        }
        if self.n > 0 {
            n = self.stages.0.process_block(None, &mut y[..n]);
        }
        debug_assert_eq!(n, y.len() >> self.n);
        n
    }
}

/// Half-band interpolation filter cascade with optimal taps.
///
/// See [HBF_TAPS].
/// Only in-place processing is implemented.
/// Supports rate changes of 1, 2, 4, 8, and 16.
#[derive(Copy, Clone, Debug)]
pub struct HbfIntCascade {
    n: usize,
    pub stages: (
        HbfInt<'static, { HBF_TAPS.0.len() }, { 2 * HBF_TAPS.0.len() - 1 + HBF_CASCADE_BLOCK }>,
        HbfInt<'static, { HBF_TAPS.1.len() }, { 2 * HBF_TAPS.1.len() - 1 + HBF_CASCADE_BLOCK * 2 }>,
        HbfInt<'static, { HBF_TAPS.2.len() }, { 2 * HBF_TAPS.2.len() - 1 + HBF_CASCADE_BLOCK * 4 }>,
        HbfInt<'static, { HBF_TAPS.3.len() }, { 2 * HBF_TAPS.3.len() - 1 + HBF_CASCADE_BLOCK * 8 }>,
    ),
}

impl Default for HbfIntCascade {
    fn default() -> Self {
        Self {
            n: 4,
            stages: (
                HbfInt::new(&HBF_TAPS.0),
                HbfInt::new(&HBF_TAPS.1),
                HbfInt::new(&HBF_TAPS.2),
                HbfInt::new(&HBF_TAPS.3),
            ),
        }
    }
}

impl HbfIntCascade {
    pub fn set_n(&mut self, n: usize) {
        assert!(n <= 4);
        self.n = n;
    }

    pub fn n(&self) -> usize {
        self.n
    }
}

impl Filter for HbfIntCascade {
    type Item = f32;

    #[inline]
    fn block_size(&self) -> (usize, usize) {
        (
            1 << self.n,
            match self.n {
                0 => usize::MAX,
                1 => self.stages.0.block_size().1,
                2 => self.stages.1.block_size().1,
                3 => self.stages.2.block_size().1,
                _ => self.stages.3.block_size().1,
            },
        )
    }

    #[inline]
    fn response_length(&self) -> usize {
        let mut n = 0;
        if self.n > 0 {
            n = 2 * n + self.stages.0.response_length();
        }
        if self.n > 1 {
            n = 2 * n + self.stages.1.response_length();
        }
        if self.n > 2 {
            n = 2 * n + self.stages.2.response_length();
        }
        if self.n > 3 {
            n = 2 * n + self.stages.3.response_length();
        }
        n
    }

    fn process_block(&mut self, x: Option<&[f32]>, y: &mut [f32]) -> usize {
        if x.is_some() {
            unimplemented!(); // TODO: one intermediate buffer and `y`
        }

        let mut n = y.len() >> self.n;
        if self.n > 0 {
            n = self.stages.0.process_block(None, &mut y[..2 * n]);
        }
        if self.n > 1 {
            n = self.stages.1.process_block(None, &mut y[..2 * n]);
        }
        if self.n > 2 {
            n = self.stages.2.process_block(None, &mut y[..2 * n]);
        }
        if self.n > 3 {
            n = self.stages.3.process_block(None, &mut y[..2 * n]);
        }
        debug_assert_eq!(n, y.len());
        n
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rustfft::{num_complex::Complex, FftPlanner};

    #[test]
    fn test() {
        let mut h = HbfDec::<1, 5>::new(&[0.25]);
        assert_eq!(h.process_block(None, &mut []), 0);

        let mut x = [1.0; 8];
        assert_eq!((2, x.len()), h.block_size());
        let n = h.process_block(None, &mut x);
        assert_eq!(x[..n], [0.75, 1.0, 1.0, 1.0]);

        let mut h = HbfDec::<3, 9>::new(&HBF_TAPS.3);
        let mut x: Vec<_> = (0..8).map(|i| i as f32).collect();
        assert_eq!((2, x.len()), h.block_size());
        let n = h.process_block(None, &mut x);
        println!("{:?}", &x[..n]);
    }

    #[test]
    fn decim() {
        let mut h = HbfDecCascade::default();
        h.set_n(4);
        assert_eq!(h.block_size(), (1 << h.n(), HBF_CASCADE_BLOCK << h.n()));
        let mut x: Vec<_> = (0..2 << h.n()).map(|i| i as f32).collect();
        let n = h.process_block(None, &mut x);
        println!("{:?}", &x[..n]);
    }

    #[test]
    fn interp() {
        let mut h = HbfIntCascade::default();
        h.set_n(4);
        assert_eq!(h.block_size(), (1 << h.n(), HBF_CASCADE_BLOCK << h.n()));
        let k = h.block_size().0;
        let r = h.response_length();
        let mut x = vec![0.0; (r + 1 + k - 1) / k * k];
        x[0] = 1.0;
        let n = h.process_block(None, &mut x);
        println!("{:?}", &x[..n]); // interpolator impulse response
        assert!(x[r] != 0.0);
        assert_eq!(x[r + 1..], vec![0.0; x.len() - r - 1]);

        let g = (1 << h.n()) as f32;
        let mut y = Vec::from_iter(x[..n].iter().map(|&x| Complex { re: x / g, im: 0.0 }));
        // pad
        y.resize(5 << 10, Complex::default());
        FftPlanner::new().plan_fft_forward(y.len()).process(&mut y);
        // transfer function
        let p = Vec::from_iter(y.iter().map(|y| 10.0 * y.norm_sqr().log10()));
        let f = p.len() as f32 / g;
        // pass band ripple
        let p_pass = p[..(f * HBF_PASSBAND).floor() as _]
            .iter()
            .fold(0.0, |m, p| p.abs().max(m));
        assert!(p_pass < 0.00035);
        // stop band attenuation
        let p_stop = p[(f * (1.0 - HBF_PASSBAND)).ceil() as _..p.len() / 2]
            .iter()
            .fold(-200.0, |m, p| p.max(m));
        assert!(p_stop < -98.4);
    }

    /// small 32 batch size, single stage, 3 mul (11 tap) decimator
    /// 3.5 insn per input sample, > 1 GS/s on Skylake
    #[test]
    #[ignore]
    fn insn_dec() {
        const N: usize = HBF_TAPS.3.len();
        let mut h = HbfDec::<N, { 2 * N - 1 + (1 << 4) }>::new(&HBF_TAPS.3);
        let mut x = [9.0; 1 << 5];
        for _ in 0..1 << 26 {
            h.process_block(None, &mut x);
        }
    }

    /// 1k batch size, single stage, 15 mul (59 tap) decimator
    /// 5 insn per input sample, > 1 GS/s on Skylake
    #[test]
    #[ignore]
    fn insn_dec2() {
        const N: usize = HBF_TAPS.0.len();
        assert_eq!(N, 15);
        const M: usize = 1 << 10;
        let mut h = HbfDec::<N, { 2 * N - 1 + M }>::new(&HBF_TAPS.0);
        let mut x = [9.0; M];
        for _ in 0..1 << 20 {
            h.process_block(None, &mut x);
        }
    }

    /// large batch size full decimator cascade (depth 4, 1024 sampled per batch)
    /// 4.1 insns per input sample, > 1 GS/s per core
    #[test]
    #[ignore]
    fn insn_casc() {
        let mut x = [9.0; 1 << 10];
        let mut h = HbfDecCascade::default();
        h.set_n(4);
        for _ in 0..1 << 20 {
            h.process_block(None, &mut x);
        }
    }
}

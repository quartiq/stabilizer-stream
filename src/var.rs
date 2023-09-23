use core::f32::consts::PI;
use derive_builder::Builder;

#[derive(Debug, Builder, Clone, Copy, PartialEq)]
pub struct Var {
    /// exponent of `pi*f*tau` in the variance frequency response (-2 for AVAR, -4 for MVAR)
    #[builder(default = "-2")]
    x_exp: i32,
    /// Exponent of `sin(pi*f*tau)` in the variance frequency response (4 for AVAR, 6 for MVAR)
    #[builder(default = "4")]
    sinx_exp: i32,
    /// Response clip (infinite for AVAR and MVAR, 1 for the main lobe: FVAR)
    #[builder(default = "f32::MAX")]
    clip: f32,
    /// skip the first `dc_cut` bins to suppress DC window leakage
    #[builder(default = "2")]
    dc_cut: usize,
}

impl Var {
    /// Compute statistical variance estimator (AVAR, MVAR, FVAR...) from Phase PSD
    ///
    /// # Args
    /// * `phase_psd`: Phase noise PSD vector from Nyquist down
    /// * `frequencies`: PSD bin frequencies, Nyquist first
    pub fn eval(&self, phase_psd: &[f32], frequencies: &[f32], tau: f32) -> f32 {
        phase_psd
            .iter()
            .rev()
            .zip(
                frequencies
                    .iter()
                    .rev()
                    .take_while(|&f| f * tau <= self.clip),
            )
            .skip(self.dc_cut)
            // force DC bin to 0
            .fold((0.0, (0.0, 0.0)), |(accu, (a0, f0)), (&sp, &f)| {
                // frequency PSD
                let sy = sp * f * f;
                let pft = PI * (f * tau);
                // Allan variance transfer function (rectangular window: sinc**2 and differencing: 2*sin**2)
                // Cancel the 2 here with the 0.5 in the trapezoidal rule
                let hahd = pft.sin().powi(self.sinx_exp) * pft.powi(self.x_exp);
                let a = sy * hahd;
                // trapezoidal integration
                (accu + (a + a0) * (f - f0), (a, f))
            })
            .0
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn basic() {
        let mut p = [1000.0, 100.0, 1.2, 3.4, 5.6];
        let mut f = [0.0, 1.0, 3.0, 6.0, 9.0];
        p.reverse();
        f.reverse();
        let var = VarBuilder::default().build().unwrap();
        let v = var.eval(&p, &f, 2.7);
        println!("{}", v);
        assert!((0.13478442 - v).abs() < 1e-6);
    }
}

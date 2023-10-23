use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum FormatError {}

pub trait Payload {
    fn new(batches: usize, data: &[u8]) -> Result<Self, FormatError>
    where
        Self: Sized;
    fn traces(&self) -> &[Vec<f32>];
    fn traces_mut(&mut self) -> &mut [Vec<f32>];
    fn labels(&self) -> &[&str];
}

pub struct AdcDac {
    traces: [Vec<f32>; 4],
}

impl Payload for AdcDac {
    /// Extract AdcDacData from a binary data block in the stream.
    ///
    /// # Args
    /// * `batch_size` - The size of each batch in samples.
    /// * `data` - The binary data composing the stream frame.
    fn new(batches: usize, data: &[u8]) -> Result<Self, FormatError> {
        const CHANNELS: usize = 4;
        const BATCH_SIZE: usize = 8;

        // The DAC output range in bipolar mode (including the external output op-amp) is +/- 4.096
        // V with 16-bit resolution. The anti-aliasing filter has an additional gain of 2.5.
        const DAC_VOLT_PER_LSB: f32 = 4.096 * 2.5 / (1u16 << 15) as f32;
        // The ADC has a differential input with a range of +/- 4.096 V and 16-bit resolution.
        // The gain into the two inputs is 1/5.
        const ADC_VOLT_PER_LSB: f32 = 5.0 / 2.0 * 4.096 / (1u16 << 15) as f32;
        assert_eq!(DAC_VOLT_PER_LSB, ADC_VOLT_PER_LSB);

        let v = Vec::with_capacity(data.len() * BATCH_SIZE);
        let mut traces = [v.clone(), v.clone(), v.clone(), v];
        let data: &[[[[u8; 2]; BATCH_SIZE]; CHANNELS]] = bytemuck::cast_slice(data);
        assert_eq!(data.len(), batches);
        for b in data.iter() {
            traces[0].extend(
                b[0].into_iter()
                    .map(|x| i16::from_le_bytes(x) as f32 * ADC_VOLT_PER_LSB),
            );
            traces[1].extend(
                b[1].into_iter()
                    .map(|x| i16::from_le_bytes(x) as f32 * ADC_VOLT_PER_LSB),
            );
            traces[2].extend(
                b[2].into_iter().map(|x| {
                    i16::from_le_bytes(x).wrapping_add(i16::MIN) as f32 * DAC_VOLT_PER_LSB
                }),
            );
            traces[3].extend(
                b[3].into_iter().map(|x| {
                    i16::from_le_bytes(x).wrapping_add(i16::MIN) as f32 * DAC_VOLT_PER_LSB
                }),
            );
        }
        Ok(Self { traces })
    }

    fn traces(&self) -> &[Vec<f32>] {
        &self.traces
    }

    fn traces_mut(&mut self) -> &mut [Vec<f32>] {
        &mut self.traces
    }
    fn labels(&self) -> &[&str] {
        &["ADC0", "ADC1", "DAC0", "DAC1"]
    }
}

pub struct Fls {
    traces: [Vec<f32>; 4],
}

impl Payload for Fls {
    fn new(batches: usize, data: &[u8]) -> Result<Self, FormatError> {
        let data: &[[[i32; 7]; 2]] = bytemuck::cast_slice(data);
        // demod_re, demod_im, phase[2], ftw, pow_amp, pll
        assert_eq!(batches, data.len());
        let traces: [Vec<f32>; 4] = [
            data.iter()
                .map(|b| {
                    ((b[0][0] as f32).powi(2) + (b[0][1] as f32).powi(2)).sqrt() / (i32::MAX as f32)
                })
                .collect(),
            data.iter()
                .map(|b| {
                    let p: i64 = bytemuck::cast([b[0][2], b[0][3]]);
                    // FIXME: 1 >> 16 is default phase_scale[0], deal with initial phase offset and dymanic range
                    p as f32 * (core::f32::consts::TAU / (1i64 << 16) as f32)
                })
                .collect(),
            data.iter()
                .map(|b| b[1][0] as f32 / i32::MAX as f32)
                .collect(),
            data.iter()
                .map(|b| b[1][1] as f32 / i32::MAX as f32)
                .collect(),
        ];
        Ok(Self { traces })
    }

    fn labels(&self) -> &[&str] {
        &["AR", "AP", "BI", "BQ"]
    }

    fn traces(&self) -> &[Vec<f32>] {
        &self.traces
    }
    fn traces_mut(&mut self) -> &mut [Vec<f32>] {
        &mut self.traces
    }
}

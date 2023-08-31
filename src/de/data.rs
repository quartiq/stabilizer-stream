use ndarray::{ArrayView, Axis, ShapeError};
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum FormatError {
    #[error("Invalid frame payload size")]
    InvalidSize(#[from] ShapeError),
}

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
        let channels = 4;
        let samples = data.len() / batches / channels / core::mem::size_of::<i16>();
        let mut data = ArrayView::from_shape(
            (batches, channels, samples, core::mem::size_of::<i16>()),
            data,
        )?;
        data.swap_axes(0, 1); // FIXME: non-contig
        let data = data.into_shape((channels, samples * batches, core::mem::size_of::<i16>()))?;

        // The DAC output range in bipolar mode (including the external output op-amp) is +/- 4.096
        // V with 16-bit resolution. The anti-aliasing filter has an additional gain of 2.5.
        const DAC_VOLT_PER_LSB: f32 = 4.096 * 2.5 / (1u16 << 15) as f32;
        // The ADC has a differential input with a range of +/- 4.096 V and 16-bit resolution.
        // The gain into the two inputs is 1/5.
        const ADC_VOLT_PER_LSB: f32 = 5.0 / 2.0 * 4.096 / (1u16 << 15) as f32;
        assert_eq!(DAC_VOLT_PER_LSB, ADC_VOLT_PER_LSB);

        let traces: [Vec<f32>; 4] = [
            data.index_axis(Axis(0), 0)
                .axis_iter(Axis(0))
                .map(|x| {
                    i16::from_le_bytes([x[0], x[1]]).wrapping_add(i16::MIN) as f32
                        * DAC_VOLT_PER_LSB
                })
                .collect(),
            data.index_axis(Axis(0), 1)
                .axis_iter(Axis(0))
                .map(|x| {
                    i16::from_le_bytes([x[0], x[1]]).wrapping_add(i16::MIN) as f32
                        * DAC_VOLT_PER_LSB
                })
                .collect(),
            data.index_axis(Axis(0), 2)
                .axis_iter(Axis(0))
                .map(|x| i16::from_le_bytes([x[0], x[1]]) as f32 * ADC_VOLT_PER_LSB)
                .collect(),
            data.index_axis(Axis(0), 3)
                .axis_iter(Axis(0))
                .map(|x| i16::from_le_bytes([x[0], x[1]]) as f32 * ADC_VOLT_PER_LSB)
                .collect(),
        ];

        Ok(Self { traces })
    }

    fn traces(&self) -> &[Vec<f32>] {
        &self.traces[..]
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
        let data: &[[[i32; 6]; 2]] = bytemuck::cast_slice(data);
        // demod_re, demod_im, wraps, ftw, pow_amp, pll
        assert_eq!(batches, data.len());
        let traces: [Vec<f32>; 4] = [
            data.iter()
                .map(|b| b[0][0] as f32 / i32::MAX as f32)
                .collect(),
            data.iter()
                .map(|b| b[0][1] as f32 / i32::MAX as f32)
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
        &["AI", "AQ", "BI", "BQ"]
    }

    fn traces(&self) -> &[Vec<f32>] {
        &self.traces
    }
    fn traces_mut(&mut self) -> &mut [Vec<f32>] {
        &mut self.traces
    }
}

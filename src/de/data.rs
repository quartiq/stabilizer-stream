use super::Error;
use core::{f32, fmt::Debug};

pub trait Payload<'a>: Debug {
    fn new(batches: usize, data: &'a [u8]) -> Result<Self, Error>
    where
        Self: Sized;
    fn traces(&self) -> Result<Vec<(&'static str, Vec<f32>)>, Error>;
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct AdcDac<'a> {
    data: &'a [[[[u8; 2]; 8]; 4]],
}

impl<'a> Payload<'a> for AdcDac<'a> {
    /// Extract AdcDacData from a binary data block in the stream.
    ///
    /// # Args
    /// * `batch_size` - The size of each batch in samples.
    /// * `data` - The binary data composing the stream frame.
    fn new(batches: usize, data: &'a [u8]) -> Result<Self, Error> {
        let data = bytemuck::try_cast_slice(data)?;
        assert_eq!(data.len(), batches);
        Ok(Self { data })
    }

    fn traces(&self) -> Result<Vec<(&'static str, Vec<f32>)>, Error> {
        // The DAC output range in bipolar mode (including the external output op-amp) is +/- 4.096
        // V with 16-bit resolution. The anti-aliasing filter has an additional gain of 2.5.
        const DAC_VOLT_PER_LSB: f32 = 4.096 * 2.5 / (1u16 << 15) as f32;
        // The ADC has a differential input with a range of +/- 4.096 V and 16-bit resolution.
        // The gain into the two inputs is 1/5.
        const ADC_VOLT_PER_LSB: f32 = 5.0 / 2.0 * 4.096 / (1u16 << 15) as f32;
        assert_eq!(DAC_VOLT_PER_LSB, ADC_VOLT_PER_LSB);

        Ok(vec![
            (
                "ADC0",
                self.data
                    .iter()
                    .flat_map(|b| {
                        b[0].iter()
                            .map(|v| i16::from_le_bytes(*v) as f32 * ADC_VOLT_PER_LSB)
                    })
                    .collect(),
            ),
            (
                "ADC1",
                self.data
                    .iter()
                    .flat_map(|b| {
                        b[1].iter()
                            .map(|v| i16::from_le_bytes(*v) as f32 * ADC_VOLT_PER_LSB)
                    })
                    .collect(),
            ),
            (
                "DAC0",
                self.data
                    .iter()
                    .flat_map(|b| {
                        b[2].iter().map(|v| {
                            i16::from_le_bytes(*v).wrapping_add(i16::MIN) as f32 * DAC_VOLT_PER_LSB
                        })
                    })
                    .collect(),
            ),
            (
                "DAC1",
                self.data
                    .iter()
                    .flat_map(|b| {
                        b[3].iter().map(|v| {
                            i16::from_le_bytes(*v).wrapping_add(i16::MIN) as f32 * DAC_VOLT_PER_LSB
                        })
                    })
                    .collect(),
            ),
        ])
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Fls<'a> {
    data: &'a [[[[u8; 4]; 7]; 2]],
}

impl<'a> Payload<'a> for Fls<'a> {
    fn new(batches: usize, data: &'a [u8]) -> Result<Self, Error> {
        let data = bytemuck::try_cast_slice(data)?;
        // demod_re, demod_im, phase[2], ftw, pow_amp, pll
        assert_eq!(batches, data.len());
        Ok(Self { data })
    }

    fn traces(&self) -> Result<Vec<(&'static str, Vec<f32>)>, Error> {
        Ok(vec![
            (
                "AR",
                self.data
                    .iter()
                    .map(|b| {
                        ((i32::from_le_bytes(b[0][0]) as f32).powi(2)
                            + (i32::from_le_bytes(b[0][1]) as f32).powi(2))
                        .sqrt()
                            * (1.0 / (i32::MAX as f32))
                    })
                    .collect(),
            ),
            (
                "AP",
                self.data
                    .iter()
                    .map(|b| {
                        let b: &[[u8; 8]] = bytemuck::cast_slice(&b[0][2..4]);
                        // FIXME: 1 << 16 is the default phase_scale[0]
                        // TODO: deal with initial phase offset and dymanic range
                        i64::from_le_bytes(b[0]) as f32
                            * (core::f32::consts::TAU / (1i64 << 16) as f32)
                    })
                    .collect(),
            ),
            (
                "BI",
                self.data
                    .iter()
                    .map(|b| i32::from_le_bytes(b[1][0]) as f32 / i32::MAX as f32)
                    .collect(),
            ),
            (
                "BQ",
                self.data
                    .iter()
                    .map(|b| i32::from_le_bytes(b[1][1]) as f32 / i32::MAX as f32)
                    .collect(),
            ),
        ])
    }
}

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ThermostatEem<'a> {
    data: &'a [[[u8; 4]; 16 + 4]],
}

impl<'a> Payload<'a> for ThermostatEem<'a> {
    fn new(batches: usize, data: &'a [u8]) -> Result<Self, Error> {
        let data = bytemuck::try_cast_slice(data)?;
        assert_eq!(batches, data.len());
        Ok(Self { data })
    }

    fn traces(&self) -> Result<Vec<(&'static str, Vec<f32>)>, Error> {
        Ok(["T00", "T20", "I0", "I1"]
            .into_iter()
            .zip(
                [0, 8, 13, 16]
                    .into_iter()
                    .map(|i| self.data.iter().map(|b| f32::from_le_bytes(b[i])).collect()),
            )
            .collect())
    }
}

#[derive(Clone, Debug)]
pub struct Mpll<'a> {
    data: &'a [[[u8; 4]; 6]],
}

impl<'a> Payload<'a> for Mpll<'a> {
    fn new(batches: usize, data: &'a [u8]) -> Result<Self, Error> {
        let data = bytemuck::try_cast_slice(data)?;
        assert_eq!(batches, data.len());
        Ok(Self { data })
    }

    fn traces(&self) -> Result<Vec<(&'static str, Vec<f32>)>, Error> {
        Ok(vec![
            (
                "phase (rad)",
                self.data
                    .iter()
                    .map(|b| {
                        i32::from_le_bytes(b[4]) as f32 * (f32::consts::TAU / (1u64 << 32) as f32)
                    })
                    .collect(),
            ),
            (
                "frequency (kHz)",
                self.data
                    .iter()
                    .map(|b| {
                        i32::from_le_bytes(b[5]) as f32 * (1.0 / 1.28e-3 / (1u64 << 32) as f32)
                    })
                    .collect(),
            ),
            (
                "amplitude (V/G10)",
                self.data
                    .iter()
                    .map(|b| {
                        ((i32::from_le_bytes(b[0]) as f32).powi(2)
                            + (i32::from_le_bytes(b[1]) as f32).powi(2))
                        .sqrt()
                            * (10.24 / 10.0 * 2.0 * 2.0 / (1u64 << 32) as f32)
                    })
                    .collect(),
            ),
        ])
    }
}

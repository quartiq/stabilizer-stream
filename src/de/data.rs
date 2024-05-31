use super::Error;

pub trait Payload<'a> {
    fn new(batches: usize, data: &'a [u8]) -> Result<Self, Error>
    where
        Self: Sized;
    fn traces(&self) -> Result<Vec<(&'static str, Vec<f32>)>, Error>;
}

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
        const CHANNELS: usize = 4;
        const BATCH_SIZE: usize = 8;
        let data: &[[[[u8; 2]; BATCH_SIZE]; CHANNELS]] =
            bytemuck::try_cast_slice(data).map_err(Error::PayloadSize)?;
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

        let v = Vec::with_capacity(self.data.len() * 8);
        let mut traces = vec![
            ("ADC0", v.clone()),
            ("ADC1", v.clone()),
            ("DAC0", v.clone()),
            ("DAC1", v),
        ];
        for b in self.data.iter() {
            traces[0].1.extend(
                b[0].into_iter()
                    .map(|x| i16::from_le_bytes(x) as f32 * ADC_VOLT_PER_LSB),
            );
            traces[1].1.extend(
                b[1].into_iter()
                    .map(|x| i16::from_le_bytes(x) as f32 * ADC_VOLT_PER_LSB),
            );
            traces[2].1.extend(
                b[2].into_iter().map(|x| {
                    i16::from_le_bytes(x).wrapping_add(i16::MIN) as f32 * DAC_VOLT_PER_LSB
                }),
            );
            traces[3].1.extend(
                b[3].into_iter().map(|x| {
                    i16::from_le_bytes(x).wrapping_add(i16::MIN) as f32 * DAC_VOLT_PER_LSB
                }),
            );
        }
        Ok(traces)
    }
}

pub struct Fls<'a> {
    data: &'a [[[i32; 7]; 2]],
}

impl<'a> Payload<'a> for Fls<'a> {
    fn new(batches: usize, data: &'a [u8]) -> Result<Self, Error> {
        // FIXME: unportable
        let data: &[[[i32; 7]; 2]] = bytemuck::try_cast_slice(data).map_err(Error::PayloadSize)?;
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
                        ((b[0][0] as f32).powi(2) + (b[0][1] as f32).powi(2)).sqrt()
                            * (1.0 / (i32::MAX as f32))
                    })
                    .collect(),
            ),
            (
                "AP",
                self.data
                    .iter()
                    .map(|b| {
                        let p: &[i64] = bytemuck::cast_slice(&b[0][2..4]);
                        // FIXME: 1 << 16 is the default phase_scale[0]
                        // TODO: deal with initial phase offset and dymanic range
                        p[0] as f32 * (core::f32::consts::TAU / (1i64 << 16) as f32)
                    })
                    .collect(),
            ),
            (
                "BI",
                self.data
                    .iter()
                    .map(|b| b[1][0] as f32 / i32::MAX as f32)
                    .collect(),
            ),
            (
                "BQ",
                self.data
                    .iter()
                    .map(|b| b[1][1] as f32 / i32::MAX as f32)
                    .collect(),
            ),
        ])
    }
}

pub struct ThermostatEem<'a> {
    data: &'a [[f32; 16 + 4]],
}

impl<'a> Payload<'a> for ThermostatEem<'a> {
    fn new(batches: usize, data: &'a [u8]) -> Result<Self, Error> {
        let data: &[[f32; 16 + 4]] = bytemuck::try_cast_slice(data).map_err(Error::PayloadSize)?;
        assert_eq!(batches, data.len());
        Ok(Self { data })
    }

    fn traces(&self) -> Result<Vec<(&'static str, Vec<f32>)>, Error> {
        Ok(["T00", "T20", "I0", "I1"]
            .into_iter()
            .zip(
                [0, 8, 13, 16]
                    .into_iter()
                    .map(|i| self.data.iter().map(|b| b[i]).collect()),
            )
            .collect())
    }
}

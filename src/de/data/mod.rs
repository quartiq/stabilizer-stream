#[derive(Debug, Copy, Clone)]
pub enum FormatError {
    InvalidSize,
}

/// Custom type for referencing DAC output codes.
/// The internal integer is the raw code written to the DAC output register.
#[derive(Copy, Clone)]
pub struct DacCode(pub u16);

impl From<DacCode> for f32 {
    fn from(code: DacCode) -> f32 {
        // The DAC output range in bipolar mode (including the external output op-amp) is +/- 4.096
        // V with 16-bit resolution. The anti-aliasing filter has an additional gain of 2.5.
        let dac_volts_per_lsb = 4.096 * 2.5 / (1u16 << 15) as f32;

        (code.0 as i16).wrapping_add(i16::MIN) as f32 * dac_volts_per_lsb
    }
}

/// A type representing an ADC sample.
#[derive(Copy, Clone)]
pub struct AdcCode(pub u16);

impl From<i16> for AdcCode {
    /// Construct an ADC code from the stabilizer-defined code (i16 full range).
    fn from(value: i16) -> Self {
        Self(value as u16)
    }
}

impl From<AdcCode> for i16 {
    /// Get a stabilizer-defined code from the ADC code.
    fn from(code: AdcCode) -> i16 {
        code.0 as i16
    }
}

impl From<AdcCode> for f32 {
    /// Convert raw ADC codes to/from voltage levels.
    ///
    /// # Note
    /// This does not account for the programmable gain amplifier at the signal input.
    fn from(code: AdcCode) -> f32 {
        // The ADC has a differential input with a range of +/- 4.096 V and 16-bit resolution.
        // The gain into the two inputs is 1/5.
        let adc_volts_per_lsb = 5.0 / 2.0 * 4.096 / (1u16 << 15) as f32;

        i16::from(code) as f32 * adc_volts_per_lsb
    }
}

pub trait FrameData {
    fn trace_count(&self) -> usize;
    fn get_trace(&self, index: usize) -> &Vec<f32>;
    fn samples_per_batch(&self) -> usize;
    fn batch_count(&self) -> usize {
        self.get_trace(0).len() / self.samples_per_batch()
    }
    fn trace_label(&self, index: usize) -> String {
        format!("{}", index)
    }
}

pub struct AdcDacData {
    traces: [Vec<f32>; 4],
    batch_size: usize,
}

impl FrameData for AdcDacData {
    fn trace_count(&self) -> usize {
        self.traces.len()
    }

    fn get_trace(&self, index: usize) -> &Vec<f32> {
        &self.traces[index]
    }

    fn samples_per_batch(&self) -> usize {
        // Each element of the batch is 4 samples, each of which are u16s.
        self.batch_size
    }

    fn trace_label(&self, index: usize) -> String {
        match index {
            0 => "ADC0".to_string(),
            1 => "ADC1".to_string(),
            2 => "DAC0".to_string(),
            3 => "DAC1".to_string(),
            _ => panic!("Invalid trace"),
        }
    }
}

impl AdcDacData {
    /// Extract AdcDacData from a binary data block in the stream.
    ///
    /// # Args
    /// * `batch_size` - The size of each batch in samples.
    /// * `data` - The binary data composing the stream frame.
    pub fn new(batch_size: usize, data: &[u8]) -> Result<AdcDacData, FormatError> {
        let batch_size_bytes: usize = batch_size * 8;
        let num_batches = data.len() / batch_size_bytes;
        if num_batches * batch_size_bytes != data.len() {
            return Err(FormatError::InvalidSize);
        }

        let mut traces: [Vec<f32>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        for batch in 0..num_batches {
            let batch_index = batch * batch_size_bytes;

            // Batches are serialized as <ADC0><ADC1><DAC0><DAC1>, where the number of samples in
            // `<ADC/DAC[0,1]> is equal to that batch_size.
            for (i, trace) in traces.iter_mut().enumerate() {
                let trace_index = batch_index + 2 * batch_size * i;

                for sample in 0..batch_size {
                    let sample_index = trace_index + sample * 2;

                    let value = {
                        let code = u16::from_le_bytes([data[sample_index], data[sample_index + 1]]);
                        if i < 2 {
                            f32::from(AdcCode(code))
                        } else {
                            f32::from(DacCode(code))
                        }
                    };

                    trace.push(value);
                }
            }
        }

        Ok(Self { batch_size, traces })
    }
}

#[derive(Debug, Copy, Clone)]
pub enum FormatError {
    InvalidSize,
}

pub trait FrameData {
    fn trace_count(&self) -> usize;
    fn get_trace(&self, index: usize) -> &Vec<f32>;
    fn samples_per_batch(&self) -> usize;
    fn batch_count(&self) -> usize {
        self.get_trace(0).len() / self.samples_per_batch()
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

            // Deserialize the batch
            for sample in 0..batch_size {
                let sample_index = batch_index + sample * 8;
                for (i, trace) in traces.iter_mut().enumerate() {
                    let trace_index = sample_index + i * 2;
                    let value = {
                        let code = u16::from_le_bytes([data[trace_index], data[trace_index + 1]]);
                        // TODO: Convert code from u16 to floating point voltage.
                        code as f32
                    };

                    trace.push(value);
                }
            }
        }

        Ok(Self { batch_size, traces })
    }
}

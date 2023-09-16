use crate::Frame;

#[derive(Clone, Copy, Default)]
pub struct Loss {
    received: u64,
    dropped: u64,
    seq: Option<u32>,
}

impl Loss {
    pub fn update(&mut self, frame: &Frame) {
        self.received += frame.header.batches as u64;
        if let Some(seq) = self.seq {
            let missing = frame.header.seq.wrapping_sub(seq) as u64;
            self.dropped += missing;
            if missing > 0 {
                log::warn!(
                    "Lost {} batches: {:#08X} -> {:#08X}",
                    missing,
                    seq,
                    frame.header.seq,
                );
            }
        }
        self.seq = Some(frame.header.seq.wrapping_add(frame.header.batches as _));
    }

    pub fn analyze(&self) {
        if self.received > 0 {
            let loss = self.dropped as f32 / (self.received + self.dropped) as f32;
            log::info!(
                "Loss: {} % ({} of {})",
                loss * 100.0,
                self.dropped,
                self.received + self.dropped
            );
        }
    }
}

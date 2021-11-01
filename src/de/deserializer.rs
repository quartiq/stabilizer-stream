use super::data::{AdcDacData, FrameData};
use super::{Error, StreamFormat};

use std::convert::TryFrom;

// The magic word at the start of each stream frame.
const MAGIC_WORD: u16 = 0x057B;

// The size of the frame header in bytes.
const HEADER_SIZE: usize = 8;

/// A single stream frame contains multiple batches of data.
pub struct StreamFrame {
    header: FrameHeader,
    pub data: Box<dyn FrameData + Send>,
}

struct FrameHeader {
    // The format code associated with the stream binary data.
    pub format_code: StreamFormat,

    // The size of each batch contained within the binary data.
    pub batch_size: u8,

    // The sequence number of the first batch in the binary data. The sequence number increments
    // monotonically for each batch. All batches the binary data are sequential.
    pub sequence_number: u32,
}

impl FrameHeader {
    /// Parse the header of a stream frame.
    pub fn parse(header: &[u8]) -> Result<Self, Error> {
        assert_eq!(header.len(), HEADER_SIZE);

        let magic_word = u16::from_le_bytes([header[0], header[1]]);

        if magic_word != MAGIC_WORD {
            return Err(Error::InvalidHeader);
        }

        let format_code = StreamFormat::try_from(header[2]).map_err(|_| Error::UnknownFormat)?;
        let batch_size = header[3];
        let sequence_number = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        log::debug!(
            "Header: {:?}, {}, {:X}",
            format_code,
            batch_size,
            sequence_number
        );

        Ok(Self {
            format_code,
            batch_size,
            sequence_number,
        })
    }
}

impl StreamFrame {
    /// Get the format code of the current frame.
    pub fn format(&self) -> StreamFormat {
        self.header.format_code
    }

    /// Get the sequence number of the first batch in the frame.
    pub fn sequence_number(&self) -> u32 {
        self.header.sequence_number
    }

    /// Parse a stream frame from a single UDP packet.
    pub fn from_bytes(input: &[u8]) -> Result<StreamFrame, Error> {
        let (header, data) = input.split_at(HEADER_SIZE);

        let header = FrameHeader::parse(header)?;

        let data = match header.format_code {
            StreamFormat::AdcDacData => {
                let data = AdcDacData::new(header.batch_size as usize, data)?;
                Box::new(data)
            }
        };

        Ok(StreamFrame {
            header: header,
            data,
        })
    }
}

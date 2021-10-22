use super::{AdcDacData, Error, FormatError, StreamData, StreamFormat};

use std::convert::TryFrom;

const MAGIC_WORD: u16 = 0x057B;

const HEADER_SIZE: usize = 8;

pub struct StreamFrame<'a> {
    pub sequence_number: u32,
    pub data: StreamData<'a>,
}

struct FrameHeader {
    pub format_code: StreamFormat,
    pub batch_size: u8,
    pub sequence_number: u32,
}

impl FrameHeader {
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

impl<'a> StreamFrame<'a> {
    pub fn from_bytes(input: &'a [u8]) -> Result<StreamFrame<'a>, Error> {
        let (header, data) = input.split_at(HEADER_SIZE);

        let header = FrameHeader::parse(header)?;

        if data.len() % header.batch_size as usize != 0 {
            return Err(FormatError::InvalidSize.into());
        }

        let data = match header.format_code {
            StreamFormat::AdcDacData => {
                let data = AdcDacData::new(header.batch_size, data)?;
                StreamData::AdcDacData(data)
            }
        };

        Ok(StreamFrame {
            sequence_number: header.sequence_number,
            data,
        })
    }

    pub fn batch_count(&self) -> usize {
        match &self.data {
            StreamData::AdcDacData(data) => data.batch_count(),
        }
    }
}

impl<'a> AdcDacData<'a> {
    pub fn new(batch_size: u8, data: &'a [u8]) -> Result<AdcDacData<'a>, FormatError> {
        // Each element of the batch is 4 samples, each of which are u16s.
        let batch_size_bytes: usize = (batch_size * 8) as usize;
        if data.len() % batch_size_bytes != 0 {
            return Err(FormatError::InvalidSize);
        }

        Ok(Self { batch_size, data })
    }

    fn batch_count(&self) -> usize {
        self.data.len() / (self.batch_size * 8) as usize
    }
}

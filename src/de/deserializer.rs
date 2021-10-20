use super::{AdcDacData, Error, FormatError, StreamData, StreamFormat};

use std::convert::TryFrom;

const MAGIC_WORD: u16 = 0x057B;

const HEADER_SIZE: usize = 8;

pub struct StreamFrame<'a> {
    pub sequence_number: u32,
    pub batch_size: usize,
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

        let data = match header.format_code {
            StreamFormat::AdcDacData => {
                let data = AdcDacData::new(header.batch_size as usize, data)?;
                StreamData::AdcDacData(data)
            }
        };

        Ok(StreamFrame {
            sequence_number: header.sequence_number,
            batch_size: header.batch_size as usize,
            data,
        })
    }
}

impl<'a> AdcDacData<'a> {
    pub fn new(batch_size: usize, data: &'a [u8]) -> Result<AdcDacData<'a>, FormatError> {
        // Each batch is composed of 4 samples, each a u16.
        if batch_size * 8 != data.len() {
            return Err(FormatError::InvalidSize);
        }

        Ok(Self { data })
    }
}

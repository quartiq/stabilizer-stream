use super::data::{self, Payload};
use super::{Error, Format};

// The magic word at the start of each stream frame.
const MAGIC_WORD: [u8; 2] = [0x7b, 0x05];

// The size of the frame header in bytes.
const HEADER_SIZE: usize = 8;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Header {
    // The format code associated with the stream binary data.
    pub format: Format,

    // The number of batches in the payload.
    pub batches: u8,

    // The sequence number of the first batch in the binary data. The sequence number increments
    // monotonically for each batch. All batches the binary data are sequential.
    pub seq: u32,
}

impl Header {
    /// Parse the header of a stream frame.
    fn parse(header: &[u8; HEADER_SIZE]) -> Result<Self, Error> {
        if header[..2] != MAGIC_WORD {
            return Err(Error::InvalidHeader);
        }
        let format = Format::try_from(header[2]).or(Err(Error::UnknownFormat))?;
        let batches = header[3];
        let seq = u32::from_le_bytes(header[4..8].try_into().unwrap());
        Ok(Self {
            format,
            batches,
            seq,
        })
    }
}

/// A single stream frame contains multiple batches of data.
#[derive(Debug)]
pub struct Frame<'a> {
    pub header: Header,
    pub payload: Box<dyn Payload<'a> + 'a>,
}

impl<'a> Frame<'a> {
    /// Parse a stream frame from a single UDP packet.
    pub fn from_bytes(input: &'a [u8]) -> Result<Self, Error> {
        let header = Header::parse(&input[..HEADER_SIZE].try_into().unwrap())?;
        let data = &input[HEADER_SIZE..];
        let batches = header.batches as _;
        let payload: Box<dyn Payload> = match header.format {
            Format::AdcDac => Box::new(data::AdcDac::new(batches, data)?),
            Format::Fls => Box::new(data::Fls::new(batches, data)?),
            Format::ThermostatEem => Box::new(data::ThermostatEem::new(batches, data)?),
            Format::Mpll => Box::new(data::Mpll::new(header.batches as _, data)?),
        };
        Ok(Self { header, payload })
    }
}

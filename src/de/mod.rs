use num_enum::TryFromPrimitive;

pub mod deserializer;

pub struct AdcDacData<'a> {
    data: &'a [u8],
}

pub enum StreamData<'a> {
    AdcDacData(AdcDacData<'a>),
}

#[derive(TryFromPrimitive)]
#[repr(u8)]
enum StreamFormat {
    AdcDacData = 1,
}

#[derive(Debug, Copy, Clone)]
pub enum Error {
    DataFormat(FormatError),
    InvalidHeader,
    UnknownFormat,
}

#[derive(Debug, Copy, Clone)]
pub enum FormatError {
    InvalidSize,
}

impl From<FormatError> for Error {
    fn from(e: FormatError) -> Error {
        Error::DataFormat(e)
    }
}

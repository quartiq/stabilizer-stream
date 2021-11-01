use num_enum::TryFromPrimitive;

pub mod data;
pub mod deserializer;

#[derive(TryFromPrimitive, Debug, Copy, Clone, PartialEq)]
#[repr(u8)]
pub enum StreamFormat {
    AdcDacData = 1,
}

#[derive(Debug, Copy, Clone)]
pub enum Error {
    DataFormat(data::FormatError),
    InvalidHeader,
    UnknownFormat,
}

impl From<data::FormatError> for Error {
    fn from(e: data::FormatError) -> Error {
        Error::DataFormat(e)
    }
}

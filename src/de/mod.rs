use num_enum::TryFromPrimitive;
use thiserror::Error;

mod data;
pub use data::*;
mod frame;
pub use frame::*;

#[derive(TryFromPrimitive, Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
#[non_exhaustive]
pub enum Format {
    AdcDac = 1,
    Fls = 2,
}

#[derive(Debug, Clone, Error)]
pub enum Error {
    #[error("Could not parse the frame payload")]
    DataFormat(#[from] data::FormatError),
    #[error("Invalid frame header")]
    InvalidHeader,
    #[error("Unknown format ID")]
    UnknownFormat,
}

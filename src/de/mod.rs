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
    ThermostatEem = 3,
}

#[derive(Debug, Clone, Error)]
pub enum Error {
    #[error("Invalid frame header")]
    InvalidHeader,
    #[error("Unknown format ID")]
    UnknownFormat,
    #[error("Payload size")]
    PayloadSize(#[from] bytemuck::PodCastError),
}

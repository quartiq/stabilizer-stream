use thiserror::Error;

mod de;
pub use de::*;
mod hbf;
pub use hbf::*;
mod psd;
pub use psd::*;
mod loss;
pub use loss::*;

pub mod source;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Frame deserialization error")]
    Frame(#[from] de::Error),
    #[error("IO/Networt error")]
    Network(#[from] std::io::Error),
}

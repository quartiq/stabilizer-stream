use thiserror::Error;

mod de;
pub use de::*;
mod psd;
pub use psd::*;
mod loss;
pub use loss::*;
mod var;
pub use var::*;

pub mod source;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Frame deserialization error")]
    Frame(#[from] de::Error),
    #[error("IO/Networt error")]
    Network(#[from] std::io::Error),
}

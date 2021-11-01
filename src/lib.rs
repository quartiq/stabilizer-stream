pub mod de;

use async_std::net::UdpSocket;
use de::deserializer::StreamFrame;
use std::time::Duration;

/// Receives stream frames from Stabilizer over UDP.
pub struct StreamReceiver {
    socket: UdpSocket,
    buf: [u8; 2048],
}

impl StreamReceiver {
    /// Construct a new receiver.
    ///
    /// # Args
    /// * `ip` - The IP address to bind to. Should be associated with the interface that is used to
    ///   communciate with Stabilizer.
    /// * `port` - The port that livestream data is being sent to.
    pub async fn new(ip: std::net::Ipv4Addr, port: u16) -> Self {
        let socket = UdpSocket::bind((ip, port)).await.unwrap();

        Self {
            socket,
            buf: [0; 2048],
        }
    }

    /// Receive a stream frame from Stabilizer.
    pub async fn next_frame(&mut self) -> Option<StreamFrame> {
        // Read a single UDP packet.
        let len = async_std::io::timeout(Duration::from_secs(1), self.socket.recv(&mut self.buf))
            .await
            .unwrap();

        // Deserialize the stream frame.
        StreamFrame::from_bytes(&self.buf[..len])
            .map_err(|err| {
                log::warn!("Frame deserialization error: {:?}", err);
                err
            })
            .ok()
    }
}

pub mod de;

use async_std::net::UdpSocket;
use de::deserializer::StreamFrame;
use std::time::Duration;

/// Receives stream frames from Stabilizer over UDP.
pub struct StreamReceiver {
    socket: UdpSocket,
    buf: [u8; 2048],
    timeout: Option<Duration>,
}

impl StreamReceiver {
    /// Construct a new receiver.
    ///
    /// # Args
    /// * `ip` - The IP address to bind to. Should be associated with the interface that is used to
    ///   communciate with Stabilizer.
    /// * `port` - The port that livestream data is being sent to.
    pub async fn new(ip: std::net::Ipv4Addr, port: u16) -> Self {
        log::info!("Binding to {}:{}", ip, port);
        let socket = UdpSocket::bind((ip, port)).await.unwrap();

        Self {
            socket,
            timeout: None,
            buf: [0; 2048],
        }
    }

    pub fn set_timeout(&mut self, duration: Duration) {
        self.timeout.replace(duration);
    }

    /// Receive a stream frame from Stabilizer.
    pub async fn next_frame(&mut self) -> Option<StreamFrame> {
        // Read a single UDP packet.
        let len = if let Some(timeout) = self.timeout {
            async_std::io::timeout(timeout, self.socket.recv(&mut self.buf))
                .await
                .unwrap()
        } else {
            self.socket.recv(&mut self.buf).await.unwrap()
        };

        // Deserialize the stream frame.
        StreamFrame::from_bytes(&self.buf[..len])
            .map_err(|err| {
                log::warn!("Frame deserialization error: {:?}", err);
                err
            })
            .ok()
    }
}

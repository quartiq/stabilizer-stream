use std::time::{Duration, Instant};
use stream_viewer::de::deserializer::StreamFrame;

use async_std::net::{Ipv4Addr, UdpSocket};

const MIN_STREAM_EFFICIENCY: f32 = 0.95;

const STREAM_PORT: u16 = 2000;

const TEST_DURATION: Duration = Duration::from_secs(5);

struct StreamReceiver {
    socket: UdpSocket,
    buf: [u8; 2048],
}

impl StreamReceiver {
    pub async fn new(port: u16) -> Self {
        let socket = UdpSocket::bind((Ipv4Addr::LOCALHOST, port)).await.unwrap();

        Self {
            socket,
            buf: [0; 2048],
        }
    }

    pub async fn next_frame<'s>(&'s mut self) -> Option<StreamFrame<'s>> {
        // Read a single UDP packet.
        let len = async_std::io::timeout(TEST_DURATION, self.socket.recv(&mut self.buf))
            .await
            .unwrap();

        // Deserialize the stream frame.
        StreamFrame::from_bytes(&self.buf[..len])
            .or_else(|err| {
                log::warn!("Frame deserialization error: {:?}", err);
                Err(err)
            })
            .ok()
    }
}

#[async_std::main]
async fn main() {
    env_logger::init();
    log::info!("Binding to socket");
    let mut stream_receiver = StreamReceiver::new(STREAM_PORT).await;

    let frame = stream_receiver.next_frame().await.unwrap();

    let mut total_batches = frame.batch_size;
    let mut dropped_batches = 0;
    let mut last_sequence = frame.sequence_number;

    let stop = Instant::now() + TEST_DURATION;

    log::info!("Reading frames");
    while Instant::now() < stop {
        let frame = stream_receiver.next_frame().await.unwrap();

        let num_dropped = frame
            .sequence_number
            .wrapping_sub(last_sequence.wrapping_add(1));
        total_batches += (1 + num_dropped) as usize;

        if num_dropped > 0 {
            dropped_batches += num_dropped;
            log::warn!(
                "Frame drop detected: 0x{:X} -> 0x{:X} ({} batches)",
                last_sequence,
                frame.sequence_number,
                num_dropped
            )
        }

        last_sequence = frame.sequence_number;
    }

    assert!(total_batches > 0);
    let stream_efficiency = 1.0 - (dropped_batches as f32 / total_batches as f32);

    log::info!("Stream reception rate: {:.2} %", stream_efficiency * 100.0);
    log::info!("Received {} batches", total_batches);
    log::info!("Lost {} batches", dropped_batches);

    assert!(stream_efficiency > MIN_STREAM_EFFICIENCY);
}

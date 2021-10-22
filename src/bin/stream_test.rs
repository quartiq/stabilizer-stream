use async_std::net::UdpSocket;
use clap::Parser;
use serde::Serialize;
use stabilizer_streaming::{de::deserializer::StreamFrame, miniconf_client::MiniconfClient};
use std::time::{Duration, Instant};

const MIN_STREAM_EFFICIENCY: f32 = 0.95;
const STREAM_PORT: u16 = 2000;

/// Execute stabilizer stream throughput testing.
#[derive(Parser)]
struct Opts {
    /// The prefix of the stabilizer to use. E.g. dt/sinara/dual-iir/00-11-22-33-44-55
    #[clap(short, long)]
    prefix: String,

    /// The MQTT broker to connect with
    #[clap(short, long)]
    broker: String,

    /// The test duration to execute for.
    #[clap(default_value = "5")]
    duration: u64,
}

#[derive(Serialize)]
struct StreamTarget {
    pub ip: [u8; 4],
    pub port: u16,
}

struct StreamReceiver {
    socket: UdpSocket,
    buf: [u8; 2048],
}

impl StreamReceiver {
    pub async fn new(broker: std::net::Ipv4Addr, port: u16) -> Self {
        let socket = UdpSocket::bind((broker, port)).await.unwrap();

        Self {
            socket,
            buf: [0; 2048],
        }
    }

    pub async fn next_frame<'s>(&'s mut self) -> Option<StreamFrame<'s>> {
        // Read a single UDP packet.
        let len = async_std::io::timeout(Duration::from_secs(1), self.socket.recv(&mut self.buf))
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
    let opts = Opts::parse();
    let broker: std::net::Ipv4Addr = opts.broker.parse().unwrap();

    env_logger::init();
    log::info!("Binding to socket");
    let mut stream_receiver = StreamReceiver::new(broker, STREAM_PORT).await;

    let mut miniconf = MiniconfClient::new(&opts.prefix, broker.octets());

    // Configure stabilizer
    miniconf
        .configure(
            "stream_target",
            StreamTarget {
                ip: broker.octets(),
                port: STREAM_PORT,
            },
        )
        .unwrap();

    let frame = stream_receiver.next_frame().await.unwrap();

    let mut total_batches = frame.batch_count();
    let mut dropped_batches = 0;
    let mut last_sequence = frame.sequence_number;

    let stop = Instant::now() + Duration::from_secs(opts.duration);

    log::info!("Reading frames");
    while Instant::now() < stop {
        let frame = stream_receiver.next_frame().await.unwrap();

        let num_dropped = frame.sequence_number.wrapping_sub(last_sequence) as usize;
        total_batches += frame.batch_count() + num_dropped;

        if num_dropped > 0 {
            dropped_batches += num_dropped;
            log::warn!(
                "Frame drop detected: 0x{:X} -> 0x{:X} ({} batches)",
                last_sequence,
                frame.sequence_number,
                num_dropped
            )
        }

        last_sequence = frame
            .sequence_number
            .wrapping_add(frame.batch_count() as u32);
    }

    miniconf
        .configure(
            "stream_target",
            StreamTarget {
                ip: [0, 0, 0, 0],
                port: 0,
            },
        )
        .unwrap();

    assert!(total_batches > 0);
    let stream_efficiency = 1.0 - (dropped_batches as f32 / total_batches as f32);

    log::info!("Stream reception rate: {:.2} %", stream_efficiency * 100.0);
    log::info!("Received {} batches", total_batches);
    log::info!("Lost {} batches", dropped_batches);

    assert!(stream_efficiency > MIN_STREAM_EFFICIENCY);
}

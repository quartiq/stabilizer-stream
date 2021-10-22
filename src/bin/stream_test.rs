use clap::Parser;
use stabilizer_streaming::StreamReceiver;
use std::time::{Duration, Instant};

const MIN_STREAM_EFFICIENCY: f32 = 0.95;

/// Execute stabilizer stream throughput testing.
#[derive(Parser)]
struct Opts {
    /// The prefix of the stabilizer to use. E.g. dt/sinara/dual-iir/00-11-22-33-44-55
    #[clap(long)]
    prefix: String,

    /// The IP of the interface to the broker.
    #[clap(short, long, default_value = "127.0.0.1")]
    ip: String,

    #[clap(long, default_value = "2000")]
    port: u16,

    /// The test duration to execute for.
    #[clap(long, default_value = "5")]
    duration: u64,
}

#[async_std::main]
async fn main() {
    env_logger::init();

    let opts = Opts::parse();
    let ip: std::net::Ipv4Addr = opts.ip.parse().unwrap();

    log::info!("Binding to socket");
    let mut stream_receiver = StreamReceiver::new(ip, opts.port).await;
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

    assert!(total_batches > 0);
    let stream_efficiency = 1.0 - (dropped_batches as f32 / total_batches as f32);

    log::info!("Stream reception rate: {:.2} %", stream_efficiency * 100.0);
    log::info!("Received {} batches", total_batches);
    log::info!("Lost {} batches", dropped_batches);

    assert!(stream_efficiency > MIN_STREAM_EFFICIENCY);
}

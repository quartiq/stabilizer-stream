use clap::Parser;
use stabilizer_streaming::StreamReceiver;
use std::time::{Duration, Instant};

const MAX_LOSS: f32 = 0.05;

/// Execute stabilizer stream throughput testing.
/// Use `RUST_LOG=info cargo run` to increase logging verbosity.
#[derive(Parser)]
struct Opts {
    /// The local IP to receive streaming data on.
    #[clap(short, long, default_value = "0.0.0.0")]
    ip: String,

    /// The UDP port to receive streaming data on.
    #[clap(long, default_value = "9293")]
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

    let mut total_batches = 0;
    let mut dropped_batches = 0;
    let mut expect_sequence = None;

    let stop = Instant::now() + Duration::from_secs(opts.duration);

    log::info!("Reading frames");
    while Instant::now() < stop {
        let frame = stream_receiver.next_frame().await.unwrap();
        total_batches += frame.batch_count();

        if let Some(expect) = expect_sequence {
            let num_dropped = frame.sequence_number.wrapping_sub(expect) as usize;
            dropped_batches += num_dropped;

            if num_dropped > 0 {
                total_batches += num_dropped;
                log::warn!(
                    "Lost frame(s): 0x{:X} -> 0x{:X} ({} batches)",
                    expect,
                    frame.sequence_number,
                    num_dropped
                );
            }
        }

        expect_sequence = Some(
            frame
                .sequence_number
                .wrapping_add(frame.batch_count() as u32),
        );
    }

    assert!(total_batches > 0);
    let loss = dropped_batches as f32 / total_batches as f32;

    log::info!(
        "Stream loss: {:.2} % ({}/{})",
        loss * 100.0,
        dropped_batches,
        total_batches
    );

    assert!(loss < MAX_LOSS);
}

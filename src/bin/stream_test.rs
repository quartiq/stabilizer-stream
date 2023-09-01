use anyhow::Result;
use clap::Parser;
use stabilizer_streaming::{Detrend, Frame, Loss, PsdCascade};
use std::sync::mpsc;
use std::time::Duration;

/// Execute stabilizer stream throughput testing.
/// Use `RUST_LOG=info cargo run` to increase logging verbosity.
#[derive(Parser)]
struct Opts {
    /// The local IP to receive streaming data on.
    #[clap(short, long, default_value = "0.0.0.0")]
    ip: std::net::Ipv4Addr,

    /// The UDP port to receive streaming data on.
    #[clap(long, long, default_value = "9293")]
    port: u16,

    /// The test duration in seconds.
    #[clap(long, long, default_value = "5")]
    duration: f32,
}

fn main() -> Result<()> {
    env_logger::init();
    let opts = Opts::parse();

    let (cmd_send, cmd_recv) = mpsc::channel();
    let receiver = std::thread::spawn(move || {
        log::info!("Binding to {}:{}", opts.ip, opts.port);
        let socket = std::net::UdpSocket::bind((opts.ip, opts.port))?;
        socket2::SockRef::from(&socket).set_recv_buffer_size(1 << 20)?;
        socket.set_read_timeout(Some(Duration::from_millis(100)))?;
        log::info!("Receiving frames");
        let mut buf = vec![0u8; 2048];

        let mut loss = Loss::default();

        let mut dec: Vec<_> = (0..4)
            .map(|_| {
                let mut c = PsdCascade::<{ 1 << 9 }>::default();
                c.set_stage_length(3);
                c.set_detrend(Detrend::Mean);
                c
            })
            .collect();

        while cmd_recv.try_recv() == Err(mpsc::TryRecvError::Empty) {
            let len = socket.recv(&mut buf)?;
            match Frame::from_bytes(&buf[..len]) {
                Ok(frame) => {
                    loss.update(&frame);
                    for (dec, x) in dec.iter_mut().zip(frame.data.traces()) {
                        dec.process(x);
                    }
                }
                Err(e) => log::warn!("{e}"),
            };
        }

        loss.analyze();

        let (y, b) = dec[1].get(4);
        println!("{:?}, {:?}", b, y);

        Result::<()>::Ok(())
    });

    std::thread::sleep(Duration::from_millis((opts.duration * 1000.) as _));
    cmd_send.send(())?;
    receiver.join().unwrap()?;

    Ok(())
}

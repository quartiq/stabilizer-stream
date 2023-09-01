use anyhow::Result;
use clap::Parser;
use stabilizer_streaming::{
    source::{Source, SourceOpts},
    Detrend, Frame, Loss, PsdCascade,
};
use std::sync::mpsc;
use std::time::Duration;

/// Execute stabilizer stream throughput testing.
/// Use `RUST_LOG=info cargo run` to increase logging verbosity.
#[derive(Parser, Debug)]
pub struct Opts {
    #[command(flatten)]
    source: SourceOpts,

    #[arg(short, long, default_value_t = 10.0)]
    duration: f32,
}

fn main() -> Result<()> {
    env_logger::init();
    let opts = Opts::parse();

    let (cmd_send, cmd_recv) = mpsc::channel();
    let receiver = std::thread::spawn(move || {
        let mut source = Source::new(&opts.source)?;
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
            let len = source.get(&mut buf)?;
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

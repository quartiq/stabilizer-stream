use anyhow::Result;
use clap::Parser;
use stabilizer_streaming::{
    source::{Source, SourceOpts},
    Break, Detrend, Frame, Loss, PsdCascade, VarBuilder,
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
                c.set_detrend(Detrend::Mid);
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

        let (y, b) = dec[1].psd(1);
        log::info!("breaks: {:?}", b);
        log::info!("psd: {:?}", y);

        if let Some(b0) = b.last() {
            let var = VarBuilder::default().dc_cut(1).clip(1.0).build().unwrap();
            let mut fdev = vec![];
            let mut tau = 1.0;
            let f = Break::frequencies(&b);
            while tau <= (b0.effective_fft_size / 2) as f32 {
                fdev.push((tau, var.eval(&y, &f, tau).sqrt()));
                tau *= 2.0;
            }
            log::info!("fdev: {:?}", fdev);
        }

        Result::<()>::Ok(())
    });

    std::thread::sleep(Duration::from_millis((opts.duration * 1000.) as _));
    cmd_send.send(())?;
    receiver.join().unwrap()?;

    Ok(())
}

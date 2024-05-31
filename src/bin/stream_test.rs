use anyhow::Result;
use clap::Parser;
use std::sync::mpsc;
use std::time::Duration;

use stabilizer_stream::{
    source::{Source, SourceOpts},
    Break, Detrend, MergeOpts, PsdCascade, VarBuilder,
};

/// Execute stabilizer stream throughput testing.
/// Use `RUST_LOG=info cargo run` to increase logging verbosity.
#[derive(Parser, Debug)]
struct Opts {
    #[command(flatten)]
    source: SourceOpts,

    #[arg(short, long, default_value_t = 10.0)]
    duration: f32,

    #[arg(short, long, default_value_t = 0)]
    trace: usize,
}

fn main() -> Result<()> {
    env_logger::init();
    let Opts {
        source,
        duration,
        trace,
    } = Opts::parse();
    let merge_opts = MergeOpts::default();

    let (cmd_send, cmd_recv) = mpsc::channel();
    let receiver = std::thread::spawn(move || {
        let mut source = Source::new(source)?;

        let mut dec: Vec<_> = (0..4)
            .map(|_| {
                let mut c = PsdCascade::<{ 1 << 9 }>::new(3);
                c.set_detrend(Detrend::Midpoint);
                c
            })
            .collect();

        while cmd_recv.try_recv() == Err(mpsc::TryRecvError::Empty) {
            match source.get() {
                Ok(traces) => {
                    for (dec, x) in dec.iter_mut().zip(traces) {
                        dec.process(&x.1);
                    }
                }
                Err(e) => log::warn!("{e}"),
            };
        }

        let (y, b) = dec[trace].psd(&merge_opts);
        log::info!("breaks: {:?}", b);
        log::info!("psd: {:?}", y);

        if let Some(b0) = b.last() {
            let var = VarBuilder::default().dc_cut(1).clip(1.0).build().unwrap();
            let mut fdev = vec![];
            let mut tau = 1.0;
            let f = Break::frequencies(&b);
            while tau <= (b0.effective_fft_size() / 2) as f32 {
                fdev.push((tau, var.eval(&y, &f, tau).sqrt()));
                tau *= 2.0;
            }
            log::info!("fdev: {:?}", fdev);
        }

        source.finish();

        Result::<()>::Ok(())
    });

    std::thread::sleep(Duration::from_millis((duration * 1000.) as _));
    cmd_send.send(()).ok();
    receiver.join().unwrap()?;

    Ok(())
}

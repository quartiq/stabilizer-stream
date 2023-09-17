use anyhow::Result;
use clap::Parser;
use stabilizer_streaming::source::{Source, SourceOpts};
use std::io::Write;

#[derive(Parser, Debug)]
pub struct Opts {
    #[command(flatten)]
    source: SourceOpts,

    #[arg(short, long, default_value_t = 0)]
    trace: usize,
}

fn main() -> Result<()> {
    env_logger::init();
    let Opts { trace, source } = Opts::parse();

    let mut source = Source::new(source)?;
    let mut stdout = std::io::BufWriter::new(std::io::stdout());

    loop {
        let t = &source.get()?[trace];
        stdout.write_all(bytemuck::cast_slice(&t[..]))?;
    }

    // source.finish();
}

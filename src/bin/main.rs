#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use anyhow::Result;
use clap::Parser;
use eframe::egui;
use eframe::egui::plot::{Legend, Line, Plot, PlotPoints};
use std::sync::mpsc;
use std::time::Duration;

use stabilizer_streaming::{
    source::{Source, SourceOpts},
    Break, Detrend, PsdCascade,
};

#[derive(Clone, Copy, Debug)]
enum Cmd {
    Exit,
    Reset,
    Send,
}

struct Trace {
    breaks: Vec<Break>,
    psd: Vec<[f64; 2]>,
}

#[derive(Parser, Debug)]
pub struct Opts {
    #[command(flatten)]
    source: SourceOpts,

    /// Exclude PSD stages with less than or equal this averaging level
    #[arg(short, long, default_value_t = 1)]
    min_avg: usize,

    /// Segment detrending method
    #[arg(short, long, default_value = "mid")]
    detrend: Detrend,

    /// Sample rate in Hertz
    #[arg(short, long, default_value_t = 1.0f32)]
    fs: f32,

    /// Exponential averaging, negative for constant time, positive for constant count accross stages
    #[arg(short, long, default_value_t = 100000)]
    max_avg: isize,

    /// Integrate jitter (linear)
    #[arg(short, long)]
    integrate: bool,
}

fn main() -> Result<()> {
    env_logger::init();
    let Opts {
        source,
        min_avg,
        detrend,
        fs,
        max_avg,
        integrate,
    } = Opts::parse();

    let logfs = fs.log10();

    let (cmd_send, cmd_recv) = mpsc::channel();
    let (trace_send, trace_recv) = mpsc::sync_channel(1);
    let mut source = Source::new(source)?;
    let mut dec = Vec::with_capacity(4);

    let receiver = std::thread::spawn(move || {
        loop {
            if dec.is_empty() {
                // TODO max 4 traces hardcoded
                dec.extend((0..4).map(|_| {
                    let mut c = PsdCascade::<{ 1 << 9 }>::default();
                    c.set_stage_depth(3);
                    c.set_detrend(detrend);
                    c.set_avg(max_avg);
                    c
                }));
            }

            match source.get() {
                Ok(traces) => {
                    for (dec, x) in dec.iter_mut().zip(traces) {
                        dec.process(&x);
                    }
                }
                Err(e) => log::warn!("source: {}", e),
            }

            match cmd_recv.try_recv() {
                Err(mpsc::TryRecvError::Disconnected) | Ok(Cmd::Exit) => break,
                Ok(Cmd::Reset) => dec.clear(),
                Err(mpsc::TryRecvError::Empty) => {}
                Ok(Cmd::Send) => {
                    let trace = dec
                        .iter()
                        .map(|dec| {
                            let (p, b) = dec.psd(min_avg);
                            let f = Break::frequencies(&b, min_avg);
                            let (mut p0, mut f0) = (0.0, 0.0);
                            Trace {
                                breaks: b,
                                psd: f
                                    .iter()
                                    .zip(p.iter())
                                    .rev()
                                    .skip(1) // skip DC
                                    .map(|(f, p)| {
                                        let fsf = fs * f;
                                        p0 += p * (fsf - f0);
                                        f0 = fsf;
                                        [
                                            fsf.log10() as f64,
                                            (if integrate {
                                                p0
                                            } else {
                                                10.0 * (p.log10() - logfs)
                                            }) as f64,
                                        ]
                                    })
                                    .collect(),
                            }
                        })
                        .collect();
                    match trace_send.try_send(trace) {
                        Ok(()) => {}
                        Err(mpsc::TrySendError::Full(_)) => {
                            // log::warn!("full");
                        }
                        Err(e) => {
                            log::error!("{:?}", e);
                        }
                    }
                }
            };
        }

        source.finish();

        Result::<()>::Ok(())
    });

    let options = eframe::NativeOptions {
        initial_window_size: Some((1000.0, 700.0).into()),
        ..Default::default()
    };
    eframe::run_native(
        "FLS",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::light());
            Box::new(FLS::new(trace_recv, cmd_send))
        }),
    )
    .unwrap();

    receiver.join().unwrap()?;

    Ok(())
}

pub struct FLS {
    trace_recv: mpsc::Receiver<Vec<Trace>>,
    cmd_send: mpsc::Sender<Cmd>,
    current: Vec<Trace>,
}

impl FLS {
    fn new(trace_recv: mpsc::Receiver<Vec<Trace>>, cmd_send: mpsc::Sender<Cmd>) -> Self {
        Self {
            trace_recv,
            cmd_send,
            current: vec![],
        }
    }
}

impl eframe::App for FLS {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.cmd_send.send(Cmd::Exit).unwrap();
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.trace_recv.try_recv() {
                Err(mpsc::TryRecvError::Empty) => {}
                Ok(new) => {
                    self.current = new;
                    ctx.request_repaint_after(Duration::from_millis(100))
                }
                Err(mpsc::TryRecvError::Disconnected) => panic!("lost data processing thread"),
            };

            let plot: Plot = Plot::new("")
                .height(ui.available_height() - 20.0)
                // TODO proper log axis
                // .x_grid_spacer(log_grid_spacer(10))
                // .x_axis_formatter(log_axis_formatter())
                .legend(Legend::default());
            plot.show(ui, |plot_ui| {
                // TODO trace names
                for (trace, name) in self.current.iter().zip("ABCD".chars()) {
                    if trace.psd.first().is_some_and(|v| v[1].is_finite()) {
                        plot_ui.line(Line::new(PlotPoints::from(trace.psd.clone())).name(name));
                    }
                }
            });

            ui.horizontal(|ui| {
                if ui.button("Reset").clicked() {
                    self.cmd_send.send(Cmd::Reset).unwrap();
                }
                self.current
                    .first()
                    .and_then(|t| t.breaks.first())
                    .map(|bi| ui.label(format!("{:.2e} top level averages", bi.count as f32)));
            });
        });
        self.cmd_send.send(Cmd::Send).unwrap();
    }
}

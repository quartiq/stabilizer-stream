#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use anyhow::Result;
use clap::Parser;
use eframe::egui::plot::{Legend, Line, Plot, PlotPoints};
use eframe::egui::{self, ComboBox, Slider};
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
    Send(AcqOpts),
}

struct Trace {
    breaks: Vec<Break>,
    psd: Vec<[f64; 2]>,
}

#[derive(Parser, Debug)]
pub struct Opts {
    #[command(flatten)]
    source: SourceOpts,

    #[command(flatten)]
    acq: AcqOpts,
}

#[derive(Parser, Debug, Copy, Clone)]
pub struct AcqOpts {
    /// Exclude PSD stages with less than or equal this averaging level
    #[arg(short, long, default_value_t = 1)]
    min_avg: usize,

    /// Segment detrending method
    #[arg(short, long, default_value = "mid")]
    detrend: Detrend,

    /// Sample rate in Hertz
    #[arg(short, long, default_value_t = 1.0f32)]
    fs: f32,

    /// Exponential averaging
    #[arg(short, long, default_value_t = 100000)]
    max_avg: usize,

    /// Enable for constant time averaging across stages
    /// Disable for constant count averaging across stages
    #[arg(short, long)]
    scale_avg: bool,

    /// Integrate jitter (linear)
    #[arg(short, long)]
    integrate: bool,
}

fn main() -> Result<()> {
    env_logger::init();
    let Opts { source, mut acq } = Opts::parse();

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
                    c.set_detrend(acq.detrend);
                    c.set_avg(acq.scale_avg, acq.max_avg);
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
                Ok(Cmd::Send(opts)) => {
                    acq = opts;
                    let logfs = acq.fs.log10();
                    let trace = dec
                        .iter()
                        .map(|dec| {
                            let (p, b) = dec.psd(acq.min_avg);
                            let f = Break::frequencies(&b, acq.min_avg);
                            let (mut p0, mut f0) = (0.0, 0.0);
                            Trace {
                                breaks: b,
                                psd: f
                                    .iter()
                                    .zip(p.iter())
                                    .rev()
                                    .skip(1) // skip DC
                                    .map(|(f, p)| {
                                        let fsf = acq.fs * f;
                                        p0 += p * (fsf - f0);
                                        f0 = fsf;
                                        [
                                            fsf.log10() as f64,
                                            (if acq.integrate {
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
        initial_window_size: Some((1200.0, 700.0).into()),
        ..Default::default()
    };
    eframe::run_native(
        "FLS",
        options,
        Box::new(move |cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::light());
            Box::new(FLS::new(trace_recv, cmd_send, acq))
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
    acq: AcqOpts,
}

impl FLS {
    fn new(
        trace_recv: mpsc::Receiver<Vec<Trace>>,
        cmd_send: mpsc::Sender<Cmd>,
        acq: AcqOpts,
    ) -> Self {
        Self {
            trace_recv,
            cmd_send,
            current: vec![],
            acq,
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

            ui.horizontal(|ui| {
                ComboBox::from_label("Detrend")
                    .selected_text(format!("{:?}", self.acq.detrend))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.acq.detrend, Detrend::None, "None");
                        ui.selectable_value(&mut self.acq.detrend, Detrend::Mid, "Mid");
                        ui.selectable_value(&mut self.acq.detrend, Detrend::Span, "Span");
                        ui.selectable_value(&mut self.acq.detrend, Detrend::Mean, "Mean");
                        // ui.selectable_value(&mut self.acq.detrend, Detrend::Linear, "Linear");
                    })
                    .response
                    .on_hover_text("Segment detrending method");
                ui.separator();
                ui.add(
                    Slider::new(&mut self.acq.max_avg, 1..=1_000_000)
                        .text("Averages")
                        .logarithmic(true),
                ).on_hover_text("Averaging count: when below, the averaging is boxcar, when above it continues with exponential averaging");
                ui.separator();
                ui.checkbox(&mut self.acq.scale_avg, "Constant time avg").on_hover_text("Scale stage averaging count by stage dependent sample rate");
                ui.separator();
                ui.add(
                    Slider::new(&mut self.acq.min_avg, 0..=self.acq.max_avg)
                        .text("Min Count")
                        .logarithmic(true),
                ).on_hover_text("Minimum averaging count to show data from a stage");
            });
            ui.horizontal(|ui| {
                ui.add(
                    Slider::new(&mut self.acq.fs, 1e-3..=1e9)
                        .text("Sample rate")
                        .custom_formatter(|x, _| format!("{:.2e}", x))
                        .suffix(" Hz")
                        .logarithmic(true),
                )
                .on_hover_text("Input sample rate");
                ui.separator();
                ui.checkbox(&mut self.acq.integrate, "Integrate").on_hover_text("Integrate PSD into cumulative sum");
                ui.separator();
                self.current
                    .first()
                    .and_then(|t| t.breaks.first())
                    .map(|bi| ui.label(format!("{:.2e}", bi.count as f32)).on_hover_text("Top level average count"));
                ui.separator();
                if ui
                    .button("Reset")
                    .on_hover_text("Reset PSD stages and begin anew")
                    .clicked()
                {
                    self.cmd_send.send(Cmd::Reset).unwrap();
                }
            });

            Plot::new("plot")
                // TODO proper log axis
                // .x_grid_spacer(log_grid_spacer(10))
                // .x_axis_formatter(log_axis_formatter())
                .legend(Legend::default())
                .show(ui, |plot_ui| {
                    // TODO trace names
                    for (trace, name) in self.current.iter().zip("ABCD".chars()) {
                        if trace.psd.first().is_some_and(|v| v[1].is_finite()) {
                            plot_ui.line(Line::new(PlotPoints::from(trace.psd.clone())).name(name));
                        }
                    }
                });
        });
        self.cmd_send.send(Cmd::Send(self.acq)).unwrap();
    }
}

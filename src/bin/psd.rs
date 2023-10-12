#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use anyhow::Result;
use clap::Parser;
use eframe::egui::{self, ComboBox, ProgressBar, Slider};
use egui_plot::{GridInput, GridMark, Legend, Line, Plot, PlotPoint, PlotPoints};
use std::time::Duration;
use std::{ops::RangeInclusive, sync::mpsc};

use stabilizer_stream::{
    source::{Source, SourceOpts},
    AvgOpts, Break, Detrend, MergeOpts, PsdCascade,
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
struct Opts {
    #[command(flatten)]
    source: SourceOpts,

    #[command(flatten)]
    acq: AcqOpts,
}

#[derive(Parser, Debug, Copy, Clone)]
struct AcqOpts {
    /// Segment detrending method
    #[arg(short, long, default_value = "mean")]
    detrend: Detrend,

    /// Sample rate in Hertz
    #[arg(short, long, default_value_t = 1.0f32)]
    fs: f32,

    /// Boxcar/Exponential averaging count
    #[arg(short, long, default_value_t = 1000)]
    max_avg: u32,

    /// Enable for constant time averaging across stages
    /// Disable for constant count averaging across stages
    #[arg(short, long)]
    scale_avg: bool,

    /// Exclude PSD stages with less than or equal this averaging level
    #[arg(short, long, default_value_t = 1)]
    min_avg: u32,

    /// Don't remove low resolution bins
    #[arg(short, long)]
    keep_overlap: bool,

    /// Integrate jitter (linear)
    #[arg(short, long)]
    integrate: bool,
}

impl AcqOpts {
    fn avg_opts(&self) -> AvgOpts {
        AvgOpts {
            scale: self.scale_avg,
            count: self.max_avg,
        }
    }
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
                    let mut dec = PsdCascade::<{ 1 << 9 }>::new(3);
                    dec.set_detrend(acq.detrend);
                    dec.set_avg(acq.avg_opts());
                    dec
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
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) | Ok(Cmd::Exit) => break,
                Ok(Cmd::Reset) => dec.clear(),
                Ok(Cmd::Send(opts)) => {
                    acq = opts;
                    for dec in dec.iter_mut() {
                        dec.set_detrend(acq.detrend);
                        dec.set_avg(acq.avg_opts());
                    }
                    let merge_opts = MergeOpts {
                        remove_overlap: !acq.keep_overlap,
                        min_count: acq.min_avg,
                    };
                    let logfs = acq.fs.log10();
                    let trace = dec
                        .iter()
                        .map(|dec| {
                            let (p, b) = dec.psd(&merge_opts);
                            let f = Break::frequencies(&b, &merge_opts);
                            let (mut p0, mut p1, mut f1) = (0.0, 0.0, 0.0);
                            Trace {
                                breaks: b,
                                psd: f
                                    .iter()
                                    .zip(p.iter())
                                    .rev() // for stable integration offset/sign
                                    .filter_map(|(f, p)| {
                                        // Trapezoidal integration
                                        p0 += (p + p1) * 0.5 * (f - f1);
                                        (p1, f1) = (*p, *f);
                                        if f.is_normal() {
                                            Some([
                                                (f.log10() + logfs) as f64,
                                                (if acq.integrate {
                                                    p0.sqrt()
                                                } else {
                                                    10.0 * (p.log10() - logfs)
                                                })
                                                    as f64,
                                            ])
                                        } else {
                                            None
                                        }
                                    })
                                    .collect(),
                            }
                        })
                        .collect();
                    match trace_send.try_send(trace) {
                        Ok(()) => {}
                        Err(mpsc::TrySendError::Full(_)) => {}
                        Err(e) => Err(e)?,
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
        "PSD",
        options,
        Box::new(move |cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::light());
            Box::new(App::new(trace_recv, cmd_send, acq))
        }),
    )
    .unwrap();

    receiver.join().unwrap()?;

    Ok(())
}

struct App {
    trace_recv: mpsc::Receiver<Vec<Trace>>,
    cmd_send: mpsc::Sender<Cmd>,
    current: Vec<Trace>,
    acq: AcqOpts,
    repaint: f32,
}

impl App {
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
            repaint: 0.1,
        }
    }
}

impl eframe::App for App {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.cmd_send.send(Cmd::Exit).unwrap();
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        match self.trace_recv.try_recv() {
            Err(mpsc::TryRecvError::Empty) => {}
            Ok(new) => {
                self.current = new;
                ctx.request_repaint_after(Duration::from_secs_f32(self.repaint))
            }
            Err(mpsc::TryRecvError::Disconnected) => panic!("lost data processing thread"),
        };

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.add(
                    Slider::new(&mut self.repaint, 0.01..=10.0)
                        .text("Repaint")
                        .suffix(" s")
                        .logarithmic(true),
                )
                .on_hover_text("Request repaint after timeout (seconds)");
                ui.separator();
                ComboBox::from_label("Detrend")
                    .selected_text(format!("{:?}", self.acq.detrend))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.acq.detrend, Detrend::None, "None");
                        ui.selectable_value(&mut self.acq.detrend, Detrend::Midpoint, "Midpoint");
                        ui.selectable_value(&mut self.acq.detrend, Detrend::Span, "Span");
                        ui.selectable_value(&mut self.acq.detrend, Detrend::Mean, "Mean");
                        // ui.selectable_value(&mut self.acq.detrend, Detrend::Linear, "Linear");
                    })
                    .response
                    .on_hover_text("Segment detrending method");
            });
            ui.horizontal(|ui| {
                ui.add(
                    Slider::new(&mut self.acq.max_avg, 1..=1_000_000)
                        .text("Averages")
                        .logarithmic(true),
                )
                .on_hover_text(
                    "Averaging count:\nthe averaging starts as boxcar,\nthen continues exponential",
                );
                ui.separator();
                ui.checkbox(&mut self.acq.scale_avg, "Scale averages")
                    .on_hover_text("Scale stage averaging count\nby stage dependent sample rate");
                ui.separator();
                ui.add(
                    Slider::new(&mut self.acq.min_avg, 0..=self.acq.max_avg)
                        .text("Min averages")
                        .logarithmic(true),
                )
                .on_hover_text("Minimum averaging count to\nshow data from a stage");
                ui.separator();
                ui.checkbox(&mut self.acq.keep_overlap, "Keep overlap")
                    .on_hover_text("Do not remove low-resolution bins");
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
                ui.checkbox(&mut self.acq.integrate, "Integrate")
                    .on_hover_text("Integrate PSD into linear\ncumulative sum");
                if let Some(t) = self.current.first() {
                    if let Some(bi) = t.breaks.last() {
                        ui.separator();
                        ui.add(
                            ProgressBar::new(bi.pending as f32 / bi.fft_size as f32)
                                .desired_width(180.0)
                                .text(format!(
                                    "{} averages, {:.2e} samples",
                                    bi.count,
                                    bi.processed as f32 * (1u64 << bi.decimation) as f32
                                )),
                                // .show_percentage(),
                        )
                        .on_hover_text("Bottom buffer fill,\nnumber of averages, and\neffective number of samples processed");
                    }
                }
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
                .x_axis_label("Modulation frequency (Hz)")
                .x_grid_spacer(log10_grid_spacer)
                .x_axis_formatter(log10_axis_formatter)
                .y_axis_width(4)
                .y_axis_label("Power spectral density (dB/Hz) or integrated RMS")
                .legend(Legend::default())
                .label_formatter(log10x_formatter)
                .show(ui, |plot_ui| {
                    // TODO trace names
                    for (trace, name) in self.current.iter().zip("ABCD".chars()) {
                        if trace.psd.last().is_some_and(|v| v[1].is_finite()) {
                            plot_ui.line(Line::new(PlotPoints::from(trace.psd.clone())).name(name));
                        }
                    }
                });
        });

        self.cmd_send.send(Cmd::Send(self.acq)).unwrap();
    }
}

fn log10_grid_spacer(input: GridInput) -> Vec<GridMark> {
    let base = 10u32;
    assert!(base >= 2);
    let basef = base as f64;

    let step_size = basef.powi(input.base_step_size.abs().log(basef).ceil() as i32);
    let (min, max) = input.bounds;
    let mut steps = vec![];

    for step_size in [step_size, step_size * basef, step_size * basef * basef] {
        if step_size == basef.powi(-1) {
            // FIXME: float comparison
            let first = ((min / (step_size * basef)).floor() as i64) * base as i64;
            let last = (max / step_size).ceil() as i64;

            let mut logs = vec![0.0; base as usize - 2];
            for (i, j) in logs.iter_mut().enumerate() {
                *j = basef * ((i + 2) as f64).log(basef);
            }

            steps.extend((first..last).step_by(base as usize).flat_map(|j| {
                logs.iter().map(move |i| GridMark {
                    value: (j as f64 + i) * step_size,
                    step_size,
                })
            }));
        } else {
            let first = (min / step_size).ceil() as i64;
            let last = (max / step_size).ceil() as i64;

            steps.extend((first..last).map(move |i| GridMark {
                value: (i as f64) * step_size,
                step_size,
            }));
        }
    }
    steps
}

fn log10_axis_formatter(tick: f64, max_digits: usize, _range: &RangeInclusive<f64>) -> String {
    let base = 10u32;
    let basef = base as f64;

    let s = format!("{:.0e}", basef.powf(tick));
    if s.len() > max_digits {
        // || !s.starts_with(|c| ['1', '2', '5'].contains(&c)) {
        "".to_string()
    } else {
        s
    }
}

fn log10x_formatter(name: &str, value: &PlotPoint) -> String {
    let base = 10u32;
    let basef = base as f64;
    format!(
        "{}\nx: {:.d$e}\ny: {:.d$e}",
        name,
        basef.powf(value.x),
        value.y,
        d = 3
    )
}

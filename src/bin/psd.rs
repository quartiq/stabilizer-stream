#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use anyhow::Result;
use clap::Parser;
use eframe::{
    egui::{self, ComboBox, Slider, Ui},
    epaint::Color32,
};
use egui_plot::{
    Bar, BarChart, GridInput, GridMark, Legend, Line, LineStyle, Plot, PlotPoint, PlotPoints, Text,
    VLine,
};
use std::{ops::RangeInclusive, sync::mpsc, time::Duration};

use stabilizer_stream::{
    source::{Source, SourceOpts},
    AvgOpts, Break, Detrend, MergeOpts, PsdCascade,
};

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

    /// Averaging limit
    #[arg(short, long, default_value_t = 1000)]
    avg_max: u32,

    /// Exclude PSD stages with less than or equal this averaging level
    #[arg(short, long, default_value_t = 1)]
    avg_min: u32,

    /// Extrapolated averaging count at the top stage
    #[arg(short, long, default_value_t = u32::MAX)]
    avg: u32,

    /// Don't remove low resolution bins
    #[arg(short, long)]
    keep_overlap: bool,

    /// Don't remove bins in the filter transition band
    #[arg(short, long)]
    keep_transition_band: bool,

    /// Integrate jitter (linear)
    #[arg(short, long)]
    integrate: bool,

    #[arg(short, long, default_value_t = 1e-6)]
    integral_start: f32,

    #[arg(short, long, default_value_t = 0.5)]
    integral_end: f32,
}

impl AcqOpts {
    fn avg_opts(&self) -> AvgOpts {
        AvgOpts {
            limit: self.avg_max.saturating_sub(1),
            count: self.avg.saturating_sub(1),
        }
    }

    fn merge_opts(&self) -> MergeOpts {
        MergeOpts {
            remove_overlap: !self.keep_overlap,
            min_count: self.avg_min,
            remove_transition_band: !self.keep_transition_band,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Cmd {
    Exit,
    Reset,
    Send(AcqOpts),
}

/// Trapezoidal integrator for irregular sampling
#[derive(Default, Copy, Clone, Debug, PartialEq, PartialOrd)]
struct Trapezoidal {
    x: f32,
    y: f32,
    i: f32,
}

impl Trapezoidal {
    fn push(&mut self, x: f32, y: f32) -> f32 {
        let di = (y + self.y) * 0.5 * (x - self.x);
        self.x = x;
        self.y = y;
        self.i += di;
        di
    }

    fn get(&self) -> f32 {
        self.i
    }
}

struct Trace {
    breaks: Vec<Break>,
    psd: Vec<f32>,
    frequencies: Vec<f32>,
}

impl Trace {
    fn into_plot(self, acq: &AcqOpts) -> (f32, Vec<[f64; 2]>, Vec<Break>) {
        let logfs = acq.fs.log10();
        let mut p0 = Trapezoidal::default();
        let mut pi = 0.0;
        let plot = self
            .psd
            .into_iter()
            .zip(self.frequencies)
            .filter_map(|(p, f)| {
                // TODO: check at stage breaks
                let dp = p0.push(f, p);
                if (acq.integral_start..=acq.integral_end).contains(&(acq.fs * f)) {
                    // TODO: correctly interpolate at range limits
                    pi += dp;
                }
                if f.is_normal() {
                    Some([
                        (f.log10() + logfs) as f64,
                        (if acq.integrate {
                            p0.get().sqrt()
                        } else {
                            10.0 * (p.log10() - logfs)
                        }) as f64,
                    ])
                } else {
                    None
                }
            })
            .collect();
        (pi.sqrt(), plot, self.breaks)
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let Opts { source, mut acq } = Opts::parse();
    acq.integral_end *= acq.fs;
    acq.integral_start *= acq.fs;

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
                    let merge_opts = acq.merge_opts();
                    let trace = dec
                        .iter()
                        .map(|dec| {
                            let (psd, breaks) = dec.psd(&merge_opts);
                            let frequencies = Break::frequencies(&breaks);
                            Trace {
                                breaks,
                                psd,
                                frequencies,
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
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 700.0])
            .with_drag_and_drop(true),
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
    current: Vec<(String, Vec<[f64; 2]>, Vec<Break>)>,
    acq: AcqOpts,
    repaint: f32,
    hold: bool,
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
            hold: false,
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
            Ok(trace) => {
                if !self.hold {
                    self.current = trace
                        .into_iter()
                        .zip("ABCDEFGH".chars())
                        .map(|(t, n)| {
                            let (pi, t, b) = t.into_plot(&self.acq);
                            (format!("{n}: {:.2e}", pi), t, b)
                        })
                        .collect();
                    ctx.request_repaint_after(Duration::from_secs_f32(self.repaint));
                }
            }
            Err(mpsc::TryRecvError::Disconnected) => panic!("lost data processing thread"),
        };

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| self.row0(ui));
            ui.horizontal(|ui| self.row1(ui));
            ui.horizontal(|ui| self.row2(ui));
            self.plot(ui);
        });

        self.cmd_send.send(Cmd::Send(self.acq)).unwrap();
    }
}

impl App {
    fn plot(&mut self, ui: &mut Ui) {
        Plot::new("stages")
            .link_axis("plots", true, false)
            .show_axes([false, true])
            .y_axis_width(4)
            .y_axis_label("X")
            .y_axis_formatter(|_, _, _| "".to_string())
            .show_grid(false)
            .show_x(false)
            .show_y(false)
            .include_y(0.0)
            .include_y(1.0)
            .show_background(false)
            .allow_boxed_zoom(false)
            .allow_double_click_reset(false)
            .allow_drag(false)
            .allow_scroll(false)
            .allow_zoom(false)
            .height(20.0)
            .show(ui, |plot_ui| {
                let ldfs = self.acq.fs.log10() as f64;
                // TODO: single-trace data
                for (_name, _line, breaks) in self.current.iter().take(1) {
                    let mut end = f32::NEG_INFINITY;
                    let mut texts = Vec::with_capacity(breaks.len());
                    plot_ui.bar_chart(BarChart::new(
                        breaks
                            .iter()
                            .map(|b| {
                                let bins = b.bins();
                                let rbw = b.rbw();
                                let start = (bins.start.max(1) as f32 * rbw).log10().max(end);
                                end = (bins.end as f32 * rbw).log10();
                                let pos = ldfs + ((start + end) * 0.5) as f64;
                                let width = (end - start) as f64;
                                let buf = b.pending as f32 / b.fft_size as f32;
                                texts.push(Text::new(
                                    [pos, 0.5].into(),
                                    format!(
                                        "{}.{:01}, {:.1e} Hz",
                                        b.count,
                                        (buf * 10.0) as i32,
                                        self.acq.fs * rbw
                                    ),
                                ));
                                Bar::new(
                                    pos,
                                    if b.count == 0 {
                                        buf
                                    } else {
                                        b.count as f32 / (b.avg + 1) as f32
                                    } as _,
                                )
                                .width(width)
                                .base_offset(0.0)
                                .stroke((0.0, Color32::default()))
                                .fill(if b.count == 0 {
                                    Color32::LIGHT_RED
                                } else {
                                    Color32::LIGHT_GRAY
                                })
                            })
                            .collect(),
                    ));
                    for t in texts.into_iter() {
                        plot_ui.text(t);
                    }
                }
            });

        Plot::new("plot")
            .x_axis_label("Modulation frequency (Hz)")
            .x_grid_spacer(log10_grid_spacer)
            .x_axis_formatter(log10_axis_formatter)
            .link_axis("plots", false, false)
            .auto_bounds([true; 2].into())
            .y_axis_width(4)
            .y_axis_label("Power spectral density (dB/Hz) or integrated RMS")
            .legend(Legend::default())
            .label_formatter(log10x_formatter)
            .show(ui, |plot_ui| {
                plot_ui.vline(
                    VLine::new(self.acq.integral_start.log10())
                        .stroke((1.0, Color32::DARK_GRAY))
                        .style(LineStyle::dashed_loose()),
                );
                plot_ui.vline(
                    VLine::new(self.acq.integral_end.log10())
                        .stroke((1.0, Color32::DARK_GRAY))
                        .style(LineStyle::dashed_loose()),
                );
                for (name, trace, _) in self.current.iter() {
                    plot_ui.line(Line::new(PlotPoints::from(trace.clone())).name(name));
                }
            });
    }

    fn row0(&mut self, ui: &mut Ui) {
        ui.checkbox(&mut self.hold, "Hold")
            .on_hover_text("Stop updating plot\nAcquiusition continues in background");
        ui.separator();
        if ui
            .button("Reset")
            .on_hover_text("Reset PSD stages and begin anew")
            .clicked()
        {
            self.cmd_send.send(Cmd::Reset).unwrap();
        }
        ui.add(
            Slider::new(&mut self.repaint, 0.01..=10.0)
                .text("Repaint")
                .suffix(" s")
                .logarithmic(true),
        )
        .on_hover_text("Request repaint after timeout (seconds)");
        ui.separator();
        ui.add(
            Slider::new(&mut self.acq.fs, 1e-3..=1e9)
                .text("Sample rate")
                .custom_formatter(|x, _| format!("{:.2e}", x))
                .suffix(" Hz")
                .logarithmic(true),
        )
        .on_hover_text("Input sample rate");
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
    }

    fn row1(&mut self, ui: &mut Ui) {
        ui.add(
            Slider::new(&mut self.acq.avg, 1..=1_000_000_000)
                .text("Averages")
                .logarithmic(true),
        )
        .on_hover_text("Target averaging count at the top stage");
        ui.separator();
        ui.add(
            Slider::new(&mut self.acq.avg_max, 1..=1_000_000.min(self.acq.avg))
                .text("Averaging limit")
                .logarithmic(true),
        )
        .on_hover_text("Averaging limit:\nClip averaging at each stage to this\nThe averaging starts as boxcar,\nthen continues exponential");
        ui.separator();
        ui.add(
            Slider::new(&mut self.acq.avg_min, 0..=self.acq.avg_max)
                .text("Min averages")
                .logarithmic(true),
        )
        .on_hover_text("Minimum averaging count to\nshow data from a stage");
    }

    fn row2(&mut self, ui: &mut Ui) {
        ui.checkbox(&mut self.acq.keep_overlap, "Keep overlap")
            .on_hover_text("Do not remove low-resolution bins");
        ui.separator();
        ui.checkbox(&mut self.acq.keep_transition_band, "Keep t-band")
            .on_hover_text("Do not remove bins in the filter transition band");
        ui.separator();
        ui.checkbox(&mut self.acq.integrate, "Integrate")
            .on_hover_text("Show integrated PSD as linear cumulative sum");
        ui.separator();
        ui.add(
            Slider::new(&mut self.acq.integral_start, 0.0..=self.acq.integral_end)
                .text("Start")
                .custom_formatter(|x, _| format!("{:.2e}", x))
                .suffix(" Hz")
                .logarithmic(true),
        )
        .on_hover_text("Integration range lower frequency");
        ui.add(
            Slider::new(
                &mut self.acq.integral_end,
                self.acq.integral_start..=self.acq.fs * 0.5,
            )
            .text("End")
            .custom_formatter(|x, _| format!("{:.2e}", x))
            .suffix(" Hz")
            .logarithmic(true),
        )
        .on_hover_text("Integration range uppder frequency");
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

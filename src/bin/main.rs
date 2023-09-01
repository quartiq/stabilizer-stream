#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use anyhow::Result;
use clap::Parser;
use eframe::egui;
use eframe::egui::plot::{Legend, Line, Plot, PlotPoints};
// use std::io::{Read, Seek};
use std::sync::mpsc;
use std::time::Duration;

use stabilizer_streaming::{Detrend, Frame, Loss, PsdCascade};

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
}

#[derive(Clone, Copy, Debug)]
enum Cmd {
    Exit,
    Reset,
}

struct Trace {
    psd: Vec<[f64; 2]>,
}

impl Trace {
    fn new(psd: Vec<[f64; 2]>) -> Self {
        Self { psd }
    }
}

fn main() -> Result<()> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let opts = Opts::parse();

    let (cmd_send, cmd_recv) = mpsc::channel();
    let (trace_send, trace_recv) = mpsc::sync_channel(1);
    let receiver = std::thread::spawn(move || {
        log::info!("Binding to {}:{}", opts.ip, opts.port);
        let socket = std::net::UdpSocket::bind((opts.ip, opts.port))?;
        socket2::SockRef::from(&socket).set_recv_buffer_size(1 << 20)?;
        socket.set_read_timeout(Some(Duration::from_millis(1000)))?;
        log::info!("Receiving frames");

        let mut loss = Loss::default();
        let mut dec = Vec::with_capacity(4);

        // let mut fil = std::fs::File::open("/tmp/fls2x.raw")?;

        let mut buf = vec![0; 2048];
        let mut i = 0;
        loop {
            match cmd_recv.try_recv() {
                Err(mpsc::TryRecvError::Disconnected) | Ok(Cmd::Exit) => break,
                Ok(Cmd::Reset) => dec.clear(),
                Err(mpsc::TryRecvError::Empty) => {}
            };

            if dec.is_empty() {
                dec.extend((0..4).map(|_| {
                    let mut c = PsdCascade::<{ 1 << 9 }>::default();
                    c.set_stage_length(3);
                    c.set_detrend(Detrend::Mean);
                    c
                }));
            }

            // let len = fil.read(&mut buf[..1400])?;
            // if len == 0 {
            //     fil.seek(std::io::SeekFrom::Start(0))?;
            //     continue;
            // }

            let len = socket.recv(&mut buf)?;
            match Frame::from_bytes(&buf[..len]) {
                Ok(frame) => {
                    loss.update(&frame);
                    for (dec, x) in dec.iter_mut().zip(frame.data.traces()) {
                        // let x = (0..1<<10).map(|_| (rand::random::<f32>()*2.0 - 1.0)).collect::<Vec<_>>();
                        dec.process(x);
                    }
                    i += 1;
                }
                Err(e) => log::warn!("{e} {:?}", &buf[..8]),
            };
            if i >= 50 {
                i = 0;
                let trace = dec
                    .iter()
                    .map(|dec| {
                        let (p, b) = dec.get(1);
                        let f = dec.f(&b);
                        Trace::new(
                            f.iter()
                                .zip(p.iter())
                                .rev()
                                .skip(1) // DC
                                .map(|(f, p)| [f.log10() as f64, 10.0 * p.log10() as f64])
                                .collect(),
                        )
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
        }

        loss.analyze();

        Result::<()>::Ok(())
    });

    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(640.0, 500.0)),
        ..Default::default()
    };
    eframe::run_native(
        "FLS",
        options,
        Box::new(|cc| Box::new(FLS::new(cc, trace_recv, cmd_send))),
    )
    .unwrap();

    receiver.join().unwrap()?;

    Ok(())
}

pub struct FLS {
    trace_recv: mpsc::Receiver<Vec<Trace>>,
    cmd_send: mpsc::Sender<Cmd>,
    current: Option<Vec<Trace>>,
}

impl FLS {
    fn new(
        cc: &eframe::CreationContext<'_>,
        trace_recv: mpsc::Receiver<Vec<Trace>>,
        cmd_send: mpsc::Sender<Cmd>,
    ) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::light());

        Self {
            trace_recv,
            cmd_send,
            current: None,
        }
    }
}

impl eframe::App for FLS {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.cmd_send.send(Cmd::Exit).ok();
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.trace_recv.try_recv() {
                Err(mpsc::TryRecvError::Empty) => {}
                Ok(new) => {
                    self.current = Some(new);
                    ctx.request_repaint_after(Duration::from_millis(100));
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    panic!("lost data processing thread")
                }
            };
            ui.heading("FLS");
            ui.add_space(20.0);
            ui.horizontal(|ui| {
                ui.add_space(20.0);
                let plot = Plot::new("")
                    .width(600.0)
                    .height(400.0)
                    // .x_grid_spacer(log_grid_spacer(10))
                    .legend(Legend::default());
                plot.show(ui, |plot_ui| {
                    if let Some(traces) = &mut self.current {
                        for (trace, name) in traces.iter().zip(["AI", "AQ", "BI", "BQ"].into_iter())
                        {
                            plot_ui.line(Line::new(PlotPoints::from(trace.psd.clone())).name(name));
                        }
                    }
                });
            });
            ui.add_space(20.0);
            ui.horizontal(|ui| {
                ui.add_space(20.0);
                if ui.button("Reset").clicked() {
                    self.cmd_send.send(Cmd::Reset).unwrap();
                }
                ui.label("Every");
            });
        });
    }
}

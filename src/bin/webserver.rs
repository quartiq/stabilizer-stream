use clap::Parser;
use serde::{Deserialize, Serialize};
use stabilizer_streaming::{de::deserializer::StreamFrame, de::StreamFormat, StreamReceiver};
use tide::{Body, Response};

// TODO: Expose this as a configurable parameter and/or add it to the stream frame.
const SAMPLE_RATE_HZ: f32 = 100e6 / 128.0;

const SAMPLE_PERIOD: f32 = 1.0 / SAMPLE_RATE_HZ;

/// Execute stabilizer stream throughput testing.
/// Use `RUST_LOG=info cargo run` to increase logging verbosity.
#[derive(Parser)]
struct Opts {
    /// The local IP to receive streaming data on.
    #[clap(short, long, default_value = "0.0.0.0")]
    ip: String,

    /// The UDP port to receive streaming data on.
    #[clap(long, default_value = "9293")]
    port: u16,
}

// TODO: Perhaps refactor this to be a state machine to simplify transitional logic.
#[derive(Serialize, Copy, Clone, PartialEq)]
enum TriggerState {
    /// The trigger is idle.
    Idle,

    /// The trigger is armed and waiting for trigger conditions.
    Armed,

    /// The trigger has occurred and data is actively being captured.
    Triggered,

    /// The trigger is complete and data is available for query.
    Stopped,
}

#[derive(Deserialize)]
struct CaptureSettings {
    /// The duration to capture data for in seconds.
    capture_duration_secs: f32,
}

struct ServerState {
    // StreamData cannot implement a const-fn constructor, so we wrap it in an option instead.
    pub data: async_std::sync::Mutex<StreamData>,
}

struct StreamData {
    current_format: Option<StreamFormat>,
    trigger: TriggerState,

    max_size: usize,
    timebase: Vec<u64>,
    traces: Vec<Trace>,
}

#[derive(Serialize, Clone, Debug)]
struct Trace {
    label: String,
    data: Vec<f32>,
}

#[derive(Serialize, Debug)]
struct TraceData {
    time: Vec<f32>,
    traces: Vec<Trace>,
}

impl StreamData {
    const fn new() -> Self {
        Self {
            current_format: None,
            timebase: Vec::new(),
            traces: Vec::new(),

            max_size: SAMPLE_RATE_HZ as usize,

            trigger: TriggerState::Idle,
        }
    }

    /// Ingest an incoming stream frame.
    pub fn add_frame(&mut self, frame: StreamFrame) {
        // If the stream format has changed, clear all data buffers.
        if let Some(format) = self.current_format {
            if frame.format() != format {
                self.flush()
            }
        }

        self.current_format.replace(frame.format());

        // If we aren't triggered, there's nothing more we want to do.
        if self.trigger != TriggerState::Triggered {
            return;
        }

        // Next, extract all of the data traces
        for i in 0..frame.data.trace_count() {
            if self.traces.len() < frame.data.trace_count() {
                self.traces.push(Trace {
                    data: Vec::new(),
                    label: frame.data.trace_label(i),
                });
            }

            // TODO: Decimate the data as requested.
            let trace = frame.data.get_trace(i);
            self.traces[i].data.extend(trace);

            // For the first trace, also extend the timebase.
            if i == 0 {
                let base = (frame.sequence_number() as u64)
                    .wrapping_mul(frame.data.samples_per_batch() as u64);
                for sample_index in 0..trace.len() {
                    self.timebase.push(base.wrapping_add(sample_index as u64))
                }
            }
        }

        // Drain the data/timebase queues to remain within our maximum size.
        if self.timebase.len() > self.max_size {
            self.timebase.drain(self.max_size..);

            for trace in &mut self.traces {
                trace.data.drain(self.max_size..);
            }

            // Stop the capture now that we've filled up our buffers.
            self.trigger = TriggerState::Stopped;
        }
    }

    /// Get the current trace data.
    pub fn get_data(&self) -> TraceData {
        let mut times: Vec<f32> = Vec::new();
        let time_offset = if self.timebase.len() > 0 {
            self.timebase[0]
        } else {
            0
        };

        for time in self.timebase.iter() {
            times.push(time.wrapping_sub(time_offset) as f32 * SAMPLE_PERIOD)
        }

        TraceData {
            time: times,
            traces: self.traces.clone(),
        }
    }

    /// Remove all data from buffers.
    pub fn flush(&mut self) {
        self.timebase.clear();
        self.traces.clear();
        self.current_format.take();
    }
}

/// Stabilizer stream frame reception thread.
///
/// # Note
/// This task executes forever, continuously receiving stabilizer stream frames for processing.
///
/// # Args
/// * `state` - The server state
/// * `state` - A receiver for reading stabilizer stream frames.
async fn receive(state: &ServerState, mut receiver: StreamReceiver) {
    loop {
        // Get a stream frame from Stabilizer.
        let frame = receiver.next_frame().await.unwrap();

        // Add the frame data to the traces.
        let mut data = state.data.lock().await;
        data.add_frame(frame);
    }
}

/// Get all available data traces.
///
/// # Note
/// There is no guarantee that the data will be complete. Poll the current trigger state to ensure
/// all data is available.
///
/// # Args
/// `request` - Unused
///
/// # Returns
/// All of the data as a json-serialized `TraceData`.
async fn get_traces(request: tide::Request<&ServerState>) -> tide::Result<impl Into<Response>> {
    log::info!("Got data request");
    let state = request.state();
    let data = state.data.lock().await;
    let response = data.get_data();
    log::debug!("Response: {:?}", response);
    Ok(Response::builder(200).body(Body::from_json(&response)?))
}

/// Configure the current capture settings
///
/// # Args
/// * `request` - An HTTP request containing json-serialized `CaptureSettings`.
async fn configure_capture(
    mut request: tide::Request<&ServerState>,
) -> tide::Result<impl Into<Response>> {
    let config: CaptureSettings = request.body_json().await?;
    let state = request.state();

    // Clear any pre-existing data in the buffers.
    let mut data = state.data.lock().await;
    data.flush();

    log::info!("Arming trigger");
    data.trigger = TriggerState::Armed;

    log::info!("Forcing trigger");
    data.trigger = TriggerState::Triggered;

    if config.capture_duration_secs < 0. {
        return Ok(Response::builder(400).body("Negative capture duration not supported"));
    }

    let samples: f32 = SAMPLE_RATE_HZ * config.capture_duration_secs;
    if samples > usize::MAX as f32 {
        return Ok(Response::builder(400).body("Too many samples requested"));
    }

    // TODO: Configure decimation
    data.max_size = samples as usize;

    Ok(Response::builder(200))
}

/// Get the current trigger state.
///
/// # Args
/// * `request` - Unused.
///
/// # Returns
/// JSON containing the current trigger state as a string.
async fn get_trigger(request: tide::Request<&ServerState>) -> tide::Result<impl Into<Response>> {
    let state = request.state();
    let data = state.data.lock().await;
    Ok(Response::builder(200).body(Body::from_json(&data.trigger)?))
}

/// Force a trigger condition.
///
/// # Args
/// * `request` - Unused.
async fn force_trigger(request: tide::Request<&ServerState>) -> tide::Result<impl Into<Response>> {
    let state = request.state();
    let mut data = state.data.lock().await;
    log::info!("Forcing trigger");
    data.trigger = TriggerState::Triggered;
    Ok(Response::new(200))
}

#[async_std::main]
async fn main() -> tide::Result<()> {
    env_logger::init();

    let opts = Opts::parse();
    let ip: std::net::Ipv4Addr = opts.ip.parse().unwrap();
    let stream_receiver = StreamReceiver::new(ip, opts.port).await;

    // Populate the initial receiver data.
    static STATE: ServerState = ServerState {
        data: async_std::sync::Mutex::new(StreamData::new()),
    };

    async_std::task::spawn(receive(&STATE, stream_receiver));

    let mut webapp = tide::with_state(&STATE);

    // Route configuration and queries.
    webapp.at("/traces").get(get_traces);
    webapp.at("/trigger").get(get_trigger).post(force_trigger);
    webapp.at("/capture").post(configure_capture);
    webapp.at("/").serve_file("frontend/dist/index.html").unwrap();
    webapp.at("/").serve_dir("frontend/dist").unwrap();

    // Start up the webapp.
    webapp.listen("127.0.0.1:8080").await?;

    Ok(())
}

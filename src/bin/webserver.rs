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

    max_size: usize,
    timebase: BufferedData<u64>,
    traces: Vec<BufferedTrace>,
}

struct BufferedData<T> {
    index: usize,
    data: Vec<T>,
}

impl<T: Clone + Default + Copy> BufferedData<T> {
    pub fn new(size: usize) -> Self {
        let mut storage = Vec::with_capacity(size);
        storage.resize(size, T::default());
        Self {
            data: storage,
            index: 0,
        }
    }

    pub fn overflowing_write(&mut self, mut data: &[T]) {
        // Continuously append data into the buffer in an overflowing manner (old data is
        // overwritten).
        while data.len() > 0 {
            let write_length = if data.len() > self.data.len() - self.index {
                self.data.len() - self.index
            } else {
                data.len()
            };

            self.data[self.index..][..write_length].copy_from_slice(&data[..write_length]);
            self.index = (self.index + write_length) % self.data.len();
            data = &data[write_length..];
        }
    }

    pub fn push(&mut self, data: T) {
        self.data[self.index] = data;
        self.index = (self.index + 1) % self.data.len();
    }

    pub fn oldest_data(&self) -> T {
        let index = if self.index == self.data.len() {
            0
        } else {
            self.index + 1
        };

        self.data[index]
    }

    pub fn resize(&mut self, size: usize) {
        self.index = 0;
        self.data.resize(size, T::default());
    }
}

#[derive(Serialize, Clone, Debug)]
struct Trace {
    label: String,
    data: Vec<f32>,
}

struct BufferedTrace {
    label: String,
    data: BufferedData<f32>,
}

impl From<&BufferedTrace> for Trace {
    fn from(bt: &BufferedTrace) -> Trace {
        Trace {
            label: bt.label.clone(),
            data: bt.data.data.clone(),
        }
    }
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
            timebase: BufferedData {
                data: Vec::new(),
                index: 0,
            },
            traces: Vec::new(),

            max_size: SAMPLE_RATE_HZ as usize,
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

        // Next, extract all of the data traces
        for i in 0..frame.data.trace_count() {
            if self.traces.len() < frame.data.trace_count() {
                self.traces.push(BufferedTrace {
                    data: BufferedData::new(self.max_size),
                    label: frame.data.trace_label(i),
                });
            }

            // TODO: Decimate the data as requested.
            let trace = frame.data.get_trace(i);
            self.traces[i].data.overflowing_write(trace);

            // For the first trace, also extend the timebase.
            if i == 0 {
                let base = (frame.sequence_number() as u64)
                    .wrapping_mul(frame.data.samples_per_batch() as u64);
                for sample_index in 0..trace.len() {
                    self.timebase.push(base.wrapping_add(sample_index as u64))
                }
            }
        }
    }

    /// Get the current trace data.
    pub fn get_data(&self) -> TraceData {
        let mut times: Vec<f32> = Vec::new();

        // Find the smallest sequence number in the timebase.
        let mut earliest_timestamp = self.timebase.oldest_data();
        for (i, time) in self.timebase.data.iter().enumerate() {
            let delta = earliest_timestamp.wrapping_sub(*time);
            if delta < u64::MAX / 4 {
                earliest_timestamp = *time;
            }
        }

        for time in self.timebase.data.iter() {
            times.push(time.wrapping_sub(earliest_timestamp) as f32 * SAMPLE_PERIOD)
        }

        let offset = self.timebase.index;

        let mut traces: Vec<Trace> = Vec::new();
        for trace in self.traces.iter() {
            let mut trace: Trace = trace.into();
            trace.data.rotate_left(offset);
            traces.push(trace)
        }

        times.rotate_left(offset);

        TraceData {
            time: times,
            traces,
        }
    }

    /// Remove all data from buffers.
    pub fn flush(&mut self) {
        log::info!("Flushing");
        self.traces.clear();
        self.current_format.take();
        self.timebase.resize(self.max_size)
    }

    pub fn resize(&mut self, max_samples: usize) {
        self.max_size = max_samples;
        self.flush();
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

    if config.capture_duration_secs < 0. {
        return Ok(Response::builder(400).body("Negative capture duration not supported"));
    }

    let samples: f32 = SAMPLE_RATE_HZ * config.capture_duration_secs;
    if samples > usize::MAX as f32 {
        return Ok(Response::builder(400).body("Too many samples requested"));
    }

    // Clear any pre-existing data in the buffers.
    let mut data = state.data.lock().await;

    // TODO: Configure decimation
    data.resize(samples as usize);

    Ok(Response::builder(200))
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

    STATE.data.lock().await.flush();

    async_std::task::spawn(receive(&STATE, stream_receiver));

    let mut webapp = tide::with_state(&STATE);

    // Route configuration and queries.
    webapp.at("/traces").get(get_traces);
    webapp.at("/capture").post(configure_capture);
    webapp
        .at("/")
        .serve_file("frontend/dist/index.html")
        .unwrap();
    webapp.at("/").serve_dir("frontend/dist").unwrap();

    // Start up the webapp.
    webapp.listen("127.0.0.1:8080").await?;

    Ok(())
}

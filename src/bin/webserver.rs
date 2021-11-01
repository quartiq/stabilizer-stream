use clap::Parser;
use serde::Serialize;
use stabilizer_streaming::{de::deserializer::StreamFrame, de::StreamFormat, StreamReceiver};
use std::collections::VecDeque;
use tide::{Body, Response};

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

struct StreamData {
    current_format: Option<StreamFormat>,

    max_size: usize,
    timebase: VecDeque<u64>,
    data: Vec<VecDeque<f32>>,
}

#[derive(Serialize, Debug)]
struct TraceData {
    time: Vec<f32>,
    data: Vec<Vec<f32>>,
}

impl StreamData {
    fn new() -> Self {
        Self {
            current_format: None,
            timebase: VecDeque::new(),
            data: Vec::new(),

            // TODO: Base this on the sample frequency.
            max_size: 1024,
        }
    }

    pub fn add_frame(&mut self, frame: StreamFrame) {
        // If the stream format has changed, clear all data buffers.
        if let Some(format) = self.current_format {
            if frame.format() != format {
                self.timebase.clear();
                self.data.clear();
                self.current_format.replace(frame.format());
            }
        } else {
            self.current_format.replace(frame.format());
        }

        // TODO: Determine whether or not we actually want to accept the current frame (e.g.
        // trigger state). We may just want to silently drop it at this point if we aren't armed.

        // Next, extract all of the data traces
        for i in 0..frame.data.trace_count() {
            if self.data.len() < frame.data.trace_count() {
                self.data.push(VecDeque::new());
            }

            // TODO: Decimate the data as requested.
            let trace = frame.data.get_trace(i);
            self.data[i].extend(trace);

            // For the first trace, also extend the timebase.
            if i == 0 {
                let base = (frame.sequence_number() as u64)
                    .wrapping_mul(frame.data.samples_per_batch() as u64);
                for sample_index in 0..trace.len() {
                    self.timebase
                        .push_back(base.wrapping_add(sample_index as u64))
                }
            }
        }

        // Drain the data/timebase queues to remain within our maximum size.
        if self.timebase.len() > self.max_size {
            let drain_size = self.timebase.len() - self.max_size;
            self.timebase.drain(0..drain_size);
            for trace in &mut self.data {
                trace.drain(0..drain_size);
            }
        }
    }

    pub fn get_data(&self) -> TraceData {
        let mut times: Vec<f32> = Vec::new();
        let time_offset = if self.timebase.len() > 0 {
            self.timebase[0]
        } else {
            0
        };

        for time in self.timebase.iter() {
            times.push(time.wrapping_sub(time_offset) as f32)
        }

        let mut data = Vec::new();
        for trace in self.data.iter() {
            let mut vec = Vec::new();
            let (front, back) = trace.as_slices();
            vec.extend_from_slice(front);
            vec.extend_from_slice(back);
            data.push(vec);
        }

        TraceData { time: times, data }
    }
}

struct TriggerState;

struct ServerState {
    trigger: TriggerState,

    // StreamData cannot implement a const-fn constructor, so we wrap it in an option instead.
    pub data: async_std::sync::Mutex<Option<StreamData>>,
}

static STATE: ServerState = ServerState {
    trigger: TriggerState {},
    data: async_std::sync::Mutex::new(None),
};

async fn receive(state: &ServerState, mut receiver: StreamReceiver) {
    loop {
        // Get a stream frame from Stabilizer.
        let frame = receiver.next_frame().await.unwrap();

        // TODO: Loop until we acquire mutex.
        // Add the frame data to the traces.
        let mut data = state.data.lock().await;

        data.as_mut().unwrap().add_frame(frame);
        drop(data);
    }
}

async fn get_traces(request: tide::Request<&ServerState>) -> tide::Result<impl Into<Response>> {
    log::info!("Got data request");
    let state = request.state();
    let data = state.data.lock().await;
    let response = data.as_ref().unwrap().get_data();
    log::debug!("Response: {:?}", response);
    Ok(Response::builder(200).body(Body::from_json(&response)?))
}

#[async_std::main]
async fn main() -> tide::Result<()> {
    env_logger::init();

    let opts = Opts::parse();
    let ip: std::net::Ipv4Addr = opts.ip.parse().unwrap();
    let stream_receiver = StreamReceiver::new(ip, opts.port).await;

    // Populate the initial receiver data.
    {
        let mut stream_data = STATE.data.lock().await;
        stream_data.replace(StreamData::new());
    }

    let child = async_std::task::spawn(receive(&STATE, stream_receiver));

    let mut webapp = tide::with_state(&STATE);
    webapp.at("/data").get(get_traces);
    webapp.at("/").get(|_| async { Ok("Hello World!") });

    webapp.listen("127.0.0.1:8080").await?;

    Ok(())
}

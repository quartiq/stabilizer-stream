use crate::{Frame, Loss};
use anyhow::Result;
use clap::Parser;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use socket2::{Domain, Protocol, Socket, Type};
use std::{
    fs::File,
    io::{BufReader, ErrorKind, Read, Seek},
    net::{Ipv4Addr, SocketAddr},
    time::Duration,
};

/// Stabilizer stream source options
#[derive(Parser, Debug, Clone)]
pub struct SourceOpts {
    /// The local IP to receive streaming data on.
    #[arg(short, long, default_value = "0.0.0.0")]
    ip: std::net::Ipv4Addr,

    /// The UDP port to receive streaming data on.
    #[arg(short, long, default_value_t = 9293)]
    port: u16,

    /// Use frames from the given file
    #[arg(short, long)]
    file: Option<String>,

    /// Frame size in file (8 + n_batches*n_channel*batch_size)
    #[arg(short, long, default_value_t = 8 + 30 * 2 * 6 * 4)]
    frame_size: usize,

    /// On a file, wrap around and repeat
    #[arg(short, long)]
    repeat: bool,

    /// Single le f32 raw trace, architecture dependent endianess
    #[arg(short, long)]
    single: Option<String>,

    /// Power law noise with psd f^noise.
    #[arg(short, long)]
    noise: Option<i32>,
}

#[derive(Debug)]
enum Data {
    Udp(Socket),
    File(BufReader<File>),
    Single(BufReader<File>),
    Noise((SmallRng, bool, Vec<f32>)),
}

pub struct Source {
    opts: SourceOpts,
    data: Data,
    loss: Loss,
}

impl Source {
    pub fn new(opts: SourceOpts) -> Result<Self> {
        let data = if let Some(noise) = opts.noise {
            Data::Noise((
                SmallRng::seed_from_u64(0x7654321),
                noise > 0,
                vec![0.0; noise.unsigned_abs() as _],
            ))
        } else if let Some(file) = &opts.file {
            Data::File(BufReader::with_capacity(1 << 20, File::open(file)?))
        } else if let Some(single) = &opts.single {
            Data::Single(BufReader::with_capacity(1 << 20, File::open(single)?))
        } else {
            log::info!("Binding to {}:{}", opts.ip, opts.port);
            let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
            socket.set_read_timeout(Some(Duration::from_millis(1000)))?;
            socket.set_recv_buffer_size(1 << 20)?;
            socket.set_reuse_address(true)?;
            if opts.ip.is_multicast() {
                socket.join_multicast_v4(&opts.ip, &Ipv4Addr::UNSPECIFIED)?;
            }
            #[cfg(windows)]
            let bind = Ipv4Addr::UNSPECIFIED;
            #[cfg(not(windows))]
            let bind = opts.ip;
            socket.bind(&SocketAddr::new(bind.into(), opts.port).into())?;
            Data::Udp(socket)
        };
        Ok(Self {
            opts,
            data,
            loss: Loss::default(),
        })
    }

    pub fn get(&mut self) -> Result<Vec<Vec<f32>>> {
        Ok(match &mut self.data {
            Data::Noise((rng, diff, state)) => {
                vec![rng
                    .sample_iter(rand::distributions::Open01)
                    .map(|mut x| {
                        x = (x - 0.5) * 12.0f32.sqrt(); // zero mean, RMS = 1
                        state.iter_mut().fold(x, |mut x, s| {
                            (x, *s) = if *diff { (x - *s, x) } else { (*s, x + *s) };
                            x
                        })
                    })
                    .take(4096)
                    .collect()]
            }
            Data::File(fil) => loop {
                let mut buf = [0u8; 2048];
                match fil.read_exact(&mut buf[..self.opts.frame_size]) {
                    Ok(()) => {
                        let frame = Frame::from_bytes(&buf[..self.opts.frame_size])?;
                        self.loss.update(&frame);
                        break frame.data.traces().into();
                    }
                    Err(e) if e.kind() == ErrorKind::UnexpectedEof && self.opts.repeat => {
                        fil.seek(std::io::SeekFrom::Start(0))?;
                    }
                    Err(e) => Err(e)?,
                }
            },
            Data::Single(fil) => loop {
                let mut buf = [0u8; 2048];
                match fil.read(&mut buf[..]) {
                    Ok(len) => {
                        if len == 0 && self.opts.repeat {
                            fil.seek(std::io::SeekFrom::Start(0))?;
                            continue;
                        }
                        let v: &[[u8; 4]] = bytemuck::cast_slice(&buf[..len / 4 * 4]);
                        break vec![v.iter().map(|b| f32::from_le_bytes(*b)).collect()];
                    }
                    Err(e) => Err(e)?,
                }
            },
            Data::Udp(socket) => {
                let mut buf = [0u8; 2048];
                let len = socket.recv(unsafe { core::mem::transmute(&mut buf[..]) })?; // meh
                let frame = Frame::from_bytes(&buf[..len])?;
                self.loss.update(&frame);
                frame.data.traces().into()
            }
        })
    }

    pub fn finish(&self) {
        self.loss.analyze()
    }
}

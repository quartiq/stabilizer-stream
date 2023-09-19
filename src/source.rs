use crate::{Frame, Loss};
use anyhow::Result;
use clap::Parser;
use std::io::ErrorKind;
use std::time::Duration;
use std::{
    fs::File,
    io::{BufReader, Read, Seek},
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

    /// Single f32 raw trace in file, architecture dependent
    #[arg(short, long)]
    single: Option<String>,
}

#[derive(Debug)]
enum Data {
    Udp(std::net::UdpSocket),
    File(BufReader<File>),
    Single(BufReader<File>),
}

pub struct Source {
    opts: SourceOpts,
    data: Data,
    loss: Loss,
}

impl Source {
    pub fn new(opts: SourceOpts) -> Result<Self> {
        let data = if let Some(file) = &opts.file {
            Data::File(BufReader::with_capacity(1 << 20, File::open(file)?))
        } else if let Some(single) = &opts.single {
            Data::Single(BufReader::with_capacity(1 << 20, File::open(single)?))
        } else {
            log::info!("Binding to {}:{}", opts.ip, opts.port);
            let socket = std::net::UdpSocket::bind((opts.ip, opts.port))?;
            socket2::SockRef::from(&socket).set_recv_buffer_size(1 << 20)?;
            socket.set_read_timeout(Some(Duration::from_millis(1000)))?;
            Data::Udp(socket)
        };
        Ok(Self {
            opts,
            data,
            loss: Loss::default(),
        })
    }

    pub fn get(&mut self) -> Result<Vec<Vec<f32>>> {
        let mut buf = [0u8; 2048];
        Ok(match &mut self.data {
            Data::File(fil) => loop {
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
                let len = socket.recv(&mut buf[..])?;
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

# stabilizer-stream

Host-side stream utilities for interacting with Stabilizer's data stream

## PSD

Graphical frontend to real-time cascaded power-spectral density (PSD) measurements.

* Low-latency online PSD monitoring
* Arbitrarily large effective FFT sizes
* More than 200 MS/s on a single CPU core
* Supports [`stabilizer`](https://github.com/quartiq/stabilizer)
  `dual-iir`/`lockin`/`fls` formats as well as device-independent raw streams

See the following real time video of a 200 MS/s stream being analyzed (4.8e9 samples in 24 seconds):

https://github.com/quartiq/stabilizer-streaming/assets/1338946/14333e17-61ef-4ca2-a5d2-c314b70714ad

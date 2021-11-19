#!/usr/bin/python3
"""
Author: Vertigo Designs, Ryan Summers

Description: Bokeh application for serving Stabilizer stream visuals.
"""
import argparse
import asyncio
import enum
import logging
import socket
from typing import List, Mapping

import tornado

import bokeh.plotting
import bokeh.layouts
import bokeh.document
import bokeh.palettes
import bokeh.io
import bokeh.models
import bokeh.server.server
import numpy as np

from miniconf import Miniconf
from stabilizer.stream import StabilizerStream, Trace

# The sample rate of stabilizer.
SAMPLE_RATE_HZ = 100e6 / 128


def _get_ip(broker):
    """ Get the IP of the local device.

    Args:
        broker: The broker IP of the test. Used to select an interface to get the IP of.

    Returns:
        The IP as a string.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect((broker, 1883))
        return sock.getsockname()[0]


class TriggerState(enum.Enum):
    """ The state of the scope trigger. """
    IDLE = enum.auto()
    ARMED = enum.auto()
    TRIGGERED = enum.auto()
    STOPPED = enum.auto()


class StreamReceiver:
    """ Handles asynchronously receiving Stabilizer stream frames. """

    @classmethod
    async def start(cls, remote):
        """ Start receiving data on the stabilizer stream. """
        transport, stream = await StabilizerStream.open(remote)
        return cls(stream)


    def __init__(self, stream):
        """ Initialize the receiver on the provided remote interface. """
        self.traces = dict()
        self.trace_index = 0
        self.times = None
        self._current_format = None
        self.trigger = TriggerState.IDLE
        self.max_size = int(SAMPLE_RATE_HZ)
        self.capture_future = None

        # Add our ingress task to the async event loop to process incoming frames.
        loop = asyncio.get_event_loop()
        loop.create_task(self.process(stream))


    def force_trigger(self):
        """ Force the trigger state. """
        self.trigger = TriggerState.TRIGGERED


    def capture(self, duration: float) -> asyncio.Future:
        """ Configure a capture for the specified number of seconds. """
        self.max_size = int(SAMPLE_RATE_HZ * duration) + 1
        self.trace_index = 0
        self.times = np.empty(self.max_size, dtype=np.uint32)
        self.traces = dict()
        logging.info('Starting capture for %f seconds (%d samples) - arming trigger',
                     duration, self.max_size)
        self.trigger = TriggerState.ARMED

        self.capture_future = asyncio.Future()

        return self.capture_future


    async def process(self, stream):
        """ Async task to continuously capture frames. """
        while True:
            self._ingest_frame(await stream.queue.get())


    def _append_traces(self, frame):
        # Next, put the trace data into our trace buffers.
        for trace in frame.to_traces():
            if trace.label not in self.traces:
                self.traces[trace.label] = Trace(scale=trace.scale,
                                                 label=trace.label,
                                                 values=np.empty(self.max_size))

            num_elements = len(trace.values)

            if self.trace_index + num_elements > len(self.traces[trace.label].values):
                num_elements = len(self.traces[trace.label].values) - self.trace_index

            self.traces[trace.label].values[
                self.trace_index:self.trace_index+num_elements] = trace.values[:num_elements]

        # Extend the timebase
        sample_index = frame.header.sequence * frame.header.batch_size
        self.times[self.trace_index:self.trace_index+num_elements] = \
            [np.uint32(sample_index + offset & 0xFFFFFFFF) for offset in range(num_elements)]

        self.trace_index += num_elements


    def _ingest_frame(self, frame):
        """ Ingest a single livestream frame. """
        # Ingest stream frames.
        if self.trigger is not TriggerState.TRIGGERED:
            return

        if frame.format_id != self._current_format:
            self.traces = dict()
            self.times = np.empty(self.max_size, dtype=np.uint32)
            self._current_format = frame.format_id

        # Append trace data
        self._append_traces(frame)

        # Check for the capture buffer being filled.
        if self.trace_index >= self.max_size:
            logging.info('Got %d samples - stopping trigger', self.max_size)
            self.trigger = TriggerState.STOPPED
            self.capture_future.set_result(self.get_traces())


    def get_traces(self):
        """ Get the traces and times to display in the graph. """
        return (self.times - self.times[0]), self.traces


class StreamVisualizer:
    """ A visualizer for stabilizer's livestream data. """

    def __init__(self, receiver: StreamReceiver, document: bokeh.document.Document):
        """ Initialize the visualizer.

        Args:
            receiver: The StreamReceiver object for managing stream reception.
            document: The Bokeh document to draw into.
        """

        self.receiver = receiver
        self.figure = None

        self._data_store = bokeh.models.ColumnDataSource(data={
            'time': []
        })

        self.recreate_figure([])

        # Add a trigger button
        trigger_button = bokeh.models.Button(label='Single', button_type='primary')

        trigger_button.on_click(lambda: tornado.ioloop.IOLoop.current().add_future(self.start_capture(),
                                                                   self.finish_capture))

        force_button = bokeh.models.Button(label='Force', button_type='primary')
        force_button.on_click(self.receiver.force_trigger)

        self._capture_duration_input = bokeh.models.TextInput(title='Capture Duration',
                                                              value='0.001',
                                                              width=100,
                                                              sizing_mode='fixed')
        self._capture_duration_input.on_change('value', self.handle_duration)

        # TODO: Register trigger state changes.
        self._trigger_state = bokeh.models.Div(text='Trigger State: <b>IDLE</b>')

        control_layout = bokeh.layouts.column(self._trigger_state, trigger_button, force_button,
                                              self._capture_duration_input)

        self.layout = bokeh.layouts.row(self.figure, control_layout, sizing_mode='stretch_height')

        document.add_root(self.layout)

        self.doc = document


    def recreate_figure(self, trace_names):
        self.figure = bokeh.plotting.figure(output_backend="webgl", sizing_mode='stretch_both')
        self.figure.legend.location = 'top_left'
        self.figure.legend.click_policy = 'hide'
        self.figure.x_range.range_padding = 0
        self.figure.x_range.on_change('start', self.handle_resize)
        self.figure.x_range.on_change('end', self.handle_resize)

        if not trace_names:
            return

        # Update traces
        palette = bokeh.palettes.d3['Category10'][len(trace_names)]
        for (trace, color) in zip(trace_names, palette):
            self.figure.circle(x='time', y=trace, source=self._data_store, color=color,
                               legend_label=trace)

        self.layout.children[0] = self.figure


    def update_trigger_state(self, trigger_state):
        """ Update the trigger state display. """
        self._trigger_state.text = f'Trigger State: <b>{trigger_state.value.upper()}</b>'


    def handle_resize(self, _attr, _old, _new):
        start = self.figure.x_range.start
        end = self.figure.x_range.end
        logging.info('%s %s %s, %s %s', _attr, _old, _new, start, end)

        if start is not None and end is not None:
            self._capture_duration_input.value = str(float(end - start))


    def handle_duration(self, _attr, old_value, new_value):
        """ Handle updates to the capture duration input. """
        try:
            self._capture_duration_input.value = str(float(new_value))
        except ValueError:
            self._capture_duration_input.value = str(float(old_value))


    def _redraw(self, times: List[float], traces: Mapping[str, Trace]):
        """ Redraw plots.

        # Args
            times: The list of timestamps of trace points.
            traces: A dictionary of all traces to display.
        """
        # If the traces have changed, we need to re-generate the figure.
        trace_diff = set(self._data_store.column_names).symmetric_difference(set(traces.keys()))
        if trace_diff != set(['time']):
            self._data_store.data = dict()
            self.recreate_figure(list(traces.keys()))

        # Update the data store atomically
        new_data = {
            'time': times / SAMPLE_RATE_HZ,
        }

        for trace in traces.values():
            new_data[trace.label] = trace.values * trace.scale

        self._data_store.data = new_data
        self.figure.x_range.start = min(new_data['time'])
        self.figure.x_range.end = max(new_data['time'])
        logging.info('Update complete')


    def start_capture(self, duration: float = None) -> asyncio.Future:
        """ Initialize a data capture from Stabilizer's datastream.

        Args:
            duration: The duration to capture for.

        Returns:
            A future for when the capture completes.
        """
        if duration is None:
            duration = float(self._capture_duration_input.value)

        future = self.receiver.capture(duration)

        # TODO: Force the trigger for now. Once we allow triggered waveforms, this force will be
        # removed.
        self.receiver.force_trigger()

        return future


    @bokeh.document.without_document_lock
    def finish_capture(self, future: asyncio.Future):
        """ Finish the capture and redraw the display. """
        times, traces = future.result()
        self.doc.add_next_tick_callback(lambda: self._redraw(times, traces))


async def configure_streaming(prefix, broker, port, local_ip=None):
    """ Set up Stabilizer streamining to the desired endpoint.

    Args:
        broker: The IP address of the broker.
        prefix: The miniconf prefix of Stabilizer.
        port: The port to direct the stream to.
        local_ip: The destination IP of the stream. If not specified, local IP will be automatically
            determined.
    """
    if not local_ip:
        local_ip = list(map(int, _get_ip(broker).split('.')))

    interface = await Miniconf.create(prefix, broker)

    # Configure the stream
    logging.debug(f'Configuring stream to target {".".join(map(str, local_ip))}:{port}')
    await interface.command('stream_target', {'ip': local_ip, 'port': port}, retain=False)


def main():
    """ Main entry point. """
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description='Stabilizer livestream viewer')
    parser.add_argument('--prefix', type=str, help='The MQTT topic prefix of the target. '
                                                   'If provided, streaming will be configured')
    parser.add_argument('--broker', '-b', default='mqtt', type=str,
                        help='The MQTT broker address')
    parser.add_argument('--port', '-p', default=2000, type=int,
                        help='The UDP port to use for streaming')
    args = parser.parse_args()

    async def stream_visualizer():
        ip = _get_ip(args.broker)
        receiver = await StreamReceiver.start((ip, args.port))

        logging.info('Starting livestream view server on http://localhost:5006')
        server = bokeh.server.server.Server(lambda document: StreamVisualizer(receiver, document),
                                            io_loop=tornado.ioloop.IOLoop.current())
        server.start()

    async def configure_stream():
        await configure_streaming(args.prefix, args.broker, args.port)

    async def unconfigure_stream():
        await configure_streaming(args.prefix, args.broker, 0, [0, 0, 0, 0])

    # Run the visualizer
    loop = tornado.ioloop.IOLoop.current()
    try:
        if args.prefix:
            loop.run_sync(configure_stream)

        loop.add_callback(stream_visualizer)
        loop.start()
    finally:
        if args.prefix:
            loop.run_sync(unconfigure_stream)


if __name__ == '__main__':
    main()

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

import bokeh.plotting
import bokeh.layouts
import bokeh.document
import bokeh.palettes
import bokeh.io
import bokeh.models
import bokeh.server.server
import numpy as np

from miniconf import Miniconf
from stabilizer.stream import StabilizerStream

# The sample rate of stabilizer.
SAMPLE_RATE_HZ = 100e6 / 128


def _get_ip(broker):
    """ Get the IP of the local device.

    Args:
        broker: The broker IP of the test. Used to select an interface to get the IP of.

    Returns:
        The IP as an array of integers.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((broker, 1883))
        address = sock.getsockname()[0]
    finally:
        sock.close()

    return list(map(int, address.split('.')))


class TriggerState(enum.Enum):
    """ The state of the scope trigger. """
    IDLE = enum.auto()
    ARMED = enum.auto()
    TRIGGERED = enum.auto()
    STOPPED = enum.auto()


class StreamReceiver:
    """ Handles asynchronously receiving Stabilizer stream frames. """

    def __init__(self, port):
        """ Initialize the receiver on the provided port. """
        self.traces = dict()
        self.trace_index = 0
        self.times = []
        self._current_format = None

        self.trigger = TriggerState.IDLE

        self._stream = StabilizerStream(port)
        self.max_size = int(SAMPLE_RATE_HZ)

        # Add our ingress task to the async event loop to process incoming frames.
        loop = asyncio.get_event_loop()
        loop.create_task(self.process())


    def force_trigger(self):
        """ Force the trigger state. """
        self.trigger = TriggerState.TRIGGERED


    def capture(self, duration: float):
        """ Configure a capture for the specified number of seconds. """
        self.max_size = int(SAMPLE_RATE_HZ * duration)
        self.trace_index = 0
        self.times = []
        self.traces = dict()
        logging.info('Starting capture for %f seconds (%d samples) - arming trigger',
                     duration, self.max_size)
        self.trigger = TriggerState.ARMED


    async def process(self):
        """ Async task to continuously capture frames. """
        while True:
            await self._update()

            # Cooperatively give up execution if we're not capturing data.
            if self.trigger is not TriggerState.TRIGGERED:
                await asyncio.sleep(0)


    async def _update(self):
        """ Ingest a single livestream frame. """
        # Ingest stream frames.
        frame = await self._stream.next_frame()

        if self.trigger is not TriggerState.TRIGGERED:
            return

        if frame.format != self._current_format:
            self.traces = dict()
            self.times = []
            self._current_format = frame.format


        # Append trace data
        for trace, data in frame.traces.items():
            if trace not in self.traces:
                self.traces[trace] = np.empty(self.max_size)

            num_elements = len(data)

            if self.trace_index + num_elements > len(self.traces[trace]):
                num_elements = len(self.traces[trace]) - self.trace_index

            self.traces[trace][self.trace_index:self.trace_index+num_elements] = data[:num_elements]

        self.trace_index += num_elements

        # Extend the timebase
        self.times += [frame.sequence_number + offset for offset in range(num_elements)]

        # Drain the traces to be defined by max size.
        if self.trace_index >= self.max_size:
            logging.info('Got %d samples - stopping trigger', self.max_size)
            self.times = self.times[:self.max_size]

            self.trigger = TriggerState.STOPPED


    def get_traces(self):
        """ Get the traces and times to display in the graph. """
        # Convert the times
        times = []
        start_time = self.times[0]
        for time in self.times:
            times.append((time - start_time) / SAMPLE_RATE_HZ)

        return times, self.traces


class StreamVisualizer:
    """ A visualizer for stabilizer's livestream data. """

    def __init__(self, port: int, document: bokeh.document.Document):
        """ Initialize the visualizer.

        Args:
            port: The UDP port to connect to
            document: The Bokeh document to draw into.
        """
        self.receiver = StreamReceiver(port)

        figure = bokeh.plotting.figure(output_backend="webgl", sizing_mode='stretch_both')

        # Add a trigger button
        trigger_button = bokeh.models.Button(label='Single', button_type='primary')
        trigger_button.on_click(self.capture)

        force_button = bokeh.models.Button(label='Force', button_type='primary')
        force_button.on_click(self.receiver.force_trigger)

        self._capture_duration_input = bokeh.models.TextInput(title='Capture Duration',
                                                              value='0.001',
                                                              width=100,
                                                              sizing_mode='fixed')
        self._capture_duration_input.on_change('value', self.handle_duration)
        self._trigger_state = bokeh.models.Div(text='Trigger State: <b>IDLE</b>')

        control_layout = bokeh.layouts.column(self._trigger_state, trigger_button, force_button,
                                              self._capture_duration_input)

        self.layout = bokeh.layouts.row(figure, control_layout, sizing_mode='stretch_height')

        document.theme = 'dark_minimal'
        document.add_root(self.layout)

        self.doc = document
        self._callback = None

        self._data_store = bokeh.models.ColumnDataSource(data={
            'time': []
        })


    def handle_duration(self, _attr, old_value, new_value):
        """ Handle updates to the capture duration input. """
        try:
            self._capture_duration_input.value = str(float(new_value))
        except ValueError:
            self._capture_duration_input.value = str(float(old_value))


    def update(self):
        """ Periodic check for capture status. """
        trigger = self.receiver.trigger
        self._trigger_state.text = f'Trigger State: <b>{trigger.name.upper()}</b>'

        if trigger is TriggerState.STOPPED:
            times, traces = self.receiver.get_traces()
            self._redraw(time=times, traces=traces)

            # Disable the document callback now that we have redrawn the plot.
            if self._callback is not None:
                self.doc.remove_periodic_callback(self._callback)
                self._callback = None


    def _redraw(self, time: List[float], traces: Mapping[str, List[int]]):
        """ Redraw plots.

        # Args
            time: The list of timestamps of trace points.
            traces: A dictionary of all traces to display.
        """
        # Update the data store atomically
        new_datastore = {
            'time': time,
        }

        for trace, data in traces.items():
            new_datastore[trace] = data

        self._data_store = new_datastore

        figure = bokeh.plotting.figure(output_backend="webgl", sizing_mode="stretch_both")

        # Update traces
        palette = bokeh.palettes.d3['Category10'][len(traces.keys())]
        for (trace, color) in zip(traces, palette):
            figure.circle(x='time', y=trace, source=self._data_store, color=color,
                          legend_label=trace)

        figure.legend.location = 'top_left'
        figure.legend.click_policy = 'hide'

        self.layout.children[0] = figure


    def capture(self, duration: float = None):
        """ Initialize a data capture from Stabilizer's datastream.

        Args:
            duration: The duration to capture for.
        """
        if duration is None:
            duration = float(self._capture_duration_input.value)

        self.receiver.capture(duration)

        # TODO: Force the trigger for now. Once we allow triggered waveforms, this force will be
        # removed.
        self.receiver.force_trigger()

        # Start periodicly reading trigger state.
        self._callback = self.doc.add_periodic_callback(self.update,
                                                        period_milliseconds=1000/60)


async def configure_streaming(prefix, broker, port):
    """ Set up Stabilizer streamining to the desired endpoint.

    Args:
        port: The port to direct the stream to.
        broker: The IP address of the broker.
        prefix: The miniconf prefix of Stabilizer.
    """
    local_ip = _get_ip(broker)
    interface = await Miniconf.create(prefix, broker)

    # Configure the stream
    print(f'Configuring stream to target {".".join(map(str, local_ip))}:{port}')
    print('')
    await interface.command('stream_target', {'ip': local_ip, 'port': port}, retain=False)


def main():
    """ Main entry point. """
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Stabilizer livestream viewer')
    parser.add_argument('--prefix', type=str, help='The MQTT topic prefix of the target. '
                                                   'If provided, streaming will be configured')
    parser.add_argument('--broker', '-b', default='mqtt', type=str,
                        help='The MQTT broker address')
    parser.add_argument('--port', '-p', default=2000, type=int,
                        help='The UDP port to use for streaming')
    args = parser.parse_args()

    if args.prefix:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(configure_streaming(args.prefix, args.broker, args.port))

    logging.info('Starting livestream view server on http://localhost:5006')
    server = bokeh.server.server.Server(lambda document: StreamVisualizer(args.port, document))
    server.start()
    server.run_until_shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/python3
"""
Author: Vertigo Designs, Ryan Summers

Description: Bokeh application for serving Stabilizer stream visuals.
"""
import bokeh.plotting
import bokeh.layouts
import bokeh.document
import bokeh.palettes
import bokeh.io
import bokeh.models
import requests
import logging
import json
from typing import List

DEFAULT_SERVER = 'http://127.0.0.1:8080'


class ReceiverApi:

    def __init__(self, server: str):
        self.server = server


    def get_json(self, path: str) -> dict:
        response = requests.get(self.server + path)
        assert response.ok, f'GET {path} failed: {response.text}'
        return response.json()


    def post_json(self, path: str, payload: dict = None):
        if payload is None:
            payload = dict()

        response = requests.post(self.server + path, data=json.dumps(payload))
        assert response.ok, f'POST {path} ({payload}) failed: {response.text}'


    def get_trigger(self) -> str:
        return self.get_json('/trigger')


    def start_capture(self, duration: float):
        request = {
            'capture_duration_secs': duration,
        }

        self.post_json('/capture', request)


class StreamVisualizer:

    def __init__(self, document: bokeh.document.Document, server: str = DEFAULT_SERVER):
        figure = bokeh.plotting.figure(output_backend="webgl", sizing_mode='stretch_both')

        # Add a trigger button
        trigger_button = bokeh.models.Button(label='Single', button_type='primary')
        trigger_button.on_click(self.capture)

        force_button = bokeh.models.Button(label='Force', button_type='primary')
        force_button.on_click(lambda: self.api.post_json('/trigger'))

        self._capture_duration_input = bokeh.models.TextInput(title='Capture Duration',
                value='0.001', width=100, sizing_mode='fixed')
        self._capture_duration_input.on_change('value', self.handle_duration)
        self._trigger_state = bokeh.models.Div(text='Trigger State: <b>IDLE</b>')

        control_layout = bokeh.layouts.column(self._trigger_state, trigger_button, force_button,
                self._capture_duration_input)

        self.layout = bokeh.layouts.row(figure, control_layout, sizing_mode='stretch_height')
        document.add_root(self.layout)

        # TODO: Add a capture duration input

        self.doc = document
        self._callback = None

        self.api = ReceiverApi(server)
        self._data_store = bokeh.models.ColumnDataSource(data={
            'time': []
        })


    def handle_duration(self, _attr, old_value, new_value):
        try:
            self._capture_duration_input.value = str(float(new_value))
        except ValueError:
            self._capture_duration_input.value = str(float(old_value))
            pass


    def update(self):
        trigger = self.api.get_json('/trigger')
        self._trigger_state.text = f'Trigger State: <b>{trigger.upper()}</b>'
        if trigger == "Stopped":
            trace_data = self.api.get_json('/traces')
            self._redraw(**trace_data)

            # Disable the document callback now that we have redrawn the plot.
            if self._callback is not None:
                self.doc.remove_periodic_callback(self._callback)
                self._callback = None


    def _redraw(self, time: List[float], traces: List[dict]):
        # Update the data store atomically
        new_datastore = {
            'time': time,
        }

        for trace in traces:
            new_datastore[trace['label']] = trace['data']

        self._data_store = new_datastore

        figure = bokeh.plotting.figure(output_backend="webgl", sizing_mode="stretch_both")

        # Update traces
        palette = bokeh.palettes.d3['Category10'][len(traces)]
        for (trace, color) in zip(traces, palette):
            figure.circle(x='time', y=trace['label'], source=self._data_store, color=color,
                               legend_label=trace['label'])

        figure.legend.location = 'top_left'
        figure.legend.click_policy = 'hide'

        self.layout.children[0] = figure


    def capture(self, duration: float = None):
        if duration is None:
            duration = float(self._capture_duration_input.value)

        self.api.start_capture(duration)

        # Start periodicly reading trigger state.
        self._callback = self.doc.add_periodic_callback(self.update,
                                                        period_milliseconds=100)


def main():
    logging.info('Startup')

    document = bokeh.io.curdoc()
    document.theme = 'dark_minimal'

    visualizer = StreamVisualizer(document)

    # Debug: Force a trigger
    visualizer.capture(0.001)
    visualizer.api.post_json('/trigger')


main()

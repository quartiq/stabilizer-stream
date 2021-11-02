#!/usr/bin/python3
"""
Author: Vertigo Designs, Ryan Summers

Description: Bokeh application for serving Stabilizer stream visuals.
"""
import bokeh.plotting
import bokeh.layouts
import bokeh.document
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
        self.figure = bokeh.plotting.figure()

        # Add a trigger button
        self._trigger_button = bokeh.models.Button(label='Capture', button_type='primary')
        self._trigger_button.on_click(self.capture)

        document.add_root(bokeh.layouts.row(self.figure, self._trigger_button))

        # TODO: Add a capture duration input

        self.doc = document
        self._callback = None

        self.api = ReceiverApi(server)
        self._data_store = bokeh.models.ColumnDataSource(data={
            'time': []
        })


    def update(self):
        if self.api.get_json('/trigger') == "Stopped":
            trace_data = self.api.get_json('/traces')
            self._redraw(**trace_data)

            # Disable the document callback now that we have redrawn the plot.
            if self._callback is not None:
                self.doc.remove_periodic_callback(self._callback)
                self._callback = None


    def _redraw(self, time: List[float], traces: List[dict]):
        # Remove any existing trace data from the store.
        for key in self._data_store.data:
            # TODO: Do we need to remove the plot from the figure?
            if key != 'time':
                del self._data_store.data[key]

        # Update the timebase
        self._data_store.data['time'] = time

        # Update traces
        for trace in traces:
            self._data_store.data[trace['label']] = trace['data']
            self.figure.circle(x='time', y=trace['label'], source=self._data_store)


    def capture(self, duration: float = None):
        if duration is None:
            # TODO: If the duration is not explicitly provided, get it from the UI.
            duration = 1

        self.api.start_capture(duration)

        # Start periodicly reading trigger state.
        self._callback = self.doc.add_periodic_callback(self.update,
                                                        period_milliseconds=100)


def main():
    logging.info('Startup')

    document = bokeh.io.curdoc()

    visualizer = StreamVisualizer(document)

    # Debug: Force a trigger
    visualizer.capture(1.0)
    visualizer.api.post_json('/trigger')


main()

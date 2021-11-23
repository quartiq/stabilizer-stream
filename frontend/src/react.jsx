import "bootstrap";
import "bootstrap/dist/css/bootstrap.css";

import CandyGraph, {
    createCartesianCoordinateSystem,
    createLinearScale,
    createLineStrip,
    createDefaultFont,
    createOrthoAxis,
    createText,
} from "candygraph";

import ReactDOM from 'react-dom';
import React from "react";

class Trigger extends React.Component {
    constructor(props) {
        super(props)
        this.onChange = this.onChange.bind(this)
        this.state = {
            capture_duration: 0.001,
            timer: null,
        }
        this.capturing = false

        this.setCaptureDuration(this.state.capture_duration)
    }

    getTraces() {
        if (this.capturing) {
            return;
        }

        this.capturing = true;

        fetch('http://localhost:8080/traces').then(response => {
            if (response.ok) {
                return response.json();
            } else {
                return Promise.reject(`Data request failed: ${response.text()}`)
            }
        }).then(data => {
            this.props.onData(data.time, data.traces)
            data = null
            this.capturing = false;
        })
    }

    onChange(evt) {
        const duration = Number(evt.target.value)
        this.setState({capture_duration: duration})
        this.setCaptureDuration(duration)
    }

    setCaptureDuration(duration) {
        const postData = JSON.stringify({
            capture_duration_secs: duration
        });

        fetch('http://localhost:8080/capture', {method: 'POST', body: postData})
            .then(response => {
                if (!response.ok) {
                    console.log(`Config error: ${error}`)
                }
            })
    }

    toggleCapture() {
        if (this.state.timer == null) {
            this.setState({timer: setInterval(() => {this.getTraces()}, 1000/30)})
        } else {
            clearInterval(this.state.timer)
            this.setState({timer: null})
        }
    }

    render() {
        return (
          <div className="trigger">
            <div className="input-group input-group-sm mb-3">
              <span className="input-group-text">Duration</span>
              <input className="form-control" type="number" value={this.state.capture_duration} onChange={this.onChange} />
            </div>

            <button className="btn btn-outline-primary" onClick={() => this.toggleCapture()}> {(this.state.timer != null)? "Stop" : "Run"} </button>
            <button className="btn btn-outline-primary" onClick={() => this.getTraces()}> Capture </button>
          </div>
        );
    }
}

class Oscilloscope extends React.Component {
    constructor(props) {
        super(props);
        this.cg = new CandyGraph()
        this.font = null
        this.canvas = null

        createDefaultFont(this.cg).then(font => {
            this.font = font
            this.drawGraph([0, 1], [])
        })
    }

    componentDidMount() {
        const width = this.divElement.clientWidth
        this.canvas = document.getElementById("oscilloscope-display")
        this.canvas.width = width
    }

    drawGraph(times, traces) {
        if (this.font == null || this.canvas == null) {
            return;
        }

        this.cg.canvas.width = this.canvas.width
        this.cg.canvas.height = this.canvas.height

        const viewport = {
            x: 0,
            y: 0,
            width: this.canvas.width,
            height: this.canvas.height,
        }

        this.cg.clear([1, 1, 1, 1])

        const max_time = Math.max(...times)

        const coords = createCartesianCoordinateSystem(
            createLinearScale([0, max_time], [32, viewport.width - 16]),
            createLinearScale([-10.24, 10.24], [32, viewport.height - 16]),
        );

        // Create the various traces for the display
        const colors = [
            [1, 0, 0, 1.0],
            [0, 1, 0, 1.0],
            [0, 0, 1, 1.0],
            [1, 0, 1, 1.0],
            [1, 1, 0, 1.0],
            [1, 1, 1, 1.0],
        ]
        var lines = []
        for (var i = 0; i < traces.length; i += 1) {
            const line = createLineStrip(this.cg, times, traces[i].data, {
                colors: colors[i],
                widths: 3,
            })

            lines.push(line)

            const label = createText(this.cg, this.font, traces[i].label, [max_time / 10, 8 - i * 0.7], {
                color: colors[i],
            })

            lines.push(label)
        }

        const xAxis = createOrthoAxis(this.cg, coords, "x", this.font, {
            labelSide: 1,
            tickOffset: -2.5,
            tickLength: 6,
            tickStep: max_time / 5,
            labelFormatter: (n) => n.toExponential(2),
        })

        const yAxis = createOrthoAxis(this.cg, coords, "y", this.font, {
            tickOffset: -2.5,
            tickLength: 6,
            tickStep: 2.0,
            labelFormatter: (n) => n.toFixed(1),
        })

        // Render the display to an HTML element.
        lines.push(xAxis)
        lines.push(yAxis)

        this.cg.render(coords, viewport, lines)

        // Copy the plot to a new canvas and add it to the document.
        this.cg.copyTo(viewport, this.canvas)
    }

    render() {
        return (
          <div className="oscilloscope" style={{width: "80%"}}>
            <div ref={ (divElement) => { this.divElement = divElement } } >
              <canvas id="oscilloscope-display" height="500px"/>
            </div>
            <Trigger onData={(times, traces) => this.drawGraph(times, traces)} />
          </div>
        );
    }
}

ReactDOM.render(<Oscilloscope />, document.getElementById("root"))

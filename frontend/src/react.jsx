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
            trigger: 'Idle',
            timer: null,
            capture_duration: 0.001,
        }
    }

    pollTrigger() {

        fetch('http://localhost:8080/trigger').then(response => {
            if (response.ok) {
                return response.json();
            } else {
                return Promise.reject(`Trigger rerequest failed: ${response.text()}`)
            }
        }).then(body => {
            console.log(`Trigger state: ${body}`)

            this.setState({trigger: body})
            if (body == "Stopped") {
                clearInterval(this.state.timer)
                this.setState({timer: null})
                this.props.onTrigger()
            }
        })
    }

    startCapture() {
        const postData = JSON.stringify({
            capture_duration_secs: this.state.capture_duration
        });

        fetch('http://localhost:8080/capture', {method: 'POST', body: postData})
            .then(response => {
                if (response.ok) {
                    // Begin polling the trigger state.
                    this.setState({timer: setInterval(() => this.pollTrigger(), 100)})
                } else {
                    console.log(`Capture error: ${error}`)
                }
            })
    }

    forceTrigger() {
        fetch('http://localhost:8080/trigger', {method: 'POST'})
    }

    onChange(evt) {
        this.setState({capture_duration: Number(evt.target.value)})
    }

    render() {
        return (
          <div className="trigger">
            <div className="input-group input-group-sm mb-3">
              <span className="input-group-text">Duration</span>
              <input className="form-control" type="number" value={this.state.capture_duration} onChange={this.onChange} />
            </div>

            <div><b>{this.state.trigger}</b></div>

            <button className="btn btn-outline-primary" onClick={() => this.startCapture()}> Capture </button>
            <button className="btn btn-outline-primary" onClick={() => this.forceTrigger()}> Force Trigger </button>
          </div>
        );
    }
}

class Oscilloscope extends React.Component {
    constructor(props) {
        super(props);
        this.times = [1, 2, 3]
        this.traces = [{'label': '', 'data': [0, 0.5, 1]}]
        this.cg = new CandyGraph()
        this.font = null
        this.state = {
            width: 384,
            height: 384,
        }

        createDefaultFont(this.cg).then(font => {
            this.font = font
            this.drawGraph()
        })
    }

    componentDidMount() {
        const height = this.divElement.clientHeight
        const width = this.divElement.clientWidth
        console.log(width, height)
        this.setState({height, width: width})
    }

    getTraces() {
        fetch('http://localhost:8080/traces').then(response => {
            if (response.ok) {
                return response.json();
            } else {
                console.log(response)
                return Promise.reject(`Data request failed: ${response.text()}`)
            }
        }).then(data => {
            this.times = data.time
            this.traces = data.traces
            this.drawGraph()
        })
    }

    drawGraph() {
        if (this.font == null || this.state.width == 0 || this.state.height == 0) {
            return;
        }

        if (this.canvas == null) {
            this.canvas = document.getElementById("oscilloscope-display")
        }

        this.canvas.width = this.state.width
        this.cg.canvas.width = this.state.width
        this.cg.canvas.height = this.canvas.height

        const viewport = {
            x: 0,
            y: 0,
            width: this.state.width,
            height: this.canvas.height,
        }

        console.log(viewport)

        this.cg.clear([1, 1, 1, 1])

        const max_time = Math.max(...this.times)

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
        for (var i = 0; i < this.traces.length; i += 1) {
            const line = createLineStrip(this.cg, this.times, this.traces[i].data, {
                colors: colors[i],
                widths: 3,
            })

            lines.push(line)

            const label = createText(this.cg, this.font, this.traces[i].label, [max_time / 10, 8 - i * 0.7], {
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

        console.log('Redrawing plot')
        this.cg.render(coords, viewport, lines)

        // Copy the plot to a new canvas and add it to the document.
        this.cg.copyTo(viewport, this.canvas)
    }

    render() {
        this.drawGraph()

        return (
          <div className="oscilloscope" style={{width: "80%"}}>
            <div
            ref={ (divElement) => { this.divElement = divElement } } >
              <canvas id="oscilloscope-display" height="500px"/>
            </div>
            <Trigger
              onTrigger={() => this.getTraces()}
            />
          </div>
        );
    }
}

ReactDOM.render(<Oscilloscope />, document.getElementById("root"))

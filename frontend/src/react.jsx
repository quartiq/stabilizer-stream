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
            <label>
              Duration:
              <input type="number" value={this.state.capture_duration} onChange={this.onChange} />
            </label>

            <div>{this.state.trigger}</div>

            <button onClick={() => this.startCapture()}> Capture </button>
            <button onClick={() => this.forceTrigger()}> Force Trigger </button>
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

        createDefaultFont(this.cg).then(font => {
            this.font = font
            this.drawGraph()
        })
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
        if (this.font == null) {
            return;
        }

        this.cg.canvas.width = this.cg.canvas.height = 384;

        const viewport = {
            x: 0,
            y: 0,
            width: 384,
            height: 384,
        }

        this.cg.clear([1, 1, 1, 1])

        const max_time = Math.max(...this.times)

        const coords = createCartesianCoordinateSystem(
            createLinearScale([0, max_time], [32, viewport.width - 16]),
            createLinearScale([-10.24, 10.24], [32, viewport.width - 16]),
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
        if (this.canvas == null) {
            this.canvas = this.cg.copyTo(viewport)
            const element = document.getElementById("oscilloscope-display")
            element.parentNode.replaceChild(this.canvas, element)
        } else {
            this.cg.copyTo(viewport, this.canvas)
        }
    }

    render() {
        this.drawGraph()

        return (
          <div className="oscilloscope">
            <Trigger
              onTrigger={() => this.getTraces()}
            />
          </div>
        );
    }
}

ReactDOM.render(<Oscilloscope />, document.getElementById("root"))

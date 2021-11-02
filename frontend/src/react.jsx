import CandyGraph, {
    createCartesianCoordinateSystem,
    createLinearScale,
    createLineStrip,
    createDefaultFont,
    createOrthoAxis,
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
            capture_duration: 1,
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
            if (body == "Triggered") {
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
        // TODO: Validate this is a float first
        this.setState({duration: evt.target.value})
    }

    render() {
        return (
          <div className="trigger">
            <label>
              Duration:
              <input type="number" value={this.state.duration} onChange={this.onChange} />
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
        this.traces = [[0, 0.5, 1]]
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

        const coords = createCartesianCoordinateSystem(
            createLinearScale([0, this.times[this.times.length - 1]], [32, viewport.width - 16]),
            createLinearScale([-10.24, 10.24], [32, viewport.width - 16]),
        );

        // Create the various traces for the display
        var lines = []
        for (var i = 0; i < this.traces.length; i += 1) {
            const line = createLineStrip(this.cg, this.times, this.traces[i], {
                colors: [1, 0.5, 0.0, 1.0],
                widths: 3,
            })

            lines.push(line)
        }


        const xAxis = createOrthoAxis(this.cg, coords, "x", this.font, {
            labelSide: 1,
            tickOffset: -2.5,
            tickLength: 6,
            tickStep: 0.2,
            labelFormatter: (n) => n.toFixed(1),
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
        if (this.canvas == null) {
            this.canvas = this.cg.copyTo(viewport)
            const element = document.getElementById("oscilloscope-display")
            element.parentNode.replaceChild(this.canvas, element)
        } else {
            this.cg.copyTo(viewport, this.canvas)
        }
    }

    example() {
        if (this.font == null) {
            return;
        }

        this.cg.canvas.width = this.cg.canvas.height = 384;

        // Generate some x & y data.
        const xs = [];
        const ys = [];
        for (let x = 0; x <= 1; x += 0.001) {
          xs.push(x);
          ys.push(0.5 + 0.25 * Math.sin(x * 2 * Math.PI));
        }

        // Create a viewport. Units are in pixels.
        const viewport = {
          x: 0,
          y: 0,
          width: this.cg.canvas.width,
          height: this.cg.canvas.height,
        };

        // Create a coordinate system from two linear scales. Note
        // that we add 32 pixels of padding to the left and bottom
        // of the viewport, and 16 pixels to the top and right.
        const coords = createCartesianCoordinateSystem(
          createLinearScale([0, 1], [32, viewport.width - 16]),
          createLinearScale([0, 1], [32, viewport.height - 16])
        );

        // Load the default Lato font
        //const font = await createDefaultFont(cg);

        // Clear the viewport.
        this.cg.clear([1, 1, 1, 1]);

        // Render the a line strip representing the x & y data, and axes.
        this.cg.render(coords, viewport, [
          createLineStrip(this.cg, xs, ys, {
            colors: [1, 0.5, 0.0, 1.0],
            widths: 3,
          }),
          createOrthoAxis(this.cg, coords, "x", this.font, {
            labelSide: 1,
            tickOffset: -2.5,
            tickLength: 6,
            tickStep: 0.2,
            labelFormatter: (n) => n.toFixed(1),
          }),
          createOrthoAxis(this.cg, coords, "y", this.font, {
            tickOffset: 2.5,
            tickLength: 6,
            tickStep: 0.2,
            labelFormatter: (n) => n.toFixed(1),
          }),
        ]);

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

import CandyGraph, {
    createCartesianCoordinateSystem,
    createLinearScale,
    createLineStrip,
} from "candygraph";

import ReactDOM from 'react-dom';
import React from "react";
import http from 'stream-http';

class Display extends React.Component {
    render() {
        const viewport = {
            x: 0,
            y: 0,
            width: this.props.width,
            height: this.props.height,
        }

        const cg = new CandyGraph()

        const coords = createCartesianCoordinateSystem(
            createLinearScale([0, this.props.times[-1]], [32, viewport.width - 16]),
            createLinearScale([-10.24, 10.24], [32, viewport.width - 16]),
        );

        // Create the various traces for the display
        var lines = []
        console.log(this.props)
        for (var i = 0; i < this.props.traces.length; i += 1) {
            lines += createLineStrip(cg, this.props.times, this.props.traces[i])
        }

        // Render the display to an HTML element.
        cg.render(coords, viewport, lines)
        cg.copyTo(viewport, document.getElementById("oscilloscope-display"))

        return (
          <div className="display">
            <canvas id="oscilloscope-display" />
          </div>
        );
    }
}

class Trigger extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            trigger: 'Idle',
            timer: null,
            capture_duration: 1,
        }
    }

    pollTrigger() {
        http.get('http://localhost:8080/trigger', res => {
            res.on('data', data => {
                const body = JSON.parse(data)
                console.log("Trigger state: ${body}")

                this.setState({trigger: body})
                if (body == "Triggered") {
                    clearInterval(this.state.timer)
                    this.setState({timer: null})
                    this.props.onTrigger()
                }
            })
        })
    }

    startCapture() {
        const postData = JSON.stringify({
            capture_duration_secs: this.state.capture_duration
        });

        const req = http.put('http://localhost:8080', {path: '/capture', method: 'POST'}, (res) => {
            res.on('end', _ => {
                // Begin polling the trigger state.
                self.setState({timer: setInterval(pollTrigger, 100)})
            })

            res.on('error', error => {
                console.log('Capture error: ${error}')
            })
        })

        req.write(postData)
        req.end()
    }

    render() {
        return (
          <div className="trigger">
            <label>
              Duration:
              <input type="number" value={this.state.duration} onChange={evt => this.setState({duration: evt.target.value})} />
            </label>

            <div>{this.state.trigger}</div>

            <button onClick={() => this.startCapture()}> Capture </button>
          </div>
        );
    }
}

class Oscilloscope extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            width: props.width,
            height: props.height,
            times: [],
            traces: []
        }
    }

    getTraces() {
        http.get('http://localhost:8080/data', (res) => {
            res.on('data', data => {
                const body = JSON.parse(data)
                this.setState({times: body.times, traces: body.traces})
            })
        })
    }

    render() {
        return (
          <div className="oscilloscope">
            <Display
              width={this.state.width}
              height={this.state.height}
              times={this.state.times}
              traces={this.state.traces}
            />
            <Trigger
              onTrigger={() => this.getTraces()}
            />
          </div>
        );
    }
}

ReactDOM.render(<Oscilloscope />, document.getElementById("root"))

import React from "react";
import ReactDOM from "react-dom";

import "./styles.css";
const tf = require('@tensorflow/tfjs');

const models = {
  yolov5s: {
    modelJson: process.env.PUBLIC_URL + "/yolov5s/model.json"
  }
}

const names = ['total', 'top_left', 'top_right', 'bottom_left']

class App extends React.Component {
  state = {
    selectedModel: "yolov5s",
    model: null,
    preview: "",
    predictions: [],
  };

  componentDidMount() {
    this.loadModel()
    // this.startWebcam();
  }

  loadModel() {
    const { selectedModel } = this.state;
    const modelConfig = models[selectedModel];

    tf.loadGraphModel(modelConfig.modelJson).then(model => {
      this.setState({
        model: model
      });
    });
  }

  handleModelChange = (event) => {
    const selectedModel = event.target.value;
    this.setState({ selectedModel }, () => {
      this.loadModel();
    });
  };

  startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        const video = document.getElementById("video");
        video.srcObject = stream;
        video.onloadedmetadata = () => {
          video.play();
          this.detectObjects();
        };
      })
      .catch(error => {
        console.error("Error accessing webcam:", error);
      });
  }

  detectObjects() {
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    const modelWidth = this.state.model.inputs[0].shape[1];
    const modelHeight = this.state.model.inputs[0].shape[2];

    const confidenceThreshold = 0.75;

    setInterval(() => {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const input = tf.tidy(() => {
        return tf.image.resizeBilinear(tf.browser.fromPixels(canvas), [modelWidth, modelHeight])
          .div(255.0).expandDims(0);
      });
      this.state.model.executeAsync(input).then(res => {
        // Object detection code here
              // Font options.
        const font = "16px sans-serif";
        ctx.font = font;
        ctx.textBaseline = "top";

        const [boxes, scores, classes, valid_detections] = res;
        const boxes_data = boxes.dataSync();
        const scores_data = scores.dataSync();
        const classes_data = classes.dataSync();
        const valid_detections_data = valid_detections.dataSync()[0];

        tf.dispose(res)

        var i;
        for (i = 0; i < valid_detections_data; ++i){
          const conf = scores_data[i].toFixed(2);
          if(conf < confidenceThreshold){
            continue;
          }
          let [x1, y1, x2, y2] = boxes_data.slice(i * 4, (i + 1) * 4);
          x1 *= canvas.width;
          x2 *= canvas.width;
          y1 *= canvas.height;
          y2 *= canvas.height;
          const width = x2 - x1;
          const height = y2 - y1;
          const klass = names[classes_data[i]];
          const score = scores_data[i].toFixed(2);

          // Draw the bounding box.
          ctx.strokeStyle = "blue";
          ctx.lineWidth = 4;
          ctx.strokeRect(x1, y1, width, height);

          // Draw the label background.
          ctx.fillStyle = "#00FFFF";
          const textWidth = ctx.measureText(klass + ":" + score).width;
          const textHeight = parseInt(font, 10); // base 10
          ctx.fillRect(x1, y1, textWidth + 4, textHeight + 4);

        }
        for (i = 0; i < valid_detections_data; ++i){
          const conf = scores_data[i].toFixed(2);
          if(conf < confidenceThreshold){
            continue;
          }
          let [x1, y1, , ] = boxes_data.slice(i * 4, (i + 1) * 4);
          x1 *= canvas.width;
          y1 *= canvas.height;
          const klass = names[classes_data[i]];
          const score = scores_data[i].toFixed(2);

          // Draw the text last to ensure it's on top.
          ctx.fillStyle = "#000000";
          ctx.fillText(klass + ":" + score, x1, y1);

        }
      });
    }, 1000 / 30); // Run object detection at 30 frames per second
  }

  toggleCanvas = () => {
    this.startWebcam();
  };

  render() {

    const {selectedModel} = this.state;
    return (
        <div className="Dropzone-page">

          <div>
            <label>
              <input
                type="radio"
                value="yolov5s"
                checked={selectedModel === "yolov5s"}
                onChange={this.handleModelChange}
              />
              YOLOv5s
            </label>
          </div>

          <button onClick={this.toggleCanvas}>Show Bounding Boxes</button>
          {this.state.model ? (<div>
            <video id="video" width="640" height="480" style={{display: 'none'}}/>
            <canvas id="canvas" width="640" height="480" style={{margin: '20px'}}/>
          </div>) : (
              <div>Loading model...</div>
          )}
        </div>
    );
  }
}


const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);

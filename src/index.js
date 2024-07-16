import React from "react";
import ReactDOM from "react-dom";

import "./styles.css";
const tf = require('@tensorflow/tfjs');

const models = {
  yolov5s: {
    modelJson: process.env.PUBLIC_URL + "/yolov5s/model.json"
  },

  yolov5n: {
    modelJson: process.env.PUBLIC_URL + "/yolov5n/model.json"
  }
}

const names = ['total', 'top_left', 'top_right', 'bottom_left']

class App extends React.Component {
  state = {
    selectedModel: "yolov5n",
    mode: "detect",
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
    console.log(selectedModel);
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
    let objectCount = 0;

    let c1_x, c1_y, c2_x, c2_y, c3_x, c3_y;

    const DROP_RATE = 10;
    let dropped_frames = 0;

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
        objectCount = 0;
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

          objectCount++;
        }
        console.log(objectCount);
        if (this.state.mode === "save" && objectCount === 4) {
          console.log("GOT HERE");
          let x1, y1, x2, y2;
          let width_total, height_total, diagonal, croppedCanvas, croppedCtx, offsetX, offsetY, c0_x, c0_y;
          // Find the coordinates of the four classes respectively
          for (i = 0; i < valid_detections_data; ++i){
            const conf = scores_data[i].toFixed(2);
            if(conf < confidenceThreshold){
              continue;
            }
            const klass = names[classes_data[i]];
            [x1, y1, x2, y2] = boxes_data.slice(i * 4, (i + 1) * 4);
            x1 *= canvas.width;
            y1 *= canvas.height;
            x2 *= canvas.width;
            y2 *= canvas.height;
            if (klass === "top_left") {
              c1_x = (x1 + x2) / 2
              c1_y = (y1 + y2) / 2
            } else if (klass === "top_right") {
              c2_x = (x1 + x2) / 2
              c2_y = (y1 + y2) / 2
            } else if (klass === "bottom_left") {
              c3_x = (x1 + x2) / 2
              c3_y = (y1 + y2) / 2
            } else{
              c0_x = x1;
              c0_y = y1
              width_total = x2 - x1;
              height_total = y2 - y1;

              // Calculate the diagonal length to ensure the rotated image fits within the canvas
              diagonal = Math.sqrt(width_total ** 2 + height_total ** 2);

              croppedCanvas = document.createElement("canvas");
              croppedCtx = croppedCanvas.getContext("2d");

              croppedCanvas.width = diagonal;
              croppedCanvas.height = diagonal;

              // Clear the canvas with white background
              croppedCtx.fillStyle = 'black';
              croppedCtx.fillRect(0, 0, diagonal, diagonal);
            }
          }

          const totalCenterRadius = Math.sqrt((width_total / 2) ** 2 + (height_total / 2) ** 2);

          const isInside = (x, y, centerX, centerY, radius) => {
            return Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2) <= radius;
          };

          const allCentersInside = isInside(c1_x, c1_y, c0_x + width_total / 2, c0_y + height_total / 2, totalCenterRadius) &&
            isInside(c2_x, c2_y, c0_x + width_total / 2, c0_y + height_total / 2, totalCenterRadius) &&
            isInside(c3_x, c3_y, c0_x + width_total / 2, c0_y + height_total / 2, totalCenterRadius);

          if (!allCentersInside)
            return;

          let theta1, theta2, theta3;

          theta1 = Math.atan((c2_y - c1_y) / (c2_x - c1_x));
          theta2 = Math.atan((c1_y - c3_y) / (c1_x - c3_x));
          theta3 = Math.atan((c2_y - c3_y) / (c2_x - c3_x));
          let alpha;

          // Calculate the center of the canvas
          const centerX = diagonal / 2;
          const centerY = diagonal / 2;

          if (theta1 !== 0){
            alpha = theta1
          }
          else if (theta2 < 0){
            alpha = Math.PI
          }
          else{
            alpha = Math.PI
          }

          // Rotate the canvas around its center
          croppedCtx.translate(centerX, centerY);
          croppedCtx.rotate(-alpha);
          croppedCtx.translate(-centerX, -centerY);

          // Calculate the offset to position the original image within the canvas
          offsetX = (diagonal - width_total) / 2;
          offsetY = (diagonal - height_total) / 2;

          croppedCtx.drawImage(
            video, // original image
            c0_x, // top-left x-coordinate
            c0_y, // top-left y-coordinate
            width_total, // width of the cropped image
            height_total, // height of the cropped image
            offsetX, // destination x-coordinate on the canvas
            offsetY, // destination y-coordinate on the canvas
            width_total, // width of the drawn image on the canvas
            height_total // height of the drawn image on the canvas
          );

          if (dropped_frames === DROP_RATE){
            const croppedImageDataURL = croppedCanvas.toDataURL();
            console.log(croppedImageDataURL);
            dropped_frames = 0;
          }

          dropped_frames += 1;

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

  handleModeChange = (event) => {
    this.setState({ mode: event.target.value });
    console.log("mode changed to " + event.target.value)
  };

  render() {

    const {selectedModel} = this.state;
    const {mode} = this.state;

    return (
        <div className="Dropzone-page">

          <div className="box">
            <label>
              <input
                type="radio"
                value="yolov5s"
                checked={selectedModel === "yolov5s"}
                onChange={this.handleModelChange}
              />
              YOLOv5s
            </label>
            <label>
              <input
                type="radio"
                value="yolov5n"
                checked={selectedModel === "yolov5n"}
                onChange={this.handleModelChange}
              />
              YOLOv5n
            </label>
          </div>

           <div className="box">
              <label>
                <input
                  type="radio"
                  value="detect"
                  checked={mode === "detect"}
                  onChange={this.handleModeChange}
                />
                Detect Objects
              </label>
              <label>
                <input
                  type="radio"
                  value="save"
                  checked={mode === "save"}
                  onChange={this.handleModeChange}
                />
                Save Image
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

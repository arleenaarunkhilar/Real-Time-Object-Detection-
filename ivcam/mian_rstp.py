import os
import cv2
import math
import numpy as np
from ultralytics import YOLO
import cvzone
from flask import Flask, Response, render_template

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

# Load class names from file
classnames = []
with open(os.path.join('static', 'classes.txt'), 'r') as f:
    classnames = f.read().splitlines()

# Define the RTSP stream URL
rtsp_url = 'rtsp://your_rtsp_stream_link'

# Initialize video capture with RTSP stream
cap = cv2.VideoCapture(rtsp_url)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform object detection with YOLO
        result = model(frame)
        detections = np.empty((0, 5))
        for info in result:
            boxes = info.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                classindex = box.cls[0]
                conf = math.ceil(conf * 100)
                classindex = int(classindex)
                objectdetect = classnames[classindex]

                if objectdetect in ['person', 'bed', 'bottle', 'cell phone', 'clock', 'chair', 'pottedplant', 'laptop', 'teddy bear', 'mouse', 'keyboard', 'sofa', 'cup', 'backpack', 'handbag'] and conf > 60:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    new_detections = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, new_detections))

                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cvzone.putTextRect(frame, f'{objectdetect} {conf}%', [x1 + 8, y1 - 12], thickness=2, scale=1.5)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)

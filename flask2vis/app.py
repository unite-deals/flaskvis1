from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)
video_capture = cv2.VideoCapture(0)

camera_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
dist_coeffs = np.zeros((4, 1))

def generate_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    _, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera([np.zeros((1, 1, 2), dtype=np.float32)], [np.zeros((1, 1, 3), dtype=np.float32)], gray.shape[::-1], None, None)

    undistorted = cv2.undistort(gray, camera_matrix, dist_coeffs)

    edges = cv2.Canny(undistorted, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()

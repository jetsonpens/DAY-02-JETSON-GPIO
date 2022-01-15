from flask import Flask, render_template, Response
import cv2
import time
import numpy as np
import Jetson.GPIO as GPIO

# Set GPIO
output_pin = 12 
GPIO.setmode(GPIO.BOARD)
GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)

app = Flask(__name__)

camera = cv2.VideoCapture(0)


def play_beep():
    print("play beep...")
    for _ in range(3):
        GPIO.output(output_pin, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(output_pin, GPIO.LOW)
        time.sleep(0.02)


def detect_motion(frame, last_mean):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = np.abs(np.mean(gray) - last_mean)

    last_mean = np.mean(gray)

    if result > 0.3:
        play_beep()

    return last_mean

def gen_frames():  
    last_mean = 0

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            last_mean = detect_motion(frame, last_mean)
            output_frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host="0.0.0.0")
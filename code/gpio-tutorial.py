import Jetson.GPIO as GPIO
import time

output_pin = 12 

GPIO.setmode(GPIO.BOARD)
GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)

print("Press CTRL+C to exit")

try:
    while True:
        GPIO.output(output_pin, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(output_pin, GPIO.LOW)
        time.sleep(1)
finally:
    GPIO.cleanup()
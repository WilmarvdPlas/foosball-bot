import pigpio
import sys
import time
import signal


DIR_PIN = 27
STEP_PIN = 12

move_time = 0.29
duty_cycle = 2000
pause_time = 2

PI = pigpio.pi()

def cleanup(sig, frame):
    PI.hardware_PWM(STEP_PIN, 0, 0)
    PI.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

PI.write(DIR_PIN, pigpio.HIGH)
PI.hardware_PWM(STEP_PIN, duty_cycle, 500000)

time.sleep(move_time)

PI.hardware_PWM(STEP_PIN, 0, 0)

time.sleep(pause_time)

PI.write(DIR_PIN, pigpio.LOW)
PI.hardware_PWM(STEP_PIN, duty_cycle, 500000)

time.sleep(move_time)

cleanup(None, None)
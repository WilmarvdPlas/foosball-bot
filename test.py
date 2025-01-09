import pigpio
import time

PI = pigpio.pi()

DIR_PIN = 17
STEP_PIN = 13

for _ in range(5):
    PI.hardware_PWM(STEP_PIN, 1000, 500000)
    time.sleep(1)
    PI.hardware_PWM(STEP_PIN, 0, 0)
    time.sleep(1)
    


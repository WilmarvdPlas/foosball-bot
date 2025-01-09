import pigpio
import time
import threading
from constants import *
from shared import *

class PlayerRow:
    def __init__(self, translate_step_pin, translate_dir_pin, rotate_step_pin, rotate_dir_pin, hall_pin, pi, lock):
        self.translate_step_pin = translate_step_pin
        self.translate_dir_pin = translate_dir_pin
        self.rotate_step_pin = rotate_step_pin
        self.rotate_dir_pin = rotate_dir_pin
        self.hall_pin = hall_pin
        self.pi = pi
        self.lock = lock

        self.shooting_active = False
        self.pid_last_error = 0
        self.pid_cumulative_error = 0
        self.previous_pid_time_seconds = 0
        
        self.pi.set_mode(self.translate_step_pin, pigpio.OUTPUT)
        self.pi.set_mode(self.translate_dir_pin, pigpio.OUTPUT)
        self.pi.set_mode(self.rotate_step_pin, pigpio.OUTPUT)
        self.pi.set_mode(self.rotate_dir_pin, pigpio.OUTPUT)
        self.pi.set_mode(self.hall_pin, pigpio.INPUT)

    def check_rotate_stop_condition(self, stop_state):
        previous_state = self.pi.read(self.hall_pin)

        while True:
            current_state = self.pi.read(self.hall_pin)

            if previous_state != stop_state and current_state == stop_state:
                self.pi.hardware_PWM(self.rotate_step_pin, 0, 0)
                with self.lock:
                    self.shooting_active = False
                return
            
            previous_state = current_state

            time.sleep(0.01)
    
    def shoot_forwards(self):
        self.pi.hardware_PWM(self.rotate_step_pin, ROTATE_FREQUENCY, 500000)

        self.pi.write(self.rotate_dir_pin, pigpio.HIGH)
        time.sleep(0.075)
        self.pi.write(self.rotate_dir_pin, pigpio.LOW)

        threading.Thread(target=self.check_rotate_stop_condition, args=(1,)).start()
        
    def shoot_backwards(self):
        self.pi.hardware_PWM(self.rotate_step_pin, ROTATE_FREQUENCY, 500000)

        self.pi.write(self.rotate_dir_pin, pigpio.HIGH)
        
        threading.Thread(target=self.check_rotate_stop_condition, args=(0,)).start()

    def stop_translation(self):
        self.pi.hardware_PWM(self.translate_step_pin, 0, 0)

    def stop_rotation(self):
        self.pi.hardware_PWM(self.rotate_step_pin, 0, 0)

    def update_actuation(self, ball_pos, player_centers, active_player_index):
        if len(player_centers) == PLAYERS_IN_ROW:
            sorted_centers = sorted(player_centers, key=lambda x: x[0])
            active_player_pos = sorted_centers[active_player_index]

            if abs(ball_pos[1] - active_player_pos[0]) < (TRANSLATION_STOP_PERCENTAGE * RESIZED_HEIGHT):
                self.stop_translation()

                ball_in_range = abs(ball_pos[0] - active_player_pos[1]) < 0.1 * RESIZED_WIDTH

                if ball_in_range and not self.shooting_active:
                    with self.lock:
                        self.shooting_active = True

                    if ball_pos[0] < active_player_pos[1]:
                        threading.Thread(target=self.shoot_forwards).start()
                    else:
                        threading.Thread(target=self.shoot_backwards).start()

                return

            pid = self.compute_translation_PID(ball_pos[1], active_player_pos[0])

            self.pi.hardware_PWM(self.translate_step_pin, int(abs(pid)) + MIN_TRANSLATE_FREQUENCY, 500000)
            self.pi.write(self.translate_dir_pin, pigpio.HIGH) if pid < 0 else self.pi.write(self.translate_dir_pin, pigpio.LOW)
        else:
            self.pi.hardware_PWM(self.translate_step_pin, 0, 0)

    def compute_translation_PID(self, ball_y, player_y):
        current_time_seconds = time.time()
        elapsed_time_seconds = current_time_seconds - self.previous_pid_time_seconds

        error = ball_y - player_y
        self.pid_cumulative_error += error * elapsed_time_seconds
        self.pid_cumulative_error = clamp(self.pid_cumulative_error, MIN_TRANSLATE_FREQUENCY, MAX_TRANSLATE_FREQUENCY)
        rate_error = (error - self.pid_last_error) / elapsed_time_seconds

        output = TRANSLATION_PID_KP * error + TRANSLATION_PID_KI * self.pid_cumulative_error + TRANSLATION_PID_KD * rate_error

        self.pid_last_error = error
        self.previous_pid_time_seconds = current_time_seconds

        return clamp(output, MIN_TRANSLATE_FREQUENCY - MAX_TRANSLATE_FREQUENCY, MAX_TRANSLATE_FREQUENCY - MIN_TRANSLATE_FREQUENCY)

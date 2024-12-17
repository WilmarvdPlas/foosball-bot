import cv2
import numpy as np
import time
import math
import pigpio
import sys
import signal

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 360

RESIZED_WIDTH = 160
RESIZED_HEIGHT = 90

PLAYER_COUNT = 6
PLAYERS_IN_ROW = PLAYER_COUNT / 2

CAPTURE = cv2.VideoCapture(0)

CAPTURE.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
CAPTURE.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

BALL_LOWER_BOUND = np.array([100, 0, 200])
BALL_UPPER_BOUND = np.array([170, 90, 255])

PLAYER_LOWER_BOUND = np.array([80, 0, 80])
PLAYER_UPPER_BOUND = np.array([170, 60, 170])

PI = pigpio.pi()

TRANSLATE_DEFENSE_STEP_PIN = 12
TRANSLATE_DEFENSE_DIR_PIN = 27
ROTATE_DEFENSE_STEP_PIN = 13
ROTATE_DEFENSE_DIR_PIN = 17

PI.set_mode(TRANSLATE_DEFENSE_STEP_PIN, pigpio.OUTPUT)
PI.set_mode(TRANSLATE_DEFENSE_DIR_PIN, pigpio.OUTPUT)
PI.set_mode(ROTATE_DEFENSE_STEP_PIN, pigpio.OUTPUT)
PI.set_mode(ROTATE_DEFENSE_DIR_PIN, pigpio.OUTPUT)

TRANSLATION_STOP_PERCENTAGE = 0.05

MIN_TRANSLATE_FREQUENCY = 250
MAX_TRANSLATE_FREQUENCY = 1350

BALL_TIMEOUT_SECONDS = 1

PID_KP = 200
PID_KI = 0
PID_KD = 10

pid_last_error = 0
pid_cumulative_error = 0
previous_pid_time_seconds = 0

last_ball_sighting_seconds = 0

frame_count = 0
fps_start_time = time.time()

def compute_translation_PID(ball_y, player_y):
    global previous_pid_time_seconds, pid_last_error, pid_cumulative_error

    current_time_seconds = time.time()
    elapsed_time_seconds = current_time_seconds - previous_pid_time_seconds

    error = ball_y - player_y
    pid_cumulative_error += error * elapsed_time_seconds
    pid_cumulative_error = clamp(pid_cumulative_error, MIN_TRANSLATE_FREQUENCY, MAX_TRANSLATE_FREQUENCY)
    rate_error = (error - pid_last_error) / elapsed_time_seconds

    output = PID_KP * error + PID_KI * pid_cumulative_error + PID_KD * rate_error

    pid_last_error = error
    previous_pid_time_seconds = current_time_seconds

    return clamp(output, MIN_TRANSLATE_FREQUENCY - MAX_TRANSLATE_FREQUENCY, MAX_TRANSLATE_FREQUENCY - MIN_TRANSLATE_FREQUENCY)

def cleanup(sig, frame):
    PI.hardware_PWM(TRANSLATE_DEFENSE_STEP_PIN, 0, 0)
    PI.stop()

    CAPTURE.release()
    cv2.destroyAllWindows()

    sys.exit(0)

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def filter_mask_by_percentage(mask, percentages):
    filtered_mask = np.zeros_like(mask)

    for (min_percent, max_percent) in percentages:
        x_min = int(min_percent * RESIZED_WIDTH)
        x_max = int(max_percent * RESIZED_WIDTH)
        filtered_mask[:, x_min:x_max] = mask[:, x_min:x_max]

    return filtered_mask

def get_ball_center(frame):
    mask = cv2.inRange(frame, BALL_LOWER_BOUND, BALL_UPPER_BOUND)

    percentages = [(0.1, 0.9)]
    filtered_mask = filter_mask_by_percentage(mask, percentages)

    moments = cv2.moments(filtered_mask)

    if moments["m00"] == 0:
        return None

    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])

    return (center_x, center_y)

def get_player_centers(frame):
    mask = cv2.inRange(frame, PLAYER_LOWER_BOUND, PLAYER_UPPER_BOUND)   
    
    percentages = [(0.35, 0.45), (0.80, 0.90)]
    filtered_mask = filter_mask_by_percentage(mask, percentages)

    scaled_frame = cv2.resize(filtered_mask, (CAPTURE_WIDTH, CAPTURE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Mask', scaled_frame)

    coordinates = np.column_stack(np.where(filtered_mask > 0)).astype(np.float32)

    if (coordinates.shape[0] < PLAYER_COUNT):
        return [], []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 200.0)
    _, _, centers = cv2.kmeans(coordinates, PLAYER_COUNT, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    centers = np.int32(centers)

    attack_centers = []
    defense_centers = []

    for center in centers:
        if len(center) < 2:
            continue

        if (center[1] < RESIZED_WIDTH / 2):
            attack_centers.append(center)
        else:
            defense_centers.append(center)

    return attack_centers, defense_centers

def control_motors(ball_center, attack_centers, defense_centers):
    global last_ball_sighting_seconds

    if ball_center is None:
        if time.time() - last_ball_sighting_seconds > BALL_TIMEOUT_SECONDS:
            PI.hardware_PWM(TRANSLATE_DEFENSE_STEP_PIN, 0, 0)
        
        return
    
    last_ball_sighting_seconds = time.time()

    player_y_zone = RESIZED_HEIGHT / PLAYERS_IN_ROW
    active_player_index = clamp(math.floor(ball_center[1] / player_y_zone), 0, len(attack_centers) - 1)

    if len(attack_centers) == PLAYERS_IN_ROW:
        sorted_defense_centers = sorted(attack_centers, key=lambda x: x[0])
        defense_coords = sorted_defense_centers[active_player_index]

        if abs(ball_center[1] - defense_coords[0]) < TRANSLATION_STOP_PERCENTAGE * RESIZED_HEIGHT:
            PI.hardware_PWM(TRANSLATE_DEFENSE_STEP_PIN, 0, 0)

            if abs(ball_center[0] - defense_coords[1]) < 0.1 * RESIZED_WIDTH:
                PI.write(ROTATE_DEFENSE_DIR_PIN, pigpio.HIGH) if ball_center[0] < defense_coords[1] else PI.write(ROTATE_DEFENSE_DIR_PIN, pigpio.LOW)
                PI.hardware_PWM(ROTATE_DEFENSE_STEP_PIN, 1000, 500000)
                time.sleep(0.25)
                PI.write(ROTATE_DEFENSE_DIR_PIN, pigpio.LOW) if ball_center[0] < defense_coords[1] else PI.write(ROTATE_DEFENSE_DIR_PIN, pigpio.HIGH)
                time.sleep(0.25)
                PI.hardware_PWM(ROTATE_DEFENSE_STEP_PIN, 0, 0)

            return

        pid = compute_translation_PID(ball_center[1], defense_coords[0])

        PI.hardware_PWM(TRANSLATE_DEFENSE_STEP_PIN, int(abs(pid)) + MIN_TRANSLATE_FREQUENCY, 500000)
        PI.write(TRANSLATE_DEFENSE_DIR_PIN, pigpio.HIGH) if pid < 0 else PI.write(TRANSLATE_DEFENSE_DIR_PIN, pigpio.LOW)

def display_frame(frame, ball_center, attack_centers, defense_centers):
    for center in attack_centers:
        cv2.circle(frame, (center[1], center[0]), 5, (0, 255, 0), -1)

    for center in defense_centers:
        cv2.circle(frame, (center[1], center[0]), 5, (255, 0, 0), -1)

    if (ball_center is not None):
        cv2.circle(frame, ball_center, 5, (0, 0, 255), -1)

    scaled_frame = cv2.resize(frame, (CAPTURE_WIDTH, CAPTURE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Frame', scaled_frame)

def handle_frame_count():
    global frame_count, fps_start_time

    frame_count += 1
    elapsed_time = time.time() - fps_start_time

    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")

        frame_count = 0
        fps_start_time = time.time()

def handle_frame(frame):
    frame = cv2.resize(frame, (RESIZED_WIDTH, RESIZED_HEIGHT), interpolation=cv2.INTER_AREA)
    
    ball_center = get_ball_center(frame)
    attack_player_centers, defense_player_centers = get_player_centers(frame)

    control_motors(ball_center, attack_player_centers, defense_player_centers)
    display_frame(frame, ball_center, attack_player_centers, defense_player_centers)
    handle_frame_count()

def main():
    signal.signal(signal.SIGINT, cleanup)

    if (not CAPTURE.isOpened()):
        print("Error opening video stream or file")

    while(CAPTURE.isOpened()):
        frame_captured, frame = CAPTURE.read()

        if not frame_captured or cv2.waitKey(25) & 0xFF == ord('q'):
            break

        handle_frame(frame)

    cleanup(None, None)

if __name__ == "__main__":
    main()

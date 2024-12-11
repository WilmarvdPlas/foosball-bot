import cv2
import numpy as np
import time
import math
import pigpio
import sys
import signal

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 360

RESIZED_WIDTH = 320
RESIZED_HEIGHT = 180

PLAYER_COUNT = 6
PLAYERS_IN_ROW = PLAYER_COUNT / 2

CAPTURE = cv2.VideoCapture(0)

CAPTURE.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
CAPTURE.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

BALL_LOWER_BOUND = np.array([100, 0, 200])
BALL_UPPER_BOUND = np.array([170, 90, 255])

PLAYER_LOWER_BOUND = np.array([120, 20, 100])
PLAYER_UPPER_BOUND = np.array([170, 60, 170])

PI = pigpio.pi()
TRANSLATE_DEFENSE_STEP_PIN = 12
TRANSLATE_DEFENSE_DIR_PIN = 27

PI.set_mode(TRANSLATE_DEFENSE_STEP_PIN, pigpio.OUTPUT)
PI.set_mode(TRANSLATE_DEFENSE_DIR_PIN, pigpio.OUTPUT)

frame_count = 0
start_time = time.time()

def cleanup(sig, frame):
    PI.hardware_PWM(TRANSLATE_DEFENSE_STEP_PIN, 0, 0)
    PI.stop()

    CAPTURE.release()
    cv2.destroyAllWindows()

    sys.exit(0)

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def get_ball_center(frame):
    mask = cv2.inRange(frame, BALL_LOWER_BOUND, BALL_UPPER_BOUND)
    moments = cv2.moments(mask)

    if moments["m00"] == 0:
        return None

    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])

    return (center_x, center_y)

def get_player_centers(frame):
    mask = cv2.inRange(frame, PLAYER_LOWER_BOUND, PLAYER_UPPER_BOUND)

    # scaled_frame = cv2.resize(mask, (CAPTURE_WIDTH, CAPTURE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    # cv2.imshow('Mask', scaled_frame)

    coordinates = np.column_stack(np.where(mask > 0)).astype(np.float32)

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
    if ball_center is None:
        return

    player_y_zone = RESIZED_HEIGHT / PLAYERS_IN_ROW
    active_player_index = clamp(math.floor(ball_center[1] / player_y_zone), 0, len(attack_centers) - 1)

    if len(attack_centers) == PLAYERS_IN_ROW:
        sorted_defense_centers = sorted(attack_centers, key=lambda x: x[0])
        defense_coords = sorted_defense_centers[active_player_index]
        PI.write(TRANSLATE_DEFENSE_DIR_PIN, pigpio.LOW) if defense_coords[0] > ball_center[1] else PI.write(TRANSLATE_DEFENSE_DIR_PIN, pigpio.HIGH)

    print(active_player_index)

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
    global frame_count, start_time

    frame_count += 1
    elapsed_time = time.time() - start_time

    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")

        frame_count = 0
        start_time = time.time()

def handle_frame(frame):
    frame = cv2.resize(frame, (RESIZED_WIDTH, RESIZED_HEIGHT), interpolation=cv2.INTER_AREA)
    
    ball_center = get_ball_center(frame)
    attack_player_centers, defense_player_centers = get_player_centers(frame)

    control_motors(ball_center, attack_player_centers, defense_player_centers)
    # display_frame(frame, ball_center, attack_player_centers, defense_player_centers)
    handle_frame_count()

def main():
    signal.signal(signal.SIGINT, cleanup)
    PI.hardware_PWM(TRANSLATE_DEFENSE_STEP_PIN, 500, 500000)

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

import cv2
import numpy as np
import time
import math
import pigpio
import sys
import signal
import threading
from constants import *
from shared import *
from player_row import PlayerRow

CAPTURE = cv2.VideoCapture(0)

CAPTURE.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
CAPTURE.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

PI = pigpio.pi()
lock = threading.Lock()

DEFENSE_ROW = PlayerRow(12, 27, 13, 17, 23, PI, lock)
ATTACK_ROW = PlayerRow(18, 22, 19, 24, 16, PI, lock)

last_ball_sighting_seconds = 0

frame_count = 0
fps_start_time = time.time()

def cleanup(sig, frame):
    DEFENSE_ROW.stop_rotation()
    DEFENSE_ROW.stop_translation()
    ATTACK_ROW.stop_rotation()
    ATTACK_ROW.stop_translation()

    PI.stop()

    CAPTURE.release()
    cv2.destroyAllWindows()

    sys.exit(0)

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

        if (center[1] > RESIZED_WIDTH / 2):
            attack_centers.append(center)
        else:
            defense_centers.append(center)

    return attack_centers, defense_centers

def control_motors(ball_center, attack_centers, defense_centers):
    global last_ball_sighting_seconds

    if ball_center is None:
        DEFENSE_ROW.stop_translation()
        ATTACK_ROW.stop_translation()
        
        return
    
    last_ball_sighting_seconds = time.time()

    player_y_zone = RESIZED_HEIGHT / PLAYERS_IN_ROW
    active_player_index = clamp(math.floor(ball_center[1] / player_y_zone), 0, PLAYERS_IN_ROW - 1)

    DEFENSE_ROW.update_actuation(ball_center, defense_centers, active_player_index)
    # ATTACK_ROW.update_actuation(ball_center, attack_centers, active_player_index)

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

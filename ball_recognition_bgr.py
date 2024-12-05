import cv2
import numpy as np
import time
import math

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 360

RESIZED_WIDTH = 320
RESIZED_HEIGHT = 180

PLAYER_COUNT = 6

CAPTURE = cv2.VideoCapture(1, cv2.CAP_DSHOW)

CAPTURE.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
CAPTURE.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

BALL_LOWER_BOUND = np.array([100, 0, 200])
BALL_UPPER_BOUND = np.array([170, 90, 255])

PLAYER_LOWER_BOUND = np.array([120, 20, 80])
PLAYER_UPPER_BOUND = np.array([170, 60, 170])

frame_count = 0
start_time = time.time()

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

    scaled_frame = cv2.resize(mask, (CAPTURE_WIDTH, CAPTURE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Mask', scaled_frame)

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

        if (center[1] > RESIZED_WIDTH / 2):
            attack_centers.append(center)
        else:
            defense_centers.append(center)

    return attack_centers, defense_centers

def control_motors(ball_center, attack_centers, defense_centers):
    if ball_center is None:
        return

    player_y_zone = RESIZED_HEIGHT / PLAYER_COUNT / 2
    active_player_index = math.floor(ball_center[1] / player_y_zone)
    # defense_coords = sorted(defense_centers, key=lambda x: x[1])[active_player_index]

    # defense_left() if defense_coords[1] > ball_center[0] else defense_right

    print(active_player_index)

def defense_left():
    return

def defense_right():
    return

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
    display_frame(frame, ball_center, attack_player_centers, defense_player_centers)
    handle_frame_count()

def main():
    if (not CAPTURE.isOpened()):
        print("Error opening video stream or file")

    while(CAPTURE.isOpened()):
        frame_captured, frame = CAPTURE.read()

        if not frame_captured or cv2.waitKey(25) & 0xFF == ord('q'):
            break

        handle_frame(frame)

    CAPTURE.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

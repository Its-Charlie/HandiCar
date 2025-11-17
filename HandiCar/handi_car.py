"""
HandiCar: Hand-Gesture Controlled Car Racing (single-file)
Requirements: mediapipe, opencv-python, pygame, numpy

Controls (via webcam):
- Move your hand left/right (index finger X) -> moves the car horizontally.
- Pinch (index tip + thumb tip near each other) -> temporary BOOST (faster speed).

Run:
python handi_car.py
Press ESC or close the pygame window to quit.
"""

import cv2
import mediapipe as mp
import pygame
import sys
import random
import time
import numpy as np

# ----------------------------
# SETTINGS
# ----------------------------
SCREEN_W, SCREEN_H = 800, 600
CAR_W, CAR_H = 80, 120
LANE_PADDING = 40
FPS = 30

OBSTACLE_W_MIN, OBSTACLE_W_MAX = 50, 140
OBSTACLE_H = 40
OBSTACLE_COLOR = (200, 30, 30)
OBSTACLE_SPAWN_INTERVAL = 1.0  # seconds

BOOST_MULTIPLIER = 1.8
BOOST_DURATION = 0.6  # seconds
PINCH_THRESHOLD = 0.05  # distance (normalized) to detect pinch

SMOOTHING = 0.2  # smoothing factor for hand-to-car movement (0..1)

# ----------------------------
# Mediapipe Hand Init
# ----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ----------------------------
# Pygame Init
# ----------------------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("HandiCar — Hand-Controlled Racing")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)

# ----------------------------
# Game State
# ----------------------------
car_x = SCREEN_W // 2
car_y = SCREEN_H - CAR_H - 20
car_speed_base = 6.0
car_speed = car_speed_base
last_hand_x = None

obstacles = []
last_spawn_time = time.time()
score = 0
game_over = False
boost_active = False
boost_end_time = 0

# Simple helper functions
def draw_text(surf, text, x, y, color=(255,255,255)):
    img = font.render(text, True, color)
    surf.blit(img, (x,y))

def spawn_obstacle():
    w = random.randint(OBSTACLE_W_MIN, OBSTACLE_W_MAX)
    x = random.randint(LANE_PADDING, SCREEN_W - LANE_PADDING - w)
    y = -OBSTACLE_H
    speed = random.uniform(2.0, 5.0) + score * 0.01  # slowly increase difficulty
    obstacles.append({"x": x, "y": y, "w": w, "h": OBSTACLE_H, "speed": speed})

def rects_collide(r1, r2):
    return not (r1['x'] + r1['w'] < r2['x'] or r1['x'] > r2['x'] + r2['w'] or r1['y'] + r1['h'] < r2['y'] or r1['y'] > r2['y'] + r2['h'])

# ----------------------------
# Webcam capture (OpenCV)
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    pygame.quit()
    sys.exit()

# Flip the camera image (mirror) so movement feels natural
FLIP_CAMERA = True

# ----------------------------
# Main Loop
# ----------------------------
try:
    while True:
        # ----- Pygame events -----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt()

        # ----- Read webcam frame & detect hand -----
        ret, frame = cap.read()
        if not ret:
            # if camera fails, skip detection but keep game running
            hand_x_norm = None
            pinch = False
        else:
            if FLIP_CAMERA:
                frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            hand_x_norm = None
            pinch = False

            if results.multi_hand_landmarks:
                # use the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                # index finger tip = landmark 8, thumb tip = 4
                idx_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                # normalized x coordinate in [0,1]
                hand_x_norm = idx_tip.x
                # pinch detection: euclidean distance between index tip and thumb tip
                dx = idx_tip.x - thumb_tip.x
                dy = idx_tip.y - thumb_tip.y
                pinch_dist = (dx*dx + dy*dy) ** 0.5
                pinch = pinch_dist < PINCH_THRESHOLD

                # (optional) draw landmarks on the camera view for debugging
                # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # ----- Map hand X to car X (with smoothing) -----
        if hand_x_norm is not None:
            # convert normalized X (0..1) to pixel X center for car
            target_x = int(hand_x_norm * SCREEN_W)
            # clamp so car remains fully inside screen
            target_x = max(CAR_W//2 + 10, min(SCREEN_W - CAR_W//2 - 10, target_x))
            # smoothing (exponential)
            if last_hand_x is None:
                last_hand_x = target_x
            else:
                last_hand_x = int(last_hand_x + (target_x - last_hand_x) * SMOOTHING)
            # set car_x with smooth follow
            car_x = last_hand_x - CAR_W//2
        # else: keep last car_x

        # ----- Boost logic -----
        current_time = time.time()
        if pinch and not boost_active:
            boost_active = True
            boost_end_time = current_time + BOOST_DURATION
        if boost_active:
            if current_time < boost_end_time:
                car_speed = car_speed_base * BOOST_MULTIPLIER
            else:
                boost_active = False
                car_speed = car_speed_base

        # ----- Spawn obstacles -----
        if time.time() - last_spawn_time > OBSTACLE_SPAWN_INTERVAL:
            spawn_obstacle()
            last_spawn_time = time.time()

        # ----- Update obstacles -----
        for obs in obstacles:
            obs['y'] += obs['speed'] * (1 + (score * 0.001))  # scale with score
        # remove off-screen
        obstacles = [o for o in obstacles if o['y'] < SCREEN_H + 100]

        # ----- Car bounding rect for collision -----
        car_rect = {"x": car_x, "y": car_y, "w": CAR_W, "h": CAR_H}

        # ----- Collision detection -----
        for obs in obstacles:
            if rects_collide(car_rect, obs):
                game_over = True

        # ----- Score (increases over time) -----
        score += 0.1  # small increment per loop; adjust as needed

        # ----- Draw everything -----
        screen.fill((30, 30, 40))  # background

        # road
        pygame.draw.rect(screen, (40, 40, 40), (LANE_PADDING, 0, SCREEN_W - 2*LANE_PADDING, SCREEN_H))

        # draw car
        car_color = (50, 180, 50) if not game_over else (120, 120, 120)
        pygame.draw.rect(screen, car_color, (car_rect['x'], car_rect['y'], car_rect['w'], car_rect['h']), border_radius=8)
        # car window / simple details
        pygame.draw.rect(screen, (20,20,20), (car_rect['x'] + 10, car_rect['y'] + 20, car_rect['w'] - 20, 30))

        # draw obstacles
        for obs in obstacles:
            pygame.draw.rect(screen, OBSTACLE_COLOR, (obs['x'], obs['y'], obs['w'], obs['h']), border_radius=6)

        # HUD
        draw_text(screen, f"Score: {int(score)}", 10, 10)
        draw_text(screen, f"Boost: {'ON' if boost_active else 'OFF'} (pinch to boost)", 10, 36)
        draw_text(screen, "Press ESC to quit", 10, 60)

        if game_over:
            # show Game Over
            over_font = pygame.font.SysFont(None, 64)
            txt = over_font.render("GAME OVER", True, (240, 60, 60))
            txt_rect = txt.get_rect(center=(SCREEN_W//2, SCREEN_H//2 - 40))
            screen.blit(txt, txt_rect)
            sub = font.render(f"Final Score: {int(score)}  —  Press ESC or close window", True, (255,255,255))
            screen.blit(sub, (SCREEN_W//2 - sub.get_width()//2, SCREEN_H//2 + 10))
            pygame.display.flip()
            # pause loop but still consume events until exit
            while True:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        raise KeyboardInterrupt()
                    if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt()
                clock.tick(10)

        pygame.display.flip()
        clock.tick(FPS)

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    hands.close()
    pygame.quit()
    cv2.destroyAllWindows()
    print("Exited gracefully.")

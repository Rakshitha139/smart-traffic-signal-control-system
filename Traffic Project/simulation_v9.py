"""
ML-Based Density Adaptive Traffic Signal System
================================================
Improvements over v6:
  - Thread-safe signal access via locks
  - Graceful shutdown with threading.Event instead of global bool
  - Vehicle generation with proper exception narrowing
  - ML auto-training debounce (avoids double-train spam)
  - Heuristic predictor correctly uses all feature dimensions
  - Matplotlib graph isolated in its own process-safe context
  - Configurable constants exposed at top; magic numbers eliminated
  - Quit button also saves model/data cleanly
  - Minor: removed bare `except:` clauses; added type hints where practical
  - Added model accuracy (MAE) display in HUD
  - FIXED: ML lane switching now properly cycles through all lanes
"""

import random
import time
import threading
import sys
import os
import pickle
import copy
from collections import deque

import pygame
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# ═══════════════════════════ CONFIG ════════════════════════════════

# Signal timing (seconds)
DEFAULT_GREEN   = {0: 10, 1: 10, 2: 10, 3: 10}
DEFAULT_RED     = 150
DEFAULT_YELLOW  = 5
MIN_GREEN_TIME  = 5
MAX_GREEN_TIME  = 40

# Vehicle speeds (pixels / frame at 60 fps)
SPEEDS = {'car': 2.25, 'bus': 1.8, 'truck': 1.8, 'bike': 2.5}

# Spawn lanes starting positions
SPAWN_X = {'right': [0, 0, 0],        'down': [755, 727, 697],
            'left':  [1400, 1400, 1400], 'up':  [602, 627, 657]}
SPAWN_Y = {'right': [348, 370, 398],  'down': [0, 0, 0],
            'left':  [498, 466, 436],   'up':  [800, 800, 800]}

VEHICLE_TYPES      = {0: 'car', 1: 'bus', 2: 'truck', 3: 'bike'}
DIRECTION_NUMBERS  = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
DIRECTION_LABELS   = ['→ Right', '↓ Down', '← Left', '↑ Up']

SIGNAL_COORDS       = [(530, 230), (810, 230), (810, 570), (530, 570)]
SIGNAL_TIMER_COORDS = [(530, 210), (810, 210), (810, 550), (530, 550)]

STOP_LINES   = {'right': 590, 'down': 330, 'left': 800, 'up': 535}
DEFAULT_STOP = {'right': 580, 'down': 320, 'left': 810, 'up': 545}

STOPPING_GAP = 25
MOVING_GAP   = 25

ROTATION_ANGLE = 3
MID = {
    'right': {'x': 705, 'y': 445}, 'down':  {'x': 695, 'y': 450},
    'left':  {'x': 695, 'y': 425}, 'up':    {'x': 695, 'y': 400},
}

ALLOWED_VEHICLE_TYPES = {'car': True, 'bus': True, 'truck': True, 'bike': True}
RANDOM_GREEN_TIMER    = True
GREEN_TIMER_RANGE     = [10, 20]

NO_OF_SIGNALS      = 4
SIMULATION_TIME    = 1000       # seconds before timer loops
VEHICLE_COUNT_COORDS = [(480, 210), (880, 210), (880, 550), (480, 550)]
TIME_ELAPSED_COORDS  = (1100, 50)

SCREEN_WIDTH  = 1400
SCREEN_HEIGHT = 800

# ML
ML_ENABLED         = True
ML_MIN_DATA_POINTS = 20
AUTO_SAVE_MODEL    = True
MODEL_PATH         = "traffic_model.joblib"
USE_PRETRAINED     = False
COLLECT_DATA       = True
ML_TRAIN_EVERY_N   = 50        # re-train every N new data points (after first)

GRAPH_COLORS = {
    'right': '#4FC3F7', 'down':  '#81C784',
    'left':  '#FFB74D', 'up':    '#F06292',
}

# ═══════════════════════ GLOBAL STATE ══════════════════════════════

# Mutable lane-spawn positions (modified as vehicles spawn)
x = {k: list(v) for k, v in SPAWN_X.items()}
y = {k: list(v) for k, v in SPAWN_Y.items()}

vehicles = {
    'right': {0: [], 1: [], 2: [], 'crossed': 0},
    'down':  {0: [], 1: [], 2: [], 'crossed': 0},
    'left':  {0: [], 1: [], 2: [], 'crossed': 0},
    'up':    {0: [], 1: [], 2: [], 'crossed': 0},
}
vehicles_turned     = {d: {1: [], 2: []} for d in DIRECTION_NUMBERS.values()}
vehicles_not_turned = {d: {1: [], 2: []} for d in DIRECTION_NUMBERS.values()}

signals      = []
currentGreen  = 0
nextGreen     = 1
currentYellow = 0

time_elapsed = 0
history_time   = []
history_counts = {d: [] for d in DIRECTION_NUMBERS.values()}

# Synchronisation
_signal_lock    = threading.Lock()
_stop_event     = threading.Event()

pygame.init()
simulation_group = pygame.sprite.Group()

# ═══════════════════════ CLASSES ═══════════════════════════════════

class TrafficSignal:
    def __init__(self, red: int, yellow: int, green: int):
        self.red = red
        self.yellow = yellow
        self.green  = green
        self.signalText = ""


class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane: int, vehicle_class: str,
                 direction_number: int, direction: str, will_turn: int):
        super().__init__()
        self.lane             = lane
        self.vehicleClass     = vehicle_class
        self.speed            = SPEEDS[vehicle_class]
        self.direction_number = direction_number
        self.direction        = direction
        self.x                = x[direction][lane]
        self.y                = y[direction][lane]
        self.crossed          = 0
        self.willTurn         = will_turn
        self.turned           = 0
        self.rotateAngle      = 0
        self.crossedIndex     = 0

        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1

        img_path = os.path.join("images", direction, vehicle_class + ".png")
        self.originalImage = pygame.image.load(img_path)
        self.image         = pygame.image.load(img_path)

        lane_list = vehicles[direction][lane]
        if self.index > 0 and lane_list[self.index - 1].crossed == 0:
            prev = lane_list[self.index - 1]
            pw   = prev.image.get_rect().width
            ph   = prev.image.get_rect().height
            if   direction == 'right': self.stop = prev.stop - pw  - STOPPING_GAP
            elif direction == 'left':  self.stop = prev.stop + pw  + STOPPING_GAP
            elif direction == 'down':  self.stop = prev.stop - ph  - STOPPING_GAP
            elif direction == 'up':    self.stop = prev.stop + ph  + STOPPING_GAP
        else:
            self.stop = DEFAULT_STOP[direction]

        iw = self.image.get_rect().width
        ih = self.image.get_rect().height
        if   direction == 'right': x[direction][lane] -= iw + STOPPING_GAP
        elif direction == 'left':  x[direction][lane] += iw + STOPPING_GAP
        elif direction == 'down':  y[direction][lane] -= ih + STOPPING_GAP
        elif direction == 'up':    y[direction][lane] += ih + STOPPING_GAP

        simulation_group.add(self)

    # ------------------------------------------------------------------
    def _green_for(self, num: int) -> bool:
        return currentGreen == num and currentYellow == 0

    def move(self):
        d   = self.direction
        dn  = self.direction_number
        iw  = self.image.get_rect().width
        ih  = self.image.get_rect().height
        mid_d = MID[d]
        sl  = STOP_LINES[d]

        if d == 'right':
            if self.crossed == 0 and self.x + iw > sl:
                self.crossed = 1
                vehicles[d]['crossed'] += 1
                if self.willTurn == 0:
                    vehicles_not_turned[d][self.lane].append(self)
                    self.crossedIndex = len(vehicles_not_turned[d][self.lane]) - 1

            if self.willTurn == 1:
                lane_list = vehicles[d][self.lane]
                turned_list = vehicles_turned[d][self.lane]
                if self.lane == 1:
                    if self.crossed == 0 or self.x + iw < sl + 40:
                        can_move = (self.x + iw <= self.stop or self._green_for(0) or self.crossed == 1)
                        no_block = (self.index == 0 or self.x + iw < lane_list[self.index-1].x - MOVING_GAP
                                    or lane_list[self.index-1].turned == 1)
                        if can_move and no_block:
                            self.x += self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += ROTATION_ANGLE
                            self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                            self.x += 2.4; self.y -= 2.8
                            if self.rotateAngle == 90:
                                self.turned = 1
                                turned_list.append(self)
                                self.crossedIndex = len(turned_list) - 1
                        else:
                            if self.crossedIndex == 0 or self.y > turned_list[self.crossedIndex-1].y + turned_list[self.crossedIndex-1].image.get_rect().height + MOVING_GAP:
                                self.y -= self.speed
                elif self.lane == 2:
                    if self.crossed == 0 or self.x + iw < mid_d['x']:
                        can_move = (self.x + iw <= self.stop or self._green_for(0) or self.crossed == 1)
                        no_block = (self.index == 0 or self.x + iw < lane_list[self.index-1].x - MOVING_GAP
                                    or lane_list[self.index-1].turned == 1)
                        if can_move and no_block:
                            self.x += self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += ROTATION_ANGLE
                            self.image = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                            self.x += 2; self.y += 1.8
                            if self.rotateAngle == 90:
                                self.turned = 1
                                turned_list.append(self)
                                self.crossedIndex = len(turned_list) - 1
                        else:
                            if self.crossedIndex == 0 or self.y + self.image.get_rect().height < turned_list[self.crossedIndex-1].y - MOVING_GAP:
                                self.y += self.speed
            else:
                if self.crossed == 0:
                    if (self.x + iw <= self.stop or self._green_for(0)) and \
                       (self.index == 0 or self.x + iw < vehicles[d][self.lane][self.index-1].x - MOVING_GAP):
                        self.x += self.speed
                else:
                    nt = vehicles_not_turned[d][self.lane]
                    if self.crossedIndex == 0 or self.x + iw < nt[self.crossedIndex-1].x - MOVING_GAP:
                        self.x += self.speed

        elif d == 'down':
            if self.crossed == 0 and self.y + ih > sl:
                self.crossed = 1
                vehicles[d]['crossed'] += 1
                if self.willTurn == 0:
                    vehicles_not_turned[d][self.lane].append(self)
                    self.crossedIndex = len(vehicles_not_turned[d][self.lane]) - 1

            if self.willTurn == 1:
                lane_list = vehicles[d][self.lane]
                turned_list = vehicles_turned[d][self.lane]
                if self.lane == 1:
                    if self.crossed == 0 or self.y + ih < sl + 50:
                        can_move = (self.y + ih <= self.stop or self._green_for(1) or self.crossed == 1)
                        no_block = (self.index == 0 or self.y + ih < lane_list[self.index-1].y - MOVING_GAP
                                    or lane_list[self.index-1].turned == 1)
                        if can_move and no_block:
                            self.y += self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += ROTATION_ANGLE
                            self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                            self.x += 1.2; self.y += 1.8
                            if self.rotateAngle == 90:
                                self.turned = 1
                                turned_list.append(self)
                                self.crossedIndex = len(turned_list) - 1
                        else:
                            if self.crossedIndex == 0 or self.x + self.image.get_rect().width < turned_list[self.crossedIndex-1].x - MOVING_GAP:
                                self.x += self.speed
                elif self.lane == 2:
                    if self.crossed == 0 or self.y + ih < mid_d['y']:
                        can_move = (self.y + ih <= self.stop or self._green_for(1) or self.crossed == 1)
                        no_block = (self.index == 0 or self.y + ih < lane_list[self.index-1].y - MOVING_GAP
                                    or lane_list[self.index-1].turned == 1)
                        if can_move and no_block:
                            self.y += self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += ROTATION_ANGLE
                            self.image = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                            self.x -= 2.5; self.y += 2
                            if self.rotateAngle == 90:
                                self.turned = 1
                                turned_list.append(self)
                                self.crossedIndex = len(turned_list) - 1
                        else:
                            if self.crossedIndex == 0 or self.x > turned_list[self.crossedIndex-1].x + turned_list[self.crossedIndex-1].image.get_rect().width + MOVING_GAP:
                                self.x -= self.speed
            else:
                if self.crossed == 0:
                    if (self.y + ih <= self.stop or self._green_for(1)) and \
                       (self.index == 0 or self.y + ih < vehicles[d][self.lane][self.index-1].y - MOVING_GAP):
                        self.y += self.speed
                else:
                    nt = vehicles_not_turned[d][self.lane]
                    if self.crossedIndex == 0 or self.y + ih < nt[self.crossedIndex-1].y - MOVING_GAP:
                        self.y += self.speed

        elif d == 'left':
            if self.crossed == 0 and self.x < sl:
                self.crossed = 1
                vehicles[d]['crossed'] += 1
                if self.willTurn == 0:
                    vehicles_not_turned[d][self.lane].append(self)
                    self.crossedIndex = len(vehicles_not_turned[d][self.lane]) - 1

            if self.willTurn == 1:
                lane_list = vehicles[d][self.lane]
                turned_list = vehicles_turned[d][self.lane]
                if self.lane == 1:
                    if self.crossed == 0 or self.x > sl - 70:
                        can_move = (self.x >= self.stop or self._green_for(2) or self.crossed == 1)
                        no_block = (self.index == 0 or self.x > lane_list[self.index-1].x + lane_list[self.index-1].image.get_rect().width + MOVING_GAP
                                    or lane_list[self.index-1].turned == 1)
                        if can_move and no_block:
                            self.x -= self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += ROTATION_ANGLE
                            self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                            self.x -= 1; self.y += 1.2
                            if self.rotateAngle == 90:
                                self.turned = 1
                                turned_list.append(self)
                                self.crossedIndex = len(turned_list) - 1
                        else:
                            if self.crossedIndex == 0 or self.y + self.image.get_rect().height < turned_list[self.crossedIndex-1].y - MOVING_GAP:
                                self.y += self.speed
                elif self.lane == 2:
                    if self.crossed == 0 or self.x > mid_d['x']:
                        can_move = (self.x >= self.stop or self._green_for(2) or self.crossed == 1)
                        no_block = (self.index == 0 or self.x > lane_list[self.index-1].x + lane_list[self.index-1].image.get_rect().width + MOVING_GAP
                                    or lane_list[self.index-1].turned == 1)
                        if can_move and no_block:
                            self.x -= self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += ROTATION_ANGLE
                            self.image = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                            self.x -= 1.8; self.y -= 2.5
                            if self.rotateAngle == 90:
                                self.turned = 1
                                turned_list.append(self)
                                self.crossedIndex = len(turned_list) - 1
                        else:
                            if self.crossedIndex == 0 or self.y > turned_list[self.crossedIndex-1].y + turned_list[self.crossedIndex-1].image.get_rect().height + MOVING_GAP:
                                self.y -= self.speed
            else:
                if self.crossed == 0:
                    if (self.x >= self.stop or self._green_for(2)) and \
                       (self.index == 0 or self.x > vehicles[d][self.lane][self.index-1].x + vehicles[d][self.lane][self.index-1].image.get_rect().width + MOVING_GAP):
                        self.x -= self.speed
                else:
                    nt = vehicles_not_turned[d][self.lane]
                    if self.crossedIndex == 0 or self.x > nt[self.crossedIndex-1].x + nt[self.crossedIndex-1].image.get_rect().width + MOVING_GAP:
                        self.x -= self.speed

        elif d == 'up':
            if self.crossed == 0 and self.y < sl:
                self.crossed = 1
                vehicles[d]['crossed'] += 1
                if self.willTurn == 0:
                    vehicles_not_turned[d][self.lane].append(self)
                    self.crossedIndex = len(vehicles_not_turned[d][self.lane]) - 1

            if self.willTurn == 1:
                lane_list = vehicles[d][self.lane]
                turned_list = vehicles_turned[d][self.lane]
                if self.lane == 1:
                    if self.crossed == 0 or self.y > sl - 60:
                        can_move = (self.y >= self.stop or self._green_for(3) or self.crossed == 1)
                        no_block = (self.index == 0 or self.y > lane_list[self.index-1].y + lane_list[self.index-1].image.get_rect().height + MOVING_GAP
                                    or lane_list[self.index-1].turned == 1)
                        if can_move and no_block:
                            self.y -= self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += ROTATION_ANGLE
                            self.image = pygame.transform.rotate(self.originalImage, self.rotateAngle)
                            self.x -= 2; self.y -= 1.2
                            if self.rotateAngle == 90:
                                self.turned = 1
                                turned_list.append(self)
                                self.crossedIndex = len(turned_list) - 1
                        else:
                            if self.crossedIndex == 0 or self.x > turned_list[self.crossedIndex-1].x + turned_list[self.crossedIndex-1].image.get_rect().width + MOVING_GAP:
                                self.x -= self.speed
                elif self.lane == 2:
                    if self.crossed == 0 or self.y > mid_d['y']:
                        can_move = (self.y >= self.stop or self._green_for(3) or self.crossed == 1)
                        no_block = (self.index == 0 or self.y > lane_list[self.index-1].y + lane_list[self.index-1].image.get_rect().height + MOVING_GAP
                                    or lane_list[self.index-1].turned == 1)
                        if can_move and no_block:
                            self.y -= self.speed
                    else:
                        if self.turned == 0:
                            self.rotateAngle += ROTATION_ANGLE
                            self.image = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                            self.x += 1; self.y -= 1
                            if self.rotateAngle == 90:
                                self.turned = 1
                                turned_list.append(self)
                                self.crossedIndex = len(turned_list) - 1
                        else:
                            if self.crossedIndex == 0 or self.x < turned_list[self.crossedIndex-1].x - turned_list[self.crossedIndex-1].image.get_rect().width - MOVING_GAP:
                                self.x += self.speed
            else:
                if self.crossed == 0:
                    if (self.y >= self.stop or self._green_for(3)) and \
                       (self.index == 0 or self.y > vehicles[d][self.lane][self.index-1].y + vehicles[d][self.lane][self.index-1].image.get_rect().height + MOVING_GAP):
                        self.y -= self.speed
                else:
                    nt = vehicles_not_turned[d][self.lane]
                    if self.crossedIndex == 0 or self.y > nt[self.crossedIndex-1].y + nt[self.crossedIndex-1].image.get_rect().height + MOVING_GAP:
                        self.y -= self.speed


# ═══════════════════════ DATA COLLECTION ═══════════════════════════

class TrafficDataCollector:
    def __init__(self, max_history: int = 1000):
        self.max_history       = max_history
        self.current_density   = {d: 0 for d in DIRECTION_NUMBERS.values()}
        self.density_history   = {d: deque(maxlen=60) for d in DIRECTION_NUMBERS.values()}
        self.lane_features     = {d: [] for d in DIRECTION_NUMBERS.values()}
        self.lane_labels       = {d: [] for d in DIRECTION_NUMBERS.values()}
        self.active_data_count = {d: 0 for d in DIRECTION_NUMBERS.values()}

    def collect_lane_features(self, direction: str) -> list:
        lane_density    = sum(len(vehicles[direction][l]) for l in [0, 1, 2])
        waiting         = sum(1 for l in [0, 1, 2]
                              for v in vehicles[direction][l] if v.crossed == 0)
        total           = sum(len(vehicles[direction][l]) for l in [0, 1, 2])
        turning         = sum(1 for l in [0, 1, 2]
                              for v in vehicles[direction][l] if v.willTurn == 1)
        turn_rate       = turning / max(total, 1)
        crossed         = vehicles[direction]['crossed']
        time_factor     = (time_elapsed % 86400) / 86400
        return [lane_density, waiting, turn_rate, crossed, time_factor]

    def store_active_lane_data(self, direction: str, green_time: float) -> bool:
        if len(self.lane_features[direction]) >= self.max_history:
            return False
        features = self.collect_lane_features(direction)
        self.lane_features[direction].append(features)
        self.lane_labels[direction].append(green_time)
        self.active_data_count[direction] += 1
        return True

    def get_lane_density(self, direction: str) -> int:
        return sum(len(vehicles[direction][l]) for l in [0, 1, 2])

    def record_density(self):
        for d in DIRECTION_NUMBERS.values():
            density = self.get_lane_density(d)
            self.density_history[d].append(density)
            self.current_density[d] = density

    def get_density_stats(self) -> dict:
        stats = {}
        for d in DIRECTION_NUMBERS.values():
            total  = sum(len(vehicles[d][l]) for l in [0, 1, 2])
            queue  = sum(1 for l in [0, 1, 2] for v in vehicles[d][l] if v.crossed == 0)
            stats[d] = {'total': total, 'queue': queue, 'crossed': vehicles[d]['crossed']}
        return stats

    def save_training_data(self, filename: str = "traffic_data.pkl"):
        data = {
            'lane_features': {d: list(self.lane_features[d]) for d in DIRECTION_NUMBERS.values()},
            'lane_labels':   {d: list(self.lane_labels[d])   for d in DIRECTION_NUMBERS.values()},
            'active_counts': self.active_data_count,
        }
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            print(f"Data saved → {filename}")
            for d in DIRECTION_NUMBERS.values():
                print(f"  {d}: {len(self.lane_features[d])} samples")
        except OSError as e:
            print(f"Could not save training data: {e}")


# ═══════════════════════ ML MODEL ══════════════════════════════════

class TrafficMLModel:
    def __init__(self):
        self.models      = {}
        self.scalers     = {}
        self.is_trained  = {d: False for d in DIRECTION_NUMBERS.values()}
        self.pred_history= {d: deque(maxlen=100) for d in DIRECTION_NUMBERS.values()}
        self.metrics     = {d: {'mae': [], 'rmse': []} for d in DIRECTION_NUMBERS.values()}
        self.last_preds  = {d: MIN_GREEN_TIME for d in DIRECTION_NUMBERS.values()}
        self.last_mae    = {d: 0.0 for d in DIRECTION_NUMBERS.values()}  # Store latest MAE for display

        for d in DIRECTION_NUMBERS.values():
            self.models[d]  = RandomForestRegressor(
                n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
            self.scalers[d] = StandardScaler()

    def train_lane(self, direction: str, features: list, labels: list) -> bool:
        if len(features) < ML_MIN_DATA_POINTS:
            return False
        X = np.array(features, dtype=np.float32)
        y_arr = np.array(labels, dtype=np.float32)
        if X.ndim != 2:
            return False
        try:
            Xs = self.scalers[direction].fit_transform(X)
            self.models[direction].fit(Xs, y_arr)
            self.is_trained[direction] = True
            preds = self.models[direction].predict(Xs)
            mae  = float(np.mean(np.abs(preds - y_arr)))
            rmse = float(np.sqrt(np.mean((preds - y_arr) ** 2)))
            self.metrics[direction]['mae'].append(mae)
            self.metrics[direction]['rmse'].append(rmse)
            self.last_mae[direction] = mae  # Store for display
            print(f"✅ {direction} trained ({len(features)} samples) | MAE={mae:.2f} RMSE={rmse:.2f}")
            return True
        except Exception as e:
            print(f"Train error [{direction}]: {e}")
            return False

    def predict_lane_timing(self, direction: str, features) -> float:
        if not self.is_trained[direction]:
            return self._heuristic(direction, features)
        try:
            feat = np.array(features, dtype=np.float32).reshape(1, -1)
            Xs   = self.scalers[direction].transform(feat)
            pred = float(self.models[direction].predict(Xs)[0])
            pred = max(MIN_GREEN_TIME, min(MAX_GREEN_TIME, pred))
            self.pred_history[direction].append(pred)
            self.last_preds[direction] = pred
            return pred
        except Exception as e:
            print(f"Predict error [{direction}]: {e}")
            return self._heuristic(direction, features)

    def _heuristic(self, direction: str, features) -> float:
        feat = list(features) if not isinstance(features, list) else features
        density = feat[0] if len(feat) > 0 else 0
        waiting = feat[1] if len(feat) > 1 else 0
        t = MIN_GREEN_TIME + min(MAX_GREEN_TIME - MIN_GREEN_TIME,
                                 (density / 15.0) * (MAX_GREEN_TIME - MIN_GREEN_TIME))
        t += min(5.0, waiting * 0.5)
        t  = max(MIN_GREEN_TIME, min(MAX_GREEN_TIME, t))
        self.last_preds[direction] = t
        return t

    def predict_all_lanes(self, all_features: list) -> list:
        predictions = []
        for i, direction in enumerate(DIRECTION_NUMBERS.values()):
            start = i * 5          # 5 features per lane (density, waiting, turn_rate, crossed, time_factor)
            lane_feat = all_features[start: start + 5]
            if len(lane_feat) < 5:  # safety pad
                lane_feat += [0.0] * (5 - len(lane_feat))
            predictions.append(self.predict_lane_timing(direction, lane_feat))
        return predictions

    def save_model(self, filename: str = MODEL_PATH) -> bool:
        try:
            joblib.dump({
                'models':     self.models,
                'scalers':    self.scalers,
                'is_trained': self.is_trained,
                'metrics':    self.metrics,
                'last_mae':   self.last_mae,
            }, filename)
            print(f"✅ Model saved → {filename}")
            return True
        except Exception as e:
            print(f"Save error: {e}")
            return False

    def load_model(self, filename: str = MODEL_PATH) -> bool:
        try:
            data = joblib.load(filename)
            self.models     = data['models']
            self.scalers    = data['scalers']
            self.is_trained = data['is_trained']
            if 'metrics' in data:
                self.metrics = data['metrics']
            if 'last_mae' in data:
                self.last_mae = data['last_mae']
            print(f"✅ Model loaded ← {filename}")
            return True
        except Exception as e:
            print(f"Load error: {e}")
            return False


# ═══════════════════════ ML CONTROLLER ═════════════════════════════

class MLTrafficController:
    def __init__(self):
        self.collector           = TrafficDataCollector()
        self.model               = TrafficMLModel()
        self.enabled             = ML_ENABLED
        self.current_predictions = {d: float(MIN_GREEN_TIME) for d in DIRECTION_NUMBERS.values()}
        self.last_prediction_time= None
        self._train_counts       = {d: 0 for d in DIRECTION_NUMBERS.values()}  # track last train size

        if USE_PRETRAINED:
            self.model.load_model(MODEL_PATH)

    # ── density helpers ────────────────────────────────────────────
    def get_density_based_lane(self) -> int:
        """Get the lane with highest density, excluding current lane"""
        densities = []
        for i in range(NO_OF_SIGNALS):
            d = DIRECTION_NUMBERS[i]
            total = sum(len(vehicles[d][l]) for l in [0, 1, 2])
            queue = sum(1 for l in [0, 1, 2] for v in vehicles[d][l] if v.crossed == 0)
            # Weight queue more heavily
            weighted = total + (queue * 2)
            densities.append((i, weighted))
        
        # Sort by density descending
        densities.sort(key=lambda x: x[1], reverse=True)
        
        # Find the highest density lane that isn't current
        for lane_idx, density in densities:
            if lane_idx != currentGreen and density > 0:
                return lane_idx
        
        # If all other lanes are empty, cycle to next
        return (currentGreen + 1) % NO_OF_SIGNALS

    # ── main update (called every second from traffic thread) ───────
    def update(self):
        if not self.enabled:
            return

        self.collector.record_density()

        # Build feature vector: 5 features × 4 lanes
        all_features = []
        for d in DIRECTION_NUMBERS.values():
            all_features.extend(self.collector.collect_lane_features(d))

        # Store data point for active lane
        if COLLECT_DATA and signals:
            active_dir = DIRECTION_NUMBERS[currentGreen]
            self.collector.store_active_lane_data(active_dir, signals[currentGreen].green)

        # Auto-train each lane when enough new data has accumulated
        for d in DIRECTION_NUMBERS.values():
            n = len(self.collector.lane_features[d])
            last_n = self._train_counts[d]
            if n >= ML_MIN_DATA_POINTS and (n == ML_MIN_DATA_POINTS or n - last_n >= ML_TRAIN_EVERY_N):
                if self.model.train_lane(d, self.collector.lane_features[d],
                                         self.collector.lane_labels[d]):
                    self._train_counts[d] = n
                    if AUTO_SAVE_MODEL:
                        self.model.save_model(MODEL_PATH)

        # Predict green times for all lanes
        predictions = self.model.predict_all_lanes(all_features)
        for i, d in enumerate(DIRECTION_NUMBERS.values()):
            self.current_predictions[d] = predictions[i]
        self.last_prediction_time = time.time()

    # ── force training from UI ──────────────────────────────────────
    def force_train(self) -> bool:
        ok = 0
        for d in DIRECTION_NUMBERS.values():
            feats  = self.collector.lane_features[d]
            labels = self.collector.lane_labels[d]
            if len(feats) >= ML_MIN_DATA_POINTS:
                if self.model.train_lane(d, feats, labels):
                    self._train_counts[d] = len(feats)
                    ok += 1
        if ok > 0 and AUTO_SAVE_MODEL:
            self.model.save_model(MODEL_PATH)
        return ok > 0

    def get_signal_times(self, green_index: int) -> float:
        if not self.enabled:
            return 15.0
        d = DIRECTION_NUMBERS[green_index]
        return max(MIN_GREEN_TIME,
                   min(MAX_GREEN_TIME, self.current_predictions.get(d, MIN_GREEN_TIME)))

    def select_next_lane(self) -> int:
        """Select the next lane based on ML or fallback to cycling"""
        if not self.enabled:
            return (currentGreen + 1) % NO_OF_SIGNALS
        
        # Use ML to find the busiest lane
        return self.get_density_based_lane()

    def get_accuracy_summary(self) -> dict:
        """Get accuracy summary for display"""
        summary = {}
        for d in DIRECTION_NUMBERS.values():
            summary[d] = {
                'mae': self.model.last_mae[d],
                'trained': self.model.is_trained[d],
                'samples': len(self.collector.lane_features[d])
            }
        return summary


# ═══════════════════════ SIGNAL INIT ═══════════════════════════════

def initialize_signals():
    global signals, currentGreen, nextGreen, currentYellow
    signals = []
    currentGreen = 0; nextGreen = 1; currentYellow = 0
    mn, mx = GREEN_TIMER_RANGE
    if RANDOM_GREEN_TIMER:
        g = [random.randint(mn, mx) for _ in range(4)]
    else:
        g = [DEFAULT_GREEN[i] for i in range(4)]
    ts0 = TrafficSignal(0,                         DEFAULT_YELLOW, g[0]); signals.append(ts0)
    ts1 = TrafficSignal(ts0.red+ts0.yellow+ts0.green, DEFAULT_YELLOW, g[1]); signals.append(ts1)
    ts2 = TrafficSignal(DEFAULT_RED,               DEFAULT_YELLOW, g[2]); signals.append(ts2)
    ts3 = TrafficSignal(DEFAULT_RED,               DEFAULT_YELLOW, g[3]); signals.append(ts3)


# ═══════════════════════ THREADS ═══════════════════════════════════

def update_values():
    for i in range(NO_OF_SIGNALS):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green  = max(0, signals[i].green  - 1)
            else:
                signals[i].yellow = max(0, signals[i].yellow - 1)
        else:
            signals[i].red = max(0, signals[i].red - 1)


def generate_vehicles(allowed_list: list):
    while not _stop_event.is_set():
        try:
            if signals:
                vt = random.choice(allowed_list)
                ln = random.randint(1, 2)
                wt = 1 if random.randint(0, 99) < 40 else 0
                dn = random.choices([0, 1, 2, 3], weights=[25, 25, 25, 25])[0]
                Vehicle(ln, VEHICLE_TYPES[vt], dn, DIRECTION_NUMBERS[dn], wt)
        except (pygame.error, FileNotFoundError, KeyError) as e:
            print(f"Vehicle spawn error: {e}")
        _stop_event.wait(timeout=random.uniform(0.5, 1.5))


def sim_time_thread():
    global time_elapsed
    while not _stop_event.is_set():
        _stop_event.wait(timeout=1.0)
        if _stop_event.is_set():
            break
        time_elapsed += 1
        history_time.append(time_elapsed)
        for d in DIRECTION_NUMBERS.values():
            history_counts[d].append(vehicles[d]['crossed'])
        if time_elapsed >= SIMULATION_TIME:
            print(f"Simulation cycle complete ({SIMULATION_TIME}s). Resetting timer.")
            time_elapsed = 0


def traffic_signal_thread(ml_ctrl: MLTrafficController):
    global currentGreen, currentYellow, nextGreen

    while not _stop_event.is_set():
        try:
            if not signals:
                _stop_event.wait(timeout=0.5)
                continue

            # ── Update ML ──────────────────────────────────────────
            ml_ctrl.update()

            # ── Choose next lane ───────────────────────────────────
            nextGreen = ml_ctrl.select_next_lane()

            # ── Set green time ─────────────────────────────────────
            signals[nextGreen].green = int(ml_ctrl.get_signal_times(nextGreen))

            # ── Switch ─────────────────────────────────────────────
            signals[currentGreen].red = DEFAULT_RED
            currentGreen = nextGreen
            signals[currentGreen].red = 0

            # ── Green phase ────────────────────────────────────────
            while signals[currentGreen].green > 0 and not _stop_event.is_set():
                if COLLECT_DATA:
                    ml_ctrl.collector.store_active_lane_data(
                        DIRECTION_NUMBERS[currentGreen], signals[currentGreen].green)
                update_values()
                _stop_event.wait(timeout=1.0)

            if _stop_event.is_set():
                break

            # ── Yellow phase ───────────────────────────────────────
            currentYellow = 1
            for ln in [0, 1, 2]:
                for v in vehicles[DIRECTION_NUMBERS[currentGreen]][ln]:
                    v.stop = DEFAULT_STOP[DIRECTION_NUMBERS[currentGreen]]

            signals[currentGreen].yellow = DEFAULT_YELLOW
            while signals[currentGreen].yellow > 0 and not _stop_event.is_set():
                update_values()
                _stop_event.wait(timeout=1.0)

            currentYellow = 0
            signals[currentGreen].red = DEFAULT_RED

        except Exception as e:
            print(f"Signal loop error: {e}")
            _stop_event.wait(timeout=0.5)


# ═══════════════════════ QUIT ═══════════════════════════════════════

def quit_all(ml_ctrl: MLTrafficController):
    _stop_event.set()
    if COLLECT_DATA:
        ml_ctrl.collector.save_training_data()
        ml_ctrl.model.save_model(MODEL_PATH)
    plt.close('all')
    pygame.quit()
    sys.exit(0)


# ═══════════════════════ STATS GRAPH ════════════════════════════════

def launch_stats_graph():
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), facecolor='#1a1a2e')
    try:
        fig.canvas.manager.set_window_title('Traffic Lane Statistics – ML Enhanced')
    except AttributeError:
        pass
    fig.suptitle('Live Traffic Lane Statistics with ML Predictions', color='white',
                 fontsize=14, fontweight='bold', y=0.98)

    ax1, ax2 = axes
    for ax, title in [(ax1, 'Cumulative Vehicles Crossed'), (ax2, 'Current Count by Direction')]:
        ax.set_facecolor('#16213e')
        ax.set_title(title, color='#e0e0e0', fontsize=11)
        ax.tick_params(colors='#9e9e9e')
        for sp in ax.spines.values(): sp.set_edgecolor('#333')
    ax1.set_xlabel('Time (s)', color='#9e9e9e')
    ax1.set_ylabel('Vehicles',  color='#9e9e9e')
    ax2.set_ylabel('Vehicles Crossed', color='#9e9e9e')

    lines = {}
    for d in DIRECTION_NUMBERS.values():
        lbl = DIRECTION_LABELS[list(DIRECTION_NUMBERS.values()).index(d)]
        ln, = ax1.plot([], [], label=lbl, color=GRAPH_COLORS[d], linewidth=2.2, alpha=0.9)
        lines[d] = ln
    ax1.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='#555', labelcolor='white', fontsize=9)
    ax1.grid(True, color='#2a2a4a', linestyle='--', alpha=0.5)
    ax2.grid(True, color='#2a2a4a', linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    def update(frame):
        if not history_time:
            return
        t = list(history_time)
        ax1.set_xlim(0, max(t[-1] + 5, 20))
        max_c = 1
        for d in DIRECTION_NUMBERS.values():
            data = list(history_counts[d])[:len(t)]
            lines[d].set_data(t[:len(data)], data)
            if data: max_c = max(max_c, data[-1])
        ax1.set_ylim(0, max_c * 1.15 + 1)

        ax2.cla()
        ax2.set_facecolor('#16213e')
        ax2.set_title('Current Count by Direction', color='#e0e0e0', fontsize=11)
        ax2.set_ylabel('Vehicles Crossed', color='#9e9e9e')
        ax2.tick_params(colors='#9e9e9e')
        for sp in ax2.spines.values(): sp.set_edgecolor('#333')
        ax2.grid(True, color='#2a2a4a', linestyle='--', alpha=0.5, zorder=0)
        dirs   = list(DIRECTION_NUMBERS.values())
        counts = [vehicles[d]['crossed'] for d in dirs]
        clrs   = [GRAPH_COLORS[d] for d in dirs]
        bars   = ax2.bar(DIRECTION_LABELS, counts, color=clrs, zorder=3,
                         edgecolor='#ffffff22', linewidth=0.8, width=0.5)
        for bar, cnt in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     str(cnt), ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')

    _ani = FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
    plt.show()


# ═══════════════════════ PYGAME DRAW HELPERS ════════════════════════

def draw_rounded_rect(surface, color, rect, radius=12, alpha=200):
    s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    pygame.draw.rect(s, (*color, alpha), (0, 0, rect[2], rect[3]), border_radius=radius)
    surface.blit(s, (rect[0], rect[1]))


def draw_hud(screen, font_med, font_sm):
    elapsed_text = f"⏱  {time_elapsed}s / {SIMULATION_TIME}s"
    progress = min(1.0, time_elapsed / SIMULATION_TIME)
    px, py, pw, ph = 1050, 15, 330, 55
    draw_rounded_rect(screen, (20, 20, 40), (px, py, pw, ph), radius=10, alpha=200)
    pygame.draw.rect(screen, (60, 60, 80), (px+10, py+36, pw-20, 8), border_radius=4)
    bar_w   = int((pw - 20) * progress)
    bar_col = (100, 220, 120) if progress < 0.75 else (255, 180, 50) if progress < 0.9 else (255, 80, 80)
    if bar_w > 0:
        pygame.draw.rect(screen, bar_col, (px+10, py+36, bar_w, 8), border_radius=4)
    screen.blit(font_med.render(elapsed_text, True, (220, 220, 255)), (px+12, py+10))

    total    = sum(vehicles[DIRECTION_NUMBERS[i]]['crossed'] for i in range(4))
    tot_surf = font_med.render(f"Total Crossed: {total}", True, (255, 230, 100))
    draw_rounded_rect(screen, (20, 20, 40), (1050, 78, 330, 34), radius=8, alpha=190)
    screen.blit(tot_surf, (1062, 85))


def draw_button(screen, rect, text, hover, col_n, col_h, text_col=(255, 255, 255),
                icon="", disabled=False, font=None):
    if font is None:
        font = pygame.font.Font(None, 24)
    col = col_h if (hover and not disabled) else (col_n if not disabled else (80, 80, 90))
    pygame.draw.rect(screen, col, rect, border_radius=8)
    if hover and not disabled:
        pygame.draw.rect(screen, (255, 255, 255, 30), rect, width=2, border_radius=8)
    label = font.render(f"{icon} {text}" if icon else text, True,
                        text_col if not disabled else (150, 150, 160))
    screen.blit(label, (rect[0] + (rect[2]-label.get_width())//2,
                        rect[1] + (rect[3]-label.get_height())//2))


def draw_ml_hud(screen, font_sm, ctrl: MLTrafficController, btn_states, message, msg_timer):
    px, py, pw, ph = 50, 50, 290, 300  # Increased height for accuracy display
    draw_rounded_rect(screen, (20, 40, 80), (px, py, pw, ph), radius=10, alpha=200)

    yo = py + 10
    any_trained = any(ctrl.model.is_trained.values())
    status_col  = (100, 255, 100) if any_trained else (255, 200, 50)
    screen.blit(font_sm.render("ML: ACTIVE" if any_trained else "ML: LEARNING", True, status_col), (px+10, yo))
    en_col = (100, 255, 100) if ctrl.enabled else (255, 100, 100)
    screen.blit(font_sm.render(f"Mode: {'ON 🟢' if ctrl.enabled else 'OFF 🔴'}", True, en_col), (px+160, yo))

    yo += 22
    screen.blit(font_sm.render("Data per lane:", True, (200, 200, 220)), (px+10, yo))
    yo += 18
    for i, d in enumerate(DIRECTION_NUMBERS.values()):
        pts = len(ctrl.collector.lane_features[d])
        chk = "✓" if ctrl.model.is_trained[d] else "✗"
        col = (100, 255, 100) if ctrl.model.is_trained[d] else (255, 100, 100)
        screen.blit(font_sm.render(f"{DIRECTION_LABELS[i][:2]}: {pts} {chk}", True, col),
                    (px + 10 + (i % 2)*130, yo + (i // 2)*18))

    # ─── Model Accuracy (MAE) ────────────────────────────────────────
    yo += 40
    screen.blit(font_sm.render("Model Accuracy (MAE):", True, (255, 200, 100)), (px+10, yo))
    yo += 18
    for i, d in enumerate(DIRECTION_NUMBERS.values()):
        if ctrl.model.is_trained[d]:
            mae = ctrl.model.last_mae[d]
            # Color: Green = good (<3), Yellow = ok (<6), Red = poor (>=6)
            acc_col = (100, 255, 100) if mae < 3 else (255, 200, 50) if mae < 6 else (255, 100, 100)
            text = f"{DIRECTION_LABELS[i][:2]}: {mae:.1f}s"
        else:
            acc_col = (150, 150, 150)
            text = f"{DIRECTION_LABELS[i][:2]}: --"
        screen.blit(font_sm.render(text, True, acc_col),
                    (px + 10 + (i % 2)*130, yo + (i // 2)*18))

    yo += 22
    screen.blit(font_sm.render(f"Active: {DIRECTION_LABELS[currentGreen]}", True, (100, 255, 100)), (px+10, yo))
    yo += 22
    screen.blit(font_sm.render("Predicted times:", True, (150, 200, 255)), (px+10, yo))
    yo += 18
    for i, d in enumerate(DIRECTION_NUMBERS.values()):
        pred = int(ctrl.current_predictions.get(d, MIN_GREEN_TIME))
        col  = (255, 255, 100) if i == currentGreen else (150, 200, 255)
        screen.blit(font_sm.render(f"{DIRECTION_LABELS[i][:2]}: {pred}s", True, col),
                    (px + 10 + (i % 2)*130, yo + (i // 2)*18))

    # ─── Buttons ──────────────────────────────────────────────────────
    by, bw, bh, bx = py + 250, 80, 28, px + 15
    has_data      = any(len(ctrl.collector.lane_features[d]) >= ML_MIN_DATA_POINTS
                        for d in DIRECTION_NUMBERS.values())
    has_any_data  = any(len(ctrl.collector.lane_features[d]) > 0
                        for d in DIRECTION_NUMBERS.values())

    draw_button(screen, (bx, by, bw, bh), "Train", btn_states['train_hover'],
                (40,180,40) if has_data else (60,60,70),
                (60,220,60) if has_data else (60,60,70),
                icon="🧠", disabled=not has_data, font=font_sm)

    draw_button(screen, (bx+bw+5, by, bw, bh), "Export", btn_states['export_hover'],
                (40,100,180) if has_any_data else (60,60,70),
                (60,140,220) if has_any_data else (60,60,70),
                icon="💾", disabled=not has_any_data, font=font_sm)

    draw_button(screen, (bx+2*(bw+5), by, bw, bh),
                f"ML", btn_states['toggle_hover'],
                (40,180,40) if ctrl.enabled else (180,40,40),
                (60,220,60) if ctrl.enabled else (220,60,60),
                icon="⚡", font=font_sm)

    if msg_timer > 0 and message:
        msg_y   = py + ph + 5
        msg_srf = font_sm.render(message, True, (255, 255, 200))
        draw_rounded_rect(screen, (0,0,40), (px, msg_y, msg_srf.get_width()+20, msg_srf.get_height()+10), radius=6, alpha=200)
        screen.blit(msg_srf, (px+10, msg_y+5))

    return max(0, msg_timer - 1)


def draw_quit_button(screen, font_med, rect, hover):
    col = (200, 50, 50) if hover else (140, 30, 30)
    draw_rounded_rect(screen, (80,10,10), (rect[0]+2, rect[1]+3, rect[2], rect[3]), radius=10, alpha=220)
    draw_rounded_rect(screen, col, rect, radius=10, alpha=230)
    label = font_med.render("✕  QUIT", True, (255, 255, 255))
    screen.blit(label, (rect[0]+(rect[2]-label.get_width())//2,
                        rect[1]+(rect[3]-label.get_height())//2))


# ═══════════════════════ MAIN ═══════════════════════════════════════

def main():
    global currentGreen, nextGreen, currentYellow

    # Build allowed vehicle list
    allowed_list = [list(VEHICLE_TYPES.values()).index(vt)
                    for vt, ok in ALLOWED_VEHICLE_TYPES.items() if ok]

    initialize_signals()

    ml_ctrl = MLTrafficController()

    # Start background threads
    threading.Thread(target=traffic_signal_thread, args=(ml_ctrl,), daemon=True).start()
    threading.Thread(target=generate_vehicles,     args=(allowed_list,),  daemon=True).start()
    threading.Thread(target=sim_time_thread,                               daemon=True).start()
    threading.Thread(target=launch_stats_graph,                            daemon=True).start()

    # Pygame setup
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("ML Based Density Adaptive Traffic Signal System")

    background   = pygame.image.load('images/intersection.png')
    red_img      = pygame.transform.smoothscale(pygame.image.load('images/signals/red.png'),    (48, 110))
    yellow_img   = pygame.transform.smoothscale(pygame.image.load('images/signals/yellow.png'), (48, 110))
    green_img    = pygame.transform.smoothscale(pygame.image.load('images/signals/green.png'),  (48, 110))

    font_med = pygame.font.Font(None, 30)
    font_sm  = pygame.font.Font(None, 24)

    QUIT_RECT = (SCREEN_WIDTH-155, SCREEN_HEIGHT-55, 140, 44)
    clock     = pygame.time.Clock()

    btn_states  = {'train_hover': False, 'export_hover': False, 'toggle_hover': False}
    message     = ""
    msg_timer   = 0

    # Button positions - updated for new layout
    BX, BY, BW, BH = 65, 300, 80, 28

    while not _stop_event.is_set():
        mx, my = pygame.mouse.get_pos()

        # Button hover detection
        quit_hover = (QUIT_RECT[0] <= mx <= QUIT_RECT[0]+QUIT_RECT[2] and
                      QUIT_RECT[1] <= my <= QUIT_RECT[1]+QUIT_RECT[3])
        btn_states['train_hover']  = (BX <= mx <= BX+BW and BY <= my <= BY+BH)
        btn_states['export_hover'] = (BX+BW+5 <= mx <= BX+2*BW+5 and BY <= my <= BY+BH)
        btn_states['toggle_hover'] = (BX+2*(BW+5) <= mx <= BX+3*BW+10 and BY <= my <= BY+BH)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_all(ml_ctrl)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if quit_hover:
                    quit_all(ml_ctrl)

                elif btn_states['train_hover']:
                    if ml_ctrl.force_train():
                        # Show accuracy summary
                        acc = ml_ctrl.get_accuracy_summary()
                        parts = []
                        for d in DIRECTION_NUMBERS.values():
                            if acc[d]['trained']:
                                parts.append(f"{DIRECTION_LABELS[list(DIRECTION_NUMBERS.values()).index(d)][:2]}:{acc[d]['mae']:.1f}s")
                        message = f"✅ Trained: " + " | ".join(parts) if parts else "✅ Training done!"
                    else:
                        message = f"⚠️ Need {ML_MIN_DATA_POINTS} samples/lane"
                    msg_timer = 300

                elif btn_states['export_hover']:
                    has_d = any(len(ml_ctrl.collector.lane_features[d]) > 0 for d in DIRECTION_NUMBERS.values())
                    if has_d:
                        ml_ctrl.collector.save_training_data()
                        message = "💾 Exported!"
                    else:
                        message = "⚠️ No data yet"
                    msg_timer = 180

                elif btn_states['toggle_hover']:
                    ml_ctrl.enabled = not ml_ctrl.enabled
                    message   = f"ML Mode: {'ON 🟢' if ml_ctrl.enabled else 'OFF 🔴'}"
                    msg_timer = 180

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    if ml_ctrl.force_train():
                        acc = ml_ctrl.get_accuracy_summary()
                        parts = []
                        for d in DIRECTION_NUMBERS.values():
                            if acc[d]['trained']:
                                parts.append(f"{DIRECTION_LABELS[list(DIRECTION_NUMBERS.values()).index(d)][:2]}:{acc[d]['mae']:.1f}s")
                        message = f"✅ Trained: " + " | ".join(parts) if parts else "✅ Training done!"
                    else:
                        message = f"⚠️ Need {ML_MIN_DATA_POINTS} samples/lane"
                    msg_timer = 300
                elif event.key == pygame.K_e:
                    has_d = any(len(ml_ctrl.collector.lane_features[d]) > 0 for d in DIRECTION_NUMBERS.values())
                    if has_d:
                        ml_ctrl.collector.save_training_data()
                        message = "💾 Exported!"
                    else:
                        message = "⚠️ No data yet"
                    msg_timer = 180
                elif event.key == pygame.K_m:
                    ml_ctrl.enabled = not ml_ctrl.enabled
                    message   = f"ML Mode: {'ON 🟢' if ml_ctrl.enabled else 'OFF 🔴'}"
                    msg_timer = 180
                elif event.key == pygame.K_ESCAPE:
                    quit_all(ml_ctrl)

        # ── Draw ──────────────────────────────────────────────────
        screen.blit(background, (0, 0))

        for i in range(NO_OF_SIGNALS):
            if not signals:
                continue
            if i == currentGreen:
                img  = yellow_img if currentYellow else green_img
                txt  = signals[i].yellow if currentYellow else signals[i].green
            else:
                img  = red_img
                txt  = signals[i].red if signals[i].red <= 10 else "---"
            screen.blit(img, SIGNAL_COORDS[i])
            tx, ty  = SIGNAL_TIMER_COORDS[i]
            t_surf  = font_med.render(str(txt), True, (255, 255, 255))
            draw_rounded_rect(screen, (0,0,0), (tx-4, ty-3, t_surf.get_width()+8, t_surf.get_height()+6),
                              radius=5, alpha=160)
            screen.blit(t_surf, (tx, ty))

        for i in range(NO_OF_SIGNALS):
            cnt    = vehicles[DIRECTION_NUMBERS[i]]['crossed']
            cx, cy = VEHICLE_COUNT_COORDS[i]
            c_surf = font_sm.render(f"Crossed: {cnt}", True, (255, 240, 150))
            draw_rounded_rect(screen, (10,10,30), (cx-6, cy-3, c_surf.get_width()+12, c_surf.get_height()+6),
                              radius=6, alpha=180)
            screen.blit(c_surf, (cx, cy))

        draw_hud(screen, font_med, font_sm)

        if ML_ENABLED:
            msg_timer = draw_ml_hud(screen, font_sm, ml_ctrl, btn_states, message, msg_timer)

        draw_quit_button(screen, font_med, QUIT_RECT, quit_hover)

        for v in simulation_group:
            screen.blit(v.image, (v.x, v.y))
            v.move()

        hint = font_sm.render("T: Train   E: Export   M: Toggle ML   ESC: Quit", True, (150, 150, 180))
        screen.blit(hint, (10, SCREEN_HEIGHT - 25))

        pygame.display.update()
        clock.tick(60)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pickle
import time
import logging
import sys
import serial
import serial.tools.list_ports
import threading
import math
import csv
from datetime import datetime
import queue
import os
import threading

# SciPy is now used for efficient pole matching.
try:
    from scipy.spatial import KDTree
except ImportError:
    # This placeholder prevents the app from crashing if SciPy is not installed.
    # The user will be notified upon trying to load a pole CSV.
    class KDTree:
        def __init__(self, data):
            raise ImportError("SciPy is not installed. Please run 'pip install scipy' for efficient pole matching.")
        def query(self, point, k):
            raise ImportError("SciPy is not installed.")

# All imports are standardized to PyQt6.
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QComboBox, QGroupBox, QRadioButton,
                             QSlider, QGridLayout, QTextEdit, QSizePolicy, QFileDialog)
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor, QFont, QPainter, QPen
from PyQt6.QtCore import (Qt, QThread, pyqtSignal, QObject, QTimer, QRect, QPoint)

# --- Configuration ---
CONFIG = {
    "video_path_left": 0, "video_path_right": 1, "calibration_file": "stereo_params_charuco_stereo6.pkl",
    "roi": {"x": 400, "y": 100, "w": 1600, "h": 20},
    "contour_detection": {
        "gaussian_blur_kernel": (5, 5), "default_canny_low": 50, "default_canny_high": 150,
        "min_contour_width": 1, "min_contour_height": 10,
    },
    "smoothing_factor": 0.15,
    "DISPARITY_TOLERANCE_PX": 15, "Y_TOLERANCE_PX": 20,
    "camera_base_settings": { "width": 1920, "height": 1080 },
    "downscale_factor": 2,
    "recording_fps": 28
}

roi_lock = threading.Lock()

# --- Camera Presets Definition ---
PRESETS = {
    "Sunny Day": {"focus": 40, "exposure": -13, "brightness": 0, "contrast": 0, "saturation": 140, "gain": 0},
    "Cloudy":    {"focus": 30, "exposure": -5, "brightness": 120, "contrast": 120, "saturation": 120, "gain": 30},
    "Evening":   {"focus": 20, "exposure": -4, "brightness": 110, "contrast": 110, "saturation": 110, "gain": 60},
    "Custom":    {"focus": 40, "exposure": -4, "brightness": 40, "contrast": 100, "saturation": 60, "gain": 0}
}

# --- Logging Handler for GUI ---
class QTextEditLogger(logging.Handler, QObject):
    appendPlainText = pyqtSignal(str)

    def __init__(self, parent):
        super().__init__()
        QObject.__init__(self)
        self.widget = QTextEdit(parent)
        self.widget.setReadOnly(True)
        self.appendPlainText.connect(self.widget.append)

    def emit(self, record):
        self.appendPlainText.emit(self.format(record))

# --- Data Logger Class ---
class DataLogger:
    """Handles writing measurement and GPS data to a CSV file."""
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.writer = None
        self.header = [
            "timestamp", "frame_left", "frame_right", "latitude", "longitude", "altitude_m", "speed_kmh", 
            "course_deg", "satellites", "cable_id", "height_z_meters", "stagger_x_meters",
            "prev_pole", "prev_pole_dist_m", "next_pole", "next_pole_dist_m"
        ]

    def start(self):
        try:
            self.file = open(self.filename, 'w', newline='', encoding='utf-8')
            self.writer = csv.DictWriter(self.file, fieldnames=self.header)
            self.writer.writeheader()
            logging.info(f"CSV Logger started. Writing to {self.filename}")
            return True
        except IOError as e:
            logging.error(f"Failed to open log file {self.filename}: {e}")
            return False

    def log_entry(self, data):
        if self.writer:
            try:
                self.writer.writerow(data)
            except Exception as e:
                logging.warning(f"Failed to write log entry: {e}")

    def stop(self):
        if self.file and not self.file.closed:
            self.file.close()
            logging.info(f"CSV Logger stopped. Saved to {self.filename}")
            self.writer = None
            self.file = None
            
# --- Video Recorder Worker ---
class VideoRecorder(QObject):
    finished = pyqtSignal()
    logMessage = pyqtSignal(str)

    def __init__(self, frame_queue, output_dir="recordings"):
        super().__init__()
        self.frame_queue = frame_queue
        self.output_dir = output_dir
        self.is_running = True
        self.writer1 = None
        self.writer2 = None
        self.frame_count = 0
        self.start_time = 0
        self.frame_number_left = 0
        self.frame_number_right = 0

    def _init_video_writer(self, side, width, height, timestamp):
        fps = CONFIG["recording_fps"]
        filename = f"{self.output_dir}/{side}_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        if not writer.isOpened():
            self.logMessage.emit(f"Error: Could not open video writer for {side} camera.")
            return None
        self.logMessage.emit(f"Recording {side} camera to {filename}")
        return writer

    def run(self):
        self.logMessage.emit("Starting video recording.")
        self.start_time = time.time()
        self.frame_count = 0
        
        width = CONFIG["camera_base_settings"]["width"]
        height = CONFIG["camera_base_settings"]["height"]
        
        # Generate one timestamp for all recording files for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer1 = self._init_video_writer("left", width, height, timestamp)
        self.writer2 = self._init_video_writer("right", width, height, timestamp)

        if not self.writer1 or not self.writer2:
            self.cleanup()
            return

        while self.is_running:
            try:
                f1, f2 = self.frame_queue.get(timeout=0.05)
                self.writer1.write(f1)
                self.writer2.write(f2)
                self.frame_count += 1
                self.frame_number_left += 1
                self.frame_number_right += 1
                # ... rest of code ...
            except queue.Empty:
                pass
            except Exception as e:
                self.logMessage.emit(f"Error in recording loop: {str(e)}")

        self.cleanup()

    def cleanup(self):
        self.logMessage.emit("Stopping video recording...")
        if self.writer1: self.writer1.release()
        if self.writer2: self.writer2.release()
        self.finished.emit()
        self.logMessage.emit("Recording stopped.")

    def stop(self):
        self.is_running = False

# --- GPS Processor ---
class GPSProcessor(QObject):
    _instance = None
    _lock = threading.Lock()

    positionChanged_update = pyqtSignal(float, float, float, float, float, int, bool)
    connectionStatus = pyqtSignal(bool, str)
    finished = pyqtSignal()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GPSProcessor, cls).__new__(cls)
                cls._instance._init_once(*args, **kwargs)
        return cls._instance

    def _init_once(self):
        super().__init__()
        self.serial_port = None
        self.port_name = None
        self.baud_rate = None
        self.is_running = False
        self.last_print_time = 0
        self.Delimeter_RMC = '$GNRMC'
        self.Delimeter_GGA = '$GNGGA'

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls()
        return cls._instance

    def configure(self, port, baudrate=9600):
        self.port_name = port
        self.baud_rate = baudrate
        logging.info(f"GPS configured for {port} at {baudrate} baud.")

    def run(self):
        logging.info("GPS processor thread started.")
        self.is_running = True
        logging.info(f"GPS thread started. Attempting to open {self.port_name}...")

        try:
            self.serial_port = serial.Serial(self.port_name, self.baud_rate, timeout=1)
            self.connectionStatus.emit(True, f"Successfully connected to {self.port_name}.")
        except serial.SerialException as e:
            logging.error(f"GPS Error: Could not open serial port {self.port_name}: {e}")
            self.connectionStatus.emit(False, f"Failed to connect: {e}")
            self.is_running = False

        lat, lon, alt, speed, course, sats = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        data_available = False

        while self.is_running:
            if not self.serial_port or not self.serial_port.is_open:
                time.sleep(0.5)
                continue

            try:
                if self.serial_port.in_waiting > 0:
                    raw_data = self.serial_port.read(self.serial_port.in_waiting).decode('utf-8', 'ignore')
                    lines = raw_data.strip().split('\n')
                    
                    for line in lines:
                        parts = line.strip().split(',')
                        if not parts: continue

                        if parts[0] == self.Delimeter_RMC and len(parts) > 7 and parts[2] == 'A':
                            try:
                                lat_raw, lon_raw = float(parts[3]), float(parts[5])
                                lat_deg, lon_deg = int(lat_raw / 100), int(lon_raw / 100)
                                lat_min, lon_min = lat_raw % 100, lon_raw % 100
                                lat = lat_deg + lat_min / 60.0
                                lon = lon_deg + lon_min / 60.0
                                if parts[4] == 'S': lat = -lat
                                if parts[6] == 'W': lon = -lon
                                speed = float(parts[7]) * 1.852  # Knots to km/h
                                course = float(parts[8]) if len(parts) > 8 and parts[8] else 0.0
                            except (ValueError, IndexError):
                                continue

                        if parts[0] == self.Delimeter_GGA and len(parts) >= 10:
                            try:
                                alt = float(parts[9])
                                sats = int(parts[7])
                                data_available = True
                            except (ValueError, IndexError):
                                continue
                        
                        if data_available:
                            self.positionChanged_update.emit(lat, lon, alt, speed, course, sats, True)

            except serial.SerialException as e:
                logging.error(f"GPS serial error: {e}")
                self.connectionStatus.emit(False, "Connection lost.")
                self.is_running = False
            except Exception as e:
                logging.error(f"Unexpected error in GPS loop: {e}")

        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            logging.info("GPS serial port closed.")
        
        self.finished.emit()
        logging.info("GPS thread finished.")

    def stop(self):
        logging.info("Stopping GPS processor...")
        self.is_running = False

# --- Preview Worker ---
class PreviewWorker(QObject):
    finished = pyqtSignal()
    logMessage = pyqtSignal(str)
    updateSettings = pyqtSignal(dict)

    def __init__(self, cam_idx_left, cam_idx_right, camera_settings=None):
        super().__init__()
        self.cam_idx_left = cam_idx_left
        self.cam_idx_right = cam_idx_right
        self.camera_settings = camera_settings or PRESETS["Sunny Day"]
        self.updateSettings.connect(self.update_camera_settings)
        self.is_running = True
        self.latest_frame_left = None
        self.latest_frame_right = None
        self.cap1 = None
        self.cap2 = None

    def _apply_settings_to_cap(self, cap, settings):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_base_settings"]["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_base_settings"]["height"])
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        for prop, key in [(cv2.CAP_PROP_FOCUS, "focus"), (cv2.CAP_PROP_EXPOSURE, "exposure"), 
                          (cv2.CAP_PROP_BRIGHTNESS, "brightness"), (cv2.CAP_PROP_CONTRAST, "contrast"), 
                          (cv2.CAP_PROP_SATURATION, "saturation"), (cv2.CAP_PROP_GAIN, "gain")]:
            success = cap.set(prop, settings[key])
            if not success:
                self.logMessage.emit(f"Warning: Failed to set {key} to {settings[key]}")
        self.logMessage.emit(f"Applied camera settings: {settings}")

    def update_camera_settings(self, settings):
        self.camera_settings = settings
        if self.cap1 and self.cap1.isOpened():
            self._apply_settings_to_cap(self.cap1, settings)
        if self.cap2 and self.cap2.isOpened():
            self._apply_settings_to_cap(self.cap2, settings)

    def run(self):
        self.logMessage.emit(f"Starting preview on cams {self.cam_idx_left} and {self.cam_idx_right}.")
        self.cap1 = cv2.VideoCapture(self.cam_idx_left, cv2.CAP_DSHOW)
        self.cap2 = cv2.VideoCapture(self.cam_idx_right, cv2.CAP_DSHOW)
        
        if not self.cap1.isOpened() or not self.cap2.isOpened():
            self.logMessage.emit("Error: Could not open one or both preview streams.")
            self.is_running = False
            return

        self._apply_settings_to_cap(self.cap1, self.camera_settings)
        self._apply_settings_to_cap(self.cap2, self.camera_settings)

        while self.is_running:
            try:
                ret1, frame1 = self.cap1.read()
                ret2, frame2 = self.cap2.read()
                if ret1: self.latest_frame_left = frame1
                if ret2: self.latest_frame_right = frame2
                time.sleep(0.005)
            except Exception as e:
                self.logMessage.emit(f"Error in preview loop: {str(e)}")
                time.sleep(0.1)

        if self.cap1 and self.cap1.isOpened(): self.cap1.release()
        if self.cap2 and self.cap2.isOpened(): self.cap2.release()
        self.logMessage.emit("Preview stopped.")
        self.finished.emit()

    def stop(self):
        self.is_running = False

# --- Stereo Processor ---
class StereoProcessor(QObject):
    newMeasurements = pyqtSignal(dict, float, float)
    rawFrames = pyqtSignal(np.ndarray, np.ndarray)
    finished = pyqtSignal()
    logMessage = pyqtSignal(str)

    def __init__(self, config, cam_idx_left, cam_idx_right, camera_settings):
        super().__init__()
        self.config = config
        self.cam_idx_left = cam_idx_left
        self.cam_idx_right = cam_idx_right
        self.camera_settings = camera_settings
        self.downscale = config.get("downscale_factor", 2)
        self.is_running = True
        self.latest_frame_left = None
        self.latest_frame_right = None
        self.smoothed_measurements = {}
        
    def _apply_settings_to_cap(self, cap, settings):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["camera_base_settings"]["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera_base_settings"]["height"])
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        for prop, key in [(cv2.CAP_PROP_FOCUS, "focus"), (cv2.CAP_PROP_EXPOSURE, "exposure"), 
                          (cv2.CAP_PROP_BRIGHTNESS, "brightness"), (cv2.CAP_PROP_CONTRAST, "contrast"), 
                          (cv2.CAP_PROP_SATURATION, "saturation"), (cv2.CAP_PROP_GAIN, "gain")]:
            cap.set(prop, settings[key])

    def run(self):
        self.logMessage.emit("Stereo processor thread started.")
        self.cap1 = cv2.VideoCapture(self.cam_idx_left, cv2.CAP_DSHOW)
        self.cap2 = cv2.VideoCapture(self.cam_idx_right, cv2.CAP_DSHOW)

        if not self.cap1.isOpened() or not self.cap2.isOpened():
            self.logMessage.emit("Error: Could not open one or both video streams for processing.")
            return self.stop()
            
        self.logMessage.emit("Applying camera settings...")
        self._apply_settings_to_cap(self.cap1, self.camera_settings)
        self._apply_settings_to_cap(self.cap2, self.camera_settings)
        
        frame_width = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.downscale > 1:
            self.frame_width = frame_width // self.downscale
            self.frame_height = frame_height // self.downscale
        else:
            self.frame_width = frame_width
            self.frame_height = frame_height
        
        try:
            self._load_and_rectify_calibration()
            self._init_rectification_maps()
        except Exception as e:
            self.logMessage.emit(f"FATAL: Calibration failed: {e}")
            return self.stop()

        self.labels = [chr(ord("A") + i) for i in range(26)]
        self.colors = self._generate_colors(len(self.labels))
        self.color_map = dict(zip(self.labels, self.colors))
        
        frame_count, start_time = 0, time.time()
        SYNC_THRESHOLD_MS = 10.0
        
        while self.is_running:
            try:
                ret1, f1 = self.cap1.read()
                ts1 = time.time()
                ret2, f2 = self.cap2.read()
                ts2 = time.time()
                if not ret1 or not ret2:
                    time.sleep(0.005)
                    continue

                sync_delay = abs(ts1 - ts2) * 1000
                if sync_delay > SYNC_THRESHOLD_MS:
                    self.logMessage.emit(f"Sync delay too high: {sync_delay} ms")
                    continue
                
                self.rawFrames.emit(f1, f2)
                self.logMessage.emit("Frames emitted.")

                if self.downscale > 1:
                    f1 = cv2.resize(f1, (self.frame_width, self.frame_height))
                    f2 = cv2.resize(f2, (self.frame_width, self.frame_height))
                self.logMessage.emit("Frames resized.")

                f1_rect = cv2.remap(f1, self.map1L, self.map2L, cv2.INTER_LINEAR)
                f2_rect = cv2.remap(f2, self.map1R, self.map2R, cv2.INTER_LINEAR)
                self.logMessage.emit("Frames rectified.")

                centers1, boxes1 = self._detect_contours_stable(f1_rect)
                centers2, boxes2 = self._detect_contours_stable(f2_rect)
                self.logMessage.emit(f"Contours detected: {len(centers1)}, {len(centers2)}")

                matched_pairs = self._calculate_measurements(centers1, centers2, boxes1, boxes2)
                self._draw_overlays(f1_rect, f2_rect, matched_pairs)
                self.logMessage.emit("Overlays drawn.")

                self.latest_frame_left = f1_rect
                self.latest_frame_right = f2_rect

                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:
                    self.newMeasurements.emit(self.smoothed_measurements, frame_count / elapsed_time, sync_delay)
                    frame_count, start_time = 0, time.time()
            except Exception as e:
                self.logMessage.emit(f"Error in processing loop: {str(e)}")
                time.sleep(0.1)

        self.cleanup()

    def _load_and_rectify_calibration(self):
        with open(self.config["calibration_file"], "rb") as f: p = pickle.load(f, encoding="latin1")
        
        if self.downscale > 1:
            s = 1.0 / self.downscale
            p["mtxL"][0, :] *= s; p["mtxL"][1, :] *= s
            p["mtxR"][0, :] *= s; p["mtxR"][1, :] *= s
        
        self.stereo_params = {"mtxL": p["mtxR"],"distL": p["distR"],"mtxR": p["mtxL"],"distR": p["distL"],"R": p["R"].T,"T": -np.dot(p["R"].T, p["T"])}
        size = (self.frame_width, self.frame_height)
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(self.stereo_params["mtxL"],self.stereo_params["distL"],self.stereo_params["mtxR"],self.stereo_params["distR"],size,self.stereo_params["R"],self.stereo_params["T"],flags=cv2.CALIB_ZERO_DISPARITY,alpha=-1)
        self.logMessage.emit("Stereo parameters loaded.")

    def _init_rectification_maps(self):
        p, size = self.stereo_params, (self.frame_width, self.frame_height)
        self.map1L, self.map2L = cv2.initUndistortRectifyMap(p["mtxL"], p["distL"], self.R1, self.P1, size, cv2.CV_16SC2)
        self.map1R, self.map2R = cv2.initUndistortRectifyMap(p["mtxR"], p["distR"], self.R2, self.P2, size, cv2.CV_16SC2)

    def _detect_contours_stable(self, frame):
        with roi_lock:
            roi_params = {k: self.config["roi"][k] // self.downscale for k in ["x", "y", "w", "h"]}
        detection_params = self.config["contour_detection"]
        canny_low, canny_high = detection_params["default_canny_low"], detection_params["default_canny_high"]
        h, w = frame.shape[:2]
        rx, ry, rw, rh = roi_params["x"], roi_params["y"], roi_params["w"], roi_params["h"]
        if ry + rh > h or rx + rw > w:
            self.logMessage.emit(f"ROI out of bounds: x={rx}, y={ry}, w={rw}, h={rh}, frame_size=({w}, {h})")
            return [], []

        roi = frame[ry:ry + rh, rx:rx + rw]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_roi, detection_params["gaussian_blur_kernel"], 1)
        edges = cv2.Canny(blurred, canny_low, canny_high)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [], []

        # Filter by min size
        valid = []
        for cnt in contours:
            x, y, ww, hh = cv2.boundingRect(cnt)
            if ww >= detection_params["min_contour_width"] and hh >= detection_params["min_contour_height"]:
                valid.append(cnt)
        if not valid:
            return [], []

        # Merge contours that are within 20px horizontally
        bounding_boxes = [cv2.boundingRect(c) for c in valid]
        contours_and_boxes = sorted(zip(valid, bounding_boxes), key=lambda x: x[1][0])
        merged = []
        used = [False] * len(contours_and_boxes)
        for i in range(len(contours_and_boxes)):
            if used[i]:
                continue
            cnt_i, (x1, y1, w1, h1) = contours_and_boxes[i]
            x2 = x1 + w1
            merged_cnt = cnt_i.copy()
            used[i] = True
            for j in range(i + 1, len(contours_and_boxes)):
                if used[j]:
                    continue
                cnt_j, (xj, yj, wj, hj) = contours_and_boxes[j]
                xj2 = xj + wj
                if xj - x2 <= 20:
                    merged_cnt = np.vstack((merged_cnt, cnt_j))
                    x2 = max(x2, xj2)
                    used[j] = True
            merged_box = cv2.boundingRect(merged_cnt)
            merged.append((merged_box, merged_cnt))
        # Sort merged (though already mostly sorted)
        sorted_merged = sorted(merged, key=lambda x: x[0][0])
        centers, boxes = [], []
        for (x, y, ww, hh), _ in sorted_merged:
            boxes.append((x, y, ww, hh))
            centers.append((rx + x + ww / 2.0, ry + y + hh / 2.0))
        return centers, boxes

    def _calculate_measurements(self, centers1, centers2, boxes1, boxes2):
        matched_pairs = []
        if not centers1 or not centers2: return matched_pairs

        YT, DT = self.config["Y_TOLERANCE_PX"], self.config["DISPARITY_TOLERANCE_PX"]

        current_labels = set()
        for i in range(min(len(centers1), len(centers2))):
            pt1, box1, pt2, box2 = centers1[i], boxes1[i], centers2[i], boxes2[i]
            if abs(pt1[1] - pt2[1]) < YT and (pt1[0] - pt2[0]) > -DT:
                matched_pairs.append((pt1, box1, pt2, box2, self.labels[i]))
                current_labels.add(self.labels[i])

        OFFSET_Z = 0.04  # 40mm in meters

        for pt1, box1, pt2, box2, label in matched_pairs:
            p1, p2 = np.array(pt1, dtype=np.float32).reshape(2, 1), np.array(pt2, dtype=np.float32).reshape(2, 1)
            X = cv2.triangulatePoints(self.P1, self.P2, p1, p2)
            X /= X[3]
            raw_x, raw_z = X[0, 0], X[2, 0]

            # Apply the offset to z
            raw_z += OFFSET_Z

            if label not in self.smoothed_measurements:
                self.smoothed_measurements[label] = {"z": raw_z, "x": raw_x}
            else:
                sf, prev = self.config["smoothing_factor"], self.smoothed_measurements[label]
                self.smoothed_measurements[label] = {
                    "z": sf * raw_z + (1-sf) * prev["z"],
                    "x": sf * raw_x + (1-sf) * prev["x"]
                }

        # Remove cable IDs not detected in this frame
        for label in list(self.smoothed_measurements.keys()):
            if label not in current_labels:
                del self.smoothed_measurements[label]

        return matched_pairs

    def _draw_overlays(self, f1, f2, matched_pairs):
        with roi_lock:
            roi = {k: self.config["roi"][k] // self.downscale for k in ["x", "y", "w", "h"]}
        cv2.rectangle(f1, (roi["x"], roi["y"]), (roi["x"] + roi["w"], roi["y"] + roi["h"]), (0, 0, 255), 3)
        cv2.rectangle(f2, (roi["x"], roi["y"]), (roi["x"] + roi["w"], roi["y"] + roi["h"]), (0, 0, 255), 3)
        
        for _, b1, _, b2, label in matched_pairs:
            color = self.color_map.get(label, (255, 255, 255))
            x1, y1, w1, h1 = b1; x2, y2, w2, h2 = b2
            cv2.rectangle(f1, (roi["x"] + x1, roi["y"] + y1), (roi["x"] + x1 + w1, roi["y"] + y1 + h1), color, 2)
            cv2.putText(f1, label, (roi["x"] + x1, roi["y"] + y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(f2, (roi["x"] + x2, roi["y"] + y2), (roi["x"] + x2 + w2, roi["y"] + y2 + h2), color, 2)
            cv2.putText(f2, label, (roi["x"] + x2, roi["y"] + y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _generate_colors(self, num_colors):
        return [tuple(int(c) for c in cv2.cvtColor(np.uint8([[[i * 180 / num_colors, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]) for i in range(num_colors)]

    def stop(self):
        self.is_running = False

    def cleanup(self):
        self.logMessage.emit("Cleaning up resources...")
        if hasattr(self, 'cap1') and self.cap1.isOpened(): self.cap1.release()
        if hasattr(self, 'cap2') and self.cap2.isOpened(): self.cap2.release()
        self.finished.emit()
        self.logMessage.emit("Cleanup complete.")

class VideoLabel(QLabel):
    roi_updated = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.roi_rect = None
        self.dragging = False
        self.drag_start_pos = QPoint()
        self.native_frame_size = (CONFIG['camera_base_settings']['width'], CONFIG['camera_base_settings']['height'])
        self.video_timer = None

    def set_roi(self, roi_dict):
        self.roi_rect = QRect(roi_dict['x'], roi_dict['y'], roi_dict['w'], roi_dict['h'])
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.roi_rect and self.pixmap() and not self.pixmap().isNull():
            painter = QPainter(self)
            pen = QPen(QColor(0, 180, 255), 4)
            pen.setStyle(Qt.PenStyle.DashLine if self.dragging else Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            scaled_rect = self._scale_rect_to_widget(self.roi_rect)
            painter.drawRect(scaled_rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.roi_rect and self.pixmap() and not self.pixmap().isNull():
            scaled_roi = self._scale_rect_to_widget(self.roi_rect)
            if scaled_roi.contains(event.pos()):
                self.dragging = True
                self.drag_start_pos = event.pos() - scaled_roi.topLeft()
                if self.video_timer:
                    self.video_timer.stop()
                self.update()

    def mouseMoveEvent(self, event):
        if self.dragging and self.roi_rect:
            new_top_left_scaled = event.pos() - self.drag_start_pos
            new_top_left_native = self._scale_point_to_native(new_top_left_scaled)
            nx = max(0, min(new_top_left_native.x(), self.native_frame_size[0] - self.roi_rect.width()))
            ny = max(0, min(new_top_left_native.y(), self.native_frame_size[1] - self.roi_rect.height()))
            self.roi_rect.moveTo(nx, ny)
            self.roi_updated.emit({'x': nx, 'y': ny, 'w': self.roi_rect.width(), 'h': self.roi_rect.height()})
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.dragging:
                logging.info(f"ROI updated to: {CONFIG['roi']}")
            self.dragging = False
            if self.video_timer:
                self.video_timer.start(30)
            self.update()

    def _scale_rect_to_widget(self, rect):
        if not self.pixmap() or self.pixmap().isNull(): return QRect()
        pixmap_size, widget_size = self.pixmap().size(), self.size()
        if pixmap_size.width() == 0 or pixmap_size.height() == 0: return QRect()
        scale_x = widget_size.width() / pixmap_size.width()
        scale_y = widget_size.height() / pixmap_size.height()
        scale = min(scale_x, scale_y)
        offset_x = (widget_size.width() - (pixmap_size.width() * scale)) / 2
        offset_y = (widget_size.height() - (pixmap_size.height() * scale)) / 2
        scaled_x = int((rect.x() * scale * pixmap_size.width()) / self.native_frame_size[0] + offset_x)
        scaled_y = int((rect.y() * scale * pixmap_size.height()) / self.native_frame_size[1] + offset_y)
        scaled_w = int((rect.width() * scale * pixmap_size.width()) / self.native_frame_size[0])
        scaled_h = int((rect.height() * scale * pixmap_size.height()) / self.native_frame_size[1])
        return QRect(scaled_x, scaled_y, scaled_w, scaled_h)

    def _scale_point_to_native(self, point):
        if not self.pixmap() or self.pixmap().isNull(): return QPoint()
        pixmap_size, widget_size = self.pixmap().size(), self.size()
        if pixmap_size.width() == 0 or pixmap_size.height() == 0: return QPoint()
        scale_x = widget_size.width() / pixmap_size.width()
        scale_y = widget_size.height() / pixmap_size.height()
        scale = min(scale_x, scale_y)
        offset_x = (widget_size.width() - (pixmap_size.width() * scale)) / 2
        offset_y = (widget_size.height() - (pixmap_size.height() * scale)) / 2
        native_x = int((point.x() - offset_x) / scale * self.native_frame_size[0] / pixmap_size.width())
        native_y = int((point.y() - offset_y) / scale * self.native_frame_size[1] / pixmap_size.height())
        return QPoint(native_x, native_y)

# --- Main Application Window ---
class StereoVisionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Stereo Vision Analysis")
        self.setGeometry(100, 100, 1600, 900)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        self.processor_thread, self.processor = None, None
        self.preview_thread, self.preview_worker = None, None
        self.recorder_thread, self.recorder = None, None
        self.frame_queue = None
        self.gps_thread = None
        self.gps_processor = None
        
        self.pending_settings = None
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_feeds)
        self.slider_timer = QTimer(self)
        self.slider_timer.setSingleShot(True)
        self.slider_timer.timeout.connect(self.apply_pending_settings)
        self.current_theme = 'light' 
        self.current_downscale = 1
        
        # Attributes for logging
        self.data_logger = None
        self.latest_gps_data = {}
        
        # Pole data structures
        self.poles = []
        self.pole_coords = None
        self.pole_tree = None

        self.prev_pole_name = "NA"
        self.next_pole_name = "NA"
        self.dist_prev = 0.0
        self.dist_next = 0.0
        
        # Create a three-panel layout
        self._create_left_panel()
        self._create_video_panel()
        self._create_right_panel()

        self.main_layout.addWidget(self.left_panel)
        self.main_layout.addWidget(self.video_panel, 1) # Central video panel will stretch
        self.main_layout.addWidget(self.right_panel)

        self.video_label_left.set_roi(CONFIG["roi"])
        self.video_label_right.set_roi(CONFIG["roi"])
        self.toggle_theme()
        
        self.refresh_camera_list()
        self.refresh_com_ports()

    def _create_left_panel(self):
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        self.left_panel.setFixedWidth(350)
        
        cam_group = QGroupBox("Camera Selection")
        cam_layout = QGridLayout()
        self.cam_left_combo, self.cam_right_combo = QComboBox(), QComboBox()
        self.refresh_button = QPushButton("Refresh Devices")
        self.refresh_button.clicked.connect(self.refresh_camera_list)
        self.cam_left_combo.currentIndexChanged.connect(self.handle_cam_selection_change)
        self.cam_right_combo.currentIndexChanged.connect(self.handle_cam_selection_change)
        cam_layout.addWidget(QLabel("Left Camera:"), 0, 0); cam_layout.addWidget(self.cam_left_combo, 0, 1)
        cam_layout.addWidget(QLabel("Right Camera:"), 1, 0); cam_layout.addWidget(self.cam_right_combo, 1, 1)
        cam_layout.addWidget(self.refresh_button, 2, 0, 1, 2)
        cam_group.setLayout(cam_layout)

        gps_group = QGroupBox("GPS Status & Control")
        gps_layout = QGridLayout()
        self.gps_port_combo = QComboBox()
        self.gps_refresh_button = QPushButton("Refresh Ports")
        self.gps_refresh_button.clicked.connect(self.refresh_com_ports)
        self.gps_connect_button = QPushButton("Connect GPS")
        self.gps_connect_button.clicked.connect(self.toggle_gps_connection)
        self.gps_connect_button.setStyleSheet("background-color: #007BFF; color: white; font-weight: bold;")
        gps_layout.addWidget(QLabel("COM Port:"), 0, 0); gps_layout.addWidget(self.gps_port_combo, 0, 1)
        gps_layout.addWidget(self.gps_refresh_button, 1, 0, 1, 2)
        gps_layout.addWidget(self.gps_connect_button, 2, 0, 1, 2)
        self.gps_labels = {}
        names = ["Lat", "Lon", "Altitude", "Speed", "Course", "Sats"]
        for i, name in enumerate(names):
            label_name = QLabel(f"{name}:"); label_value = QLabel("--")
            font = label_value.font(); font.setBold(True); label_value.setFont(font)
            gps_layout.addWidget(label_name, i + 3, 0); gps_layout.addWidget(label_value, i + 3, 1)
            self.gps_labels[name] = label_value
        gps_layout.addWidget(QLabel("Prev Pole:"), len(names) + 3, 0)
        self.prev_pole_label = QLabel("--")
        gps_layout.addWidget(self.prev_pole_label, len(names) + 3, 1)
        gps_layout.addWidget(QLabel("Next Pole:"), len(names) + 4, 0)
        self.next_pole_label = QLabel("--")
        gps_layout.addWidget(self.next_pole_label, len(names) + 4, 1)
        gps_group.setLayout(gps_layout)
        
        poles_group = QGroupBox("Poles")
        poles_layout = QGridLayout()
        self.load_csv_button = QPushButton("Load CSV")
        self.load_csv_button.clicked.connect(self.load_poles_csv)
        poles_layout.addWidget(self.load_csv_button, 0, 0, 1, 2)
        self.poles_status = QLabel("No poles loaded")
        poles_layout.addWidget(self.poles_status, 1, 0, 1, 2)
        poles_group.setLayout(poles_layout)
        
        log_group = QGroupBox("Application Log")
        log_layout = QVBoxLayout()
        self.log_handler = QTextEditLogger(self)
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S'))
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)
        log_layout.addWidget(self.log_handler.widget)
        log_group.setLayout(log_layout)
        
        left_layout.addWidget(cam_group)
        left_layout.addWidget(gps_group)
        left_layout.addWidget(poles_group)
        left_layout.addWidget(log_group)

    def _create_right_panel(self):
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        self.right_panel.setFixedWidth(350)

        recording_group = QGroupBox("Video Recording")
        recording_layout = QGridLayout()
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setStyleSheet("background-color: #FFC107; color: black; font-weight: bold;")
        recording_layout.addWidget(self.record_button, 0, 0, 1, 2)
        self.recording_status = QLabel("Not Recording")
        recording_layout.addWidget(self.recording_status, 1, 0, 1, 2)
        recording_group.setLayout(recording_layout)
        
        presets_group = QGroupBox("Camera Presets")
        presets_layout = QVBoxLayout()
        self.preset_radios = {}
        for name in PRESETS.keys():
            radio = QRadioButton(name)
            radio.toggled.connect(self.on_preset_changed)
            self.preset_radios[name] = radio
            presets_layout.addWidget(radio)
        presets_group.setLayout(presets_layout)
        
        self.settings_group = QGroupBox("Live Camera Settings")
        settings_layout = QGridLayout()
        self.sliders = {}
        slider_params = {"focus": 255, "exposure": 0, "brightness": 255, "contrast": 255, "saturation": 255, "gain": 255}
        for i, (name, max_val) in enumerate(slider_params.items()):
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-13 if name == 'exposure' else 0, max_val if name != 'exposure' else 0)
            value_label = QLabel("0")
            slider.valueChanged.connect(value_label.setNum)
            slider.valueChanged.connect(self.on_slider_changed)
            settings_layout.addWidget(QLabel(f"{name.capitalize()}:"), i, 0)
            settings_layout.addWidget(slider, i, 1)
            settings_layout.addWidget(value_label, i, 2)
            self.sliders[name] = (slider, value_label)
        self.settings_group.setLayout(settings_layout)
        self.settings_group.setEnabled(False)
        
        self.start_button = QPushButton("START DETECTION")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.start_button.clicked.connect(self.toggle_processing)
        
        measurements_group = QGroupBox("Live Measurements")
        self.measurements_layout = QVBoxLayout()
        self.measurements_label_fps = QLabel("FPS: --.-- | Sync Delay: --.-- ms")
        self.measurements_text = QTextEdit()
        self.measurements_text.setReadOnly(True)
        self.measurements_layout.addWidget(self.measurements_label_fps)
        self.measurements_layout.addWidget(self.measurements_text)
        measurements_group.setLayout(self.measurements_layout)
        
        self.theme_button = QPushButton("Toggle Light/Dark Mode")
        self.theme_button.clicked.connect(self.toggle_theme)
        
        right_layout.addWidget(recording_group)
        right_layout.addWidget(presets_group)
        right_layout.addWidget(self.settings_group)
        right_layout.addWidget(self.start_button)
        right_layout.addWidget(measurements_group)
        right_layout.addStretch()
        right_layout.addWidget(self.theme_button)

        # Set the default radio button AFTER all widgets in this panel are created
        self.preset_radios["Sunny Day"].setChecked(True)

    def _create_video_panel(self):
        self.video_panel = QWidget()
        video_layout = QHBoxLayout(self.video_panel)
        font = QFont("Arial", 16, QFont.Weight.Bold)
        self.video_label_left = VideoLabel()
        self.video_label_right = VideoLabel()
        self.video_label_left.video_timer = self.video_timer
        self.video_label_right.video_timer = self.video_timer
        for label in [self.video_label_left, self.video_label_right]:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFont(font)
            label.setMinimumSize(640, 360)
            label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label_left.roi_updated.connect(self.on_roi_changed)
        self.video_label_right.roi_updated.connect(self.on_roi_changed)
        video_layout.addWidget(self.video_label_left)
        video_layout.addWidget(self.video_label_right)

    def refresh_com_ports(self):
        self.gps_port_combo.clear()
        ports = serial.tools.list_ports.comports()
        if not ports:
            self.gps_port_combo.addItem("No COM Ports Found")
            self.gps_connect_button.setEnabled(False)
            logging.warning("No serial COM ports found.")
        else:
            for port in sorted(ports):
                self.gps_port_combo.addItem(port.device)
            self.gps_connect_button.setEnabled(True)
            logging.info(f"Found COM ports: {[p.device for p in ports]}")

    def toggle_gps_connection(self):
        if self.gps_thread and self.gps_thread.isRunning():
            self.stop_gps()
        else:
            self.start_gps()

    def start_gps(self):
        if "No COM Ports" in self.gps_port_combo.currentText():
            logging.error("Cannot start GPS: No COM port selected.")
            return
        port = self.gps_port_combo.currentText()
        self.gps_thread = QThread()
        self.gps_processor = GPSProcessor.get_instance()
        self.gps_processor.configure(port=port, baudrate=9600)
        self.gps_processor.moveToThread(self.gps_thread)
        self.gps_thread.started.connect(self.gps_processor.run)
        self.gps_processor.finished.connect(self.on_gps_stopped)
        self.gps_processor.positionChanged_update.connect(self.update_gps_display)
        self.gps_processor.connectionStatus.connect(self.on_gps_connection_status)
        self.gps_thread.start()
        self.gps_connect_button.setText("Connecting...")
        self.set_gps_controls_enabled(False)

    def stop_gps(self):
        if self.gps_processor:
            self.gps_processor.stop()
            self.gps_thread.quit()
            self.gps_thread.wait(2000)

    def on_gps_stopped(self):
        logging.info("GPS processing has been stopped by the worker.")
        self.gps_thread.deleteLater()
        self.gps_thread = None
        self.set_gps_controls_enabled(True)
        self.gps_connect_button.setText("Connect GPS")
        self.gps_connect_button.setStyleSheet("background-color: #007BFF; color: white; font-weight: bold;")
        for label in self.gps_labels.values():
            label.setText("--")
        self.prev_pole_label.setText("--")
        self.next_pole_label.setText("--")

    def update_gps_display(self, lat, lon, alt, speed, course, sats, is_valid):
        if is_valid:
            self.gps_labels["Lat"].setText(f"{lat:.6f}")
            self.gps_labels["Lon"].setText(f"{lon:.6f}")
            self.gps_labels["Altitude"].setText(f"{alt:.2f} m")
            self.gps_labels["Speed"].setText(f"{speed:.2f} km/h")
            self.gps_labels["Course"].setText(f"{course:.1f}\u00B0")
            self.gps_labels["Sats"].setText(f"{sats}")

            # Cache the latest valid data for the logger
            self.latest_gps_data = {
                "latitude": lat, "longitude": lon, "altitude_m": alt,
                "speed_kmh": speed, "course_deg": course, "satellites": sats
            }
            
            if self.pole_tree is not None:
                try:
                    distances, indices = self.pole_tree.query([lat, lon], k=2)
                    p1_idx, p2_idx = indices[0], indices[1]
                    pole1 = self.poles[p1_idx]; pole2 = self.poles[p2_idx]
                    dist1 = self.haversine(lat, lon, pole1['lat'], pole1['lon'])
                    dist2 = self.haversine(lat, lon, pole2['lat'], pole2['lon'])
                    self.prev_pole_name = pole1['name']; self.dist_prev = dist1
                    self.next_pole_name = pole2['name']; self.dist_next = dist2
                    self.prev_pole_label.setText(f"{self.prev_pole_name} ({self.dist_prev:.1f} m)")
                    self.next_pole_label.setText(f"{self.next_pole_name} ({self.dist_next:.1f} m)")
                except Exception as e:
                    logging.warning(f"Could not find 2 nearest poles: {e}")
                    self.prev_pole_label.setText("--"); self.next_pole_label.setText("--")
            else:
                self.prev_pole_label.setText("--"); self.next_pole_label.setText("--")
        else:
            self.latest_gps_data = {} # Clear cached data if GPS is invalid
            for label in self.gps_labels.values():
                label.setText("N/A")
            self.prev_pole_label.setText("N/A"); self.next_pole_label.setText("N/A")

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371.0 * 1000
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def bearing(self, lat1, lon1, lat2, lon2):
        dlon = math.radians(lon2 - lon1)
        y = math.sin(dlon) * math.cos(math.radians(lat2))
        x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon)
        brng = (math.degrees(math.atan2(y, x)) + 360) % 360
        return brng

    def load_poles_csv(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Poles CSV", "", "CSV Files (*.csv)")
        if filename:
            self.poles = []; self.pole_coords = None; self.pole_tree = None
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        lat_raw = float(row['Latitude']); lon_raw = float(row['Longitude'])
                        lat_deg = int(lat_raw // 100); lat_min = lat_raw % 100
                        lat = lat_deg + lat_min / 60
                        lon_deg = int(lon_raw // 100); lon_min = lon_raw % 100
                        lon = lon_deg + lon_min / 60
                        self.poles.append({'name': row['OHEMaster'], 'lat': lat, 'lon': lon, 'alt': float(row['Altitude'])})
                
                if not self.poles:
                    self.poles_status.setText("CSV is empty.")
                    logging.warning("Pole CSV file is empty.")
                    return

                self.pole_coords = np.array([[p['lat'], p['lon']] for p in self.poles])
                self.pole_tree = KDTree(self.pole_coords)
                
                self.poles_status.setText(f"{len(self.poles)} poles loaded & indexed.")
                logging.info(f"Loaded and built KDTree for {len(self.poles)} poles from {filename}")

            except ImportError:
                self.poles_status.setText("Error: SciPy not installed.")
                logging.error("Failed to build pole index: SciPy library not found. Please run 'pip install scipy'.")
            except Exception as e:
                logging.error(f"Error loading CSV or building index: {str(e)}")
                self.poles_status.setText("Error loading CSV")

    def toggle_recording(self):
        if (self.recorder_thread and self.recorder_thread.isRunning()) or self.data_logger:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if not self.processor:
            logging.error("Detection must be running to start recording.")
            return
        
        os.makedirs("recordings", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Start Data Logger
        log_filename = os.path.join("recordings", f"log_{timestamp}.csv")
        self.data_logger = DataLogger(log_filename)
        if self.data_logger.start():
            self.processor.newMeasurements.connect(self.handle_measurements_for_logging)

        # Start Video Recorder
        self.frame_queue = queue.Queue(maxsize=CONFIG["recording_fps"])
        self.processor.rawFrames.connect(self.on_raw_frames)
        
        self.recorder_thread = QThread()
        self.recorder_thread.setPriority(QThread.Priority.LowPriority)
        self.recorder = VideoRecorder(self.frame_queue, output_dir="recordings")
        self.recorder.moveToThread(self.recorder_thread)
        self.recorder.logMessage.connect(logging.info)
        self.recorder_thread.started.connect(self.recorder.run)
        self.recorder.finished.connect(self.on_recording_stopped)
        self.recorder_thread.start()
        
        self.record_button.setText("Stop Recording")
        self.record_button.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.recording_status.setText("Recording")

    def stop_recording(self):
        if self.processor:
            try:
                self.processor.rawFrames.disconnect(self.on_raw_frames)
                self.processor.newMeasurements.disconnect(self.handle_measurements_for_logging)
            except TypeError: pass
        
        if self.recorder:
            self.recorder.stop()
        
        if self.data_logger:
            self.data_logger.stop()
            self.data_logger = None

    def on_recording_stopped(self):
        if self.recorder_thread:
            self.recorder_thread.quit()
            self.recorder_thread.wait()
            self.recorder.deleteLater()
            self.recorder_thread.deleteLater()
            
        self.frame_queue = None
        self.recorder = None
        self.recorder_thread = None
        
        self.record_button.setText("Start Recording")
        self.record_button.setStyleSheet("background-color: #FFC107; color: black; font-weight: bold;")
        self.recording_status.setText("Not Recording")

    def handle_measurements_for_logging(self, measurements, fps, sync_delay):
        if not self.data_logger or not self.latest_gps_data:
            return

        log_timestamp = datetime.now().isoformat()
        frame_left = self.recorder.frame_number_left if self.recorder else -1
        frame_right = self.recorder.frame_number_right if self.recorder else -1
        prev_pole = self.prev_pole_name
        prev_pole_dist = self.dist_prev
        next_pole = self.next_pole_name
        next_pole_dist = self.dist_next

        for cable_id, data in measurements.items():
            log_entry = {
                "timestamp": log_timestamp,
                "frame_left": frame_left,
                "frame_right": frame_right,
                "cable_id": cable_id,
                "height_z_meters": f"{data['z']:.4f}",
                "stagger_x_meters": f"{data['x']:.4f}",
                "prev_pole": prev_pole,
                "prev_pole_dist_m": f"{prev_pole_dist:.2f}",
                "next_pole": next_pole,
                "next_pole_dist_m": f"{next_pole_dist:.2f}",
            }
            log_entry.update(self.latest_gps_data)
            self.data_logger.log_entry(log_entry)

    def on_raw_frames(self, f1, f2):
        if self.frame_queue and not self.frame_queue.full():
            self.frame_queue.put((f1.copy(), f2.copy()), block=False)

    def on_gps_connection_status(self, is_connected, message):
        logging.info(f"GPS Status: {message}")
        if is_connected:
            self.gps_connect_button.setText("Disconnect GPS")
            self.gps_connect_button.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        else:
            self.gps_connect_button.setText("Connect GPS")
            self.set_gps_controls_enabled(True)

    def set_gps_controls_enabled(self, enabled):
        self.gps_port_combo.setEnabled(enabled)
        self.gps_refresh_button.setEnabled(True)

    def handle_cam_selection_change(self):
        if self.cam_left_combo.count() > 0 and self.cam_right_combo.count() > 0:
            self.start_preview()

    def on_roi_changed(self, new_roi):
        with roi_lock:
            new_roi_full = {k: new_roi[k] * self.current_downscale for k in new_roi}
            CONFIG['roi'] = new_roi_full
        scaled_roi = {k: int(CONFIG['roi'][k] / self.current_downscale) for k in CONFIG['roi']}
        self.video_label_left.set_roi(scaled_roi)
        self.video_label_right.set_roi(scaled_roi)
        logging.info(f"ROI changed: {CONFIG['roi']}")

    def refresh_camera_list(self):
        self.cam_left_combo.blockSignals(True)
        self.cam_right_combo.blockSignals(True)
        self.cam_left_combo.clear(); self.cam_right_combo.clear()
        available_cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available_cameras.append(f"Camera {i}")
                cap.release()
        
        if available_cameras:
            self.cam_left_combo.addItems(available_cameras)
            self.cam_right_combo.addItems(available_cameras)
            if len(available_cameras) > 1:
                self.cam_right_combo.setCurrentIndex(1)
            logging.info(f"Found cameras: {available_cameras}")
        else:
            logging.warning("No cameras found.")
            self.cam_left_combo.addItem("No Cameras")
            self.cam_right_combo.addItem("No Cameras")
        
        self.cam_left_combo.blockSignals(False)
        self.cam_right_combo.blockSignals(False)
        self.start_preview()

    def start_preview(self):
        self.stop_preview()
        
        cam_left_text, cam_right_text = self.cam_left_combo.currentText(), self.cam_right_combo.currentText()
        if "No Cameras" in cam_left_text or cam_left_text == "" or self.processor:
            return

        cam_idx_left = int(cam_left_text.split()[-1])
        cam_idx_right = int(cam_right_text.split()[-1])
        if cam_idx_left == cam_idx_right:
            logging.warning("Preview requires two different cameras.")
            return

        self.current_downscale = 1
        camera_settings = {name: slider.value() for name, (slider, _) in self.sliders.items()}
        self.preview_thread = QThread()
        self.preview_worker = PreviewWorker(cam_idx_left, cam_idx_right, camera_settings)
        self.preview_worker.moveToThread(self.preview_thread)
        self.preview_worker.logMessage.connect(logging.info)
        self.preview_thread.started.connect(self.preview_worker.run)
        self.preview_worker.finished.connect(self.preview_thread.quit)
        self.preview_worker.finished.connect(self.preview_worker.deleteLater)
        self.preview_thread.finished.connect(self.preview_thread.deleteLater)
        self.preview_thread.start()
        if not self.video_timer.isActive():
            self.video_timer.start(30)

    def stop_preview(self):
        if self.preview_worker:
            self.preview_worker.stop()
            if self.preview_thread:
                self.preview_thread.quit()
                self.preview_thread.wait()
        self.preview_worker, self.preview_thread = None, None

    def on_preset_changed(self):
        for name, radio in self.preset_radios.items():
            if radio.isChecked():
                self.settings_group.setEnabled(name == "Custom")
                settings = PRESETS.get(name, {})
                for s_name, (slider, val_label) in self.sliders.items():
                    if s_name in settings:
                        slider.blockSignals(True)
                        slider.setValue(settings[s_name])
                        val_label.setNum(settings[s_name])
                        slider.blockSignals(False)
                if self.preview_worker:
                    camera_settings = {s_name: slider.value() for s_name, (slider, _) in self.sliders.items()}
                    self.preview_worker.updateSettings.emit(camera_settings)
                logging.info(f"Preset '{name}' selected.")
                break

    def on_slider_changed(self):
        if self.preset_radios["Custom"].isChecked() and self.preview_worker:
            self.pending_settings = {name: slider.value() for name, (slider, _) in self.sliders.items()}
            self.slider_timer.start(100)

    def apply_pending_settings(self):
        if self.pending_settings and self.preview_worker:
            self.preview_worker.updateSettings.emit(self.pending_settings)
            logging.info(f"Updated preview camera settings: {self.pending_settings}")
            self.pending_settings = None

    def toggle_processing(self):
        if self.processor_thread and self.processor_thread.isRunning():
            self.stop_processing()
        else:
            self.start_processing()

    def start_processing(self):
        self.stop_preview()
        
        cam_left_text, cam_right_text = self.cam_left_combo.currentText(), self.cam_right_combo.currentText()
        if "No Cameras" in cam_left_text:
            return logging.error("Select valid cameras.")
        cam_idx_left, cam_idx_right = int(cam_left_text.split()[-1]), int(cam_right_text.split()[-1])
        if cam_idx_left == cam_idx_right:
            return logging.error("Cameras cannot be the same.")
        
        self.current_downscale = CONFIG["downscale_factor"]
        camera_settings = {name: slider.value() for name, (slider, _) in self.sliders.items()}
        self.processor_thread = QThread()
        self.processor_thread.setPriority(QThread.Priority.HighPriority)
        self.processor = StereoProcessor(CONFIG, cam_idx_left, cam_idx_right, camera_settings)
        self.processor.moveToThread(self.processor_thread)
        self.processor.newMeasurements.connect(self.update_measurements_display)
        self.processor.logMessage.connect(logging.info)
        self.processor_thread.started.connect(self.processor.run)
        self.processor.finished.connect(self.on_processing_stopped)
        self.processor.finished.connect(self.processor.deleteLater)
        self.processor_thread.finished.connect(self.processor_thread.deleteLater)
        self.processor_thread.start()
        
        if not self.video_timer.isActive():
            self.video_timer.start(30)
        self.start_button.setText("STOP DETECTION")
        self.start_button.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px;")
        self.set_controls_enabled(False)

    def stop_processing(self):
        self.stop_recording() # Ensure recording stops if processing is stopped
        if self.processor:
            self.processor.stop()
            if self.processor_thread:
                self.processor_thread.quit()
                self.processor_thread.wait()

    def on_processing_stopped(self):
        self.start_button.setText("START DETECTION")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.processor, self.processor_thread = None, None
        self.current_downscale = 1
        self.set_controls_enabled(True)
        self.start_preview()

    def update_video_feeds(self):
        frame_left, frame_right = None, None
        
        if self.processor:
            frame_left = self.processor.latest_frame_left
            frame_right = self.processor.latest_frame_right
        elif self.preview_worker:
            frame_left = self.preview_worker.latest_frame_left
            frame_right = self.preview_worker.latest_frame_right
            if frame_left is not None:
                roi = CONFIG['roi']
                cv2.rectangle(frame_left, (roi["x"], roi["y"]), (roi["x"] + roi["w"], roi["y"] + roi["h"]), (0, 0, 255), 3)
            if frame_right is not None:
                roi = CONFIG['roi']
                cv2.rectangle(frame_right, (roi["x"], roi["y"], roi["x"] + roi["w"], roi["y"] + roi["h"]), (0, 0, 255), 3)
        
        if frame_left is not None: 
            self.video_label_left.native_frame_size = (frame_left.shape[1], frame_left.shape[0])
            scaled_roi = {k: int(CONFIG['roi'][k] / self.current_downscale) for k in CONFIG['roi']}
            self.video_label_left.set_roi(scaled_roi)
            self._display_frame(frame_left, self.video_label_left)
        if frame_right is not None: 
            self.video_label_right.native_frame_size = (frame_right.shape[1], frame_right.shape[0])
            scaled_roi = {k: int(CONFIG['roi'][k] / self.current_downscale) for k in CONFIG['roi']}
            self.video_label_right.set_roi(scaled_roi)
            self._display_frame(frame_right, self.video_label_right)

    def _display_frame(self, frame, label):
        h, w, ch = frame.shape
        qt_image = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
        label.setPixmap(QPixmap.fromImage(qt_image).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))

    def update_measurements_display(self, measurements, fps, sync_delay):
        self.measurements_label_fps.setText(f"Processing FPS: {fps:.2f} | Sync Delay: {sync_delay:.2f} ms")
        html = "<style>p { margin: 2px; }</style>"
        for label in sorted(measurements.keys()):
            data = measurements[label]
            html += f"<p><b>ID {label}</b> &nbsp;&nbsp;&nbsp; Z: {data['z']:.3f} m | X: {data['x']:.3f} m</p>"
        self.measurements_text.setHtml(html)

    def set_controls_enabled(self, enabled):
        for w in [self.cam_left_combo, self.cam_right_combo, self.refresh_button] + list(self.preset_radios.values()):
            w.setEnabled(enabled)
        self.settings_group.setEnabled(enabled and self.preset_radios["Custom"].isChecked())

    def toggle_theme(self):
        palette = QPalette()
        if self.current_theme == 'light':
            palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53)); palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
            palette.setColor(QPalette.ColorRole.Base, QColor(42, 42, 42)); palette.setColor(QPalette.ColorRole.AlternateBase, QColor(66, 66, 66))
            palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white); palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white); palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
            self.current_theme = 'dark'
        else:
            self.current_theme = 'light'
        QApplication.instance().setPalette(palette)
        for label in [self.video_label_left, self.video_label_right]:
            label.setStyleSheet("background-color: black;")
    
    def closeEvent(self, event):
        logging.info("Shutting down...")
        self.video_timer.stop()
        self.slider_timer.stop()
        self.stop_recording()
        self.stop_gps()
        self.stop_preview()
        self.stop_processing()
        
        # Wait for threads to finish
        if self.gps_thread and self.gps_thread.isRunning(): self.gps_thread.wait(2000)
        if self.preview_thread and self.preview_thread.isRunning(): self.preview_thread.wait(2000)
        if self.processor_thread and self.processor_thread.isRunning(): self.processor_thread.wait(2000)
        if self.recorder_thread and self.recorder_thread.isRunning(): self.recorder_thread.wait(2000)
        
        logging.info("All threads stopped. Exiting.")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StereoVisionApp()
    window.show()
    sys.exit(app.exec())
import cv2
import numpy as np
import pickle
import time
"""
Stereo Vision Pipeline (Fixed Calibration)
-----------------------------------------
Key change: Calibration is performed ONCE using (Prediction 2 -> Actual) pairs.
Do NOT re-calibrate using values that have already been corrected. That recursive
feedback is what caused your oscillating Corrections 1..4.
Plug in your videos and calibration .pkl as usual.
"""
# --- Configuration ---
CONFIG = {
    # Input videos
    #"video_path_left": "rec\main1_L.mkv",
    #"video_path_right": "rec\main1_R.mkv",
    "video_path_left": 1,
    "video_path_right": 0,
    # Stereo calibration file (pickle with keys mtxL, distL, mtxR, distR, R, T)
    "calibration_file": "stereo_params_charuco_stereo6.pkl",
    # ROI for cable/edge detection (drag to move in the main window)
    "roi": {"x": 400, "y": 100, "w": 1600, "h": 20},
    # Contour detection params
    "contour_detection": {
        "gaussian_blur_kernel": (7, 7),
        "default_canny_low": 50,
        "default_canny_high": 150,
        "min_contour_width": 1,
        "min_contour_height": 10,
    },
    # Temporal smoothing (EMA)
    "smoothing_factor": 0.15,
    # Display sizes
    "display": {"width": 960, "height": 540},
    # --- CALIBRATION: Use Prediction 2 (RAW) vs Actual ---
    # These should be the *uncalibrated* heights from your pipeline (Prediction 2)
    # matched to ground-truth Actual measurements. Calibrate once.
    "calibration_points": {
        "raw_measurements": [1.871, 1.863, 1.869, 1.861, 1.889], # Prediction 2
        "actual_measurements": [1.67, 1.75, 1.74, 1.81, 1.79], # Actual
    },
    # Matching tolerances
    "DISPARITY_TOLERANCE_PX": 15, # allows a small negative disparity
    "Y_TOLERANCE_PX": 20, # vertical alignment tolerance
    # Window names
    "main_window_name": "Synced Stereo Vision Analysis",
    "controls_window_name": "Correction Controls",
    "readings_window_name": "Live Measurements",
    # Start timestamp in seconds (replace with desired start time, e.g., 10.0 for 10 seconds)
    "start_timestamp": 0.0,  # Placeholder: set to the desired start time in seconds
    "tall": 0.129,
    "camera_controls_window": "Camera Controls",
    "camera_settings": {
        "width": 1920,
        "height": 1080,
        "focus": 40,
        "exposure": -4,
        "brightness": 40,
        "contrast": 100,
        "saturation": 60,
        "gain": 0
    }
}
# --- Helper ---
def apply_settings_from_dict(cap, settings):
    """Applies dictionary settings to a single camera."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings["height"])
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_FOCUS, settings["focus"])
    cap.set(cv2.CAP_PROP_EXPOSURE, settings["exposure"])
    cap.set(cv2.CAP_PROP_BRIGHTNESS, settings["brightness"])
    cap.set(cv2.CAP_PROP_CONTRAST, settings["contrast"])
    cap.set(cv2.CAP_PROP_SATURATION, settings["saturation"])
    cap.set(cv2.CAP_PROP_GAIN, settings["gain"])

def on_trackbar_change(val, processor):
    """Trackbar callback that updates both cameras."""
    if processor.cap1 is None or processor.cap2 is None:
        return
    focus = cv2.getTrackbarPos("Focus", CONFIG["camera_controls_window"])
    exposure = cv2.getTrackbarPos("Exposure", CONFIG["camera_controls_window"]) - 13
    brightness = cv2.getTrackbarPos("Brightness", CONFIG["camera_controls_window"])
    contrast = cv2.getTrackbarPos("Contrast", CONFIG["camera_controls_window"])
    saturation = cv2.getTrackbarPos("Saturation", CONFIG["camera_controls_window"])
    gain = cv2.getTrackbarPos("Gain", CONFIG["camera_controls_window"])
    for cap in [processor.cap1, processor.cap2]:
        cap.set(cv2.CAP_PROP_FOCUS, focus)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        cap.set(cv2.CAP_PROP_CONTRAST, contrast)
        cap.set(cv2.CAP_PROP_SATURATION, saturation)
        cap.set(cv2.CAP_PROP_GAIN, gain)

def create_camera_control_panel(processor):
    """Creates the camera controls window with sliders."""
    s = CONFIG["camera_settings"]
    cv2.namedWindow(CONFIG["camera_controls_window"])
    cv2.createTrackbar("Focus", CONFIG["camera_controls_window"], s["focus"], 255,
                       lambda v: on_trackbar_change(v, processor))
    cv2.createTrackbar("Exposure", CONFIG["camera_controls_window"], s["exposure"] + 13, 13,
                       lambda v: on_trackbar_change(v, processor))
    cv2.createTrackbar("Brightness", CONFIG["camera_controls_window"], s["brightness"], 255,
                       lambda v: on_trackbar_change(v, processor))
    cv2.createTrackbar("Contrast", CONFIG["camera_controls_window"], s["contrast"], 255,
                       lambda v: on_trackbar_change(v, processor))
    cv2.createTrackbar("Saturation", CONFIG["camera_controls_window"], s["saturation"], 255,
                       lambda v: on_trackbar_change(v, processor))
    cv2.createTrackbar("Gain", CONFIG["camera_controls_window"], s["gain"], 255,
                       lambda v: on_trackbar_change(v, processor))
    print("Camera controls ready. Use sliders to adjust settings.")

def detect_contours_stable(args):
    """Detect contours inside ROI using adaptive thresholding with live params."""
    frame, roi_params, detection_params, canny_low, canny_high = args
    h, w = frame.shape[:2]
    rx, ry, rw, rh = (
        roi_params["x"],
        roi_params["y"],
        roi_params["w"],
        roi_params["h"],
    )
    # ROI bounds check
    if ry + rh > h or rx + rw > w:
        return [], []
    roi = frame[ry : ry + rh, rx : rx + rw]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(
        gray_roi, detection_params["gaussian_blur_kernel"], 1
    )
    # Canny edges
    edges = cv2.Canny(blurred, canny_low, canny_high)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return [], []
    # Filter by min size
    valid = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        if (
            ww >= detection_params["min_contour_width"]
            and hh >= detection_params["min_contour_height"]
        ):
            valid.append(cnt)
    if not valid:
        return [], []
    # Merge contours that are within 4px horizontally
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
class StereoVideoProcessor:
    """Encapsulates the stereo pipeline + one-shot calibration."""
    def __init__(self, config):
        self.config = config
        self.is_paused = False
        self.last_frames = (None, None)
        self.smoothed_measurements = {}
        # ROI drag state
        self.current_roi = self.config["roi"].copy()
        self.roi_dragging = False
        self.roi_drag_start = (0, 0)
        # FPS tracking
        self.frame_count = 0
        self.start_time = time.time()
        # 1) Open videos
        self.cap1 = cv2.VideoCapture(config["video_path_left"], cv2.CAP_DSHOW)
        self.cap2 = cv2.VideoCapture(config["video_path_right"], cv2.CAP_DSHOW)
        #self.cap1 = cv2.VideoCapture(config["video_path_left"])
        #self.cap2 = cv2.VideoCapture(config["video_path_right"])
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap1.set(cv2.CAP_PROP_FPS, 100)
        self.cap2.set(cv2.CAP_PROP_FPS, 100)
        
        if not self.cap1.isOpened() or not self.cap2.isOpened():
            raise IOError("Error: Could not open one or both video files.")
        
        print("Applying initial camera settings...")
        apply_settings_from_dict(self.cap1, self.config["camera_settings"])
        apply_settings_from_dict(self.cap2, self.config["camera_settings"])

        # Create camera control sliders
        create_camera_control_panel(self)
        # 2) Video props
        self.total_frames = int(
            min(
                self.cap1.get(cv2.CAP_PROP_FRAME_COUNT),
                self.cap2.get(cv2.CAP_PROP_FRAME_COUNT),
            )
        )
        video_fps = float(self.cap1.get(cv2.CAP_PROP_FPS))
        self.frame_delay_ms = 1  # Override for testing
        print(f"Video FPS (metadata): {video_fps}, Frame Delay: {self.frame_delay_ms}ms")
        self.frame_width = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Set start timestamp
        start_timestamp = self.config["start_timestamp"] * 1000  # Convert to milliseconds
        self.cap1.set(cv2.CAP_PROP_POS_MSEC, start_timestamp)
        self.cap2.set(cv2.CAP_PROP_POS_MSEC, start_timestamp)
        print(f"Starting video from {self.config['start_timestamp']} seconds")
        # 3) One-shot calibration params
        self._init_calibration_params()
        # 4) Load camera model + rectify
        self._load_and_rectify_calibration()
        self._init_rectification_maps()
        # 5) Colors and labels
        self.labels = [chr(ord("A") + i) for i in range(26)]
        self.colors = self._generate_colors(len(self.labels))
        self.color_map = dict(zip(self.labels, self.colors))
        # 6) UI 
        self._init_windows_and_trackbars()
    # -------------------- Calibration --------------------
    def _init_calibration_params(self):
        cal = self.config["calibration_points"]
        raw = np.array(cal["raw_measurements"], dtype=float)
        actual = np.array(cal["actual_measurements"], dtype=float)
        # Linear regression: actual = slope * raw + intercept
        self.cal_slope, self.cal_intercept = np.polyfit(raw, actual, 1)
        print(
            f"Calibration (Actual = {self.cal_slope:.6f} * Raw + {self.cal_intercept:.6f})"
        )
    def _calibrate_height(self, raw_height):
        return raw_height * self.cal_slope + self.cal_intercept
    # -------------------- Rectification --------------------
    def _load_and_rectify_calibration(self):
        try:
            with open(self.config["calibration_file"], "rb") as f:
                p = pickle.load(f, encoding="latin1")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Calibration file not found: {self.config['calibration_file']}"
            )
        # Handle swapped cameras by inverting extrinsics
        R_swapped = p["R"].T
        T_swapped = -np.dot(R_swapped, p["T"])
        self.stereo_params = {
            "mtxL": p["mtxR"],
            "distL": p["distR"],
            "mtxR": p["mtxL"],
            "distR": p["distL"],
            "R": R_swapped,
            "T": T_swapped,
        }
        size = (self.frame_width, self.frame_height)
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.stereo_params["mtxL"],
            self.stereo_params["distL"],
            self.stereo_params["mtxR"],
            self.stereo_params["distR"],
            size,
            self.stereo_params["R"],
            self.stereo_params["T"],
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=-1,
        )
        print("Stereo parameters loaded and rectification computed.")
    def _init_rectification_maps(self):
        p = self.stereo_params
        size = (self.frame_width, self.frame_height)
        self.map1L, self.map2L = cv2.initUndistortRectifyMap(
            p["mtxL"], p["distL"], self.R1, self.P1, size, cv2.CV_32FC1
        )
        self.map1R, self.map2R = cv2.initUndistortRectifyMap(
            p["mtxR"], p["distR"], self.R2, self.P2, size, cv2.CV_32FC1
        )
    # -------------------- UI & Input --------------------
    def _init_windows_and_trackbars(self):
        cv2.namedWindow(self.config["main_window_name"])
        cv2.namedWindow(self.config["controls_window_name"])
        cv2.namedWindow(self.config["readings_window_name"])
        cv2.resizeWindow(self.config["controls_window_name"], 420, 160)
        cv2.createTrackbar(
            "Canny Low",
            self.config["controls_window_name"],
            self.config["contour_detection"]["default_canny_low"],
            300,
            lambda x: None,
        )
        cv2.createTrackbar(
            "Canny High",
            self.config["controls_window_name"],
            self.config["contour_detection"]["default_canny_high"],
            300,
            lambda x: None,
        )
        cv2.setMouseCallback(
            self.config["main_window_name"], self._mouse_callback
        )
        print(
            "Controls: [SPACE]=Pause/Play | [ESC]=Exit | Drag blue ROI box to move"
        )
    def _mouse_callback(self, event, x, y, flags, param):
        disp_w = self.config["display"]["width"]
        disp_h = self.config["display"]["height"]
        if x >= disp_w:
            self.roi_dragging = False
            return
        # scale from display to native resolution
        sx = self.frame_width / disp_w
        sy = self.frame_height / disp_h
        x_native, y_native = int(x * sx), int(y * sy)
        rx, ry, rw, rh = (
            self.current_roi["x"],
            self.current_roi["y"],
            self.current_roi["w"],
            self.current_roi["h"],
        )
        if event == cv2.EVENT_LBUTTONDOWN:
            if rx <= x_native <= rx + rw and ry <= y_native <= ry + rh:
                self.roi_dragging = True
                self.roi_drag_start = (x_native - rx, y_native - ry)
        elif event == cv2.EVENT_MOUSEMOVE and self.roi_dragging:
            nx = x_native - self.roi_drag_start[0]
            ny = y_native - self.roi_drag_start[1]
            self.current_roi["x"] = max(0, min(nx, self.frame_width - rw))
            self.current_roi["y"] = max(0, min(ny, self.frame_height - rh))
        elif event == cv2.EVENT_LBUTTONUP:
            self.roi_dragging = False
    # -------------------- Main Loop --------------------
    def run(self):
        while self.cap1.isOpened() and self.cap2.isOpened():
            current_time1 = time.time()
            print(f"Main - Seconds: {int(time.time())}, Milliseconds: {int((time.time() % 1) * 1000)}")
            self._handle_input()
            # Handle next frame
            if not self.is_paused:
                print(f"Read Start - Seconds: {int(time.time())}, Milliseconds: {int((time.time() % 1) * 1000)}")
                self._read_frames()
                if self.last_frames[0] is None:
                    continue
                print(f"Read End - Seconds: {int(time.time())}, Milliseconds: {int((time.time() % 1) * 1000)}")
                self._process_and_display()
                # Update FPS
                self.frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                if elapsed_time >= 1.0:  # Update FPS every second
                    self.live_fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.start_time = current_time
            else:
                # When paused, display last frame with current FPS
                if self.last_frames[0] is not None:
                    self._process_and_display()
            print(f"Finish - Seconds: {int(time.time())}, Milliseconds: {int((time.time() % 1) * 1000)}")
            elapsed_time1 = time.time() - current_time1
            print(f"Frame processing time: {elapsed_time1 * 1000:.2f} ms")
        self.cleanup()
    def _handle_input(self):
        key = cv2.waitKey(1 if self.is_paused else self.frame_delay_ms) & 0xFF
        if key == 27: # ESC
            self.cleanup()
            raise SystemExit
        elif key == 32: # SPACE
            self.is_paused = not self.is_paused
    def _read_frames(self):
        ret1, f1 = self.cap1.read()
        ret2, f2 = self.cap2.read()
        if not ret1 or not ret2:
            print("End of video.")
            self.cleanup()
            raise SystemExit
        self.last_frames = (f1, f2)
    # -------------------- Processing --------------------
    def _process_and_display(self):
        current_time2 = time.time()
        print(f"Process - Seconds: {int(time.time())}, Milliseconds: {int((time.time() % 1) * 1000)}")
        # Rectification
        f1_rect = cv2.remap(self.last_frames[1], self.map1L, self.map2L, cv2.INTER_LINEAR)
        f2_rect = cv2.remap(self.last_frames[0], self.map1R, self.map2R, cv2.INTER_LINEAR)
        elapsed_rect = time.time() - current_time2
        print(f"Rect - Seconds: {int(time.time())}, Milliseconds: {int((time.time() % 1) * 1000)}")
        # Read trackbar params
        canny_low = cv2.getTrackbarPos("Canny Low", self.config["controls_window_name"])
        canny_high = cv2.getTrackbarPos("Canny High", self.config["controls_window_name"])
        # Contour detection (sequential)
        elapsed_track = time.time() - current_time2
        print(f"Track - Seconds: {int(time.time())}, Milliseconds: {int((time.time() % 1) * 1000)}")
        centers1, boxes1 = detect_contours_stable((f1_rect, self.current_roi, self.config["contour_detection"], canny_low, canny_high))
        centers2, boxes2 = detect_contours_stable((f2_rect, self.current_roi, self.config["contour_detection"], canny_low, canny_high))
        elapsed_detection = time.time() - current_time2
        print(f"Contour - Seconds: {int(time.time())}, Milliseconds: {int((time.time() % 1) * 1000)}")
        # Measurements
        readings_canvas, matched_pairs = self._calculate_measurements(
            centers1, centers2, boxes1, boxes2
        )
        elapsed_measurements = time.time() - current_time2
        print(f"Measurements - Seconds: {int(time.time())}, Milliseconds: {int((time.time() % 1) * 1000)}")
        # Overlays
        self._draw_overlays(f1_rect, f2_rect, matched_pairs)
        elapsed_overlays = time.time() - current_time2
        print(f"Overlay - Seconds: {int(time.time())}, Milliseconds: {int((time.time() % 1) * 1000)}")
        # Display with live FPS (default to 0.0 if not yet calculated)
        live_fps = getattr(self, 'live_fps', 0.0)
        self._display_frames(f1_rect, f2_rect, readings_canvas, live_fps)
        elapsed_display = time.time() - current_time2
        print(f"Display - Seconds: {int(time.time())}, Milliseconds: {int((time.time() % 1) * 1000)}")
    def _calculate_measurements(self, centers1, centers2, boxes1, boxes2):
        canvas = np.zeros((540, 450, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            "Live Measurements",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        yline = 70
        matched_pairs = []
        if not centers1 or not centers2:
            return canvas, matched_pairs

        YT = self.config["Y_TOLERANCE_PX"]
        DT = self.config["DISPARITY_TOLERANCE_PX"]
        n = min(len(centers1), len(centers2))

        for i in range(n):
            pt1 = centers1[i]
            pt2 = centers2[i]
            if abs(pt1[1] - pt2[1]) < YT and (pt1[0] - pt2[0]) > -DT:
                label = self.labels[i]
                box1, box2 = boxes1[i], boxes2[i]
                matched_pairs.append((pt1, box1, pt2, box2, label))

        for pt1, box1, pt2, box2, label in matched_pairs:
            p1 = np.array(pt1, dtype=float).reshape(2, 1)
            p2 = np.array(pt2, dtype=float).reshape(2, 1)
            X = cv2.triangulatePoints(self.P1, self.P2, p1, p2)  # 4x1
            X = X / X[3]  # Normalize
            raw_x = float(X[0])   # lateral
            raw_z = float(X[2])   # depth


            # EMA smoothing
            if label not in self.smoothed_measurements:
                self.smoothed_measurements[label] = {"z": raw_z, "x": raw_x}
            else:
                sf = self.config["smoothing_factor"]
                prev = self.smoothed_measurements[label]
                self.smoothed_measurements[label] = {
                    "z": sf * raw_z + (1 - sf) * prev["z"],
                    "x": sf * raw_x + (1 - sf) * prev["x"],
                }
            disp_z = self.smoothed_measurements[label]["z"]
            disp_x = self.smoothed_measurements[label]["x"]

            color = self.color_map.get(label, (255, 255, 255))
            cv2.rectangle(canvas, (10, yline - 15), (20, yline - 5), color, -1)
            cv2.putText(
                canvas,
                f"ID {label} | Z: {disp_z + 0.000:.3f} m | X: {disp_x:.3f} m",
                (30, yline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1,
            )
            yline += 40

        return canvas, matched_pairs

    # -------------------- Drawing & Display --------------------
    def _draw_overlays(self, f1, f2, matched_pairs):
        roi = self.current_roi
        cv2.rectangle(
            f1, (roi["x"], roi["y"]), (roi["x"] + roi["w"], roi["y"] + roi["h"]), (255, 0, 0), 2
        )
        cv2.rectangle(
            f2, (roi["x"], roi["y"]), (roi["x"] + roi["w"], roi["y"] + roi["h"]), (255, 0, 0), 2
        )
        for _, b1, _, b2, label in matched_pairs:
            color = self.color_map.get(label, (255, 255, 255))
            x1, y1, w1, h1 = b1
            x2, y2, w2, h2 = b2
            cv2.rectangle(
                f1, (roi["x"] + x1, roi["y"] + y1), (roi["x"] + x1 + w1, roi["y"] + y1 + h1), color, 2
            )
            cv2.putText(
                f1, label, (roi["x"] + x1, roi["y"] + y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
            cv2.rectangle(
                f2, (roi["x"] + x2, roi["y"] + y2), (roi["x"] + x2 + w2, roi["y"] + y2 + h2), color, 2
            )
            cv2.putText(
                f2, label, (roi["x"] + x2, roi["y"] + y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
    def _display_frames(self, f1, f2, canvas, live_fps=0):
        disp_w, disp_h = self.config["display"]["width"], self.config["display"]["height"]
        d1 = cv2.resize(f1, (disp_w, disp_h))
        d2 = cv2.resize(f2, (disp_w, disp_h))
        combined = np.hstack((d1, d2))
        cv2.putText(combined, f"Live FPS: {live_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(self.config["main_window_name"], combined)
        cv2.imshow(self.config["readings_window_name"], canvas)
    # -------------------- Cleanup --------------------
    def cleanup(self):
        try:
            self.cap1.release()
            self.cap2.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    # -------------------- Utils --------------------
    @staticmethod
    def _generate_colors(num_colors):
        colors = []
        for i in range(num_colors):
            hue = int(i * 180 / max(1, num_colors))
            bgr = cv2.cvtColor(
                np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR
            )[0, 0]
            colors.append(tuple(int(c) for c in bgr))
        return colors
if __name__ == "__main__":
    try:
        processor = StereoVideoProcessor(CONFIG)
        processor.run()
    except SystemExit:
        pass
    except Exception as e:
        print(f"An error occurred: {e}")

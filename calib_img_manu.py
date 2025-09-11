
# capture_calibration_images_manual.py
# Manual capture tool for stereo ChArUco calibration with a live coverage map and camera property control.

import cv2
import numpy as np
import time, os, json, pickle

# --- CONFIG ---
CAM_INDEX_L = 0
CAM_INDEX_R = 2
BOARD_SQUARES_X = 16
BOARD_SQUARES_Y = 23
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
NUM_PAIRS_TO_CAPTURE = 25
CALIBRATION_FILE_FOR_DISPLAY = 'non_existent_file.pkl'  # path to params if available
IMAGE_DIR = 'calibration_images_stereo_full'

# Camera property defaults (can be changed via sliders)
CAMERA_SETTINGS = {
    "width": 1920,
    "height": 1080,
    "focus": 40,
    "exposure": 0,
    "brightness": 40,
    "contrast": 70,
    "saturation": 60,
    "gain": 0
}

capL, capR = None, None  # global for slider callbacks

def apply_settings_from_dict(cap, settings):
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
    cap.set(cv2.CAP_PROP_FPS, 30)

def on_trackbar_change(val):
    if capL is None or capR is None:
        return
    focus = cv2.getTrackbarPos('Focus', 'Controls')
    exposure = cv2.getTrackbarPos('Exposure', 'Controls') - 13
    brightness = cv2.getTrackbarPos('Brightness', 'Controls')
    contrast = cv2.getTrackbarPos('Contrast', 'Controls')
    saturation = cv2.getTrackbarPos('Saturation', 'Controls')
    gain = cv2.getTrackbarPos('Gain', 'Controls')
    for cap in [capL, capR]:
        cap.set(cv2.CAP_PROP_FOCUS, focus)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        cap.set(cv2.CAP_PROP_CONTRAST, contrast)
        cap.set(cv2.CAP_PROP_SATURATION, saturation)
        cap.set(cv2.CAP_PROP_GAIN, gain)

def create_control_panel():
    cv2.namedWindow('Controls')
    cv2.createTrackbar('Focus', 'Controls', CAMERA_SETTINGS['focus'], 255, on_trackbar_change)
    cv2.createTrackbar('Exposure', 'Controls', CAMERA_SETTINGS['exposure'] + 13, 13, on_trackbar_change)
    cv2.createTrackbar('Brightness', 'Controls', CAMERA_SETTINGS['brightness'], 255, on_trackbar_change)
    cv2.createTrackbar('Contrast', 'Controls', CAMERA_SETTINGS['contrast'], 255, on_trackbar_change)
    cv2.createTrackbar('Saturation', 'Controls', CAMERA_SETTINGS['saturation'], 255, on_trackbar_change)
    cv2.createTrackbar('Gain', 'Controls', CAMERA_SETTINGS['gain'], 255, on_trackbar_change)

# --- Setup ---
board = cv2.aruco.CharucoBoard(
    (BOARD_SQUARES_X, BOARD_SQUARES_Y),
    0.03577, 0.02531,
    ARUCO_DICT
)

os.makedirs(os.path.join(IMAGE_DIR, 'left'), exist_ok=True)
os.makedirs(os.path.join(IMAGE_DIR, 'right'), exist_ok=True)
print(f"Images will be saved in '{IMAGE_DIR}'")

# Create a blank image for the coverage map visualization
coverage_map = np.zeros((CAMERA_SETTINGS['height'], CAMERA_SETTINGS['width'], 3), dtype=np.uint8)
cv2.putText(coverage_map, "Capture Coverage Map", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3)
cv2.putText(coverage_map, "(Shows where board corners have been detected)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)


# Try to load calibration for rectified preview
use_rectify = False
try:
    with open(CALIBRATION_FILE_FOR_DISPLAY, 'rb') as f:
        params = pickle.load(f)
    mtxL, distL = params['mtxL'], params['distL']
    mtxR, distR = params['mtxR'], params['distR']
    R, T = params['R'], params['T']
    image_size = tuple(params['image_size'])
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtxL, distL, mtxR, distR, image_size, R, T, alpha=1)
    mapL1, mapL2 = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, image_size, cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, image_size, cv2.CV_32FC1)
    use_rectify = True
    print("Loaded calibration for rectified preview.")
except Exception:
    print("Display will be UNRECTIFIED (normal for new calibration).")

# Init cameras
capL = cv2.VideoCapture(CAM_INDEX_L, cv2.CAP_DSHOW)
capR = cv2.VideoCapture(CAM_INDEX_R, cv2.CAP_DSHOW)
if not capL.isOpened() or not capR.isOpened():
    raise RuntimeError("Cannot open both cameras")

print("Applying initial settings...")
apply_settings_from_dict(capL, CAMERA_SETTINGS)
apply_settings_from_dict(capR, CAMERA_SETTINGS)
create_control_panel()

aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())
capture_count = 0

while capture_count < NUM_PAIRS_TO_CAPTURE:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        print("Frame grab failed.")
        break

    displayL, displayR = frameL.copy(), frameR.copy()
    if use_rectify:
        displayL = cv2.remap(displayL, mapL1, mapL2, cv2.INTER_LINEAR)
        displayR = cv2.remap(displayR, mapR1, mapR2, cv2.INTER_LINEAR)

    # Detect ChArUco in both frames
    gray_l = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    corners_l, ids_l, _ = aruco_detector.detectMarkers(gray_l)
    corners_r, ids_r, _ = aruco_detector.detectMarkers(gray_r)

    ret_l = ret_r = False
    charuco_corners_l, charuco_ids_l = None, None
    charuco_corners_r, charuco_ids_r = None, None
    if ids_l is not None and len(ids_l) > 4:
        ret_l, charuco_corners_l, charuco_ids_l = cv2.aruco.interpolateCornersCharuco(corners_l, ids_l, gray_l, board)
        if ret_l:
            cv2.aruco.drawDetectedCornersCharuco(displayL, charuco_corners_l, charuco_ids_l)
    if ids_r is not None and len(ids_r) > 4:
        ret_r, charuco_corners_r, charuco_ids_r = cv2.aruco.interpolateCornersCharuco(corners_r, ids_r, gray_r, board)
        if ret_r:
            cv2.aruco.drawDetectedCornersCharuco(displayR, charuco_corners_r, charuco_ids_r)

    # --- HUD and Status ---
    board_detected = ret_l and ret_r
    status_text, status_color = "", (255, 255, 255)
    
    if board_detected:
        status_text, status_color = "READY - Press SPACE to capture", (0, 255, 0)  # Green
    else:
        status_text, status_color = "BOARD NOT DETECTED", (0, 0, 255)  # Red

    cv2.putText(displayL, f"Captured: {capture_count}/{NUM_PAIRS_TO_CAPTURE}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(displayL, status_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv2.putText(displayR, status_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    # --- Display ---
    combined = cv2.hconcat([displayL, displayR])
    cv2.imshow("Stereo Capture - Press SPACE to capture, Q to quit", cv2.resize(combined, (640, 480)))
    cv2.imshow("Capture Coverage", cv2.resize(coverage_map, (960, 540)))
    
    key = cv2.waitKey(1) & 0xFF

    # --- Controls ---
    if key == ord('q'):
        break
    
    # --- MANUAL CAPTURE LOGIC ---
    if board_detected and key == ord(' '):
        ts = time.strftime('%Y%m%d-%H%M%S')
        left_path = os.path.join(IMAGE_DIR, 'left', f'left_{ts}.png')
        right_path = os.path.join(IMAGE_DIR, 'right', f'right_{ts}.png')
        cv2.imwrite(left_path, frameL)
        cv2.imwrite(right_path, frameR)

        meta = {
            'timestamp': time.time(),
            'left_ids': charuco_ids_l.flatten().tolist() if ret_l and charuco_ids_l is not None else [],
            'right_ids': charuco_ids_r.flatten().tolist() if ret_r and charuco_ids_r is not None else []
        }
        with open(os.path.join(IMAGE_DIR, f'meta_{ts}.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"Saved pair {capture_count + 1}/{NUM_PAIRS_TO_CAPTURE}: {ts}")

        # Update the coverage map with the newly captured corners
        if ret_l and charuco_corners_l is not None:
            # Draw a filled circle for each detected corner
            for corner in charuco_corners_l:
                pt = (int(corner[0][0]), int(corner[0][1]))
                cv2.circle(coverage_map, pt, 5, (0, 255, 0), -1)  # Green dot for each corner

        capture_count += 1

print("Capture finished.")
capL.release()
capR.release()
cv2.destroyAllWindows()

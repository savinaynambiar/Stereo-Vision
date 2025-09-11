# calibrate_charuco.py
# Robust stereo calibration using ChArUco, with an added subpixel refinement step.

import cv2
import numpy as np
import glob
import pickle
import os

# --- CONFIG ---
# Camera settings used during capture (for reference)
CAMERA_SETTINGS = {
    "width": 1920,
    "height": 1080,
    "focus": 40,
    "exposure": -4,
    "brightness": 40,
    "contrast": 100,
    "saturation": 60,
    "gain": 0
}

# IMPORTANT: Double-check these values to ensure they match your physical board!
BOARD_SQUARES_X = 8
BOARD_SQUARES_Y = 11
SQUARE_LENGTH_M = 0.035
MARKER_LENGTH_M = 0.025
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
IMAGE_DIR = 'calibration_images_stereo_lap1'
OUTPUT_FILE = 'stereo_params_charuco_stereo_lap_best.pkl' # Changed output file name
MIN_COMMON_CORNERS = 10
# -------------

print("Starting stereo calibration with explicit subpixel refinement...")

# Create the ChArUco board object
board = cv2.aruco.CharucoBoard((BOARD_SQUARES_X, BOARD_SQUARES_Y),
                               SQUARE_LENGTH_M, MARKER_LENGTH_M, ARUCO_DICT)
aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())

# Get image paths
images_left = sorted(glob.glob(os.path.join(IMAGE_DIR, 'left', '*.png')))
images_right = sorted(glob.glob(os.path.join(IMAGE_DIR, 'right', '*.png')))
assert len(images_left) == len(images_right) and len(images_left) > 0, "Mismatched or empty image sets"

# Prepare lists for calibration
all_corners_l, all_corners_r, obj_points = [], [], []
image_size = None

# Iterate through image pairs
for i, (lp, rp) in enumerate(zip(images_left, images_right)):
    img_l, img_r = cv2.imread(lp), cv2.imread(rp)
    gray_l, gray_r = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = gray_l.shape[::-1]

    # Detect markers
    corners_l, ids_l, _ = aruco_detector.detectMarkers(gray_l)
    corners_r, ids_r, _ = aruco_detector.detectMarkers(gray_r)
    if ids_l is None or ids_r is None:
        print(f"Skipping pair {i}: Markers not found in one or both images.")
        continue

    # Interpolate to find ChArUco corners
    ret_l, char_l, ids_l = cv2.aruco.interpolateCornersCharuco(corners_l, ids_l, gray_l, board)
    ret_r, char_r, ids_r = cv2.aruco.interpolateCornersCharuco(corners_r, ids_r, gray_r, board)
    
    # --- NEW: Explicit Subpixel Refinement ---
    # Even though interpolateCornersCharuco is subpixel, we can refine it further.
    if ret_l:
        # Define criteria for the subpixel refinement algorithm
        subpixel_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Refine the corner locations for the left image
        cv2.cornerSubPix(gray_l, char_l, winSize=(5, 5), zeroZone=(-1, -1), criteria=subpixel_criteria)
    if ret_r:
        subpixel_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Refine the corner locations for the right image
        cv2.cornerSubPix(gray_r, char_r, winSize=(5, 5), zeroZone=(-1, -1), criteria=subpixel_criteria)
    # --- END NEW ---
    
    if not ret_l or not ret_r:
        print(f"Skipping pair {i}: ChArUco interpolation failed.")
        continue

    # Find common corners detected in both images
    ids_l, ids_r = ids_l.flatten(), ids_r.flatten()
    common = np.intersect1d(ids_l, ids_r)
    if len(common) < MIN_COMMON_CORNERS:
        print(f"Skipping pair {i}: Not enough common corners ({len(common)} < {MIN_COMMON_CORNERS}).")
        continue

    # Create maps of ID -> corner position for quick lookup
    map_l = {id_val: corner for id_val, corner in zip(ids_l, char_l)}
    map_r = {id_val: corner for id_val, corner in zip(ids_r, char_r)}

    # Get the image points (pixels) for the common corners
    cornersL = np.array([map_l[id_val] for id_val in common], dtype=np.float32)
    cornersR = np.array([map_r[id_val] for id_val in common], dtype=np.float32)

    # Get the 3D object points from the board definition
    all_obj_points = board.getChessboardCorners()
    objs = np.array([all_obj_points[id_val] for id_val in common], dtype=np.float32)

    all_corners_l.append(cornersL)
    all_corners_r.append(cornersR)
    obj_points.append(objs)

print(f"\nUsing {len(obj_points)} valid pairs for calibration.")
if len(obj_points) < 15:
    raise RuntimeError("Not enough valid pairs for calibration. Check images and board parameters.")

# Calibrate each camera individually
print("Calibrating intrinsics for each camera...")
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_points, all_corners_l, image_size, None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_points, all_corners_r, image_size, None, None)

# Perform stereo calibration
print("Performing stereo calibration...")
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    obj_points, all_corners_l, all_corners_r,
    mtxL, distL, mtxR, distR, image_size,
    criteria=criteria, flags=flags
)

print(f"Stereo calibration RMS error: {rms:.4f}")

# Perform stereo rectification
print("Performing stereo rectification...")
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtxL, distL, mtxR, distR, image_size, R, T, alpha=0)

# Store all calibration data
data = dict(
    mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR,
    R=R, T=T, E=E, F=F,
    R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
    image_size=image_size, rms_error=rms,
    camera_settings=CAMERA_SETTINGS
)

# Save the data to a file
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(data, f)

print("Saved refined calibration data to", OUTPUT_FILE)
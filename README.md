Real-Time Stereo Vision Pipeline for 3D Measurement
This project is an end-to-end computer vision system built with Python and OpenCV for capturing, calibrating, and performing real-time 3D measurements using a stereo camera setup.

The pipeline is composed of three main components: a manual tool for capturing high-quality calibration images, a robust script for computing stereo calibration parameters, and a final application that uses these parameters to perform live 3D detection and measurement.

Key Features
High-Accuracy Calibration: Utilizes a ChArUco board for robust stereo camera calibration, including a subpixel refinement step to enhance precision.

Interactive Calibration Capture: A GUI tool (calib_img_manu.py) provides live, rectified previews, direct camera controls (focus, exposure, brightness), and a visual coverage map to ensure a high-quality dataset for calibration.

Real-Time 3D Measurement: The main application (detect_2.py) processes rectified stereo streams to detect objects via contour analysis and calculates their 3D position (height and distance) using triangulation.

Interactive Detection GUI: The main application features a movable Region of Interest (ROI), live controls for Canny edge detection thresholds, and separate display windows for video feeds and measurement readouts.

Measurement Correction: Implements a simple linear regression model to calibrate raw, triangulated measurements against ground-truth values, improving real-world accuracy.

Modular and Configurable: The entire pipeline is broken into logical scripts that are well-documented and can be easily configured through Python dictionaries.

Project Components
calib_img_manu.py: An interactive tool to manually capture stereo image pairs of a ChArUco calibration board. This ensures you get a diverse and high-quality set of images covering the entire field of view.

calib_char_pkl.py: This script processes the images captured by the first tool. It detects the board corners, calibrates each camera individually, performs the stereo calibration, and saves all intrinsic and extrinsic parameters to a .pkl file.

detect_2.py: The main real-time application. It loads the calibration .pkl file, opens the stereo camera streams, and performs live 3D measurement on objects detected within the specified ROI.

Workflow & Usage
Step 0: Prerequisites
Hardware:

Two webcams connected to your computer.

A printed ChArUco calibration board.

Software:

Python 3

OpenCV and NumPy libraries. Install them using pip:

pip install opencv-python numpy

Step 1: Capture Calibration Images
First, you need to capture pairs of images from your stereo cameras to be used for calibration.

Configure calib_img_manu.py by setting the camera indices (CAM_INDEX_L, CAM_INDEX_R) and the output directory (IMAGE_DIR).

Run the script:

python calib_img_manu.py

A window will appear showing the live feeds from both cameras. Move the ChArUco board around, capturing it at different angles, positions, and distances.

When the board is clearly visible in both feeds, press the SPACEBAR to save a stereo pair. The coverage map window will update to show you which areas of the frame you have covered.

Repeat until you have captured the desired number of pairs (NUM_PAIRS_TO_CAPTURE). Press 'q' to quit.

Step 2: Perform Stereo Calibration
Next, process the captured images to generate the calibration file.

Configure calib_char_pkl.py to match your setup. You must set the correct board dimensions (BOARD_SQUARES_X, BOARD_SQUARES_Y), the real-world measurements (SQUARE_LENGTH_M, MARKER_LENGTH_M), and the IMAGE_DIR where you saved your images.

Run the script:

python calib_char_pkl.py

The script will analyze the image pairs, calculate the calibration parameters, and save them to an output file (e.g., stereo_params_charuco_stereo_lap_best.pkl).

Step 3: Run the Real-Time Detection
Finally, use the generated calibration file to perform live measurements.

Configure detect_2.py.

Set video_path_left and video_path_right to your camera indices.

Update calibration_file to the path of the .pkl file you just created.

(Optional) Calibrate the measurement system by populating calibration_points with your own raw vs. actual measurements to fine-tune accuracy.

Run the main application:

python detect_2.py

The application will open, showing the synced stereo feeds. You can:

Drag the blue ROI box to the area you want to analyze.

Use the "Correction Controls" window to adjust Canny edge thresholds for better object detection.

View live height and distance measurements in the "Live Measurements" window.

Press SPACEBAR to pause/resume the video feed.

Press ESC to exit.

License
This project is licensed under the MIT License. See the LICENSE file for details.

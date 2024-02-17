import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pytransform3d.transform_manager import TransformManager
from pytransform3d.rotations import matrix_from_euler_xyz
from pytransform3d.transformations import transform_from, plot_transform

class StereoVisualOdometry:
    def __init__(self, calib_file):
        self.calib_data = self.load_calibration(calib_file)
        self.disparity = self.setup_disparity()
        self.feature_detector = cv2.ORB_create(nfeatures=500)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_image_left = None
        self.prev_image_right = None
        self.tm = TransformManager()

    def load_calibration(self, filepath):
        with open(filepath, 'r') as f:
            calib = json.load(f)
        # Process calibration data
        return calib

    def setup_disparity(self):
        # Setup StereoSGBM matcher
        return cv2.StereoSGBM_create()

    def process_frame(self, img_left, img_right):
        # Convert images to grayscale
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        # Feature detection, matching, and disparity computation
        keypoints_left, descriptors_left = self.feature_detector.detectAndCompute(gray_left, None)
        keypoints_right, descriptors_right = self.feature_detector.detectAndCompute(gray_right, None)
        
        matches = self.bf_matcher.match(descriptors_left, descriptors_right)
        # Filter matches and triangulate points to get 3D points
        # Estimate camera pose
        
        # Update previous images
        self.prev_image_left = gray_left
        self.prev_image_right = gray_right

    def visualize_path(self):
        # Use matplotlib or pytransform3d to visualize the path

def main():
    calib_file = 'calibration_data.json'
    svo = StereoVisualOdometry(calib_file)
    
    # Setup video capture
    cap_left = cv2.VideoCapture(0)  # Adjust ID for your camera
    cap_right = cv2.VideoCapture(1)  # Adjust ID for your camera

    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        if not ret_left or not ret_right:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        svo.process_frame(frame_left, frame_right)

        # Visualization can be updated here if real-time display is needed
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

    # After processing, visualize the final path and pose
    svo.visualize_path()

if __name__ == "__main__":
    main()
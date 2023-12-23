import json

import cv2
import cv2.aruco as aruco
import numpy as np


def find_aruco_markers(image, camera_matrix, dist_coeffs):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
    aruco_params = aruco.DetectorParameters_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(
        gray, aruco_dict, parameters=aruco_params)

    centers = []
    if ids is not None:
        for i, corner in enumerate(corners):
            center = tuple(np.mean(corner, axis=1)[0].astype(int))
            # (center coordinates, marker ID)
            centers.append((center, ids[i][0]))

    return centers


# Load the stereo calibration parameters from the JSON file
with open('calibration_data.json', 'r') as file:
    calibration_data = json.load(file)

left_camera_matrix = np.array(calibration_data['left_camera_matrix'])
left_dist_coeffs = np.array(calibration_data['left_dist_coeffs'])
right_camera_matrix = np.array(calibration_data['right_camera_matrix'])
right_dist_coeffs = np.array(calibration_data['right_dist_coeffs'])
R = np.array(calibration_data['R'])
T = np.array(calibration_data['T'])

left_cam_index = 0  # Replace with your left camera index
right_cam_index = 2  # Replace with your right camera index

left_cap = cv2.VideoCapture(left_cam_index)
right_cap = cv2.VideoCapture(right_cam_index)

while True:
    ret_left, left_frame = left_cap.read()
    ret_right, right_frame = right_cap.read()

    if not ret_left or not ret_right:
        print("Error: Unable to capture video")
        break

    left_markers = find_aruco_markers(
        left_frame, left_camera_matrix, left_dist_coeffs)
    right_markers = find_aruco_markers(
        right_frame, right_camera_matrix, right_dist_coeffs)

    for left_center, left_id in left_markers:
        cv2.circle(left_frame, left_center, 2, (0, 0, 255), 3)

        for right_center, right_id in right_markers:
            if left_id == right_id:
                cv2.circle(right_frame, right_center, 2, (0, 0, 255), 3)

                disparity = abs(left_center[0] - right_center[0])
                Z = (left_camera_matrix[0, 0] * T[0]) / disparity
                X = (left_center[0] - left_camera_matrix[0, 2]
                     ) * Z / left_camera_matrix[0, 0]
                Y = (left_center[1] - left_camera_matrix[1, 2]
                     ) * Z / left_camera_matrix[0, 0]

                coord_text = f"ID {left_id}: X: {float(X):.2f}, Y: {float(Y):.2f}, Z: {float(Z):.2f}"
                cv2.putText(left_frame, coord_text, (
                    left_center[0] + 10, left_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Left Frame', left_frame)
    cv2.imshow('Right Frame', right_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

left_cap.release()
right_cap.release()
cv2.destroyAllWindows()

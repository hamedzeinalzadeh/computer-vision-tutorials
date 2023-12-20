import cv2
import numpy as np
import json
import open3d as o3d


def load_calibration_parameters(file_path):
    with open(file_path, 'r') as file:
        calibration_data = json.load(file)
    return calibration_data


def compute_disparity(frame_left, frame_right, stereo_matcher):
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    disparity = stereo_matcher.compute(gray_left, gray_right)
    return disparity


def generate_point_cloud(disparity, Q):
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    return points_3D


def save_point_cloud(point_cloud, filename):
    mask = point_cloud[:, :, 2] > 0
    point_cloud = point_cloud[mask]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.io.write_point_cloud(filename, pcd)


def main():
    file_path = './calibration_data.json'
    calibration_data = load_calibration_parameters(file_path)

    # Extract Q matrix from calibration data
    Q = np.array(calibration_data['Q'])

    # Initialize stereo cameras and stereo matcher
    cap_left = cv2.VideoCapture(0)  # Adjust camera index if needed
    cap_right = cv2.VideoCapture(2)  # Adjust camera index if needed
    stereo_matcher = cv2.StereoSGBM_create()

    try:
        while True:
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()

            if not ret_left or not ret_right:
                break

            disparity = compute_disparity(
                frame_left, frame_right, stereo_matcher)
            point_cloud = generate_point_cloud(disparity, Q)
            save_point_cloud(point_cloud, "output_point_cloud.ply")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap_left.release()
        cap_right.release()


if __name__ == '__main__':
    main()

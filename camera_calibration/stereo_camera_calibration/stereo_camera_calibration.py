from pathlib import Path
import json
import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm


def find_chessboard_corners(image_path, chessboard_size):
    image = cv2.imread(str(image_path))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)
    return ret, corners, gray_image, image


def save_stereo_maps(file_name, stereo_map_l, stereo_map_r):
    cv2_file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_WRITE)
    cv2_file.write('stereoMapL_x', stereo_map_l[0])
    cv2_file.write('stereoMapL_y', stereo_map_l[1])
    cv2_file.write('stereoMapR_x', stereo_map_r[0])
    cv2_file.write('stereoMapR_y', stereo_map_r[1])
    cv2_file.release()


def calculate_reprojection_error(objpoints, imgpoints, camera_matrix, dist_coeffs, rvecs, tvecs):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2,
                         cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)
    return mean_error


def save_calibration_parameters(file_name, camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, R, T, Q):
    calibration_data = {
        'left_camera_matrix': camera_matrix_l.tolist(),
        'left_dist_coeffs': dist_coeffs_l.tolist(),
        'right_camera_matrix': camera_matrix_r.tolist(),
        'right_dist_coeffs': dist_coeffs_r.tolist(),
        'R': R.tolist(),  # Rotation matrix
        'T': T.tolist(),   # Translation vector
        'Q': Q.tolist()   # Q matrix
    }

    with open(file_name, 'w') as file:
        json.dump(calibration_data, file, indent=4)


def main():
    # Parameters
    chessboard_size = (5, 7)
    frame_size = (640, 480)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    size_of_chessboard_squares_mm = 40

    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]
                           ].T.reshape(-1, 2) * size_of_chessboard_squares_mm

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []

    # Define the paths
    synched_dir = Path.cwd() / 'synched'
    images_left = sorted(synched_dir.glob('img_cam0_*.png'))
    images_right = sorted(synched_dir.glob('img_cam1_*.png'))

    progress_bar = tqdm(total=len(images_left), desc='Finding Corners')
    for img_left, img_right in zip(images_left, images_right):
        ret_l, corners_l, gray_l, img_l = find_chessboard_corners(
            img_left, chessboard_size)
        ret_r, corners_r, gray_r, img_r = find_chessboard_corners(
            img_right, chessboard_size)

        if ret_l and ret_r:
            objpoints.append(objp)
            corners_l = cv2.cornerSubPix(
                gray_l, corners_l, (11, 11), (-1, -1), criteria)
            imgpoints_l.append(corners_l)
            corners_r = cv2.cornerSubPix(
                gray_r, corners_r, (11, 11), (-1, -1), criteria)
            imgpoints_r.append(corners_r)

            cv2.drawChessboardCorners(img_l, chessboard_size, corners_l, ret_l)
            cv2.drawChessboardCorners(img_r, chessboard_size, corners_r, ret_r)
            cv2.imshow('img left', img_l)
            cv2.imshow('img right', img_r)
            cv2.waitKey(10)

        # Update the progress after each iteration
        progress_bar.update(1)  # TODO: Fix the extra progress bar issue

    cv2.destroyAllWindows()

    # Close the progress bar after the loop
    progress_bar.close()

    # Calibrate cameras
    ret_l, camera_matrix_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints, imgpoints_l, frame_size, None, None)
    ret_r, camera_matrix_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints, imgpoints_r, frame_size, None, None)

    # Calculate reprojection errors
    error_l = calculate_reprojection_error(
        objpoints, imgpoints_l, camera_matrix_l, dist_l, rvecs_l, tvecs_l)
    error_r = calculate_reprojection_error(
        objpoints, imgpoints_r, camera_matrix_r, dist_r, rvecs_r, tvecs_r)
    logger.info(f"Left_Camera Reprojection Error: {error_l}")
    logger.info(f"Right_Camera Reprojection Error: {error_r}")

    # Stereo Calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS +
                       cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret_stereo, new_camera_matrix_l, dist_l, new_camera_matrix_r, dist_r, rot, trans, essential_matrix, fundamental_matrix = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, camera_matrix_l, dist_l, camera_matrix_r, dist_r, gray_l.shape[::-1], criteria_stereo, flags)

    # Stereo Rectification
    rectify_scale = 1
    rect_l, rect_r, proj_matrix_l, proj_matrix_r, Q, roi_l, roi_r = cv2.stereoRectify(
        new_camera_matrix_l, dist_l, new_camera_matrix_r, dist_r, gray_l.shape[::-1], rot, trans, rectify_scale)

    # Comment explaining the unused variables
    # NOTE: Essential matrix, fundamental matrix, Q matrix, roi_l, and roi_r are computed as part of the stereo calibration
    # and rectification process. These are not used in the current reprojection error calculation but are valuable for
    # advanced stereo vision applications like 3D reconstruction, feature matching, and depth estimation.

    # Undistort and Rectify Maps
    stereo_map_l = cv2.initUndistortRectifyMap(
        new_camera_matrix_l, dist_l, rect_l, proj_matrix_l, gray_l.shape[::-1], cv2.CV_16SC2)
    stereo_map_r = cv2.initUndistortRectifyMap(
        new_camera_matrix_r, dist_r, rect_r, proj_matrix_r, gray_r.shape[::-1], cv2.CV_16SC2)


    # logger.info(f"Stereo Reprojection Error: {stereo_error}")
    logger.info(f"ret_stereo: {ret_stereo}")

    # Save Stereo Maps
    save_stereo_maps('stereo_map.xml', stereo_map_l, stereo_map_r)
    logger.info("Stereo maps saved.")


    # Save the calibration parameters
    save_calibration_parameters('calibration_data.json', new_camera_matrix_l, dist_l, new_camera_matrix_r, dist_r, rot, trans, Q)
    logger.info('Calibration parameters(.json) saved')


if __name__ == "__main__":
    main()

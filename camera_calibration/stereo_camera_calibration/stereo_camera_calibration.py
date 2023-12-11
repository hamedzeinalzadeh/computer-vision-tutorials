from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm


def find_chessboard_corners(image_path, chessboard_size):
    """
    Find chessboard corners in the image.

    :param image_path: Path to the image.
    :param chessboard_size: Size of the chessboard (number of inner corners per a chessboard row and column).
    :return: Tuple (ret, corners) - ret is a boolean indicating success, corners are the detected corners.
    """
    image = cv2.imread(str(image_path))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)
    return ret, corners, gray_image, image


def calibrate_single_camera(objpoints, imgpoints, frame_size):
    """
    Calibrate a single camera.

    :param objpoints: 3D points in real world space.
    :param imgpoints: 2D points in image plane.
    :param frame_size: Size of the image frame.
    :return: Calibration parameters including camera matrix and distortion coefficients.
    """
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, frame_size, None, None)
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs


def stereo_calibration(objpoints, imgpoints_l, imgpoints_r, camera_matrix_l, dist_l, camera_matrix_r, dist_r, image_shape, criteria_stereo, flags):
    """
    Perform stereo calibration.

    :param objpoints: 3D points in real world space.
    :param imgpoints_l: 2D points in image plane for left camera.
    :param imgpoints_r: 2D points in image plane for right camera.
    :param camera_matrix_l: Camera matrix for left camera.
    :param dist_l: Distortion coefficients for left camera.
    :param camera_matrix_r: Camera matrix for right camera.
    :param dist_r: Distortion coefficients for right camera.
    :param image_shape: Shape of the images used for calibration.
    :param criteria_stereo: Criteria for stereo calibration.
    :param flags: Flags for stereo calibration.
    :return: Stereo calibration parameters.
    """
    ret_stereo, camera_matrix_l, dist_l, camera_matrix_r, dist_r, rot, trans, essential_matrix, fundamental_matrix = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, camera_matrix_l, dist_l, camera_matrix_r, dist_r, image_shape, criteria_stereo, flags)
    return ret_stereo, camera_matrix_l, dist_l, camera_matrix_r, dist_r, rot, trans, essential_matrix, fundamental_matrix


def stereo_rectification(camera_matrix_l, dist_l, camera_matrix_r, dist_r, image_shape, rot, trans, rectify_scale):
    """
    Perform stereo rectification.

    :param camera_matrix_l: Camera matrix for left camera.
    :param dist_l: Distortion coefficients for left camera.
    :param camera_matrix_r: Camera matrix for right camera.
    :param dist_r: Distortion coefficients for right camera.
    :param image_shape: Shape of the images used for calibration.
    :param rot: Rotation matrix from stereo calibration.
    :param trans: Translation vector from stereo calibration.
    :param rectify_scale: Scale parameter for stereoRectify.
    :return: Rectification parameters including rectification transforms, projection matrices and disparity-to-depth mapping matrix.
    """
    rect_l, rect_r, proj_matrix_l, proj_matrix_r, Q, roi_l, roi_r = cv2.stereoRectify(
        camera_matrix_l, dist_l, camera_matrix_r, dist_r, image_shape, rot, trans, rectify_scale, (0, 0))
    return rect_l, rect_r, proj_matrix_l, proj_matrix_r, Q, roi_l, roi_r


def init_undistort_rectify_map(camera_matrix, dist_coeffs, rect, proj_matrix, image_shape):
    """
    Initialize undistortion and rectification transformation map.

    :param camera_matrix: Camera matrix.
    :param dist_coeffs: Distortion coefficients.
    :param rect: Rectification transform.
    :param proj_matrix: Projection matrix.
    :param image_shape: Shape of the images used for calibration.
    :return: Undistortion and rectification transformation map.
    """
    stereo_map = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, rect, proj_matrix, image_shape, cv2.CV_16SC2)
    return stereo_map


def save_stereo_maps(file_name, stereo_map_l, stereo_map_r):
    """
    Save stereo maps to a file.

    :param file_name: Name of the file to save the maps.
    :param stereo_map_l: Stereo map for left camera.
    :param stereo_map_r: Stereo map for right camera.
    """
    cv2_file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_WRITE)
    cv2_file.write('stereoMapL_x', stereo_map_l[0])
    cv2_file.write('stereoMapL_y', stereo_map_l[1])
    cv2_file.write('stereoMapR_x', stereo_map_r[0])
    cv2_file.write('stereoMapR_y', stereo_map_r[1])
    cv2_file.release()


def calculate_reprojection_error(objpoints, imgpoints, camera_matrix, dist_coeffs, rvecs, tvecs):
    """
    Calculate the reprojection error for a camera.

    :param objpoints: 3D object points.
    :param imgpoints: 2D image points.
    :param camera_matrix: Camera matrix.
    :param dist_coeffs: Distortion coefficients.
    :param rvecs: Rotation vectors.
    :param tvecs: Translation vectors.
    :return: Mean reprojection error.
    """
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2,
                         cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)
    return mean_error


def calculate_stereo_reprojection_error(objpoints, imgpoints_l, imgpoints_r, camera_matrix_l, dist_l, camera_matrix_r, dist_r, rot, trans):
    """
    Calculate a more accurate stereo reprojection error.

    :param objpoints: 3D object points.
    :param imgpoints_l: 2D image points for the left camera.
    :param imgpoints_r: 2D image points for the right camera.
    :param camera_matrix_l: Camera matrix for the left camera.
    :param dist_l: Distortion coefficients for the left camera.
    :param camera_matrix_r: Camera matrix for the right camera.
    :param dist_r: Distortion coefficients for the right camera.
    :param rot: Rotation matrix from the stereo calibration.
    :param trans: Translation vector from the stereo calibration.
    :return: Mean stereo reprojection error.

    Note:
    - This function assumes that 'rot' and 'trans' are the rotation and translation matrices
      describing the transformation from the left to the right camera.
    - The function uses the first set of image points to determine the image size for stereo rectification,
      so it's important that all image points are consistent in size.
    - This approach provides a more comprehensive measurement of the stereo system's accuracy, taking into
      account the relative positions and orientations of the two cameras.
    """
    total_error = 0

    height, width = imgpoints_l[0].shape[:2]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(camera_matrix_l, dist_l, camera_matrix_r, dist_r, (width, height), rot, trans)

    for i in range(len(objpoints)):
        # Undistort points
        undistorted_points_l = cv2.undistortPoints(
            imgpoints_l[i], camera_matrix_l, dist_l, P=P1)
        undistorted_points_r = cv2.undistortPoints(
            imgpoints_r[i], camera_matrix_r, dist_r, P=P2)

        # Triangulate points in 3D space
        points_4d = cv2.triangulatePoints(
            P1, P2, undistorted_points_l, undistorted_points_r)
        points_3d = points_4d / np.tile(points_4d[-1, :], (4, 1))

        # Reproject points back to 2D
        reprojected_points_l, _ = cv2.projectPoints(
            points_3d[:3].T, R1, trans, camera_matrix_l, dist_l)
        reprojected_points_r, _ = cv2.projectPoints(
            points_3d[:3].T, R2, -trans, camera_matrix_r, dist_r)

        # Calculate error
        error_l = cv2.norm(
            imgpoints_l[i], reprojected_points_l, cv2.NORM_L2) / len(reprojected_points_l)
        error_r = cv2.norm(
            imgpoints_r[i], reprojected_points_r, cv2.NORM_L2) / len(reprojected_points_r)
        total_error += (error_l + error_r) / 2

    mean_error = total_error / len(objpoints)  # Calculate the mean error
    return mean_error


def main():
    chessboard_size = (4, 4)
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
            cv2.waitKey(100)

        # Update the progress after each iteration
        progress_bar.update(1)

    cv2.destroyAllWindows()

    # Close the progress bar after the loop
    progress_bar.close()

    # Calibrate cameras
    ret_l, camera_matrix_l, dist_l, rvecs_l, tvecs_l = calibrate_single_camera(
        objpoints, imgpoints_l, frame_size)
    ret_r, camera_matrix_r, dist_r, rvecs_r, tvecs_r = calibrate_single_camera(
        objpoints, imgpoints_r, frame_size)

    # Calculate reprojection errors
    error_l = calculate_reprojection_error(
        objpoints, imgpoints_l, camera_matrix_l, dist_l, rvecs_l, tvecs_l)
    error_r = calculate_reprojection_error(
        objpoints, imgpoints_r, camera_matrix_r, dist_r, rvecs_r, tvecs_r)
    logger.info(f"Left Camera Reprojection Error: {error_l}")
    logger.info(f"Right Camera Reprojection Error: {error_r}")

    # Stereo Calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS +
                       cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret_stereo, new_camera_matrix_l, dist_l, new_camera_matrix_r, dist_r, rot, trans, essential_matrix, fundamental_matrix = stereo_calibration(
        objpoints, imgpoints_l, imgpoints_r, camera_matrix_l, dist_l, camera_matrix_r, dist_r, gray_l.shape[::-1], criteria_stereo, flags)

    # Stereo Rectification
    rectify_scale = 1
    rect_l, rect_r, proj_matrix_l, proj_matrix_r, Q, roi_l, roi_r = stereo_rectification(
        new_camera_matrix_l, dist_l, new_camera_matrix_r, dist_r, gray_l.shape[::-1], rot, trans, rectify_scale)

    # Comment explaining the unused variables
    # NOTE: Essential matrix, fundamental matrix, Q matrix, roi_l, and roi_r are computed as part of the stereo calibration
    # and rectification process. These are not used in the current reprojection error calculation but are valuable for
    # advanced stereo vision applications like 3D reconstruction, feature matching, and depth estimation.

    # Undistort and Rectify Maps
    stereo_map_l = init_undistort_rectify_map(
        new_camera_matrix_l, dist_l, rect_l, proj_matrix_l, gray_l.shape[::-1])
    stereo_map_r = init_undistort_rectify_map(
        new_camera_matrix_r, dist_r, rect_r, proj_matrix_r, gray_r.shape[::-1])

    # Calculate stereo reprojection error
    stereo_error = calculate_stereo_reprojection_error(
        objpoints, imgpoints_l, imgpoints_r, camera_matrix_l, dist_l, camera_matrix_r, dist_r, rot, trans)

    logger.info(f"Stereo Reprojection Error: {stereo_error}")

    # Save Stereo Maps
    save_stereo_maps('stereo_map.xml', stereo_map_l, stereo_map_r)
    logger.info("Stereo maps saved.")


if __name__ == "__main__":
    main()

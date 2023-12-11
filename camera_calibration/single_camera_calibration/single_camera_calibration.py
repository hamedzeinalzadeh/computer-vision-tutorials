import glob

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm


def find_chessboard_corners(images, chessboard_size, chessboard_grid_size_mm, criteria):
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0],
                           0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= chessboard_grid_size_mm

    objpoints, imgpoints = [], []

    for image in tqdm(images, desc='Finding Corners'):
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(100)

    cv2.destroyAllWindows()
    return objpoints, imgpoints


def calibrate_camera(objpoints, imgpoints, frameSize):
    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, frameSize, None, None)
    np.savez("single_camera_calib_params", cameraMatrix=cameraMatrix,
             dist=dist, rvecs=rvecs, tvecs=tvecs)
    return cameraMatrix, dist, rvecs, tvecs


def undistort_image(image_path, cameraMatrix, dist):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
        cameraMatrix, dist, (w, h), 1, (w, h))

    # Method 1: Undistort
    dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('caliResult1.png', dst)

    # Method 2: Undistort with Remapping
    mapx, mapy = cv2.initUndistortRectifyMap(
        cameraMatrix, dist, None, newCameraMatrix, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('caliResult2.png', dst)


def calculate_reprojection_error(objpoints, imgpoints, cameraMatrix, dist, rvecs, tvecs):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv2.norm(imgpoints[i], imgpoints2,
                         cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)
    logger.info(
        f"Total Error: {total_error:.3f}, Mean Error: {mean_error:.3f}")


def main():
    chessboard_size = (4, 4)
    frameSize = (640, 480)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    chessboard_grid_size_mm = 40
    images = glob.glob('./single_camera_images/*.jpg')

    objpoints, imgpoints = find_chessboard_corners(
        images, chessboard_size, chessboard_grid_size_mm, criteria)
    cameraMatrix, dist, rvecs, tvecs = calibrate_camera(
        objpoints, imgpoints, frameSize)
    undistort_image(
        './single_camera_images/2023-12-09-014609.jpg', cameraMatrix, dist)
    calculate_reprojection_error(
        objpoints, imgpoints, cameraMatrix, dist, rvecs, tvecs)


if __name__ == '__main__':
    main()

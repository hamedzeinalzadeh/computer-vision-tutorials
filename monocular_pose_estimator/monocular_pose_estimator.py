import glob

import cv2
import numpy as np


# Load the camera calibration parameters from "*.npz" file
def load_calibration_parameters(filename='single_camera_calib_params.npz'):
    with np.load(filename) as f:
        mtx, dist, rvecs, tvecs = [f[i] for i in (
            'cameraMatrix', 'dist', 'rvecs', 'tvecs')]
    return mtx, dist, rvecs, tvecs

# Draw the 3D world coordinate axes on the image


def draw_axes(img, corner, imgpts):
    """
    Draw 3D axes on the image.

    Args:
    - img: The original image.
    - corner: The first detected chessboard corner in the image.
    - imgpts: Projected 3D points into the image plane.

    Returns:
    - The image with the 3D axes drawn.
    """

    # Colors for the axes: Blue, Green, Red (in BGR format)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    # Drawing each axis line
    for pt, color in zip(imgpts, colors):
        img = cv2.line(img, tuple(corner.ravel()),
                       tuple(pt.ravel()), color, 10)

    return img



def draw_boxes(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


def process_images(image_files, objp, axis_boxes, mtx, dist):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for image_file in image_files:
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (24, 17), None)

        if ret:
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
            imgpts, jac = cv2.projectPoints(
                axis_boxes, rvecs, tvecs, mtx, dist)

            img = draw_boxes(img, corners2, imgpts)
            cv2.imshow('img', img)

            k = cv2.waitKey(0) & 0xFF
            if k == ord('q'):
                cv2.imwrite(f'pose_{image_file}', img)

    cv2.destroyAllWindows()


def main():
    mtx, dist = load_calibration_parameters('single_camera_calib_params.npz')
    objp = np.zeros((24*17, 3), np.float32)
    objp[:, :2] = np.mgrid[0:24, 0:17].T.reshape(-1, 2)
    axis_boxes = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                             [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

    image_files = glob.glob('undistorted*.png')
    process_images(image_files, objp, axis_boxes, mtx, dist)


if __name__ == '__main__':
    main()

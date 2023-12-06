import glob

import cv2
import numpy as np

"""
1 - Find chessboard corners(object points and imagepoints)
"""

chessboard_size = (3, 3)
frameSize = (640, 480)


# termination criteria(OpenCv default)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(2,2,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0],
                       0:chessboard_size[1]].T.reshape(-1, 2)

chessboard_grid_size_mm = 40
objp = objp * chessboard_grid_size_mm


# Lists of object-points and image-points of all the images
objpoints = []  # 3d point in real world coordinate system
imgpoints = []  # 2d points in image plane

# Add the paths of single-camera calibration images
path_to_search = './single_camera_calibration/*.png'
images = glob.glob(path_to_search)

for image in images:

    img = cv2.imread(image)
    gray = cv2.cv2tColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, after refining them, add object-points and image-points
    if ret:

        objpoints.append(objp)
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(1000)


cv2.destroyAllWindows()


############## CALIBRATION #######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, frameSize, None, None)


############## UNDISTORTION #####################################################

img = cv2.imread('cali5.png')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
    cameraMatrix, dist, (w, h), 1, (w, h))


# Undistort
dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('caliResult1.png', dst)


# Undistort with Remapping
mapx, mapy = cv2.initUndistortRectifyMap(
    cameraMatrix, dist, None, newCameraMatrix, (w, h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('caliResult2.png', dst)


# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error/len(objpoints)))

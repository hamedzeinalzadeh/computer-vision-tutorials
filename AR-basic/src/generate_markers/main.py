import cv2
from cv2 import aruco

marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

marker_size = 400 # Pixels

for id in range(20):
    marker_image = aruco.drawMarker(marker_dict, id, marker_size)
    cv2.imwrite(f"./markers/marker_{id}.png", marker_image)
# arucoParams = cv2.aruco.DetectorParameters_create()
# (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict,
#                                                    parameters=arucoParams)

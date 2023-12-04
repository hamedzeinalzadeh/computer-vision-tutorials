import cv2
import numpy as np
from src.images import IMAGE_DIR
from src.utils.object_detector import HomogenBgDectector

img = cv2.imread(str(IMAGE_DIR / 'test2.png'))

# Load aruco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
corners, _, _ = cv2.aruco.detectMarkers(img, dictionary=aruco)
detector = HomogenBgDectector()
internal_contours = detector.detect_internal_contours(img)

# Detect and Draw objects' boundaries
for cnt in internal_contours:
    # Draw polygon
    #cv2.polylines(img, [cnt], isClosed=True, color=(255, 0, 0), thickness=2)

    # Get min-area rectangle around the objects
    rect = cv2.minAreaRect(cnt)
    (x_center, y_center), (w, h), angle = rect

    # Draw the object center
    cv2.circle(img, center=(int(x_center), int(y_center)),
               radius=5, color=(0, 0, 255), thickness=-1)

    # retrieve the 4 corner points of the rectangle
    box = cv2.boxPoints(rect)
    box = box.astype(np.int)

    # Draw the rectangle
    cv2.polylines(img, [box], isClosed=True, color=(255, 0, 0), thickness=2)

    # Put text to mention the size of the object
    cv2.putText(img, text=f"Width: {w:.2f}", org=(int(x_center), int(y_center) - 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.5, color=(50,200,0), thickness=2)
    cv2.putText(img, text=f"Height: {h:.2f}", org=(int(x_center), int(
        y_center) + 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.5, color=(50, 200, 0), thickness=2)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
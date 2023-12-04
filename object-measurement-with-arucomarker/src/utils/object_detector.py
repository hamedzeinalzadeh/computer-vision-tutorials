import cv2
from loguru import logger

from src.images import IMAGE_DIR


class HomogenBgDectector:
    def __init__(self):
        pass

    def detect_internal_contours(self, image) -> tuple:
        """Detect the objects in the given image and returns the contours coordinates

        Args:
            image (np.ndarray)

        Returns:
            tuple: returns <contours> coordinates
        """

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # apply binary thresholding
        ret, thresh = cv2.threshold(
            img_gray, 150, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(
            image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)


        # Detect Internal Contours
        # last column in the array is -1 if an external contour (no contours inside of it)
        # Extract the external contours from the list of contours
        internal_contours = tuple(contours[i] for i in range(len(contours)) if (hierarchy[0][i][3] != -1))

        return internal_contours

if __name__ == '__main__':
    detector = HomogenBgDectector()
    image = cv2.imread(str(IMAGE_DIR / 'test0.jpg'))
    internal_contours = detector.detect_internal_contours(image)
    print(internal_contours[0])
    logger.info('module has no error!')

from pathlib import Path

import cv2
from loguru import logger


def initialize_camera(camera_id, width=640, height=480):
    """
    Initialize a camera with given ID and resolution.

    :param camera_id: The system path of the camera device.
    :param width: The desired width of the camera frame.
    :param height: The desired height of the camera frame.
    :return: The initialized camera object.
    """
    camera = cv2.VideoCapture()
    camera.open(camera_id)

    # Set camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return camera


def save_synched_frames(cam1, cam2, dir_path, counter):
    """
    Capture and save synchronized frames from two cameras.

    :param cam1: The first camera object.
    :param cam2: The second camera object.
    :param dir_path: Directory path to save the images.
    :param counter: A counter to format the image file names.
    """
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    # Save frames if both cameras successfully captured images
    if ret1 and ret2:
        filename1 = dir_path / f'img_cam0_{counter:02d}.png'
        filename2 = dir_path / f'img_cam1_{counter:02d}.png'

        cv2.imwrite(str(filename1), frame1)
        cv2.imwrite(str(filename2), frame2)

        logger.info(f'{filename1} and {filename2} saved.')


def save_stereo_synched_frames():
    """
    Open both cameras and save synchronized frames when 's' is pressed.
    The images are saved in a './synched' directory.
    """
    # Camera device paths
    camera_0_id = '/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6:1.0-video-index0'
    camera_1_id = '/dev/v4l/by-path/pci-0000:00:14.0-usb-0:8:1.0-video-index0'

    # Initialize cameras
    cam1 = initialize_camera(camera_0_id)
    cam2 = initialize_camera(camera_1_id)

    # Counter for naming images
    counter = 0

    # Check and create the directory for synchronized images
    synched_images_dir = Path('synched')
    synched_images_dir.mkdir(exist_ok=True)

    while cam1.isOpened() and cam2.isOpened():
        # Display camera feeds
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        cv2.imshow('camera 0', frame1)
        cv2.imshow('camera 1', frame2)

        # Handle keypress events
        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC key to exit
            break
        elif key == ord('s'):  # 's' key to save frames
            save_synched_frames(cam1, cam2, synched_images_dir, counter)
            counter += 1

    # Release resources
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    save_stereo_synched_frames()

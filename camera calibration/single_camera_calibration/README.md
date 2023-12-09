
# Single Camera Calibration

## Overview
This script is designed for camera calibration using a single camera setup. The process involves taking photos with a chessboard pattern, which the script then uses to compute intrinsic camera parameters. This is particularly useful for applications in computer vision and 3D reconstruction where precise camera calibration is necessary.

## Getting Started

### Prerequisites
- Python 3.x
- OpenCV library installed
- A chessboard for taking calibration photos

### Installation
1. Clone the repository or download the script.
2. Ensure Python 3.x is installed on your system.
3. Install OpenCV using pip:
   ```
   pip install opencv-python
   ```

### Usage
1. Place the chessboard in various orientations and take multiple photographs.
2. Save these photographs in a specified directory.
3. Run the script and point it to the directory containing the chessboard images.

   Example command:
   ```
   python single_camera_calibration.py --images_path /path/to/images
   ```

The script will process the images and return the camera matrices as output, which includes the camera intrinsic matrix and distortion coefficients.

## Output
The output of this script includes:
- Camera Intrinsic Matrix: Contains focal length and optical center.
- Distortion Coefficients: Essential for correcting camera lens distortion.

## Contributing
Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)

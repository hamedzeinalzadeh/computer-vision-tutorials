import cv2

# Replace 'your_ip_camera_url' with the URL of your IP webcam stream
ip_camera_url = 'http://192.168.1.96/video'

# Open the IP webcam stream
cap = cv2.VideoCapture(ip_camera_url)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Create a window to display the webcam feed
cv2.namedWindow('IP Webcam', cv2.WINDOW_NORMAL)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
file:///home/hamed/Downloads/keras_yolo3_position_estimatioin%20(1)/keras_yolo3_position_estimatioin/yolo.py

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the frame
    cv2.imshow('IP Webcfile:///home/hamed/Downloads/keras_yolo3_position_estimatioin%20(1)/keras_yolo3_position_estimatioin/yolo.py
am', frame)

    # Press 'q' to exit the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

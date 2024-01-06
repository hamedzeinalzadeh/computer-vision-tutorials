'''
Script #1: 
Connecting a Function for Drawing on Image
'''

import cv2
import numpy as np

# Function based on a CV2 Event (Left button click)
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),100,(0,255,0),-1)

# Create blank image
img = np.zeros((500,500,3))

# Naming the window to reference it 
cv2.namedWindow(winname='my_drawing')

# Connect the mouse button to callback function
cv2.setMouseCallback('my_drawing',draw_circle)

# Run forever until pressing Esc
while True:
    
    # Shows the image window
    cv2.imshow('my_drawing',img)
    
    if (cv2.waitKey(20) & 0xFF) == 27:
        break

# It closes all windows (just in case you have multiple windows called)
cv2.destroyAllWindows()


'''
Script #2:     
Adding different event choices
'''

# import cv2
# import numpy as np

# # Function based on a CV2 Events
# def draw_circle(event,x,y,flags,param):
    
#     if event == cv2.EVENT_LBUTTONDOWN:
#         cv2.circle(img,(x,y),100,(0,255,0),-1)
#     elif event == cv2.EVENT_RBUTTONDOWN:
#         cv2.circle(img,(x,y),100,(0,0,255),-1)
        

# # Create a blank image
# img = np.zeros((500,500,3))

# # Naming the window to reference it 
# cv2.namedWindow(winname='my_drawing')

# # Connect the mouse button to callback function
# cv2.setMouseCallback('my_drawing',draw_circle)

# while True: #Runs forever until we break with Esc key on keyboard
#     # Shows the image window
#     cv2.imshow('my_drawing',img)
   
#     if (cv2.waitKey(20) & 0xFF) == 27:
#         break

# # It closes all windows (just in case you have multiple windows called)
# cv2.destroyAllWindows()


'''
Script #3:     
Drag and Drop
'''

# import cv2
# import numpy as np


# # Function based on a CV2 Events
# drawing = False # True if mouse is pressed down
# ix,iy = -1,-1

# # mouse callback function
# def draw_rectangle(event,x,y,flags,param):
#     global ix,iy,drawing,mode

#     if event == cv2.EVENT_LBUTTONDOWN:
#         # True if click DOWN with left mouse button
#         drawing = True
#         # Take note of where that mouse was located
#         ix,iy = x,y

#     elif event == cv2.EVENT_MOUSEMOVE:
#         # Now the mouse is moving
#         if drawing == True:
#             # If drawing is True, it means you've already clicked on the left mouse button
#             # We draw a rectangle from the previous position to the x,y where the mouse is
#             cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
           

#     elif event == cv2.EVENT_LBUTTONUP:
#         # Once you lift the mouse button, drawing is False
#         drawing = False
#         # we complete the rectangle.
#         cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        
        

# # Create a blank image
# img = np.zeros((500,500,3))

# # Naming the window to reference it 
# cv2.namedWindow(winname='my_drawing')

# # Connect the mouse button to callback function
# cv2.setMouseCallback('my_drawing',draw_rectangle)

# while True: #Runs forever until we break with Esc key on keyboard
#     # Shows the image window
#     cv2.imshow('my_drawing',img)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# # It closes all windows (just in case you have multiple windows called)
# cv2.destroyAllWindows()


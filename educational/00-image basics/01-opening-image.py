import cv2

img = cv2.imread('DATA/sun_flowers.jpeg')

while True:

    cv2.imshow('Sun Flower',img)

    # EXPLANATION FOR THIS LINE OF CODE:
    # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1/39201163
    
    # If waiting at least 1 ms AND pressing the Esc
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

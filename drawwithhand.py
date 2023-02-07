import cv2
import numpy as np
import math
import HandTrackingModule

# initialize the video capture object
cap = cv2.VideoCapture(0)

# initialize the color variables
#Check BGR colors
red = (0, 0, 255)
torqua = (0, 20, 0)
green = (0, 255, 0)
yellow = (0, 255, 50)
blue = (255, 0, 0)
white = (255, 255, 255)

# initialize the hand detection object
detector = HandTrackingModule.handDetector()

# create a black window with dimensions 720 x 1280 x 4
win = np.zeros([720, 1280, 4])

# initialize an empty list for hand positions
positions = []

# Start the loop for capturing video
while True:
    # read the video frame
    success, img = cap.read()

    # find the hands in the frame using the hand detection object
    hand = detector.findHands(img, draw=False)

    # get the finger  locations from the frame using the module
    fingerlocation = detector.findPosition(img)

    # check if there are any hands detected
    if len(fingerlocation) != 0:
        # get the finger coordinates
        finger = fingerlocation[8] #index finger location
        midFinger = fingerlocation[12] #middle finger location


        #this section  to iniate color based on the
        # check if finger[1] is even and if so, draw a red circle on the win window
        #finger[1] indecatce the position X of the finger , from the area [finger Index , Position x , postition Y]
        if (finger[1] % 2) == 0:
            cv2.circle(win, (finger[1], finger[2]), 20, red, cv2.FILLED)

        # check if finger[1] is divisible by 3 and if so, draw a blue circle on the win window
        elif (finger[1] % 3) == 0:
            cv2.circle(win, (finger[1], finger[2]), 20, blue, cv2.FILLED)

        # check if finger[2] is even and if so, draw a yellow circle on the win window
        elif (finger[2] % 2) == 0:
            cv2.circle(win, (finger[1], finger[2]), 20, yellow, cv2.FILLED)

        # check if sin(finger[1]) is positive and if so, draw a green circle on the win window
        elif math.sin(finger[1]) > 0:
            cv2.circle(win, (finger[1], finger[2]), 20, green, cv2.FILLED)

        # otherwise, draw a torqua circle on the win window
        else:
            cv2.circle(win, (finger[1], finger[2]), 20, torqua, cv2.FILLED)

    # flip the win window horizontally
    flipHorizontal = cv2.flip(win, 1)

    # display the win window
    cv2.imshow("win", flipHorizontal)

    # wait for 3 milliseconds
    cv2.waitKey(3)

    # check if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close

import cv2
import mediapipe as mp
import time

# Class for hand detection
class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplex=1, detectCon=0.5, trackCon=0.5):
        # Initialize the parameters
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplex
        self.detectCon = detectCon
        self.trackCon = trackCon

        # Load the media pipe hands solution
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectCon, self.trackCon)

        # Load the media pipe drawing solution
        self.mpDraw = mp.solutions.drawing_utils

    # Function to detect hands in an image
    def findHands(self, img, draw=True):
        # Convert the image to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run the hand detection process
        self.results = self.hands.process(imgRGB)

        # Draw the landmarks if specified
        if self.results.multi_hand_landmarks:
            for handlm in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlm, self.mpHands.HAND_CONNECTIONS)

        # Return the processed image
        return img

    # Function to find the hand position
    def findPosition(self, img, hanNo=0, draw=True):
        marks = []
        # Check if there are any hand landmarks detected
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks
            for id, lm in enumerate(myhand[hanNo].landmark):
                h, w, c = img.shape
                # Convert the landmark coordinates to image dimensions
                cx, cy = int(lm.x * w), int(lm.y * h)
                marks.append([id, cx, cy])
                if draw:
                    # Draw a circle on the image at the landmark position
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
        # Return the hand landmarks
        return marks

# Main function for the hand detection program
def main():
    pTime = 0
    cTime = 0

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Initialize the hand detector object
    detector = handDetector()

    while True:
        # Read the webcam frame
        success, img = cap.read()

        # Detect hands in the image
        img = detector.findHands(img, draw=True)

        # Find the hand position
        lmlist = detector.findPosition(img, draw=True)
        print(lmlist)

        # Calculate the frame rate
        cTime = time.time()
        fbs = 1 / (cTime - pTime)

        pTime = cTime
        cv2.putText(img, str(int(fbs)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("video", img)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()

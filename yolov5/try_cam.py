import numpy as np
import cv2



cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret,frame = cap.read()
    cv2.rectangle(frame, (100, 100), (200, 200), [255, 0, 0], 2)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows() 

import numpy as np
import cv2

cap = cv2.VideoCapture('video/1.mp4')
out_path = 'video/images/'

frame_id = 0
while(cap.isOpened()):
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    cv2.imwrite(out_path+str(frame_id)+'.jpg', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id = frame_id + 1

cap.release()
cv2.destroyAllWindows()
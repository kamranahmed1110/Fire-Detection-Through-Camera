import cv2
import numpy as np 
from playsound import playsound

cap = cv2.VideoCapture(0)

""" it help to identify the fire in the video stream"""

lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255 ,255]) 

# Alert sound  

Alert_sound = "alarm.mp3"

# Cool down frames for setting the alert sound 
alert_flag = False
cooldown_frames = 100
current_cooldown = cooldown_frames
min_contour_area = 1000

# main loop 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # converted each frame into HSV 
    # create a mask to isolate the red region  
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Apply morphological operation to reduce noise
    kernel = np.ones((5 , 5), np.uint8)
    mask = cv2.dilate(mask, kernel , iterations=1)
    mask = cv2.erode(mask, kernel , iterations=1)

    contours, _ = cv2.findContours(mask , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue
    
        # filter contours based on other properties if needed

        if not alert_flag and current_cooldown == cooldown_frames:
            playsound(Alert_sound)
            print("Fire Detected")
            alert_flag = True
    
    if alert_flag:
        current_cooldown -= 1
        if current_cooldown == 0:
            alert_flag = False
            current_cooldown = cooldown_frames

    cv2.drawContours(frame, contours , -1 , (0 , 255 , 0), 2)
    cv2.imshow("Fire Detection" , frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Correct indentation for the break statement

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
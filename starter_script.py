import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import math
import requests
import json
import time

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

URL = 'http://127.0.0.1:5000/api/starter'

# Initialize the video feed from local webcam
cap = cv.VideoCapture(0)  # 0 is usually the default webcam

if not cap.isOpened():
    raise ValueError("Could not open webcam. Please check if it's connected properly.")

# You might want to set specific dimensions for consistency
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

x_offset = 25
y_offset = 26

def getCorners(corners):
    point_dict = {}
    for marker in corners:
        id = marker[0][0]
        if id == 0:
            point_dict[id] = (marker[1][0][0][0] - x_offset, marker[1][0][0][1] - y_offset)
        elif id == 1:
            point_dict[id] = (marker[1][0][1][0] + x_offset,marker[1][0][1][1] - y_offset)
        elif id == 2:
            point_dict[id] = (marker[1][0][2][0] + x_offset,marker[1][0][2][1] + y_offset)
        elif id == 3:
            point_dict[id] = (marker[1][0][3][0] - x_offset,marker[1][0][3][1] + y_offset)

    return point_dict

def correctPerspective(frame):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(image=gray_frame, dictionary=aruco_dict, parameters=parameters)
    markers_found = False
    if ids is not None and len(ids) == 4:
        combined = tuple(zip(ids,corners))
        point_dict = getCorners(combined)
        points_src = np.array([point_dict[0], point_dict[3], point_dict[1], point_dict[2]])
        points_dst = np.float32([[0, 0], [0, 500], [500, 0], [500, 500]])

        matrix, _ = cv.findHomography(points_src, points_dst)
        image_out = cv.warpPerspective(frame, matrix, (500, 500))
        frame = image_out
        markers_found = True

    return frame, markers_found

def sendData(image):
    _, img_encoded = cv.imencode('.jpg', image)
    files = {'image': img_encoded.tobytes()}
    response = requests.post(URL, files=files)
    return response

# Display instructions to user
print("Place the target with ArUco markers in view of the webcam")
print("Press 'q' to exit if needed")

# Optional: Show webcam feed for user to position the target
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam")
        break
        
    # Show the current view for alignment
    cv.imshow("Align Target", frame)
    
    # Optional: Attempt to detect markers and draw them
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(image=gray_frame, dictionary=aruco_dict, parameters=parameters)
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        cv.imshow("Markers Detected", frame)
    
    key = cv.waitKey(1)
    if key == ord('q'):
        break
        
    # Once all 4 markers are detected, process and send the image
    if ids is not None and len(ids) == 4:
        corrected_image, target_detected = correctPerspective(frame)
        if target_detected:
            try:
                response = sendData(corrected_image)
                print(f"Target initialized. Server response: {response.status_code}")
                break  # Exit after successfully sending
            except Exception as e:
                print(f"Error sending data: {e}")

# Clean up
cap.release()
cv.destroyAllWindows()
print("Starter script exited")
import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import math
import requests
import json
import time

# Initialize ArUco marker detection
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Global variables
prev_frame = None
fps_limit = 5  # Frames per second limit
start_time = time.time()
score = {10: [], 9: [], 8: [], 7: [], 6: [], 5: [], 4: [], 3: [], 2: [], 1: []}
angles = {10: [], 9: [], 8: [], 7: [], 6: [], 5: [], 4: [], 3: [], 2: [], 1: []}
score_sum = 0
URL = 'http://127.0.0.1:5000/api/score'

# Ring parameters
k_size = 11
params = {'alpha': 10.0, 'beta': -9.0, 'gamma': 0}
cny_lower = 50
cny_upper = 100
center_x = 254
center_y = 247  # Adjusted from the original
largest_radius = 0
ring_delta = 22
rings_radius = []

# Initialize webcam
cap = cv.VideoCapture(0)  # Use default webcam (usually 0)

# Set webcam properties for consistent results
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Check if webcam is opened successfully
if not cap.isOpened():
    raise ValueError("Could not open webcam. Please check if it's connected properly.")

x_offset = 25
y_offset = 26

def getCorners(corners):
    point_dict = {}
    for marker in corners:
        id = marker[0][0]
        if id == 0:
            point_dict[id] = (marker[1][0][0][0] - x_offset, marker[1][0][0][1] - y_offset)
        elif id == 1:
            point_dict[id] = (marker[1][0][1][0] + x_offset, marker[1][0][1][1] - y_offset)
        elif id == 2:
            point_dict[id] = (marker[1][0][2][0] + x_offset, marker[1][0][2][1] + y_offset)
        elif id == 3:
            point_dict[id] = (marker[1][0][3][0] - x_offset, marker[1][0][3][1] + y_offset)

    return point_dict

def correctPerspective(frame):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(image=gray_frame, dictionary=aruco_dict, parameters=parameters)
    markers_found = False
    if ids is not None and len(ids) == 4:
        combined = tuple(zip(ids, corners))
        point_dict = getCorners(combined)
        points_src = np.array([point_dict[0], point_dict[3], point_dict[1], point_dict[2]])
        points_dst = np.float32([[0, 0], [0, 500], [500, 0], [500, 500]])

        matrix, _ = cv.findHomography(points_src, points_dst)
        image_out = cv.warpPerspective(frame, matrix, (500, 500))
        frame = image_out
        markers_found = True
    else:
        print("No valid ArUco markers found. Make sure all 4 markers are visible.")

    return frame, markers_found

def getBullets(th1, output_frame, draw=True):
    mask = th1
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.dilate(mask, kernel, iterations=2)
    contours = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    bullets = []
    
    for contour in contours:
        area = cv.contourArea(contour)
        # Filter by area
        if 5 <= area <= 400:  # Adjusted for typical bullet hole sizes
            approx = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True), True)
            ((x, y), radius) = cv.minEnclosingCircle(contour)
            
            # Additional filter - ensure it's somewhat circular
            if len(approx) >= 5:  # More points suggests a more circular shape
                bullets.append((int(x), int(y)))
                if draw:
                    cv.circle(output_frame, (int(x), int(y)), (int(radius)), (0, 0, 255), 2)

    return bullets

def calculateDistance(x1, y1):
    x2 = center_x
    y2 = center_y
    radius = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    return radius

def calculateAngle(x, y):
    delta_x = (x - center_x)
    delta_y = (y - center_y)
    if delta_x == 0:
        angle = -90
    else:
        angle = round(math.atan2(delta_y, delta_x) * 180 / math.pi)
    return angle

def updateScore(bullets):
    global score_sum, score, angles
    
    for x, y in bullets:
        dist = calculateDistance(x, y)
        angle = calculateAngle(x, y)
        
        # Score based on distance from center
        if 0 <= dist <= 23:
            score[10].append((x, y))
            score_sum += 10
            angles[10].append(angle)
        elif 23 < dist <= 53:
            score[9].append((x, y))
            score_sum += 9
            angles[9].append(angle)
        elif 53 < dist <= 79:
            score[8].append((x, y))
            score_sum += 8
            angles[8].append(angle)
        elif 79 < dist <= 105:
            score[7].append((x, y))
            score_sum += 7
            angles[7].append(angle)
        elif 105 < dist <= 135:
            score[6].append((x, y))
            score_sum += 6
            angles[6].append(angle)
        elif 135 < dist <= 162:
            score[5].append((x, y))
            score_sum += 5
            angles[5].append(angle)
        elif 162 < dist <= 188:
            score[4].append((x, y))
            score_sum += 4
            angles[4].append(angle)
        elif 188 < dist <= 215:
            score[3].append((x, y))
            score_sum += 3
            angles[3].append(angle)
        elif 215 < dist <= 242:
            score[2].append((x, y))
            score_sum += 2
            angles[2].append(angle)
        elif 242 < dist <= 268:
            score[1].append((x, y))
            score_sum += 1
            angles[1].append(angle)

def drawFrame(frame):
    i = 1
    for points in score.keys():
        frame = cv.putText(frame, f"{points}: {len(score[points])}", (10, 30 * i), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        i += 1
    
    # Add total score
    frame = cv.putText(frame, f"Total: {score_sum}", (10, 30 * i), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

def sendData(image, angles):
    _, img_encoded = cv.imencode('.jpg', image)
    files = {'image': img_encoded.tobytes(),
             'angles': (None, json.dumps(angles), 'application/json')}
    try:
        response = requests.post(URL, files=files)
        return response
    except Exception as e:
        print(f"Error sending data: {e}")
        return None

def drawRings(canvas):
    cv.circle(canvas, (center_x, center_y), 3, (0, 0, 255), -1)  # Center point
    cv.circle(canvas, (center_x, center_y), 23, (255, 0, 255), 2)  # 10
    cv.circle(canvas, (center_x, center_y), 53, (255, 0, 255), 2)  # 9
    cv.circle(canvas, (center_x, center_y), 79, (255, 0, 255), 2)  # 8
    cv.circle(canvas, (center_x, center_y), 105, (255, 0, 255), 2) # 7
    cv.circle(canvas, (center_x, center_y), 135, (255, 0, 255), 2) # 6
    cv.circle(canvas, (center_x, center_y), 162, (255, 0, 255), 2) # 5
    cv.circle(canvas, (center_x, center_y), 188, (255, 0, 255), 2) # 4
    cv.circle(canvas, (center_x, center_y), 215, (255, 0, 255), 2) # 3
    cv.circle(canvas, (center_x, center_y), 242, (255, 0, 255), 2) # 2
    cv.circle(canvas, (center_x, center_y), 268, (255, 0, 255), 2) # 1
    return canvas

# Main execution starts here
print("Starting scoring script with webcam")
print("Press 'q' to exit the application")

# First, let's get a frame to initialize
ret, frame = cap.read()
if not ret:
    print("Failed to capture initial frame from webcam")
    cap.release()
    cv.destroyAllWindows()
    exit(1)

# Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error with webcam")
        break

    curr_time = time.time()
    if (curr_time - start_time) > (1.0 / fps_limit):  # Limit processing rate
        # Process the frame
        corrected_image, target_detected = correctPerspective(frame)
        
        if target_detected:
            output_frame = corrected_image.copy()
            output_frame = drawRings(output_frame)  # Draw target rings
            
            # Look for changes between frames to detect new bullet holes
            if prev_frame is not None:
                # Compute difference between current and previous frame
                diff = cv.absdiff(prev_frame, corrected_image)
                gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
                blur = cv.medianBlur(gray, 5)
                _, th1 = cv.threshold(blur, 30, 255, cv.THRESH_BINARY)
                
                # Detect bullets from the difference image
                bullets = getBullets(th1, output_frame)
                
                # If new bullet holes are detected, update score
                if bullets:
                    updateScore(bullets)
                    print(f"New bullet points detected: {bullets}")
                
                # Display intermediate processing results
                cv.imshow("Difference", diff)
                cv.imshow("Threshold", th1)
            
            # Update previous frame for next iteration
            prev_frame = corrected_image.copy()
            
            # Draw scores on the output frame
            output_frame = drawFrame(output_frame)
            
            # Send data to the server
            sendData(output_frame, angles)
            
            # Show the processed frame
            cv.imshow("Target Analysis", output_frame)
        else:
            # If no target detected, show the raw frame with instructions
            cv.putText(frame, "No target detected - Position ArUco markers in view", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.imshow("Webcam", frame)
        
        # Reset timer for frame rate limiting
        start_time = time.time()

    # Check for exit command
    if cv.waitKey(1) & 0xFF == ord('q'):
        print("Exit signal received, closing the application.")
        break

# Final data send before exit
if target_detected:
    sendData(output_frame, angles)

# Clean up
cap.release()
cv.destroyAllWindows()
print("Scoring script exited")
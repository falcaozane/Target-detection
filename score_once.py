import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import math
import requests
import json

# Initialize ArUco marker detection
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Initialize score dictionaries
score = {10: [], 9: [], 8: [], 7: [], 6: [], 5: [], 4: [], 3: [], 2: [], 1: []}
angles = {10: [], 9: [], 8: [], 7: [], 6: [], 5: [], 4: [], 3: [], 2: [], 1: []}
score_sum = 0
URL = 'http://127.0.0.1:5000/api/score'

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

def calculateDistance(x1, y1, x2=255, y2=244):
    radius = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    return radius

def detectWhiteRingBullets(frame, canvas):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(frame, 5)
    ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    mask = th1
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.dilate(mask, kernel, iterations=2)
    contours = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]

    min_area = 0
    max_area = 400

    bullets = []

    for contour in contours:
        area = cv.contourArea(contour)
        if min_area <= area <= max_area:
            approx = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True), True)
            ((x, y), radius) = cv.minEnclosingCircle(contour)
            if (268 > calculateDistance(int(x), int(y)) > 105):
                bullets.append((int(x), int(y)))
                print(area)
                cv.circle(canvas, (int(x), int(y)), (int(radius)), (0, 0, 255), -1)

    return canvas, bullets

def detectBlackRingBullets(frame, canvas):
    # Convert to HSV color space
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Define the range for copper/reddish color of the bullet holes
    # Lower bound - more orange/copper tone
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([20, 255, 255])

    # Upper bound - more reddish tone
    lower_red2 = np.array([150, 100, 100])
    upper_red2 = np.array([200, 255, 255])

    # Create masks for both ranges
    mask1 = cv.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv_frame, lower_red2, upper_red2)

    # Combine the masks
    red_mask = cv.bitwise_or(mask1, mask2)

    # Add some noise reduction
    kernel = np.ones((1, 1), np.uint8)
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel)
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_CLOSE, kernel)

    # Perform a bitwise AND to highlight the detected areas
    red_output = cv.bitwise_and(frame, frame, mask=red_mask)

    # Convert the output to grayscale
    gray_red = cv.cvtColor(red_output, cv.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv.threshold(gray_red, 30, 255, cv.THRESH_BINARY)

    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Adjust these values based on the actual size of bullet holes in your image
    min_area = 2 # Reduced minimum area
    max_area = 400  # Increased maximum area

    bullets = []

    # Debug - draw all detected contours in red
    cv.drawContours(canvas, contours, -1, (0, 0, 255), 2)

    for contour in contours:
        area = cv.contourArea(contour)
        if min_area <= area <= max_area:
            ((x, y), radius) = cv.minEnclosingCircle(contour)

            # Only consider points in the black region
            if calculateDistance(int(x), int(y), 255, 244) <= 105:  # Adjust this value based on your target size
                bullets.append((int(x), int(y)))
                # Draw detected bullet holes on the canvas in blue
                cv.circle(canvas, (int(x), int(y)), int(radius), (255, 0, 0), -1)

    # Add debug displays
    cv.imshow('Mask', red_mask)
    cv.imshow('Threshold', thresh)
    cv.imshow('Detection Result', canvas)

    return canvas, bullets

def drawRings(canvas, center_x=254, center_y=247):
    cv.circle(canvas, (center_x, center_y), (23), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (53), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (79), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (105), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (135), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (162), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (188), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (215), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (242), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (268), (255, 0, 255), 2)

    return canvas

def sendData(image, angles):
    _, img_encoded = cv.imencode('.jpg', image)
    files = {'image': img_encoded.tobytes(), 'angles': (None, json.dumps(angles), 'application/json')}
    return requests.post(URL, files=files)

# Main processing loop
print("Reading from webcam...")
ret, frame = cap.read()

if not ret:
    print("Failed to capture frame from webcam")
    cap.release()
    cv.destroyAllWindows()
    exit(1)

# Process the frame
corrected_image, target_detected = correctPerspective(frame)
output_frame = corrected_image.copy()

if target_detected:
    # Process the corrected image
    print("Target detected, processing...")
    output_frame, white_ring_bullets = detectWhiteRingBullets(corrected_image, output_frame)
    output_frame, black_ring_bullets = detectBlackRingBullets(corrected_image, output_frame)
    
    print("Detected bullet points:", angles)
    
    try:
        response = sendData(output_frame, angles)
        print(f"Server response: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error sending data: {e}")

    # Display the output
    cv.imshow('Processed Frame', output_frame)
    cv.waitKey(0)  # Wait for a key press before closing

else:
    print("No target detected. Make sure the ArUco markers are visible.")
    
# Clean up
cap.release()
cv.destroyAllWindows()
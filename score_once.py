import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import math
import requests
import json



# calibration_data = np.load('calibration_params.npz')
# mtx = calibration_data['mtx']
# dist = calibration_data['dist']


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
score = {10: [], 9: [], 8: [], 7: [], 6: [], 5: [], 4: [], 3: [], 2: [], 1: []}
angles = {10: [], 9: [], 8: [], 7: [], 6: [], 5: [], 4: [], 3: [], 2: [], 1: []}
score_sum = 0
URL = 'http://127.0.0.1:5000/api/score'



# Initialize the video feed

def selected_ip():
    try:
        response = requests.get('http://127.0.0.1:5000/api/selected_ip')  # Flask endpoint
        if response.status_code == 200:
            return response.json().get('selected_ip')
        else:
            print("Error fetching selected IP")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

selected_ip =  "192.168.1.19"  # Fetch selected IP from the Flask app

if selected_ip is None:
    raise ValueError("No selected IP provided. Please ensure you have selected the device in the app.")

# Use the selected IP in the video stream
cap = cv.VideoCapture(f"http://{selected_ip}:8000/video_feed")  # Use dynamic IP address


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
            if calculateDistance(int(x), int(y)) <= 105:  # Adjust this value based on your target size
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
    # cv.circle(canvas, (center_x, center_y), (250), (255, 0, 255), 2)

    return canvas


# def calculateDistance(x1, y1, x2=255, y2=244):
#     radius = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
#     return radius


# def displayScore(score, canvas):
#     cv.putText(canvas, f'Score: {score}', (200, 40), cv.FONT_HERSHEY_PLAIN, 3, (80, 127, 255), 3)
#     return canvas


def sendData(image, angles):
    _, img_encoded = cv.imencode('.jpg', image)
    files = {'image': img_encoded.tobytes(), 'angles': (None, json.dumps(angles), 'application/json')}
    return requests.post(URL, files=files)


# def calculateAngle(x, y):
#     delta_x = (x - 255)
#     delta_y = (y - 244)
#     if delta_x == 0:
#         return -90
#     else:
#         # slope_hole = delta_y/delta_x
#         # angle = round(math.atan(abs(slope_hole-slope_base/(1+ slope_hole*slope_base))) * 180 / math.pi)
#         angle = round(math.atan2(delta_y, delta_x) * 180 / math.pi)
#         return angle


# def updateScore(bullets):
#     global score_sum
#     for x, y in bullets:
#         dist = calculateDistance(x, y)
#         angle = calculateAngle(x, y)
#         if 10 <= dist <= 23:
#             score[10].append((x, y))
#             score_sum += 10
#             angles[10].append(angle)
#         elif 23 < dist <= 53:
#             score[9].append((x, y))
#             score_sum += 9
#             angles[9].append(angle)
#         elif 53 < dist <= 79:
#             score[8].append((x, y))
#             score_sum += 8
#             angles[8].append(angle)
#         elif 79 < dist <= 105:
#             score[7].append((x, y))
#             score_sum += 7
#             angles[7].append(angle)
#         elif 105 < dist <= 135:
#             score[6].append((x, y))
#             score_sum += 6
#             angles[6].append(angle)
#         elif 135 < dist <= 162:
#             score[5].append((x, y))
#             score_sum += 5
#             angles[5].append(angle)
#         elif 162 < dist <= 188:
#             score[4].append((x, y))
#             score_sum += 4
#             angles[4].append(angle)
#         elif 188 < dist <= 215:
#             score[3].append((x, y))
#             score_sum += 3
#             angles[3].append(angle)
#         elif 215 < dist <= 242:
#             score[2].append((x, y))
#             score_sum += 2
#             angles[2].append(angle)
#         elif 242 < dist <= 268:
#             score[1].append((x, y))
#             score_sum += 1
#             angles[1].append(angle)


ret, frame = cap.read()

# frame = cv.undistort (frame, mtx, dist, None)
corrected_image, target_detected = correctPerspective(frame)
output_frame = corrected_image.copy()

if target_detected:
    # output_frame = drawRings(output_frame)
    output_frame, white_ring_bullets = detectWhiteRingBullets(corrected_image, output_frame)
    output_frame, black_ring_bulllets = detectBlackRingBullets(corrected_image, output_frame)
    # updateScore(white_ring_bullets)
    # updateScore(black_ring_bulllets)
    print(angles)
    new_score = 0
    score_dict = {"bullets": [], "total_score": 0}

    try:
        # response = sendData(output_frame, score_dict)
        response = sendData(output_frame, angles)
        print(f"Server response: {response.status_code}, {response.text}")
        pass
    except Exception as e:
        print("The error is: ", e)

    # output_frame = displayScore(score_sum, output_frame)

    cv.imshow('frame', output_frame)
#
    cv.waitKey(1)

cap.release()
cv.destroyAllWindows()
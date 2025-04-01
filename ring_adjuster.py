import cv2 as cv
import numpy as np
import cv2.aruco as aruco

cap = cv.VideoCapture('http://192.168.1.16:8000/video_feed')
cv.namedWindow("Frame")

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()



center_x = 254
center_y = 244

ring_11 = 0
ring_10 = 23
ring_9 = 53
ring_8 = 79
ring_7 = 105
ring_6 = 135
ring_5 = 162
ring_4 = 188
ring_3 = 215
ring_2 = 242
ring_1 = 268

# List to store points
corner_points = []

x_offset = 10
y_offset = 10

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
        combined = tuple(zip(ids, corners))
        point_dict = getCorners(combined)
        points_src = np.array([point_dict[0], point_dict[3], point_dict[1], point_dict[2]])
        points_dst = np.float32([[0, 0], [0, 500], [500, 0], [500, 500]])

        matrix, _ = cv.findHomography(points_src, points_dst)
        image_out = cv.warpPerspective(frame, matrix, (500, 500))
        frame = image_out
        markers_found = True

        # Append the points to corner_points list
        corner_points.append(points_src)
    return frame, markers_found

def drawRings(canvas):
    cv.circle(canvas, (center_x, center_y), (1), (0, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_11), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_10), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_9), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_8), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_7), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_6), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_5), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_4), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_3), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_2), (255, 0, 255), 2)
    cv.circle(canvas, (center_x, center_y), (ring_1), (255, 0, 255), 2)
    return canvas

def nothing(_):
    pass

# Trackbars for adjusting the center and rings dynamically
cv.createTrackbar('Center X', 'Frame', 254, 400, nothing)
cv.createTrackbar('Center Y', 'Frame', 244, 400, nothing)
cv.createTrackbar('Ring 11', 'Frame', 0, 400, nothing)  # New trackbar for ring 11
cv.createTrackbar('Ring 10', 'Frame', 23, 400, nothing)
cv.createTrackbar('Ring 9', 'Frame', 53, 300, nothing)
cv.createTrackbar('Ring 8', 'Frame', 79, 300, nothing)
cv.createTrackbar('Ring 7', 'Frame', 105, 300, nothing)
cv.createTrackbar('Ring 6', 'Frame', 135, 300, nothing)
cv.createTrackbar('Ring 5', 'Frame', 162, 300, nothing)
cv.createTrackbar('Ring 4', 'Frame', 188, 300, nothing)
cv.createTrackbar('Ring 3', 'Frame', 215, 300, nothing)
cv.createTrackbar('Ring 2', 'Frame', 242, 300, nothing)
cv.createTrackbar('Ring 1', 'Frame', 268, 300, nothing)



while True:
    ret, frame = cap.read()
    if not ret:
        print("Error with camera")
        break

    # Undistort the captured frame if needed
    # frame = cv.undistort(frame, mtx, dist, None)

    # Process frame to detect markers and correct perspective
    processed_frame, target_detected = correctPerspective(frame)

    if target_detected:
        # Get the trackbar positions to dynamically adjust the rings
        center_x = cv.getTrackbarPos('Center X', 'Frame')
        center_y = cv.getTrackbarPos('Center Y', 'Frame')
        ring_11 = cv.getTrackbarPos('Ring 11', 'Frame')
        ring_10 = cv.getTrackbarPos('Ring 10', 'Frame')
        ring_9 = cv.getTrackbarPos('Ring 9', 'Frame')
        ring_8 = cv.getTrackbarPos('Ring 8', 'Frame')
        ring_7 = cv.getTrackbarPos('Ring 7', 'Frame')
        ring_6 = cv.getTrackbarPos('Ring 6', 'Frame')
        ring_5 = cv.getTrackbarPos('Ring 5', 'Frame')
        ring_4 = cv.getTrackbarPos('Ring 4', 'Frame')
        ring_3 = cv.getTrackbarPos('Ring 3', 'Frame')
        ring_2 = cv.getTrackbarPos('Ring 2', 'Frame')
        ring_1 = cv.getTrackbarPos('Ring 1', 'Frame')

        # Draw the rings on the processed frame
        processed_frame = drawRings(processed_frame)

    # Display the processed frame
    cv.imshow("Frame", processed_frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        # Print the stored points when 'q' is pressed
        print("Corner Points Collected:")
        for points in corner_points:
            print(points)
        break

cap.release()
cv.destroyAllWindows()

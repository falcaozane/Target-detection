import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import time

print("Target Webcam Calibration Utility")
print("=================================")
print("This utility helps you set up your webcam and ArUco markers for the target scoring system.")
print("Instructions:")
print("1. Print out 4 ArUco markers (ID: 0, 1, 2, 3) and place them at the corners of your target")
print("2. Position your webcam to capture the entire target")
print("3. Follow the on-screen instructions to calibrate")
print("\nControls:")
print("  R: Reset calibration")
print("  S: Save current configuration")
print("  Q: Quit application")

# Initialize ArUco detector
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Initialize webcam
cap = cv.VideoCapture(0)  # Use default webcam
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Could not open webcam. Please check connection.")
    exit(1)

# Offsets for marker detection
x_offset = 25
y_offset = 26

# Target center and ring sizes 
center_x = 250
center_y = 250
ring_sizes = [23, 53, 79, 105, 135, 162, 188, 215, 242, 268]  # 10, 9, 8, ... down to 1

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
    frame_out = frame.copy()
    
    # Draw detected markers on original image
    if ids is not None:
        aruco.drawDetectedMarkers(frame_out, corners, ids)
    
    # Apply perspective correction if all 4 markers are found
    if ids is not None and len(ids) == 4:
        combined = tuple(zip(ids, corners))
        point_dict = getCorners(combined)
        points_src = np.array([point_dict[0], point_dict[3], point_dict[1], point_dict[2]])
        points_dst = np.float32([[0, 0], [0, 500], [500, 0], [500, 500]])

        matrix, _ = cv.findHomography(points_src, points_dst)
        corrected_image = cv.warpPerspective(frame, matrix, (500, 500))
        markers_found = True
        return frame_out, corrected_image, markers_found
    
    return frame_out, frame, markers_found

def drawRings(canvas):
    # Draw center point
    cv.circle(canvas, (center_x, center_y), 3, (0, 0, 255), -1)
    
    # Draw rings with labels
    for i, radius in enumerate(ring_sizes):
        score = 10 - i
        cv.circle(canvas, (center_x, center_y), radius, (255, 0, 255), 2)
        
        # Add score label at 45 degrees
        label_x = int(center_x + radius * 0.7)
        label_y = int(center_y - radius * 0.7)
        cv.putText(canvas, str(score), (label_x, label_y), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return canvas

def createCalibrationWindows():
    cv.namedWindow("Original")
    cv.namedWindow("Corrected Target")
    
    # Create trackbars for ring center adjustment
    cv.createTrackbar("Center X", "Corrected Target", center_x, 500, lambda x: globals().update(center_x=x))
    cv.createTrackbar("Center Y", "Corrected Target", center_y, 500, lambda x: globals().update(center_y=x))

def saveCalibration():
    # Save calibration parameters to a file
    calibration = {
        "center_x": center_x,
        "center_y": center_y,
        "x_offset": x_offset,
        "y_offset": y_offset,
        "ring_sizes": ring_sizes
    }
    
    np.save("target_calibration.npy", calibration)
    print("Calibration saved to 'target_calibration.npy'")
    
    # Also save as human-readable text file
    with open("target_calibration.txt", "w") as f:
        f.write("Target Calibration Parameters\n")
        f.write("============================\n")
        f.write(f"Center X: {center_x}\n")
        f.write(f"Center Y: {center_y}\n")
        f.write(f"X Offset: {x_offset}\n")
        f.write(f"Y Offset: {y_offset}\n")
        f.write("Ring Sizes:\n")
        for i, size in enumerate(ring_sizes):
            f.write(f"  Ring {10-i}: {size}\n")
    print("Calibration also saved as text to 'target_calibration.txt'")

# Main function
def main():
    global center_x, center_y
    
    createCalibrationWindows()
    
    last_perspective_time = 0
    perspective_interval = 0.5  # Limit expensive perspective operations
    
    print("Webcam activated. Looking for ArUco markers...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from webcam.")
            break
        
        current_time = time.time()
        
        # Process the frame with ArUco detection
        frame_with_markers, corrected_image, markers_found = correctPerspective(frame)
        
        # Display the original frame with detected markers
        cv.putText(frame_with_markers, "Original", (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow("Original", frame_with_markers)
        
        # If markers found, show the corrected image with rings
        if markers_found:
            # Create a copy to draw on
            display_image = corrected_image.copy()
            
            # Draw target rings using current center
            display_image = drawRings(display_image)
            
            # Add guidance text
            cv.putText(display_image, "Adjust Center X/Y with trackbars", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv.imshow("Corrected Target", display_image)
        else:
            # If markers not found, show message
            blank = np.zeros((500, 500, 3), dtype=np.uint8)
            cv.putText(blank, "No markers detected", (150, 250), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.imshow("Corrected Target", blank)
        
        # Process key commands
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting calibration utility.")
            break
        elif key == ord('r'):
            print("Resetting calibration to defaults.")
            center_x = 250
            center_y = 250
            cv.setTrackbarPos("Center X", "Corrected Target", center_x)
            cv.setTrackbarPos("Center Y", "Corrected Target", center_y)
        elif key == ord('s'):
            saveCalibration()
    
    # Clean up
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

"""

Press   :   Function
  P     :   prints wrapped points
  O     :   Shows Original Image
  W     :   Shows Wrapped Image
  V     :   Shows TrackBar For Wrapping points
  R     :   Shows Rings on Image
  B     :   Shows Binary Threshold Image
  S     :   Shows Sharpened Image
  Q     :   Quits Entire Program

NOTE : If a window is already on, pressing its corresponding Button will Close it
NOTE : Sharpened Image ('S') will only show up if Binary Image is active.
"""

# TODO: Add Multi Threading
print("hello")
import sys

import cv2
import numpy as np

output_pts = np.float32([[0, 0], [0, 500], [500, 500], [500, 0]])

show_Rings = False
show_Wrapped = False
show_wrapPoints = False
show_Params = False
sharpening = False
show_OG = True
show_Fish = False

Thresh = 127
k_size = 3
lim = 0

focal = 0
cx = 0
cy = 0
k1 = 0
k2 = 0

wrap_points = {
        'x1': 18, 'y1': 113,
        'x2': 83, 'y2': 607,
        'x3': 563, 'y3': 613,
        'x4': 616, 'y4': 110
    }

rings = {
    'center_x': 250, 'center_y': 250,
    'ring_11':8,
    'ring_10':18, 'ring_9':46, 'ring_8':74,
    'ring_7':100, 'ring_6': 122, 'ring_5': 147, 'ring_4': 174,
    'ring_3': 199, 'ring_2': 219, 'ring_1': 244
}

cap = cv2.VideoCapture("http://192.168.1.5:8000/video_feed")
_, frame = cap.read()
height, width, _ = frame.shape

fish_eye = {
        'focal': [1500, 1500],
        'cx': [427, 1500], 'cy': [0, 600],
        'k1': [94, 100], 'k2': [8, 100]
    }

params = {'alpha': 1.5, 'beta': -0.5, 'gamma': 0}

key_actions = {
    ord('W'): 'show_Wrapped',
    ord('O'): 'show_OG',
    ord('V'): 'show_wrapPoints',
    ord('R'): 'show_Rings',
    ord('B'): 'show_Params',
    ord('S'): 'sharpening',
    ord('F'): 'show_Fish'
}


def manage_window(window_name, should_show, show_func=None, close_func=None, draw_func=None):

    if should_show:
        if draw_func:
            draw_func()
        if show_func and cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            show_func()
    else:
        if close_func:
            close_func(window_name)
        else:
            cv2.destroyWindow(window_name)


def closeWindow(name: str):
    if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow(name)


def startWrapPoints():
    cv2.namedWindow("WrapPoints")
    cv2.resizeWindow("WrapPoints", (400, 300))

    for point, initial_val in wrap_points.items():
        cv2.createTrackbar(point, 'WrapPoints', initial_val, lim,
                           lambda val, key=point: update_point(val, key, wrap_points))


def startParams():
    cv2.namedWindow("Params")
    cv2.resizeWindow("Params", (400, 300))
    for key1, initial_val in params.items():
        cv2.createTrackbar(key1, 'Params', int(initial_val*2+20), 40,
                           lambda val, key=key1: update_point(((val / 2) - 10), key, params))

    cv2.createTrackbar("Thresh", "Params", Thresh, 255, lambda x: globals().update(Thresh=x))
    cv2.createTrackbar("k_size", "Params", k_size, 255, lambda x: globals().update(k_size=(x + (0 if x % 2 else 1))))


def imageProcessor(frame1, sharp: bool):

    frame1 = frame1.copy()
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise and improve detection
    gray_blurred = cv2.GaussianBlur(gray, (k_size, k_size), 2)

    output_image = gray_blurred
    if sharp:
        output_image = sharpened_image = cv2.addWeighted(gray, params['alpha'], gray_blurred, params['beta'], params['gamma'])
        cv2.imshow("Sharpened Image", sharpened_image)
    else:
        closeWindow("Sharpened Image")

    _, binary_image = cv2.threshold(output_image, Thresh, 255, cv2.THRESH_BINARY)

    cv2.imshow("Binary Image", binary_image)


def startRings():
    cv2.namedWindow("Ring_Points")
    cv2.resizeWindow("Ring_Points", (400, 750))
    for key1, initial_val in rings.items():
        cv2.createTrackbar(key1, 'Ring_Points', initial_val, 400, lambda val, key=key1: update_point(val, key, rings))


def drawRings(canvas):
    center_x, center_y = rings['center_x'], rings['center_y']
    cv2.circle(canvas, (center_x, center_y), 1, (0, 0, 255), 2)

    for i in range(1, 12):
        cv2.circle(canvas, (center_x, center_y), rings["ring_" + str(i)], (255, 0, 255), 2)

    cv2.imshow("Rings", canvas)


def update_point(val, key, cus_dict):
    cus_dict[key] = val


def update_fish(val, key):
    fish_eye[key][0] = val
    print(key, ' : ', globals()[key])


def startFishPoints():

    cv2.namedWindow("FishPoints")
    cv2.resizeWindow("FishPoints", (600, 200))

    for point, initial_val in fish_eye.items():
        cv2.createTrackbar(point, 'FishPoints', int(initial_val[0]), initial_val[1],
                           lambda val, key=point: update_fish(val, key))


def printAll():
    print(f"wrap Points: {wrap_points}")
    print(rings)
    print(params)
    print(f"Threshold: {Thresh}")
    print(f"Kernal Size: {k_size}")

def correctFisheye(frame1):
    global focal, cx, cy, k1, k2
    focal = fish_eye['focal'][0] - 60
    cx = fish_eye['cx'][0] - 60
    cy = fish_eye['cy'][0] - 60
    k1 = (fish_eye['k1'][0] - 60)/100
    k2 = (fish_eye['k2'][0] - 60)/100
    K = np.array([[focal, 0, cx],
                  [0, focal, cy],
                  [0, 0, 1]], dtype=np.float32)  # Ensure matrix is of type float32

    D = np.array([k1, k2, 0, 0], dtype=np.float32)  # Ensure the distortion coefficients are also float32
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (width, height), np.eye(3), balance=1)
    undistorted_image = cv2.fisheye.undistortImage(frame1, K, D=D, Knew=new_K)
    points = np.array([
        [wrap_points['x1'], wrap_points['y1']],
        [wrap_points['x2'], wrap_points['y2']],
        [wrap_points['x3'], wrap_points['y3']],
        [wrap_points['x4'], wrap_points['y4']]
    ], dtype=np.int32)
    cv2.polylines(undistorted_image, [points], True, (0, 255, 0), 3)
    cv2.imshow('Fisheye Correction', undistorted_image)
    return undistorted_image

print("hello1")
if __name__ == "__main__":
    cap = cv2.VideoCapture("http://192.168.1.16:8000/video_feed")
    print("hello5")
    _, frame = cap.read()
    lim = max(frame.shape[0], frame.shape[1])
    print("hello2")    
    while True:

        _, frame = cap.read()
   
        points = np.array([
            [wrap_points['x1'], wrap_points['y1']],
            [wrap_points['x2'], wrap_points['y2']],
            [wrap_points['x3'], wrap_points['y3']],
            [wrap_points['x4'], wrap_points['y4']]
        ], dtype=np.int32)

        manage_window("Original Image", show_OG, None, closeWindow, lambda: cv2.imshow("Original Image", frame))
        if show_Fish:
            frame = correctFisheye(frame)
            if (cv2.getWindowProperty("FishPoints", cv2.WND_PROP_VISIBLE)) < 1:
                startFishPoints()
        else:
            closeWindow("FishPoints")
            closeWindow("Fisheye Correction")
            cv2.polylines(frame, [points], True, (0, 255, 0), 3)

        input_pts = np.float32(points)
        M = cv2.getPerspectiveTransform(input_pts, output_pts)
        frameW = cv2.warpPerspective(frame, M, (500, 500))
        print("hello3")
        k = cv2.waitKey(1)

        if k == ord('P'):
            print(points)
        elif k in key_actions:
            globals()[key_actions[k]] = not globals()[key_actions[k]]
        elif k == ord('Q'):
            printAll()
            cv2.destroyAllWindows()
            break

        manage_window("WrapPoints", show_wrapPoints, startWrapPoints, closeWindow)
        manage_window("Wrapped Image", show_Wrapped, None, closeWindow, lambda: cv2.imshow("Wrapped Image", frameW))
        manage_window("Ring_Points", show_Rings, startRings, closeWindow, lambda: drawRings(frameW))
        manage_window("Params", show_Params, startParams, closeWindow, lambda: imageProcessor(frameW, sharpening))

        if not show_Rings:
            closeWindow("Rings")

        if not show_Params:
            closeWindow("Binary Image")
            closeWindow("Sharpened Image")
    sys.exit()

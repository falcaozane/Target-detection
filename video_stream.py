import cv2 as cv
import numpy as np
import requests


class VideoStream:
    def __init__(self, ip):
        self.selected_ip = ip
        self.cap = cv.VideoCapture(f"http://{self.selected_ip}:8000/video_feed")

        if not self.cap.isOpened():
            raise ValueError("Unable to open video feed. Check IP or camera settings.")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from video stream.")
        return frame

    def correct_perspective(self, frame):
        points_src = np.array([[0, 10], [49, 568], [627, 0], [578, 598]])
        points_dst = np.float32([[0, 0], [0, 580], [500, 0], [500, 580]])

        matrix, _ = cv.findHomography(points_src, points_dst)
        image_out = cv.warpPerspective(frame, matrix, (500, 580))
        return image_out

    def release(self):
        self.cap.release()
        cv.destroyAllWindows()

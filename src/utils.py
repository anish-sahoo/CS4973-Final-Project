import numpy as np
import cv2

def rad2deg(x):
    return x * 180.0 / np.pi

def deg2rad(x):
    return x * np.pi / 180.0

def draw_point_on_frame(frame, x, y, color=(0,0,255)):
    cv2.circle(frame, (int(x), int(y)), 8, color, -1)

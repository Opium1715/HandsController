import math
import handDetection
from utils.point import Point


# def overlay(p1, p2):
#     length = math.hypot((p1.x - p2.x), (p1.y - p2.y))
#     return length

def overlay(vector):
    length = math.hypot(vector.x, vector.y)
    return length


def angle_calculate(p1, p2):
    angle = math.degrees(math.acos((p1.x * p2.x + p1.y * p2.y) / (math.hypot(p1.x, p1.y) * math.hypot(p2.x, p2.y))))
    return angle


# def Vector_calculator(hand_landmark, start, end):
#     return Point((hand_landmark[end][1] - hand_landmark[start][1]),
#                  (hand_landmark[end][2] - hand_landmark[start][2]))

def Vector_calculator(handDetector, start, end):
    start_X, start_Y = handDetector.getSpecificXY(start)
    end_X, end_Y = handDetector.getSpecificXY(end)
    return Point((end_X-start_X), (end_Y - start_Y))

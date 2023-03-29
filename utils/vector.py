import math

from handtest import Point


def angle_calculate(p1, p2):
    angle = math.degrees(math.acos((p1.x * p2.x + p1.y * p2.y) / (math.hypot(p1.x, p1.y) * math.hypot(p2.x, p2.y))))
    return angle


def overlay(p1, p2):
    length = math.hypot((p1.x - p2.x), (p1.y - p2.y))
    return length


def Vector_calculator(hand_landmark, width, height, start, end):
    return Point((hand_landmark[end][1] - hand_landmark[start][1]) * width,
                 (hand_landmark[end][2] - hand_landmark[start][2]) * height)


class Vector:
    def __init__(self):
        pass

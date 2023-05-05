import time

import autopy
import cv2
import mediapipe as mp
import numpy as np
import pyautogui

import utils.vector as vector
from utils.point import Point

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 72)

# 设置线程睡眠时间
pyautogui.PAUSE = 0.01

PostureList = [False]
MOUSE_MODE = False
CONTROL_PANEL_MODE = False
CLICK_READY_MODE = False
CLICK_MODE = False

Last_Frame_Point = [Point(320, 240)]

# 平滑处理
beginX, beginY = 0, 0
smooth_ratio = 24
# 分辨率适配
ScreenX, ScreenY = pyautogui.size()


# shape 帧的高度宽度
def posture_predict(hand_landmark, shape):
    global MOUSE_MODE
    global CONTROL_PANEL_MODE
    global CLICK_MODE
    global CLICK_READY_MODE
    # 3->4
    # p3_4 = Point((hand_landmark[4][1] - hand_landmark[3][1]) * shape[1],
    #              (hand_landmark[4][2] - hand_landmark[3][2]) * shape[0])
    p3_4 = vector.Vector_calculator(hand_landmark, shape[1], shape[0], 3, 4)
    # 3->2
    p3_2 = vector.Vector_calculator(hand_landmark, shape[1], shape[0], 3, 2)
    # 17->0
    p17_0 = vector.Vector_calculator(hand_landmark, shape[1], shape[0], 17, 0)
    # 17->18
    p17_18 = vector.Vector_calculator(hand_landmark, shape[1], shape[0], 17, 18)
    # 5->6
    p5_6 = vector.Vector_calculator(hand_landmark, shape[1], shape[0], 5, 6)
    # 5->0
    p5_0 = vector.Vector_calculator(hand_landmark, shape[1], shape[0], 5, 0)
    # 7->8
    p7_8 = vector.Vector_calculator(hand_landmark, shape[1], shape[0], 7, 8)
    # 7->6
    p7_6 = vector.Vector_calculator(hand_landmark, shape[1], shape[0], 7, 6)

    # 计算角度
    angle2_3_4 = vector.angle_calculate(p3_4, p3_2)
    angle0_5_6 = vector.angle_calculate(p5_0, p5_6)
    angle0_17_18 = vector.angle_calculate(p17_18, p17_0)
    angle6_7_8 = vector.angle_calculate(p7_8, p7_6)

    # 方位判断
    direction4_2 = hand_landmark[4][1] < hand_landmark[2][1]  # 4在2左方
    direction8_7 = hand_landmark[8][2] < hand_landmark[7][2]  # 8在7上方
    direction18_17 = hand_landmark[18][2] > hand_landmark[17][2]  # 18在17上方

    # 距离计算
    length4_8 = vector.overlay(Point(x=hand_landmark[4][1] * shape[1], y=hand_landmark[4][2] * shape[0]),
                               Point(x=hand_landmark[8][1] * shape[1], y=hand_landmark[8][2] * shape[0]))
    print("4-8距离{}".format(length4_8))

    if angle0_5_6 >= 140 and angle2_3_4 >= 165 and angle0_17_18 >= 160 and direction18_17:
        CONTROL_PANEL_MODE = False
        MOUSE_MODE = False
        CLICK_MODE = False
        print("手掌张开\n")
        pyautogui.keyUp('alt')

    elif CONTROL_PANEL_MODE:
        print("握拳移动\n")
        Last_Frame_Point[0] = move_direction(Last_Frame_Point[0],
                                             Point(hand_landmark[0][1] * shape[1], hand_landmark[0][2] * shape[0]))

    elif angle0_5_6 <= 80 and angle0_17_18 <= 90 and angle2_3_4 <= 165 and angle6_7_8 < 60:
        print("握拳\n")
        pyautogui.keyDown('alt')
        pyautogui.press('tab')
        CONTROL_PANEL_MODE = True

    elif CLICK_MODE:
        print("点击{}".format(beginX, beginY))
        pyautogui.click()
        CLICK_MODE = False
        CLICK_READY_MODE = False

    elif direction4_2 and direction8_7:
        print("左击准备")
        CLICK_READY_MODE = True

    elif CLICK_READY_MODE:
        CONTROL_PANEL_MODE = False
        MOUSE_MODE = False
        if length4_8 < 50:
            CLICK_MODE = True
            CLICK_READY_MODE = False
        else:
            CLICK_READY_MODE = False

    elif MOUSE_MODE:
        targetX, targetY = smooth_process(Point(hand_landmark[8][1], hand_landmark[8][2]), shape)
        autopy.mouse.move(targetX, targetY)

    elif angle0_17_18 <= 90 and angle2_3_4 <= 165:
        print("____________鼠标模式______________")
        MOUSE_MODE = True

    # p4 = Point(hand_landmark[4][1] * shape[1], hand_landmark[4][2] * shape[0])
    # p8 = Point(hand_landmark[8][1] * shape[1], hand_landmark[8][2] * shape[0])
    # length = vector.overlay(p4, p8)
    # print("拇指关节角度=" + str(angle2_3_4) + "  食指关节点角度=" + str(angle6_7_8) + "  小拇指关节点角度=" + str(
    #     angle0_17_18) + "\n")
    # print("距离值=" + str(length))
    # print("拇指关节角度=" + str(angle2_3_4))


# def getHandAction(hand_landmark, shape):
#     global MOUSE_MODE
#     global CONTROL_PANEL_MODE
#     global CLICK_MODE
#     global CLICK_READY_MODE
#
#     # 3->4
#     # p3_4 = Point((hand_landmark[4][1] - hand_landmark[3][1]) * shape[1],
#     #              (hand_landmark[4][2] - hand_landmark[3][2]) * shape[0])
#     p3_4 = vector.Vector_calculator(hand_landmark, shape[1], shape[0], 3, 4)
#     # 3->2
#     p3_2 = vector.Vector_calculator(hand_landmark, shape[1], shape[0], 3, 2)
#     # 17->0
#     p17_0 = vector.Vector_calculator(hand_landmark, shape[1], shape[0], 17, 0)
#     # 17->18
#     p17_18 = vector.Vector_calculator(hand_landmark, shape[1], shape[0], 17, 18)
#     # 5->6
#     p5_6 = vector.Vector_calculator(hand_landmark, shape[1], shape[0], 5, 6)
#     # 5->0
#     p5_0 = vector.Vector_calculator(hand_landmark, shape[1], shape[0], 5, 0)
#     # 7->8
#     p7_8 = vector.Vector_calculator(hand_landmark, shape[1], shape[0], 7, 8)
#     # 7->6
#     p7_6 = vector.Vector_calculator(hand_landmark, shape[1], shape[0], 7, 6)
#
#     # 计算角度
#     angle2_3_4 = vector.angle_calculate(p3_4, p3_2)
#     angle0_5_6 = vector.angle_calculate(p5_0, p5_6)
#     angle0_17_18 = vector.angle_calculate(p17_18, p17_0)
#     angle6_7_8 = vector.angle_calculate(p7_8, p7_6)
#
#     # 方位判断
#     direction4_2 = hand_landmark[4][0] < hand_landmark[2][0]  # 4在2左方
#     direction8_7 = hand_landmark[8][0] > hand_landmark[7][0]  # 8在7上方
#
#     if angle0_5_6 >= 140 and angle2_3_4 >= 165 and angle0_17_18 >= 140:
#         CONTROL_PANEL_MODE = False
#         MOUSE_MODE = False
#         CLICK_MODE = False
#         print("手掌张开\n")
#         pyautogui.keyUp('alt')
#
#     elif CONTROL_PANEL_MODE:
#         print("握拳移动\n")
#         Last_Frame_Point[0] = move_direction(Last_Frame_Point[0],
#                                              Point(hand_landmark[0][1] * shape[1], hand_landmark[0][2] * shape[0]))
#
#     elif angle0_5_6 <= 80 and angle0_17_18 <= 90 and angle2_3_4 <= 165 and angle6_7_8 < 60:
#         print("握拳\n")
#         pyautogui.keyDown('alt')
#         pyautogui.press('tab')
#         CONTROL_PANEL_MODE = True
#
#     elif MOUSE_MODE:
#         targetX, targetY = smooth_process(Point(hand_landmark[8][1], hand_landmark[8][2]), shape)
#         autopy.mouse.move(targetX, targetY)
#
#     elif angle0_17_18 <= 90 and angle2_3_4 <= 165:
#         print("____________鼠标模式______________")
#         MOUSE_MODE = True
#
#     elif direction4_2 and direction8_7:
#         CLICK_READY_MODE = True
#
#     elif CLICK_READY_MODE:
#         CONTROL_PANEL_MODE = False
#         MOUSE_MODE = False
#         pass
#
#     elif CLICK_MODE:
#         pyautogui.click()
#


def move_direction(last_frame, current_frame):
    Right = True if current_frame.x > last_frame.x else False
    Down = True if current_frame.y > last_frame.y else False
    length_LR = (current_frame.x - last_frame.x) / 8
    length_UD = (current_frame.y - last_frame.y) / 8
    if abs(length_LR) > abs(length_UD):
        print("左右偏移量=" + str(length_LR) + "\n")
        # 左右偏移量更大
        if abs(length_LR) > 3.5:  # 偏移量超过阈值，认为移动
            if Right:
                pyautogui.press('right')
            else:
                pyautogui.press('left')
    else:
        print("上下偏移量=" + str(length_UD) + "\n")
        # 上下偏移量更大
        if abs(length_UD) > 3.5:
            if Down:
                pyautogui.press('down')
            else:
                pyautogui.press('up')
    return current_frame


# 平滑处理
def smooth_process(key_controller, shape):
    global beginY
    global beginX
    # 线性插值
    interp_X = np.interp(key_controller.x * shape[1], (100, shape[1] - 100), (0, ScreenX - 1))
    interp_Y = np.interp(key_controller.y * shape[0], (100, shape[0] - 100), (0, ScreenY - 1))
    # print(interp_X, interp_Y)
    # 平滑
    end_X = beginX + (interp_X - beginX) / smooth_ratio
    end_Y = beginY + (interp_Y - beginY) / smooth_ratio
    # 限制鼠标移动范围
    end_X = end_X if end_X <= 2040 else 2039
    end_X = end_X if end_X >= 1 else 1
    end_Y = end_Y if end_Y <= 1140 else 1140
    end_Y = end_Y if end_Y >= 1 else 1
    beginY = end_Y
    # holy shit beginX = begin_X
    beginX = end_X
    # print(end_X, end_Y)
    # print('\n')
    return end_X, end_Y


def handDetect():
    with mp_hands.Hands(
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            last_Point = Point(320, 240)
            time_start = time.perf_counter()
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    landmark_list = []
                    for landmark_id, finger_axis in enumerate(hand_landmarks.landmark):
                        landmark_list.append([
                            landmark_id, finger_axis.x, finger_axis.y,
                            finger_axis.z
                        ])

                    height, width, channel = image.shape
                    basic_Z = landmark_list[0][3]
                    for lm in landmark_list:
                        cx, cy, cz = int(lm[1] * width), int(lm[2] * height), lm[3] - basic_Z
                        deep_weight = abs(int(cz * 100))
                        color_basic = (deep_weight / 18) * 255
                        # print(deep_weight)
                        # print(color_basic)
                        blue = green = red = color_basic if color_basic < 255 else 255

                        # 进行均值归一化
                        # color_basic =
                        # red,green,blue =
                        cv2.circle(image, (cx, cy), deep_weight, (red, green, blue), -1, 1)
                    if landmark_list:
                        posture_predict(landmark_list, image.shape)

                    # basic_Y = landmark_list[0][2]
                    # pointer_Y = landmark_list[12][2]
                    # ratio = int((pointer_Y - basic_Y) * -10 * 3)
                    # pyautogui.scroll(ratio)

            time_frame_proc = time.perf_counter() - time_start
            fps = int(1 / time_frame_proc)
            cv2.putText(image, "FPS " + str(fps), (10, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
            cv2.imshow('HandsController', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    handDetect()
    cap.release()
    cv2.destroyAllWindows()

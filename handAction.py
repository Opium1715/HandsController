from enum import Enum

import autopy
import cv2
import numpy as np
import pyautogui

import handDetection
import utils.vector as vector
from utils.point import Point


class ActionMode(Enum):
    MOUSE_MODE = 0
    CONTROL_PANEL_MODE = 1
    CLICK_MODE = 2
    CLICK_READY_MODE = 3
    DRAG_MODE = 4
    VICE_CLICK = 5
    RESET_MODE = 6


class HandAction:
    def __init__(self):
        self.handDetector = handDetection.HandDetector()
        # 设置线程睡眠时间
        pyautogui.PAUSE = 0.01
        # 平滑处理参数
        self.smooth_ratio = 30
        # 平滑处理相对点位置
        self.beginX = 0
        self.beginY = 0
        self.image = None
        self.Screen_X = None
        self.Screen_Y = None
        self.lastFrame_X = 320
        self.lastFrame_Y = 240
        self.lastMode = None

    def posture_predict(self):
        handDetector = self.handDetector
        # 向量计算
        # 3->4
        p3_4 = vector.Vector_calculator(handDetector, 3, 4)
        # 3->2
        p3_2 = vector.Vector_calculator(handDetector, 3, 2)
        # 17->0
        p17_0 = vector.Vector_calculator(handDetector, 17, 0)
        # 17->18
        p17_18 = vector.Vector_calculator(handDetector, 17, 18)
        # 5->6
        p5_6 = vector.Vector_calculator(handDetector, 5, 6)
        # 5->0
        p5_0 = vector.Vector_calculator(handDetector, 5, 0)
        # 7->8
        p7_8 = vector.Vector_calculator(handDetector, 7, 8)
        # 7->6
        p7_6 = vector.Vector_calculator(handDetector, 7, 6)
        # 4->8
        p4_8 = vector.Vector_calculator(handDetector, 4, 8)
        # 4->2
        p4_2 = vector.Vector_calculator(handDetector, 4, 2)

        # 计算角度
        angle2_3_4 = vector.angle_calculate(p3_4, p3_2)
        angle0_5_6 = vector.angle_calculate(p5_0, p5_6)
        angle0_17_18 = vector.angle_calculate(p17_18, p17_0)
        angle6_7_8 = vector.angle_calculate(p7_8, p7_6)
        print("angle2-3-4 {}".format(angle2_3_4))

        # 方位判断
        direction4_2 = handDetector.getSpecificXY(4)[0] < handDetector.getSpecificXY(2)[0]  # 4在2左方
        direction8_7 = handDetector.getSpecificXY(8)[1] < handDetector.getSpecificXY(7)[1]  # 8在7上方
        direction18_17 = handDetector.getSpecificXY(18)[1] < handDetector.getSpecificXY(17)[1]  # 18在17上方
        # 距离计算
        length4_8 = vector.overlay(p4_8)
        length4_2 = vector.overlay(p4_2)
        print("length4-2 {}".format(length4_2))

        # 模式出口
        if angle0_5_6 >= 140 and angle2_3_4 >= 165 and angle0_17_18 >= 160 and direction18_17:
            print("手掌张开，重置")
            self.lastMode = None
            return ActionMode.RESET_MODE
        # 保证模式稳定
        # 控制面板
        elif self.lastMode == ActionMode.CONTROL_PANEL_MODE:
            return self.lastMode
        elif angle0_5_6 <= 80 and angle0_17_18 <= 90 and angle2_3_4 <= 165 and angle6_7_8 < 60:
            print("控制面板\n")
            pyautogui.keyDown('alt')
            pyautogui.press('tab')
            self.lastMode = ActionMode.CONTROL_PANEL_MODE
            return ActionMode.CONTROL_PANEL_MODE
        # 鼠标点击
        elif direction4_2 and length4_2 > 85 and direction8_7:
            print("左击准备")
            self.lastMode = ActionMode.CLICK_READY_MODE
            # return ActionMode.CLICK_READY_MODE  # need return ?
        elif self.lastMode == ActionMode.CLICK_READY_MODE:
            if length4_8 < 85:
                self.lastMode = ActionMode.CLICK_MODE
                return self.lastMode
            else:
                self.lastMode = None
        # 鼠标滑动
        elif self.lastMode == ActionMode.MOUSE_MODE:
            return self.lastMode
        elif angle0_17_18 <= 90 and angle2_3_4 <= 165:
            print("____________鼠标模式______________")
            self.lastMode = ActionMode.MOUSE_MODE
            return ActionMode.MOUSE_MODE

    def smooth_process(self, key_controller):
        shape = self.image.shape
        # 线性插值
        interp_X = np.interp(key_controller.x, (100, shape[1] - 100), (0, self.Screen_X - 1))
        interp_Y = np.interp(key_controller.y, (100, shape[0] - 100), (0, self.Screen_Y - 1))
        # 平滑
        end_X = self.beginX + (interp_X - self.beginX) / self.smooth_ratio
        end_Y = self.beginY + (interp_Y - self.beginY) / self.smooth_ratio
        # 限制移动范围
        end_X = end_X if end_X <= 2040 else 2039
        end_X = end_X if end_X >= 1 else 1
        end_Y = end_Y if end_Y <= 1140 else 1140
        end_Y = end_Y if end_Y >= 1 else 1
        self.beginY = end_Y
        self.beginX = end_X
        return end_X, end_Y

    def move_direction(self, current_frame):
        Right = True if current_frame.x > self.lastFrame_X else False
        Down = True if current_frame.y > self.lastFrame_Y else False
        length_LR = (current_frame.x - self.lastFrame_X) / 8
        length_UD = (current_frame.y - self.lastFrame_Y) / 8
        if abs(length_LR) > abs(length_UD):
            print("左右偏移量=" + str(length_LR) + "\n")
            # 左右偏移量更大
            if abs(length_LR) > 2.5:  # 偏移量超过阈值，认为移动
                if Right:
                    pyautogui.press('right')
                else:
                    pyautogui.press('left')
        else:
            print("上下偏移量=" + str(length_UD) + "\n")
            # 上下偏移量更大
            if abs(length_UD) > 2.5:
                if Down:
                    pyautogui.press('down')
                else:
                    pyautogui.press('up')
        self.lastFrame_X = current_frame.x
        self.lastFrame_Y = current_frame.y

    def process(self):
        cap = cv2.VideoCapture(0)
        self.Screen_X, self.Screen_Y = pyautogui.size()
        handDetector = self.handDetector
        # tempAction = None
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            self.image, isDetected = handDetector.handDetect(image)
            cv2.imshow('HandsController', self.image)
            if isDetected:
                action = self.posture_predict()
                if action:
                    if action == ActionMode.RESET_MODE:
                        # tempAction = None
                        pyautogui.keyUp('alt')

                    elif action == ActionMode.CONTROL_PANEL_MODE:
                        # tempAction = ActionMode.CONTROL_PANEL_MODE
                        print("握拳移动\n")
                        self.move_direction(Point(handDetector.getSpecificXY(0)[0],
                                                  handDetector.getSpecificXY(0)[1]))
                    elif action == ActionMode.CLICK_MODE:
                        print("点击于像素空间{} {}".format(self.beginX, self.beginY))
                        pyautogui.click()
                    elif action == ActionMode.CLICK_READY_MODE:
                        pass
                    elif action == ActionMode.MOUSE_MODE:
                        targetX, targetY = self.smooth_process(Point(handDetector.getSpecificXY(8)[0],
                                                                     handDetector.getSpecificXY(8)[1]))
                        autopy.mouse.move(targetX, targetY)
                    elif action == ActionMode.DRAG_MODE:
                        pass
                    elif action == ActionMode.VICE_CLICK:
                        pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

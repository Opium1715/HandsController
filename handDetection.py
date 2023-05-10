import time

import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, DeepPerception=True):
        # 临时使用的手部21点模型
        self.hand_model = mp.solutions.hands
        self.drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles
        self.hands = self.hand_model.Hands(model_complexity=1,
                                           max_num_hands=1,
                                           min_tracking_confidence=0.55,
                                           min_detection_confidence=0.7,
                                           )
        self.landmark_list = []
        self.DeepPerception = DeepPerception

    def getLandMark(self):
        return self.landmark_list

    def getSpecificXY(self, index):
        index_X = self.landmark_list[index][1]
        index_Y = self.landmark_list[index][2]
        return index_X, index_Y

    def drawHandDeepPerception(self, image):
        basic_Z = self.landmark_list[0][3]
        for lm in self.landmark_list:
            cx, cy, cz = int(lm[1]), int(lm[2]), lm[3] - basic_Z
            deep_weight = abs(int(cz * 100))
            color_basic = (deep_weight / 18) * 255
            blue = green = red = color_basic if color_basic < 255 else 255
            cv2.circle(image, (cx, cy), deep_weight, (red, green, blue), -1, 1)

    def handDetect(self, image):
        # 开始处理时间
        time_start = time.perf_counter()
        image.flags.writeable = False
        # 转换处理格式RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 镜像翻转
        image = cv2.flip(image, 1)
        # 推理21点结果
        results = self.hands.process(image)
        image.flags.writeable = True
        self.landmark_list.clear()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        isDetected = False
        if results.multi_hand_landmarks:
            isDetected = True
            for hand_landmarks in results.multi_hand_landmarks:
                self.drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.hand_model.HAND_CONNECTIONS,
                    self.drawing_styles.get_default_hand_landmarks_style(),
                    self.drawing_styles.get_default_hand_connections_style())

                for landmark_id, finger_axis in enumerate(hand_landmarks.landmark):
                    height, width, channel = image.shape
                    # 归一化的位置数据 ——> 像素空间的21点位置
                    pixel_X = finger_axis.x * width
                    pixel_Y = finger_axis.y * height
                    self.landmark_list.append([
                        landmark_id, pixel_X, pixel_Y,
                        finger_axis.z
                    ])
                if self.DeepPerception:
                    self.drawHandDeepPerception(image)
        time_frame_proc = time.perf_counter() - time_start
        fps = int(1 / time_frame_proc)
        cv2.putText(image, "FPS " + str(fps), (10, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0))
        return image, isDetected

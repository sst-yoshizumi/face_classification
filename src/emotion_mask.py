# coding: UTF-8

from statistics import mode

import cv2
import time
from keras.models import load_model
import numpy as np
import os
import pickle
import base64

from utils.datasets import get_labels
# from utils.inference import detect_faces
from utils.inference import draw_text
# from utils.inference import draw_bounding_box
from utils.inference import draw_circle
# from utils.inference import draw_transparent_circle
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# ADD kuni
import json
import requests
import time
from datetime import datetime as dt

from emotion import Emotion             # 感情データ処理用
from happymirror_const import *

import yunet_facedetect
import matplotlib.pyplot as plt
# ガベージコレクションを有効にする
import gc
gc.enable()


# parameters for loading data and images
# detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

detection_model_path = '../trained_models/detection_models/face_detection_yunet_2023mar_int8.onnx'
# Happy とその他の 2 値分類モデル
#emotion_model_path = '../trained_models/emotion_models/happymirror_mini_XCEPTION.65-0.95.hdf5'
#emotion_labels = get_labels('happymirror')  # Happy とその他の 2 値分類

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
# 顔検出用モデル YuNet の準備
print('Preparing face-detection model...')
# face_detection = load_detection_model(detection_model_path)
face_detection = yunet_facedetect.load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

time_before = time.time()                               # ループ直前の時刻を保存（デバッグ用）

emotion_data = Emotion()                                # 感情データ処理用インスタンスを生成。

# starting video streaming
cv2.namedWindow('window_frame')

# ウィンドウサイズを最大化します。
cv2.setWindowProperty('window_frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

video_capture = cv2.VideoCapture(0)
w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
face_detection.setInputSize([w, h])

while True:
    hasFrame, bgr_image = video_capture.read()
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    if not hasFrame:
        print('No frames grabbed!')
        break

    # Inference
    # faces = detect_faces(face_detection, gray_image)
    faces = face_detection.infer(bgr_image)

    if faces is None or len(faces) == 0:
        emotion_data.no_faces()                         # 顔を検出していない情報を emotion_data に知らせる。
    print(f'{len(faces)} faces found.')

    face_num = 0
    for face_coordinates in faces:
        face_num += 1

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]

        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        # 赤
        if emotion_text == 'angry':
            color = np.asarray((254, 0, 0))
        # 青
        elif emotion_text == 'sad':
            color = np.asarray((0, 0, 254))
        # 黄
        elif emotion_text == 'happy':
            color = np.asarray((254, 254, 0))
        # 水色
        elif emotion_text == 'surprise':
            color = np.asarray((0, 254, 254))
        # 紫
        elif emotion_text == 'disgust':
            color = np.asarray((128, 0, 128))
        # 緑 
        elif emotion_text == 'fear':
            color = np.asarray((0, 128, 0))
        # 白
        elif emotion_text == 'neutral':
            color = np.asarray((254, 254, 254))
        # defalt 黄緑 ここには、入らない。
        else:
            color = emotion_probability * np.asarray((0, 254, 0))

        # debug
        time_now = time.time()
        time_before = time_now

        # 2 値分類を従来の 7 分類に置き換えます。
        # emotion_prediction[0][0] = その他（neutral）  = 6 番目
        # emotion_prediction[0][1] = Happy              = 3 番目
        emotion_prediction7 = [0, 0, 0, emotion_prediction[0][1], 0, 0, emotion_prediction[0][0]]

        # 感情値（各感情の確率）を蓄積します。
        # EmotionFlower の add_emotion_rate() と同じような役割をします。
#        emotion_data.accumurate(emotion_prediction[0])
        emotion_data.accumurate(emotion_prediction7)

        color = color.astype(int).tolist()

        # color の色の順序が RGB なので、 BGR に変換します。
        bgr_color = [color[2], color[1], color[0]]

        draw_circle(face_coordinates, bgr_image, color)
        draw_text(face_coordinates, bgr_image, emotion_text, bgr_color, 0, -45, 1, 1)

#        # ウィンドウ全体に太い枠を描画します。
#        THICKNESS = 20
#        height, width = rgb_image.shape[:2]
#        cv2.rectangle(rgb_image, (0 + THICKNESS, 0 + THICKNESS), (width - THICKNESS, height - THICKNESS), color, thickness=20)

    if face_num == 0:
        emotion_data.no_faces()                         # 顔が小さい（遠くにいる）ときは、検出なし扱いとします。

    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # ガベージコレクション実行
    gc.collect()


video_capture.release()
cv2.destroyAllWindows()

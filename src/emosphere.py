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
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import draw_circle
from utils.inference import draw_transparent_circle
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# ADD kuni
import json
import requests
import time
from datetime import datetime as dt

#next# from notifier import Notifier           # ハートビート、顔検出通知用
from emotion import Emotion             # 感情データ処理用
from happymirror_const import *
#next# from happymirror_neopixel import HappyMirrorLed # LED 制御用

#import yunet.yunet_facedetect
import yunet_facedetect
import matplotlib.pyplot as plt
# ガベージコレクションを有効にする
import gc
gc.enable()


def draw_stacked_bar_chart(base_image, data_list, position='bottom'):
    """
    積み上げ横棒グラフを画像の下部に描画します。

    Parameters
    ----------
    base_image : np.ndarray
        描画するベースとなる画像。 BGR 形式。
    data_list : list
        各データのリスト。感情ごとの値を持つリスト。
    position : str or int, optional
        グラフの位置。'bottom' または Y軸の位置を指定します。デフォルトは 'bottom'。
    
    Returns
    -------
    np.ndarray
        描画されたグラフを含む画像。
    """
    # BGR format colors
    colors = [[0, 0, 254], [254, 0, 0], [0, 254, 254], [254, 254, 0], [128, 0, 128], [0, 128, 0], [254, 254, 254]]
    labels = ['angry', 'sad', 'happy', 'surprise', 'disgust', 'fear', 'neutral']
    category_name = 'Atmosphere'

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, value in enumerate(data_list):
        # 各感情の値を積み上げていく
        if i == 0:
            bottom = 0
        else:
            bottom = sum(data_list[:i])
        
        # 横棒グラフを描画
        ax.barh(category_name, value, left=bottom, color=colors[i], label=labels[i])

#    ax.set_xlabel('Emotion Rate')
#    ax.set_title(f'{category_name} Emotion Rates')
    ax.set_xlim(0, bottom * 1.1)  # Assuming emotion rates are normalized between 0 and 1
    ax.axis('off')  # 軸を非表示にする
#    ax.legend(loc='upper right')
    plt.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    graph_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
    plt.close(fig)

    # グラフを画像の下部に配置
    if position == 'bottom':
        base_height, base_width = base_image.shape[:2]
        graph_height, graph_width = graph_image.shape[:2]
        combined_image = np.zeros((base_height + graph_height, base_width, 3), dtype=np.uint8)
        combined_image[:base_height, :base_width] = base_image
        combined_image[base_height:, :graph_width] = graph_image
    else:
        # Y軸の位置を指定された場合、その位置にグラフを配置
        y_position = int(position)
        base_height, base_width = base_image.shape[:2]
        graph_height, graph_width = graph_image.shape[:2]
        combined_image = np.zeros((base_height + graph_height, base_width, 3), dtype=np.uint8)
        combined_image[:base_height, :base_width] = base_image
        combined_image[y_position:y_position + graph_height, :graph_width] = graph_image



def draw_stacked_bar_chart_old(base_image, data_list, position='bottom'):
    """
    Draw stacked bar chart.

    Parameters
    ----------
    base_image : image_array
        BGRA format image
    data_list : list
        data list
    position : str or int
        position to draw the chart
        'bottom' or Y-coordinate (int)

    Returns
    -------
    image : image_array
    """
    fig, ax = plt.subplots(figsize=(6, 1.5))
    bottom = 0
#    colors = plt.cm.tab20(np.linspace(0, 1, len(data_series_list)))
#    colors = [[254, 0, 0], [0, 0, 254], [254, 254, 0], [0, 254, 254], [128, 0, 128], [0, 128, 0], [254, 254, 254]]
    # BGR format colors
    colors = [[0, 0, 254], [254, 0, 0], [0, 254, 254], [254, 254, 0], [128, 0, 128], [0, 128, 0], [254, 254, 254]]
    labels = ['angry', 'sad', 'happy', 'surprise', 'disgust', 'fear', 'neutral']
    category_name = 'Atmosphere'

    for i, val in enumerate(data_list):
        ax.barh(category_name, val, left=bottom, color=colors[i], label=labels[i])
        bottom += val

    ax.set_xlim(0, bottom * 1.1)
    ax.axis('off')
    plt.tight_layout(pad=0)
#    ax.set_title('Atmosphere')
#    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#    plt.tight_layout()

    # Convert to image
    fig.canvas.draw()
    w, h = fig.get_size_inches() * fig.dpi
    w, h = int(w), int(h)
#    fig.patch.set_alpha(0.0)  # 図全体の背景を透明に
    graph_image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
    plt.close(fig)
    # matplotlib は ARGB 形式で画像を保存するので、BGRA に変換する必要があります。
    # Convert ARGB to BGRA
    graph_image = graph_image[:, :, [3, 2, 1, 0]]
    # 得られた graph_image のアルファチャネルは 255 で、完全に不透明です。
#    # アルファチャネルの値を 60% に設定します。
#    graph_image[:, :, 3] = (graph_image[:, :, 3] * 0.6).astype(np.uint8)

    # graph_image のサイズを base_image に合わせてリサイズします。
    # base_image のサイズを取得
    base_h, base_w = base_image.shape[:2]
    # graph image size
    if graph_image.shape[1] > base_w:
        scale = base_w / graph_image.shape[1]
        graph_image = cv2.resize(graph_image, (base_w, int(graph_image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    h, w = graph_image.shape[:2]

    # position
    if position == 'bottom':
        y = base_h - h
    elif isinstance(position, int):
        y = position
    else:
        y = base_h - h
    
    if y < 0:
        extended = np.ones((base_h + abs(y), base_w, 3), dtype=np.uint8) * 255
        extended[:base_h, :] = base_image
        base_image = extended
        y = base_h
    
    # alpha blending
    # base_image の ROI を取得し、graph_image のアルファチャンネルを使って合成します。
    # ROI とは、base_image の graph_image のサイズと同じ領域のことです。
    roi = base_image[y:y+h, :w].copy()
    # graph_image のアルファチャンネルを取得
    alpha_graph = graph_image[:, :, 3:] / 255.0
    # base_image の ROI のアルファチャンネルを取得
    alpha_base = roi[:, :, 3:] / 255.0 * (1 - alpha_graph)
    # 各チャンネルごとに合成します。
    for c in range(3):
        roi[:, :, c] = (graph_image[:, :, c] * alpha_graph[:, :, 0] + roi[:, :, c] * alpha_base[:, :, 0]).astype(np.uint8)
    # アルファチャンネルを更新します。
    roi[:, :, 3] = ((alpha_graph[:, :, 0] + alpha_base[:, :, 0]) * 255).astype(np.uint8)
    # base_image の ROI を更新します。
    base_image[y:y+h, :w] = roi
    
    return base_image





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

#next# heartbeat_notifier = Notifier(HEARTBEAT_HOLD_TIME)      # ハートビートのインスタンスを生成
#next# face_detect_notifier = Notifier(FACEDETECT_HOLD_TIME)   # 顔検出通知用インスタンスを生成
emotion_data = Emotion()                                # 感情データ処理用インスタンスを生成。
#next# happy_led = HappyMirrorLed()                            # HappyMirror 用 NeoPixel LED 制御インスタンスを生成

# starting video streaming
cv2.namedWindow('window_frame')

# ウィンドウサイズを最大化します。
cv2.setWindowProperty('window_frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

video_capture = cv2.VideoCapture(0)
w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
face_detection.setInputSize([w, h])

#next# heartbeat_notifier.init()                               # ハートビート初期化
#next# face_detect_notifier.init()                             # 顔検出通知を初期化
#next# happy_led.all_led_off()                                 # 念のため LED をすべて消灯します。


while True:
    # bgr_image = video_capture.read()[1]
    hasFrame, bgr_image = video_capture.read()
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
#RGB#    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    if not hasFrame:
        print('No frames grabbed!')
        break
#next#     heartbeat_notifier.notice()                         # カメラ画像の読み取りができればハートビートを打つ

    # Inference
    # faces = detect_faces(face_detection, gray_image)
    faces = face_detection.infer(bgr_image)

    if faces is None or len(faces) == 0:
        emotion_data.no_faces()                         # 顔を検出していない情報を emotion_data に知らせる。
    print(f'{len(faces)} faces found.')

    face_num = 0
    for face_coordinates in faces:
        # 一番大きい顔を取得し、その値が 100 以上なら表情判定に進みます。
        # それ以外は遠くの人や横から覗いている人と考えて、無視するようにします。
        # 展示会で多くの人を認識したり、遠くのパネルなどの誤認識をはじくため。
        # face_coordinates はそれぞれ以下のようなリストになっていて、[X座標, Y座標, サイズ, サイズ] のようなので、
        # 3 番目の要素 face_coordinates[2] を大きさの判定材料にします。
        # [239 230 132 132]
        # [222 326  53  53]
#        print(face_coordinates)
#next#         if face_coordinates[2] < 100:
#next#             continue

        face_num += 1
#next#         face_detect_notifier.notice()                   # 顔を検出したら通知する。

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        #pred_time_now = time.time()
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        #pred_time_before = time.time()
        #print("pred_timetime span: ", (pred_time_before - pred_time_now))   # 認識時間

#        # for debug
#        for i in range(len(emotion_prediction[0])):
#            print(str(i), emotion_labels[i], emotion_prediction[0][i])

        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        # 赤
        if emotion_text == 'angry':
            #color = emotion_probability * np.asarray((255, 0, 0))
            color = np.asarray((254, 0, 0))
        # 青
        elif emotion_text == 'sad':
            #color = emotion_probability * np.asarray((0, 0, 255))
            color = np.asarray((0, 0, 254))
        # 黄
        elif emotion_text == 'happy':
            #color = emotion_probability * np.asarray((255, 255, 0))
            color = np.asarray((254, 254, 0))
        # 水色
        elif emotion_text == 'surprise':
            #color = emotion_probability * np.asarray((0, 255, 255))
            color = np.asarray((0, 254, 254))
        # 紫
        elif emotion_text == 'disgust':
            #color = emotion_probability * np.asarray((128, 0, 128))
            color = np.asarray((128, 0, 128))
        # 緑 
        elif emotion_text == 'fear':
            #color = emotion_probability * np.asarray((0, 128, 0))
            color = np.asarray((0, 128, 0))
        # 白
        elif emotion_text == 'neutral':
            #color = emotion_probability * np.asarray((255, 255, 255))
            color = np.asarray((254, 254, 254))
        # defalt 黄緑 ここには、入らない。
        else:
            color = emotion_probability * np.asarray((0, 254, 0))

#        print(emotion_text,color,emotion_probability)

        # debug
        time_now = time.time()
#        print("time span: ", (time_now - time_before))   # 前回からの経過時間
        time_before = time_now

        # 2 値分類を従来の 7 分類に置き換えます。
        # emotion_prediction[0][0] = その他（neutral）  = 6 番目
        # emotion_prediction[0][1] = Happy              = 3 番目
        emotion_prediction7 = [0, 0, 0, emotion_prediction[0][1], 0, 0, emotion_prediction[0][0]]

        # 感情値（各感情の確率）を蓄積します。
        # EmotionFlower の add_emotion_rate() と同じような役割をします。
#        emotion_data.accumurate(emotion_prediction[0])
        emotion_data.accumurate(emotion_prediction7)

        #     # ADD kuni
        #     url = "https://script.google.com/macros/s/AKfycbwplNBc3ILI7VaPeYWTKmOZuW8pihMEgEvIGIMsQuVwXLs-5a93qzy8YfWvlXd1U3E_yw/exec"
        #     tdatetime = dt.now()
        #     tstr = (tdatetime.now().strftime('%Y/%m/%d %H:%M:%S:%f')[:-3])
        #     # print(tstr)
        #     # JSON形式でデータを用意してdataに格納
        #     data = {
        #     	"timestamp": tstr,
        #     	"emotion"  : emotion_text,
        #     	"other"    : str(emotion_probability)
        #     }
        #     # json.dumpでデータをJSON形式として扱う
        #     r = requests.post(url, data=json.dumps(data))

        color = color.astype(int).tolist()

        # color の色の順序が RGB なので、 BGR に変換します。
        bgr_color = [color[2], color[1], color[0]]

#draw#
#        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_circle(face_coordinates, bgr_image, bgr_color)
        draw_text(face_coordinates, bgr_image, emotion_text, bgr_color, 0, -45, 1, 1)


#        # ウィンドウ全体に太い枠を描画します。
#        THICKNESS = 20
#        height, width = rgb_image.shape[:2]
#        cv2.rectangle(rgb_image, (0 + THICKNESS, 0 + THICKNESS), (width - THICKNESS, height - THICKNESS), color, thickness=20)

    if face_num == 0:
        emotion_data.no_faces()                         # 顔が小さい（遠くにいる）ときは、検出なし扱いとします。

    # LED を制御します。
    # for ブロックの中は顔を検出されないと実行されないので、for の外で行います。
#next#     happy_led.depict(heartbeat_notifier, face_detect_notifier, emotion_data)

#    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    #time.sleep(500/1000)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # ガベージコレクション実行
    gc.collect()


video_capture.release()
cv2.destroyAllWindows()

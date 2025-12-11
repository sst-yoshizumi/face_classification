import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

def load_image(image_path, grayscale=False, target_size=None):
    pil_image = image.load_img(image_path, grayscale, target_size)
    return image.img_to_array(pil_image)

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates[:4]
    cv2.rectangle(image_array, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates[:4]
    x_off, y_off = offsets
    return (int(x - x_off), int(x + width + x_off), int(y - y_off), int(y + height + y_off))

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (int(x + x_offset), int(y + y_offset)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

def get_colors(num_classes):
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    colors = np.asarray(colors) * 255
    return colors


def draw_circle(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates[:4]

    # Calculate center
    center = (int(x + w / 2), int(y + h / 2))

    # Calculate radius
#    radius = int(0.5 * np.sqrt(w**2 + h**2))
    radius = int(h * 0.5)

    # Draw cirle
    cv2.circle(image_array, center, radius, color, thickness=-1)

    # 円の周辺を同じ色で、円の中心から遠くなるにつれて徐々に透過するように描画します。
    for r in range(radius, radius + 100, 2):
        # 円の中心から遠くなるにつれて透過度を上げる
        alpha = max(0, 1 - (r - radius) / 100)
        # 背景画像と透過色を合成
        overlay_color = (int(color[0] * alpha + image_array[center[1], center[0], 0] * (1 - alpha)),
                         int(color[1] * alpha + image_array[center[1], center[0], 1] * (1 - alpha)),
                         int(color[2] * alpha + image_array[center[1], center[0], 2] * (1 - alpha)))
        cv2.circle(image_array, center, r, overlay_color, thickness=1, lineType=cv2.LINE_AA)
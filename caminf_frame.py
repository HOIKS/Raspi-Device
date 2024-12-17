import cv2
import torch
import numpy as np
import onnxruntime as ort

# from deepface import DeepFace
from picamera2 import Picamera2
from ultralytics import YOLO


# YOLOv8n-face 모델 불러오기
model = YOLO("yolov8n-face_ncnn_model")

# 성별 & 나이 추론 ONNX 모델 불러오기 
ortsession_age = ort.InferenceSession("age_model.onnx")
ortsession_gender = ort.InferenceSession("gender_model.onnx")


# Picamera2 초기화
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# 카메라로부터 프레임 읽기
capture = picam2.capture_array()

# 카메라 이미지를 모델에 넣기
inference = model(capture)

# 얼굴 박스 위치정보 가져오기
frame_box = inference[0].boxes.xyxy.cpu().tolist()

# 이미지 크롭
cropped = capture[int(frame_box[0][1]) : int(frame_box[0][3]), int(frame_box[0][0]) : int(frame_box[0][2])]
cropped_resize = cv2.resize(cropped, (224, 224))

# 이미지 float32 변환 및 정규화
input_frame = cropped_resize.astype(np.float32)
input_frame = input_frame / 255.0 # 정규화
input_frame = np.expand_dims(input_frame, axis=0) # (224, 224, 3) -> (1, 224, 224, 3) 차원 추가 및 축변경

### 성별 추론 ###
labels = ["Female", "Male"]
input_name_gender = ortsession_gender.get_inputs()[0].name
inference_raw_gender = ortsession_gender.run(None, {input_name_gender: input_frame})
result_gender = labels[np.argmax(inference_raw_gender[0])]

print(f"추론된 성별: {result_gender}")


### 나이 추론 ###
input_name_age = ortsession_age.get_inputs()[0].name
inference_raw_age = ortsession_age.run(None, {input_name_age: input_frame})
age_indexes = np.array(list(range(0, 101)))
result_age = np.sum(inference_raw_age[0] * age_indexes)

print(f"추론된 나이: {result_age}세")


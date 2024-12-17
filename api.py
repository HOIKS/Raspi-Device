import cv2
import numpy as np
import onnxruntime as ort
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from picamera2 import Picamera2
from ultralytics import YOLO

from printer import print_receipt as print_receipt

# FastAPI 로드
app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://172.20.10.6:3000",
    "http://192.168.134.41:3000",
    "http://192.168.130.129:3000",    # 프론트엔드의 도메인 혹은 IP를 여기에 추가
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 허용할 도메인 리스트
    allow_credentials=True,
    allow_methods=["*"],  # 허용할 HTTP 메서드, ["GET", "POST"] 
    allow_headers=["*"],  # 허용할 HTTP 헤더, ["X-Custom-Header"] 
)


# YOLOv8n-face 모델 불러오기
model = YOLO("yolov8n-face_ncnn_model")

# 성별 & 나이 추론 ONNX 모델 불러오기 
ortsession_age = ort.InferenceSession("age_model.onnx")
ortsession_gender = ort.InferenceSession("gender_model.onnx")


# Picamera2 초기화
picam2 = Picamera2()
# 카메라 설정을 더 상세하게 구성
picam2.set_controls({
    "AfMode": 2,  # 연속 오토포커스
    "AfTrigger": 0,  # 오토포커스 트리거
    "AfSpeed": 1,  # 오토포커스 속도 (1: 일반)
    "AfRange": 0,  # 전체 범위
    "AfMetering": 0,  # 중앙 중점 측광
    "Brightness": 0,  # 밝기
    "Contrast": 1.0,  # 대비
    "Sharpness": 1.0  # 선명도
})

# 카메라 안정화를 위한 대기 시간 추가
time.sleep(2)

# 오토포커스 활성화
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()


@app.get("/")
async def root():
    return "Hello! I'm HOIKS Kiosk System"

@app.get("/inference")
async def inference():
    # 카메라로부터 프레임 읽기
    capture = picam2.capture_array()

    # 이미지로부터 프레임 읽기
    # capture = cv2.imread("test_img/Jiwoo.JPG")

    capture = cv2.resize(capture, (1280, 720))

    # 카메라 이미지를 모델에 넣기
    inference = model(capture)

    # 얼굴 박스 위치정보 가져오기
    frame_box = inference[0].boxes.xyxy.cpu().tolist()
    if (len(frame_box) == 0) :
        return "No Face Detection"
    
    # print(frame_box[0])
    if (frame_box[0][1] > 360):
        position = "LOW"
    else:
        position = "NORM"
        

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

    ### 나이 추론 ###
    input_name_age = ortsession_age.get_inputs()[0].name
    inference_raw_age = ortsession_age.run(None, {input_name_age: input_frame})[0]
    age_indexes = np.arange(0, 101)
    result_age = int(np.argmax(inference_raw_age))

    result_output = {'age' : result_age,
                     'gender' : result_gender,
                     'position' : position}

    return result_output 

    

@app.get("/yolo_capture")
async def yolo_capture():
    # Capture frame-by-frame
    frame = picam2.capture_array()

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imwrite("yolo_capture.jpg", annotated_frame)

    return FileResponse("yolo_capture.jpg")


@app.get("/shot")
async def shot():
    capture = picam2.capture_array()
    
    height, width = capture.shape[:2]
    size = min(height, width)  # 정사각형의 한 변 길이
    start_x = (width - size) // 2
    start_y = (height - size) // 2
    cropped = capture[start_y:start_y+size, start_x:start_x+size]
    bright_value = 50
    cropped = cv2.convertScaleAbs(cropped, alpha=1, beta=bright_value)
    downsized = cv2.resize(cropped, (360, 360))
    cv2.imwrite("shot.jpg", cropped)
    cv2.imwrite("shot_downsized.jpg", downsized)
    return FileResponse("shot.jpg")


@app.get("/print")
async def printer():
    print_receipt("shot_downsized.jpg")
    return "Print Request Success"

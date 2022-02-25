import cv2
import math

def output_keypoints(frame, net, threshold, BODY_PARTS, now_frame, total_frame):
    global points

    # 입력 이미지의 사이즈 정의
    image_height = 368
    image_width = 368

    # 네트워크에 넣기 위한 전처리
    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(input_blob)

    # 결과 받아오기
    out = net.forward()
    # output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
    out_height = out.shape[2]
    out_width = out.shape[3]

    # 원본 이미지의 높이, 너비를 받아오기
    frame_height, frame_width = frame.shape[:2]

    # 포인트 리스트 초기화
    points = []

    print(f"============================== frame: {now_frame:.0f} / {total_frame:.0f} ==============================")
    for i in range(len(BODY_PARTS)):

        # 신체 부위의 confidence map
        prob_map = out[0, i, :, :]

        # 최소값, 최대값, 최소값 위치, 최대값 위치
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        # 원본 이미지에 맞게 포인트 위치 조정
        x = (frame_width * point[0]) / out_width
        x = int(x)
        y = (frame_height * point[1]) / out_height
        y = int(y)

        if prob > threshold:  # [pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)

            points.append((x, y))

        else:  # [not pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)

            points.append(None)

    return frame

def calculate_degree_neck(point_1, point_2, frame):
    # 역탄젠트 구하기
    dx = point_2[0] - point_1[0]
    dy = point_2[1] - point_1[1]
    rad = math.atan2(abs(dy), abs(dx))

    # radian 을 degree 로 변환
    deg = rad * 180 / math.pi

    # degree 가 75'보다 작으면 자세가 안좋다고 판단
    if deg < 75:
        string = "Bad"
        cv2.putText(frame, string, (800, 200), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255))
        print(f"[degree] {deg} ({string})")

def calculate_degree_Shoulder(point_1, point_2, frame):
    # 역탄젠트 구하기
    dx = point_2[0] - point_1[0]
    dy = point_2[1] - point_1[1]
    rad = math.atan2(abs(dy), abs(dx))

    # radian 을 degree 로 변환
    deg = rad * 180 / math.pi

    # degree 가 10'보다 크면 자세가 안좋다고 판단
    if deg > 10:
        string = "Bad"
        cv2.putText(frame, string, (800, 200), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255))
        print(f"[degree] {deg} ({string})")
    
        
def output_keypoints_with_lines(frame, POSE_PAIRS):
    frame = frame.copy()
    
    if (points[0] is not None) and (points[1] is not None):
        calculate_degree_neck(point_1=points[0], point_2=points[1], frame=frame)
    
    if (points[2] is not None) and (points[5] is not None):
        calculate_degree_Shoulder(point_1=points[2], point_2=points[5], frame=frame)
         
    for pair in POSE_PAIRS:
        part_a = pair[0]  # 0 (Head)
        part_b = pair[1]  # 1 (Neck)
        if points[part_a] and points[part_b]:   
            cv2.line(frame, points[part_a], points[part_b], (0, 255, 0), 3)

    return frame



def output_keypoints_with_lines_video(proto_file, weights_file, video_path, threshold, BODY_PARTS, POSE_PAIRS):
    
    # 네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
    
    # GPU 사용
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # 비디오 읽어오기
    capture = cv2.VideoCapture(video_path)
    
    #재생할 파일의 넓이와 높이
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output2.avi', fourcc, 5.0, (int(width), int(height)))

    while True:
        now_frame_boy = capture.get(cv2.CAP_PROP_POS_FRAMES)
        total_frame_boy = capture.get(cv2.CAP_PROP_FRAME_COUNT)

        if now_frame_boy == total_frame_boy:
            break

        ret, frame_boy = capture.read()
        frame_boy = output_keypoints(frame=frame_boy, net=net, threshold=threshold, BODY_PARTS=BODY_PARTS, now_frame=now_frame_boy, total_frame=total_frame_boy)
        frame_boy = output_keypoints_with_lines(frame=frame_boy, POSE_PAIRS=POSE_PAIRS)
        cv2.imshow("Output_Keypoints", frame_boy)
        out.write(frame_boy)

        if cv2.waitKey(10) == 27:  # esc 입력시 종료
            break

    capture.release()
    cv2.destroyAllWindows()

BODY_PARTS_COCO = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                   5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                   10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
                   15: "LEye", 16: "REar", 17: "LEar", 18: "Background"}

POSE_PAIRS_COCO = [[0, 1], [0, 14], [0, 15], [1, 2], [1, 5], [1, 8], [1, 11], [2, 3], [3, 4],
                   [5, 6], [6, 7], [8, 9], [9, 10], [12, 13], [11, 12], [14, 16], [15, 17]]

# 신경 네트워크의 구조를 지정하는 prototxt 파일 (다양한 계층이 배열되는 방법 등)
protoFile_coco = "C:\\Users\\best1\\SP1\\sc_app\\pose_deploy_linevec.prototxt"

# 훈련된 모델의 weight 를 저장하는 caffemodel 파일
weightsFile_coco = "C:\\Users\\best1\\SP1\\sc_app\\views\\pose_iter_440000.caffemodel"

# 경로
man = "C:\\Users\\best1\\SP1\\Timeline.mov"

# 키포인트를 저장할 빈 리스트
points = []

output_keypoints_with_lines_video(proto_file=protoFile_coco, weights_file=weightsFile_coco, video_path=man,
                                  threshold=0.1, BODY_PARTS=BODY_PARTS_COCO, POSE_PAIRS=POSE_PAIRS_COCO)
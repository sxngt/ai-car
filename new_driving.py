"""
도로 중앙 추종 방식 + 장애물 감지 정지
밝은 영역(도로)의 중심을 찾아 화면 중앙과의 차이로 조향
전방에 장애물(어두운 물체) 감지 시 정지
"""
import mycamera
import cv2
import numpy as np
from gpiozero import DigitalOutputDevice
from gpiozero import PWMOutputDevice

# 모터 핀 설정
PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)
BIN2 = DigitalOutputDevice(24)


def motor_go(speed):
    print(f"  -> motor_go({speed})")
    # 전진: 양쪽 모터 전진
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed


def motor_left(speed):
    print(f"  -> motor_left({speed})")
    # 좌회전: A후진 B전진
    AIN1.value = 1
    AIN2.value = 0
    PWMA.value = speed
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed


def motor_right(speed):
    print(f"  -> motor_right({speed})")
    # 우회전: A전진 B후진
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = speed


def motor_stop():
    print(f"  -> motor_stop()")
    PWMA.value = 0.0
    PWMB.value = 0.0


# ===== 파라미터 (조정 가능) =====
SPEED = 1.0               # 기본 전진 속도 (최대)
TURN_SPEED = 1.0          # 회전 속도 (최대)
ROAD_THRESHOLD = 100      # 이 밝기 이상을 도로로 인식 (벽보다 밝은 영역)
CENTER_DEADZONE = 40      # 중앙에서 이 픽셀 이내면 직진 (30 → 40, 고속 안정성)

# ===== 장애물 감지 파라미터 =====
OBSTACLE_THRESHOLD = 80   # 이 밝기 이하를 장애물로 인식 (도로보다 어두운 물체)
OBSTACLE_RATIO = 0.3      # 감지 영역의 30% 이상이 장애물이면 정지
OBSTACLE_ROI_HEIGHT = 0.3 # 화면 하단 30%를 장애물 감지 영역으로 사용
OBSTACLE_ROI_WIDTH = 0.4  # 화면 중앙 40%를 장애물 감지 영역으로 사용


def find_road_center(image):
    """
    도로(밝은 영역)의 중심 x좌표를 찾음

    Returns:
        center_x: 도로 중심의 x좌표
        frame_center: 화면 중앙 x좌표
        confidence: 도로 감지 신뢰도 (0~1)
    """
    height, width = image.shape[:2]

    # 상단 노이즈 제거: 하단 50%만 사용
    roi_top = int(height * 0.5)
    roi = image[roi_top:, :]

    # 그레이스케일 변환
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi

    # 도로(밝은 영역) 마스크 생성
    road_mask = gray > ROAD_THRESHOLD

    # 도로 픽셀이 거의 없으면 중앙 반환
    road_pixels = np.sum(road_mask)
    total_pixels = road_mask.size
    confidence = road_pixels / total_pixels

    if confidence < 0.05:  # 도로가 거의 안 보이면
        return width // 2, width // 2, 0.0

    # 도로 영역의 x좌표들의 평균 (무게중심)
    road_x_coords = np.where(road_mask)[1]  # [1]은 x좌표
    if len(road_x_coords) == 0:
        return width // 2, width // 2, 0.0

    center_x = int(np.mean(road_x_coords))
    frame_center = width // 2

    return center_x, frame_center, confidence


def detect_obstacle(image):
    """
    전방 장애물 감지
    화면 하단 중앙에 도로보다 어두운 물체가 있으면 장애물로 판단

    Returns:
        is_obstacle: 장애물 감지 여부
        obstacle_ratio: 장애물 비율 (0~1)
        roi_rect: 감지 영역 좌표 (x1, y1, x2, y2)
    """
    height, width = image.shape[:2]

    # 감지 영역: 화면 하단 중앙
    roi_h = int(height * OBSTACLE_ROI_HEIGHT)
    roi_w = int(width * OBSTACLE_ROI_WIDTH)

    x1 = (width - roi_w) // 2
    x2 = x1 + roi_w
    y1 = height - roi_h
    y2 = height

    roi = image[y1:y2, x1:x2]

    # 그레이스케일 변환
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi

    # 장애물(어두운 영역) 마스크 생성
    obstacle_mask = gray < OBSTACLE_THRESHOLD

    # 장애물 비율 계산
    obstacle_pixels = np.sum(obstacle_mask)
    total_pixels = obstacle_mask.size
    obstacle_ratio = obstacle_pixels / total_pixels

    # 장애물 판정
    is_obstacle = obstacle_ratio > OBSTACLE_RATIO

    return is_obstacle, obstacle_ratio, (x1, y1, x2, y2)


def decide_steering(road_center, frame_center):
    """
    도로 중심과 화면 중심의 차이로 조향 결정

    Returns:
        direction: "go", "left", "right"
        offset: 중앙에서 벗어난 정도 (음수=왼쪽, 양수=오른쪽)
    """
    offset = road_center - frame_center

    # 데드존: 중앙 근처면 직진
    if abs(offset) < CENTER_DEADZONE:
        return "go", offset

    # 도로 중심이 오른쪽에 있으면 → 오른쪽으로 가야 함
    if offset > 0:
        return "right", offset
    else:
        return "left", offset


def draw_debug_info(image, road_center, frame_center, direction, offset, confidence,
                    is_obstacle=False, obstacle_ratio=0.0, obstacle_roi=None):
    """디버그 정보 표시"""
    height, width = image.shape[:2]
    debug_img = image.copy()

    # ROI 영역 표시
    roi_top = int(height * 0.5)
    cv2.line(debug_img, (0, roi_top), (width, roi_top), (0, 255, 255), 2)

    # 화면 중앙선 (파란색)
    cv2.line(debug_img, (frame_center, roi_top), (frame_center, height), (255, 0, 0), 2)

    # 도로 중심선 (녹색)
    cv2.line(debug_img, (road_center, roi_top), (road_center, height), (0, 255, 0), 3)

    # 데드존 표시 (노란색 영역)
    cv2.line(debug_img, (frame_center - CENTER_DEADZONE, roi_top),
             (frame_center - CENTER_DEADZONE, height), (0, 255, 255), 1)
    cv2.line(debug_img, (frame_center + CENTER_DEADZONE, roi_top),
             (frame_center + CENTER_DEADZONE, height), (0, 255, 255), 1)

    # 장애물 감지 영역 표시
    if obstacle_roi:
        x1, y1, x2, y2 = obstacle_roi
        color = (0, 0, 255) if is_obstacle else (0, 255, 0)  # 빨강: 장애물, 초록: 안전
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
        if is_obstacle:
            cv2.putText(debug_img, "OBSTACLE!", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 정보 텍스트
    if is_obstacle:
        dir_color = (0, 0, 255)  # 빨강
        direction_text = "STOP!"
    else:
        dir_color = (0, 255, 0) if direction == "go" else (0, 165, 255)
        direction_text = direction.upper()

    cv2.putText(debug_img, f"Offset: {offset:+d}px", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(debug_img, f"Conf: {confidence:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(debug_img, f"Obstacle: {obstacle_ratio:.1%}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(debug_img, f"DIR: {direction_text}", (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, dir_color, 2)

    return debug_img


def main():
    camera = mycamera.MyPiCamera(640, 480)
    car_state = "stop"
    obstacle_stopped = False  # 장애물로 인한 정지 상태

    print("=== Road Center Following + Obstacle Detection ===")
    print("UP: Start | DOWN: Stop | Q: Quit")
    print(f"SPEED={SPEED}, ROAD_THRESHOLD={ROAD_THRESHOLD}, DEADZONE={CENTER_DEADZONE}px")
    print(f"OBSTACLE: threshold={OBSTACLE_THRESHOLD}, ratio={OBSTACLE_RATIO:.0%}")

    try:
        while True:
            # 카메라 이미지 캡처
            ret, image = camera.read()
            if not ret:
                continue
            image = cv2.flip(image, -1)

            # 도로 중심 찾기
            road_center, frame_center, confidence = find_road_center(image)

            # 장애물 감지
            is_obstacle, obstacle_ratio, obstacle_roi = detect_obstacle(image)

            # 조향 결정
            direction, offset = decide_steering(road_center, frame_center)

            # 디버그 화면
            debug_img = draw_debug_info(image, road_center, frame_center, direction, offset, confidence,
                                        is_obstacle, obstacle_ratio, obstacle_roi)
            cv2.imshow('Road Center', debug_img)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 82:  # UP arrow
                print(">>> START")
                car_state = "go"
                obstacle_stopped = False
            elif key == 84:  # DOWN arrow
                print(">>> STOP")
                car_state = "stop"
                obstacle_stopped = False
                motor_stop()

            # 모터 제어
            if car_state == "go":
                # 장애물 감지 시 즉시 정지
                if is_obstacle:
                    if not obstacle_stopped:
                        print(f"[OBSTACLE DETECTED] ratio={obstacle_ratio:.1%} -> EMERGENCY STOP!")
                        obstacle_stopped = True
                    motor_stop()
                else:
                    obstacle_stopped = False
                    # offset이 크면 직진 금지, 조향만 실행
                    if direction == "left":
                        motor_left(TURN_SPEED)
                        print(f"[MOTOR] LEFT speed={TURN_SPEED}")
                    elif direction == "right":
                        motor_right(TURN_SPEED)
                        print(f"[MOTOR] RIGHT speed={TURN_SPEED}")
                    else:  # direction == "go" (중앙에 있을 때만 직진)
                        motor_go(SPEED)
                        print(f"[MOTOR] GO speed={SPEED}")

                    print(f"Road:{road_center} Frame:{frame_center} Offset:{offset:+d} -> {direction}")
            else:
                motor_stop()

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        motor_stop()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == '__main__':
    main()

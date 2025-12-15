"""
도로 중앙 추종 방식
밝은 영역(도로)의 중심을 찾아 화면 중앙과의 차이로 조향
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
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = speed


def motor_left(speed):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed


def motor_right(speed):
    AIN1.value = 1
    AIN2.value = 0
    PWMA.value = speed
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = speed


def motor_stop():
    PWMA.value = 0.0
    PWMB.value = 0.0


# ===== 파라미터 (조정 가능) =====
SPEED = 0.45              # 기본 전진 속도
TURN_SPEED = 0.5          # 회전 속도
ROAD_THRESHOLD = 100      # 이 밝기 이상을 도로로 인식 (벽보다 밝은 영역)
CENTER_DEADZONE = 30      # 중앙에서 이 픽셀 이내면 직진 (화면 너비의 약 5%)


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


def draw_debug_info(image, road_center, frame_center, direction, offset, confidence):
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

    # 정보 텍스트
    dir_color = (0, 255, 0) if direction == "go" else (0, 165, 255)
    cv2.putText(debug_img, f"Offset: {offset:+d}px", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(debug_img, f"Conf: {confidence:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(debug_img, f"DIR: {direction.upper()}", (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, dir_color, 2)

    return debug_img


def main():
    camera = mycamera.MyPiCamera(640, 480)
    car_state = "stop"

    print("=== Road Center Following ===")
    print("UP: Start | DOWN: Stop | Q: Quit")
    print(f"SPEED={SPEED}, ROAD_THRESHOLD={ROAD_THRESHOLD}, DEADZONE={CENTER_DEADZONE}px")

    try:
        while True:
            # 카메라 이미지 캡처
            ret, image = camera.read()
            if not ret:
                continue
            image = cv2.flip(image, -1)

            # 도로 중심 찾기
            road_center, frame_center, confidence = find_road_center(image)

            # 조향 결정
            direction, offset = decide_steering(road_center, frame_center)

            # 디버그 화면
            debug_img = draw_debug_info(image, road_center, frame_center, direction, offset, confidence)
            cv2.imshow('Road Center', debug_img)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 82:  # UP arrow
                print(">>> START")
                car_state = "go"
            elif key == 84:  # DOWN arrow
                print(">>> STOP")
                car_state = "stop"
                motor_stop()

            # 모터 제어
            if car_state == "go":
                if direction == "go":
                    motor_go(SPEED)
                elif direction == "left":
                    motor_left(TURN_SPEED)
                elif direction == "right":
                    motor_right(TURN_SPEED)

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

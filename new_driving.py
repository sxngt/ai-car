"""
중앙 유지 주행 코드 - 원래 모터 제어 방식 사용
"""
import time
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


# 원래 driving.py와 동일한 모터 제어 함수들
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
SPEED = 0.5               # 주행 속도
TURN_SPEED = 0.5          # 회전 속도
WALL_THRESHOLD = 50       # 벽 판정 밝기
DEAD_ZONE = 0.15          # 좌우 차이가 이 이하면 직진


def analyze_regions(image):
    """이미지를 좌/중/우 3개 영역으로 나누어 벽 비율 분석"""
    height, width = image.shape[:2]

    # 상단 노이즈 제거: 하단 55%만 사용
    roi_top = int(height * 0.45)
    roi = image[roi_top:, :]

    # 그레이스케일 변환
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi

    roi_height, roi_width = gray.shape

    # 좌/중/우 영역 정의
    third = roi_width // 3
    left_region = gray[:, :third]
    center_region = gray[:, third:third*2]
    right_region = gray[:, third*2:]

    def calc_wall_ratio(region):
        wall_pixels = np.sum(region < WALL_THRESHOLD)
        total_pixels = region.size
        return wall_pixels / total_pixels if total_pixels > 0 else 0

    left_ratio = calc_wall_ratio(left_region)
    center_ratio = calc_wall_ratio(center_region)
    right_ratio = calc_wall_ratio(right_region)

    return left_ratio, center_ratio, right_ratio


def decide_direction(left_ratio, right_ratio):
    """
    좌우 벽 비율로 방향 결정
    Returns: "go", "left", "right"
    """
    diff = left_ratio - right_ratio

    # 데드존: 차이가 작으면 직진
    if abs(diff) < DEAD_ZONE:
        return "go"

    # 왼쪽 벽이 많으면 오른쪽으로
    if diff > 0:
        return "right"
    else:
        return "left"


def draw_debug_info(image, left_ratio, center_ratio, right_ratio, direction):
    """디버그 정보를 이미지에 표시"""
    height, width = image.shape[:2]
    debug_img = image.copy()

    # ROI 영역 표시
    roi_top = int(height * 0.45)
    cv2.line(debug_img, (0, roi_top), (width, roi_top), (0, 255, 255), 2)

    # 3등분 선 표시
    third = width // 3
    cv2.line(debug_img, (third, roi_top), (third, height), (255, 255, 0), 1)
    cv2.line(debug_img, (third*2, roi_top), (third*2, height), (255, 255, 0), 1)

    # 비율 색상
    def ratio_color(ratio):
        if ratio > 0.5:
            return (0, 0, 255)
        elif ratio > 0.3:
            return (0, 165, 255)
        else:
            return (0, 255, 0)

    # 좌우 비율 표시
    cv2.putText(debug_img, f"L:{left_ratio:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ratio_color(left_ratio), 2)
    cv2.putText(debug_img, f"C:{center_ratio:.2f}", (third + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ratio_color(center_ratio), 2)
    cv2.putText(debug_img, f"R:{right_ratio:.2f}", (third*2 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ratio_color(right_ratio), 2)

    # 방향 표시
    dir_color = (0, 255, 0) if direction == "go" else (0, 165, 255)
    cv2.putText(debug_img, f"DIR: {direction.upper()}", (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, dir_color, 2)

    return debug_img


def main():
    camera = mycamera.MyPiCamera(640, 480)
    car_state = "stop"

    print("=== Simple Lane Keeping ===")
    print("UP: Start | DOWN: Stop | Q: Quit")
    print(f"SPEED={SPEED}, DEAD_ZONE={DEAD_ZONE}, WALL_THRESHOLD={WALL_THRESHOLD}")

    try:
        while True:
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

            # 카메라 이미지 캡처
            ret, image = camera.read()
            if not ret:
                continue
            image = cv2.flip(image, -1)

            # 벽 비율 분석
            left_ratio, center_ratio, right_ratio = analyze_regions(image)

            # 방향 결정
            direction = decide_direction(left_ratio, right_ratio)

            # 디버그 화면
            debug_img = draw_debug_info(image, left_ratio, center_ratio, right_ratio, direction)
            cv2.imshow('Lane Keep', debug_img)

            # 모터 제어
            if car_state == "go":
                if direction == "go":
                    motor_go(SPEED)
                elif direction == "left":
                    motor_left(TURN_SPEED)
                elif direction == "right":
                    motor_right(TURN_SPEED)

                print(f"L:{left_ratio:.2f} R:{right_ratio:.2f} diff:{left_ratio-right_ratio:+.2f} -> {direction}")
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

"""
완주 전략: 느리고 촘촘하게 방향 조절
펄스 방식 - 짧게 동작 후 멈추고 판단 반복
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
SPEED = 0.4               # 전진 속도
TURN_SPEED = 0.6          # 회전 속도 (높여서 확실히 회전)
WALL_THRESHOLD = 80       # 벽 판정 밝기
DEAD_ZONE = 0.08          # 좌우 차이 임계값

# 펄스 타이밍 (초)
PULSE_DURATION = 0.15     # 한 번 동작 시간 (늘려서 모터가 반응할 시간 확보)
PAUSE_DURATION = 0.02     # 동작 후 멈춤 시간


def analyze_regions(image):
    """이미지를 좌/중/우 3개 영역으로 나누어 벽 비율 분석"""
    height, width = image.shape[:2]

    roi_top = int(height * 0.45)
    roi = image[roi_top:, :]

    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi

    roi_width = gray.shape[1]

    third = roi_width // 3
    left_region = gray[:, :third]
    center_region = gray[:, third:third*2]
    right_region = gray[:, third*2:]

    def calc_wall_ratio(region):
        wall_pixels = np.sum(region < WALL_THRESHOLD)
        total_pixels = region.size
        return wall_pixels / total_pixels if total_pixels > 0 else 0

    return calc_wall_ratio(left_region), calc_wall_ratio(center_region), calc_wall_ratio(right_region)


def decide_direction(left_ratio, right_ratio):
    """좌우 벽 비율로 방향 결정"""
    diff = left_ratio - right_ratio

    if abs(diff) < DEAD_ZONE:
        return "go", diff
    elif diff > 0:
        return "right", diff
    else:
        return "left", diff


def pulse_move(direction):
    """짧은 펄스로 동작 실행"""
    if direction == "go":
        motor_go(SPEED)
    elif direction == "left":
        motor_left(TURN_SPEED)
    elif direction == "right":
        motor_right(TURN_SPEED)

    time.sleep(PULSE_DURATION)
    motor_stop()
    time.sleep(PAUSE_DURATION)


def draw_debug_info(image, left_ratio, center_ratio, right_ratio, direction):
    """디버그 정보 표시"""
    height, width = image.shape[:2]
    debug_img = image.copy()

    roi_top = int(height * 0.45)
    cv2.line(debug_img, (0, roi_top), (width, roi_top), (0, 255, 255), 2)

    third = width // 3
    cv2.line(debug_img, (third, roi_top), (third, height), (255, 255, 0), 1)
    cv2.line(debug_img, (third*2, roi_top), (third*2, height), (255, 255, 0), 1)

    def ratio_color(ratio):
        if ratio > 0.5:
            return (0, 0, 255)
        elif ratio > 0.3:
            return (0, 165, 255)
        else:
            return (0, 255, 0)

    cv2.putText(debug_img, f"L:{left_ratio:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ratio_color(left_ratio), 2)
    cv2.putText(debug_img, f"C:{center_ratio:.2f}", (third + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ratio_color(center_ratio), 2)
    cv2.putText(debug_img, f"R:{right_ratio:.2f}", (third*2 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, ratio_color(right_ratio), 2)

    dir_color = (0, 255, 0) if direction == "go" else (0, 165, 255)
    cv2.putText(debug_img, f"DIR: {direction.upper()}", (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, dir_color, 2)

    return debug_img


def main():
    camera = mycamera.MyPiCamera(640, 480)
    car_state = "stop"

    print("=== Slow & Steady Strategy ===")
    print("UP: Start | DOWN: Stop | Q: Quit")
    print(f"SPEED={SPEED}, PULSE={PULSE_DURATION}s")

    try:
        while True:
            # 카메라 이미지 캡처
            ret, image = camera.read()
            if not ret:
                continue
            image = cv2.flip(image, -1)

            # 벽 비율 분석
            left_ratio, center_ratio, right_ratio = analyze_regions(image)

            # 방향 결정
            direction, diff = decide_direction(left_ratio, right_ratio)

            # 디버그 화면
            debug_img = draw_debug_info(image, left_ratio, center_ratio, right_ratio, direction)
            cv2.imshow('Slow Drive', debug_img)

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

            # 모터 제어: 펄스 방식
            if car_state == "go":
                pulse_move(direction)
                print(f"L:{left_ratio:.2f} R:{right_ratio:.2f} diff:{diff:+.2f} -> {direction}")

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        motor_stop()
        cv2.destroyAllWindows()
        print("Done")


if __name__ == '__main__':
    main()

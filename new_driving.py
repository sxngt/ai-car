"""
중앙 유지 주행 코드
좌우 벽 비율 차이에 비례해서 연속적으로 조향
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


def motor_forward(left_speed, right_speed):
    """차등 구동: 좌우 바퀴 속도를 다르게 해서 부드러운 조향"""
    left_speed = max(0, min(1, left_speed))
    right_speed = max(0, min(1, right_speed))

    # 왼쪽 모터 (A) 전진
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = left_speed

    # 오른쪽 모터 (B) 전진
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = right_speed


def motor_stop():
    PWMA.value = 0.0
    PWMB.value = 0.0


# ===== 파라미터 (조정 가능) =====
BASE_SPEED = 0.5          # 기본 주행 속도
STEERING_GAIN = 0.6       # 조향 민감도
WALL_THRESHOLD = 50       # 벽 판정 밝기 (낮출수록 더 어두운 것만 벽으로 인식)
MIN_SPEED = 0.35          # 회전 시 느린쪽 바퀴 최소 속도
DEAD_ZONE = 0.15          # 좌우 차이가 이 이하면 직진 (직진성 강화)


def analyze_regions(image):
    """
    이미지를 좌/우 2개 영역으로 나누어 벽 비율 분석
    중앙은 별도로 체크 (전방 장애물)
    """
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


def calculate_steering(left_ratio, right_ratio):
    """
    좌우 벽 비율 차이로 조향값 계산

    Returns:
        steering: -1.0 (왼쪽으로) ~ 0 (직진) ~ +1.0 (오른쪽으로)
    """
    diff = left_ratio - right_ratio

    # 데드존: 차이가 작으면 직진
    if abs(diff) < DEAD_ZONE:
        return 0.0

    # 데드존 넘는 부분만 조향에 반영
    if diff > 0:
        adjusted_diff = diff - DEAD_ZONE
    else:
        adjusted_diff = diff + DEAD_ZONE

    # 민감도 적용
    steering = adjusted_diff * STEERING_GAIN

    # 범위 제한
    steering = max(-1.0, min(1.0, steering))

    return steering


def apply_steering(base_speed, steering):
    """
    조향값을 적용해서 좌우 바퀴 속도 계산

    steering > 0: 오른쪽으로 → 왼쪽 바퀴 빠르게
    steering < 0: 왼쪽으로 → 오른쪽 바퀴 빠르게
    """
    left_speed = base_speed * (1 + steering)
    right_speed = base_speed * (1 - steering)

    # 최소 속도 보장 (한쪽이 너무 느리면 회전 안됨)
    left_speed = max(MIN_SPEED, min(1.0, left_speed))
    right_speed = max(MIN_SPEED, min(1.0, right_speed))

    return left_speed, right_speed


def draw_debug_info(image, left_ratio, center_ratio, right_ratio, steering, left_spd, right_spd):
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

    # 조향값 시각화 (중앙에 화살표)
    center_x = width // 2
    center_y = height - 50
    arrow_len = int(steering * 100)
    color = (0, 255, 255) if abs(steering) < 0.3 else (0, 165, 255) if abs(steering) < 0.6 else (0, 0, 255)
    cv2.arrowedLine(debug_img, (center_x, center_y), (center_x + arrow_len, center_y), color, 3)

    # 조향값 및 속도 표시
    cv2.putText(debug_img, f"Steer:{steering:+.2f}", (10, height - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(debug_img, f"L:{left_spd:.2f} R:{right_spd:.2f}", (10, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return debug_img


def main():
    camera = mycamera.MyPiCamera(640, 480)
    car_state = "stop"

    print("=== Center-Keeping Drive ===")
    print("UP: Start | DOWN: Stop | Q: Quit")
    print(f"BASE_SPEED={BASE_SPEED}, STEERING_GAIN={STEERING_GAIN}")

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

            # 조향값 계산 (항상 중앙 유지하려고 함)
            steering = calculate_steering(left_ratio, right_ratio)

            # 좌우 바퀴 속도 계산
            left_spd, right_spd = apply_steering(BASE_SPEED, steering)

            # 디버그 화면
            debug_img = draw_debug_info(image, left_ratio, center_ratio, right_ratio, steering, left_spd, right_spd)
            cv2.imshow('Center Keep', debug_img)

            # 모터 제어
            if car_state == "go":
                motor_forward(left_spd, right_spd)
                print(f"L:{left_ratio:.2f} R:{right_ratio:.2f} -> steer:{steering:+.2f} | spd L:{left_spd:.2f} R:{right_spd:.2f}")
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

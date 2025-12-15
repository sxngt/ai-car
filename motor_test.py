"""
모터 테스트 스크립트
방향키로 직접 모터 제어 테스트
"""
import cv2
from gpiozero import DigitalOutputDevice
from gpiozero import PWMOutputDevice

# 모터 핀 설정
PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)
BIN2 = DigitalOutputDevice(24)

SPEED = 0.7


def motor_go():
    print(">>> GO (전진)")
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = SPEED
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = SPEED


def motor_back():
    print(">>> BACK (후진)")
    AIN1.value = 1
    AIN2.value = 0
    PWMA.value = SPEED
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = SPEED


def motor_left():
    print(">>> LEFT (좌회전)")
    AIN1.value = 1
    AIN2.value = 0
    PWMA.value = SPEED
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = SPEED


def motor_right():
    print(">>> RIGHT (우회전)")
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = SPEED
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = SPEED


def motor_stop():
    print(">>> STOP (정지)")
    PWMA.value = 0.0
    PWMB.value = 0.0


def main():
    print("=" * 40)
    print("모터 테스트 스크립트")
    print("=" * 40)
    print("조작법:")
    print("  ↑ (위)   : 전진")
    print("  ↓ (아래) : 후진")
    print("  ← (왼쪽) : 좌회전")
    print("  → (오른쪽): 우회전")
    print("  Space    : 정지")
    print("  Q        : 종료")
    print("=" * 40)

    # OpenCV 창 생성 (키 입력 받기 위해)
    cv2.namedWindow("Motor Test")
    img = cv2.imread("/dev/null") if False else None
    # 빈 이미지 생성
    import numpy as np
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    cv2.putText(img, "Motor Test", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "Use arrow keys", (80, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(img, "Q to quit", (130, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    try:
        while True:
            cv2.imshow("Motor Test", img)
            key = cv2.waitKey(100) & 0xFF

            if key == ord('q'):
                print("종료합니다.")
                break
            elif key == 82:  # UP arrow
                motor_go()
            elif key == 84:  # DOWN arrow
                motor_back()
            elif key == 81:  # LEFT arrow
                motor_left()
            elif key == 83:  # RIGHT arrow
                motor_right()
            elif key == 32:  # Space
                motor_stop()

    except KeyboardInterrupt:
        print("\n인터럽트")
    finally:
        motor_stop()
        cv2.destroyAllWindows()
        print("테스트 종료")


if __name__ == '__main__':
    main()

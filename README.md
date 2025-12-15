# AI Car - 자율주행 조향각 예측 모델

NVIDIA End-to-End Learning 기반 자율주행 조향각 예측 모델 학습 코드

## 환경 요구사항

- Python 3.10-3.11
- Ubuntu 22.04 + NVIDIA RTX 4090 (CUDA)
- [uv](https://docs.astral.sh/uv/) 패키지 매니저

## 설치

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync
```

## 데이터 준비

`data/video/` 폴더에 학습 이미지를 배치합니다.

### 이미지 형식
- 파일명: `_XXXXX_YYY.png` (YYY = 조향각)
- 크기: 200x66 pixels
- 조향각: 45(좌회전), 90(직진), 135(우회전)

## 학습 실행

```bash
uv run python train.py
```

## 출력

`output/` 폴더에 생성:
- `lane_navigation_check.keras` - 최적 모델 (val_loss 기준)
- `lane_navigation_final.keras` - 최종 모델
- `history.pickle` - 학습 히스토리

## 모델 사용 예시

```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('output/lane_navigation_final.keras')

def predict_steering(frame):
    image = cv2.resize(frame, (200, 66))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    angle = model.predict(image)[0][0]

    if angle < 60:
        return angle, "LEFT"
    elif angle > 120:
        return angle, "RIGHT"
    else:
        return angle, "STRAIGHT"
```

## 모델 구조

NVIDIA End-to-End Self-Driving 아키텍처:
- Conv2D (24, 36, 48, 64, 64 filters)
- Dropout (0.2)
- Dense (100, 50, 10, 1)
- 총 파라미터: 252,219개

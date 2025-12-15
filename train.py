#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
자율주행 자동차 모델 학습 스크립트
NVIDIA End-to-End Learning Model for Self-Driving Cars
M1 Max 최적화 버전
"""

import os
import random
import fnmatch
import re
import pickle
from pathlib import Path

# Data processing
import numpy as np
np.set_printoptions(formatter={'float_kind': lambda x: "%.4f" % x})

import pandas as pd
pd.set_option('display.width', 300)
pd.set_option('display.float_format', '{:,.4f}'.format)
pd.set_option('display.max_colwidth', 200)

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# sklearn
from sklearn.model_selection import train_test_split

# Imaging
import cv2

print(f"TensorFlow version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# GPU 설정 확인
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU devices: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found, using CPU")

# ====================
# 설정
# ====================
DATA_DIR = Path(__file__).parent / "data" / "video"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 100
EPOCHS = 100
STEPS_PER_EPOCH = 150
VALIDATION_STEPS = 200
LEARNING_RATE = 1e-3


def load_data(data_dir: Path) -> tuple[list[str], list[int]]:
    """이미지 경로와 조향각 로드"""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    image_paths = []
    steering_angles = []
    pattern = "*.png"

    for filename in os.listdir(data_dir):
        if fnmatch.fnmatch(filename, pattern):
            image_paths.append(str(data_dir / filename))
            # 파일명에서 조향각 추출: _XXXXX_YYY.png 형식
            match = re.search(r'_(\d+)\.png$', filename)
            if match:
                angle = int(match.group(1))
                steering_angles.append(angle)
            else:
                print(f"Warning: Could not extract angle from '{filename}'")

    print(f"Loaded {len(image_paths)} images")
    return image_paths, steering_angles


def my_imread(image_path: str) -> np.ndarray:
    """이미지 읽기"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def img_preprocess(image: np.ndarray) -> np.ndarray:
    """이미지 정규화 (0-1 범위)"""
    return image / 255.0


def nvidia_model() -> Sequential:
    """NVIDIA 자율주행 모델 구성"""
    model = Sequential(name='Nvidia_Model')

    # Input: 66x200x3 (이미지 크기가 이미 맞음)
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))  # 조향각 출력

    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='mse', optimizer=optimizer)

    return model


def image_data_generator(image_paths: list[str], steering_angles: list[int], batch_size: int):
    """배치 데이터 제너레이터"""
    while True:
        batch_images = []
        batch_steering_angles = []

        for _ in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            image = my_imread(image_paths[random_index])
            steering_angle = steering_angles[random_index]

            image = img_preprocess(image)
            batch_images.append(image)
            batch_steering_angles.append(steering_angle)

        yield np.asarray(batch_images), np.asarray(batch_steering_angles)


def main():
    print("=" * 50)
    print("자율주행 자동차 모델 학습 시작")
    print("=" * 50)

    # 1. 데이터 로드
    print("\n[1/5] 데이터 로드 중...")
    image_paths, steering_angles = load_data(DATA_DIR)

    # 2. 데이터 분할
    print("\n[2/5] 학습/검증 데이터 분할 중...")
    X_train, X_valid, y_train, y_valid = train_test_split(
        image_paths, steering_angles, test_size=0.2, random_state=42
    )
    print(f"Training data: {len(X_train)}")
    print(f"Validation data: {len(X_valid)}")

    # 3. 모델 생성
    print("\n[3/5] NVIDIA 모델 생성 중...")
    model = nvidia_model()
    model.summary()

    # 4. 콜백 설정
    checkpoint_path = OUTPUT_DIR / "lane_navigation_check.keras"
    checkpoint_callback = ModelCheckpoint(
        filepath=str(checkpoint_path),
        verbose=1,
        save_best_only=True,
        monitor='val_loss'
    )

    # 5. 학습
    print("\n[4/5] 모델 학습 시작...")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"Steps per epoch: {STEPS_PER_EPOCH}, Validation steps: {VALIDATION_STEPS}")

    history = model.fit(
        image_data_generator(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=image_data_generator(X_valid, y_valid, batch_size=BATCH_SIZE),
        validation_steps=VALIDATION_STEPS,
        verbose=1,
        callbacks=[checkpoint_callback]
    )

    # 6. 최종 모델 저장
    print("\n[5/5] 모델 저장 중...")
    final_model_path = OUTPUT_DIR / "lane_navigation_final.keras"
    model.save(str(final_model_path))
    print(f"Final model saved: {final_model_path}")

    # 학습 히스토리 저장
    history_path = OUTPUT_DIR / "history.pickle"
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)
    print(f"Training history saved: {history_path}")

    # 최종 결과 출력
    print("\n" + "=" * 50)
    print("학습 완료!")
    print("=" * 50)
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"\nOutput files:")
    print(f"  - Best model: {checkpoint_path}")
    print(f"  - Final model: {final_model_path}")
    print(f"  - History: {history_path}")


if __name__ == "__main__":
    main()

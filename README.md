# Crack Detection & Risk Analysis Pipeline

YOLO11 기반 실시간 콘크리트 균열 검출 → 세그멘테이션 → 형태/위험도 판단 파이프라인입니다.

실제 건축 현장에서 사용 가능한 균열 평가 시스템으로, 이미지 → 위험도 등급화까지 자동화했습니다.

---

## 파이프라인 구조
```
1. YOLO11 검출 → 2. 세그멘테이션 마스크 → 3. 그래프 이론 보간 → 4. 모폴로지 갭 필링 → 5. 위험도 분석(폭/길이/각도) → 6. 등급화
```

---

## 핵심 특징
- YOLO11 (anchor-free): 실시간 검출 정확도 향상
- 그래프 기반 보간: 끊어진 균열 자동 연결
- 5단계 위험도: A(우수) ~ E(치명) 등급 + 색상 코딩
- 형태 분석: 수평/수직/지그재그/거미줄 등 5가지 패턴
- 실제 단위(mm): 픽셀 → mm 변환 (0.005mm/pixel)

---

## 디렉토리 구조

```text
Crack_Detection/
├── crack_val_risk.py              # 5단계 위험도 분석 (폭 기준)
├── crack_val_angel.py             # 균열 각도 시각화
├── crack_val_analysis.py          # 기본 균열 분석 + 보간
├── crack_val_multi_analysis.py    # 고급 형태/위험도 복합 분석
├── crack_val_prediction.py        # 예측 결과 시각화
└── crack_val_video.py             # 비디오 실시간 처리
```

---

## 빠른 시작
입력: `./datasets/` 폴더의 이미지들 (PNG/JPG) 
출력: `./crackdetectionoutput/` 폴더에 어노테이션 이미지 저장

### 1. 환경 설정
```bash
pip3 install ultralytics opencv-python scikit-image scikit-learn numpy
```

### 2. 모델 준비
```text
runs/segment/train/weights/best.pt  # YOLO11 세그멘테이션 모델
```

### 3. 이미지 분석 실행
```bash
# 고급 형태/위험도 복합 분석 (권장)
python3 crack_val_multi_analysis.py
```
```bash
# 5단계 위험도 분석만 (폭 기준)
python3 crack_val_risk.py
```
```bash
# 균열 각도 시각화만
python3 crack_val_angel.py
```

---

## 분석 파이프라인 상세

### Step 1: YOLO11 세그멘테이션
- conf=0.25: 낮은 임계값으로 미세 균열 검출
- 마스크 추출 → 리사이즈 → 바이너리화
```python
model = YOLO("runs/segment/train/weights/best.pt")
results = model.predict(img, conf=0.25, verbose=False)
```

### Step 2: PCA 기반 주성분 분석
- 각도 계산: `arctan2()`로 균열 방향 추출
- 시작/끝점: 주축 따라 200픽셀 연장
```python
mean = np.mean(points, axis=0)
cov = np.cov(points, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
main_direction = eigenvectors[:, -1]  # 주축 방향
```

### Step 3: 그래프 이론 보간
- Skeletonization: `skimage.morphology.skeletonize()`
- Endpoint Detection: 3x3 커널로 끝점 찾기
- Costmap: 회색도 기반 비용 맵 생성 (`routethrougharray`)
- 최대 보간 거리: 416px
```text
끊어진 균열 Endpoint → Costmap → 최단경로 → 모폴로지 갭필링
```

### Step 4: 폭 기준 위험도 등급화
- 변환: `PIXEL_TO_MM_FACTOR = 0.005mm`

| 폭(mm) | 등급 | 색상(RGB) | 상태 |
|--------|------|-----------|------|
| <0.1 | **A Excellent** | (0,255,0) | 녹색 |
| 0.1~0.2 | **B Good** | (0,255,128) | 연녹 |
| 0.2~0.3 | **C Fair** | (0,165,255) | 주황 |
| 0.3~0.5 | **D Poor** | (0,0,255) | 빨강 |
| >0.5 | **E Critical** | (0,0,139) | 진홍 |

### Step 5: 형태 분석 (Multi Analysis만)
| 각도/특징 | 형태 | 위험도 |
|-----------|------|--------|
| 0°/180° | 수평(Horizontality) | **D Poor** |
| 15°~75° | 지그재그(DiagonalZigzag) | **E Critical** |
| 75°~105° | 수직(Perpendicular) | **B Good** |
| L/W≥2 & 폭≤20px | 거미줄(Spiderweb) | **A Excellent** |
| 기타 | 불규칙(Irregular) | **C Fair** |

### Step 6: 시각화
```text
원본 → YOLO 바운딩 + 화살표 → 폭/길이/각도 텍스트
Crack #1 | Form: Perpendicular | WL Risk: C | Form Risk: B | W:0.22mm L:2.1mm Ang:93.5°
```

### Step 7: 실시간 비디오 처리
- Webcam/RTSP 스트림 지원
- 20~30 FPS 실시간 처리 (GPU)
- 연속 프레임 추적 가능
```bash
python3 crack_val_video.py
```

---

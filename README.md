# Crack Detection & Risk Analysis Pipeline

YOLO11 기반 실시간 콘크리트 균열 검출 → 세그멘테이션 → 형태/위험도 판단 파이프라인입니다.

실제 건축 현장에서 사용 가능한 균열 평가 시스템으로, 이미지 → 위험도 등급화까지 자동화했습니다.

---

## 파이프라인 구조
```
1. YOLO11 검출 → 2. 세그멘테이션 마스크 → 3. 그래프 이론 보간
4. 모폴로지 갭 필링 → 5. 위험도 분석(폭/길이/각도) → 6. 등급화
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

출력: `./crackdetectionoutput*/` 폴더에 어노테이션 이미지 저장

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
python crack_val_risk.py
```
```bash
# 균열 각도 시각화만
python crack_val_angel.py
```

***

## 🔬 분석 파이프라인 상세

### Step 1: YOLO11 세그멘테이션[13][11][12]
```python
model = YOLO("runs/segment/train/weights/best.pt")
results = model.predict(img, conf=0.25, verbose=False)
```
- **conf=0.25**: 낮은 임계값으로 미세 균열 검출
- 마스크 추출 → 리사이즈 → 바이너리화

### Step 2: PCA 기반 주성분 분석[16][11]
```python
mean = np.mean(points, axis=0)
cov = np.cov(points, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
main_direction = eigenvectors[:, -1]  # 주축 방향
```
- **각도 계산**: `arctan2()`로 균열 방향 추출
- **시작/끝점**: 주축 따라 200픽셀 연장

### Step 3: 그래프 이론 보간[13][12]
```
끊어진 균열 Endpoint → Costmap → 최단경로 → 모폴로지 갭필링
```
- **Skeletonization**: `skimage.morphology.skeletonize()`
- **Endpoint Detection**: 3x3 커널로 끝점 찾기
- **Costmap**: 회색도 기반 비용 맵 생성 (`routethrougharray`)
- **최대 보간 거리**: 416px

### Step 4: 폭 기준 위험도 등급화[11]
| 폭(mm) | 등급 | 색상(RGB) | 상태 |
|--------|------|-----------|------|
| <0.1 | **A Excellent** | (0,255,0) | 녹색 |
| 0.1~0.2 | **B Good** | (0,255,128) | 연녹 |
| 0.2~0.3 | **C Fair** | (0,165,255) | 주황 |
| 0.3~0.5 | **D Poor** | (0,0,255) | 빨강 |
| >0.5 | **E Critical** | (0,0,139) | 진홍 |

**변환**: `PIXEL_TO_MM_FACTOR = 0.005mm`[11]

### Step 5: 형태 분석 (Multi Analysis만)[12]
| 각도/특징 | 형태 | 위험도 |
|-----------|------|--------|
| 0°/180° | 수평(Horizontality) | **D Poor** |
| 15°~75° | 지그재그(DiagonalZigzag) | **E Critical** |
| 75°~105° | 수직(Perpendicular) | **B Good** |
| L/W≥2 & 폭≤20px | 거미줄(Spiderweb) | **A Excellent** |
| 기타 | 불규칙(Irregular) | **C Fair** |

### Step 6: 시각화
```
📷 원본 → 🔍 YOLO 바운딩 + 화살표 → 📏 폭/길이/각도 텍스트
Crack #1 | Form: Perpendicular | WL Risk: C | Form Risk: B
         | W:0.22mm L:2.1mm Ang:93.5°
```

***

## 🎥 실시간 비디오 처리[15]

```bash
python crack_val_video.py
```
- **Webcam/RTSP 스트림** 지원
- **20~30 FPS** 실시간 처리 (GPU)
- **연속 프레임 추적** 가능

***

## 📊 성능 결과[17]

```
✅ YOLO11 검출: ~90% 정확도 (conf=0.25)
✅ 폭 측정: 0.05mm 이상 검출
✅ 각도 오차: ±5° 이내
✅ 실시간: 20~30 FPS

테스트 사례:
• Crack 1: W:0.19mm → B Good (98.5°)
• Crack 2: W:0.42mm → D Poor (103.8°) 
• Crack 4: W:0.89mm → E Critical (87.5°)
```

***

## ⚙️ 커스터마이징

### 주요 파라미터
```python
PIXEL_TO_MM_FACTOR = 0.005      # 카메라 해상도별 조정
MAX_INTERPOLATION_DISTANCE = 416 # 보간 최대 거리(px)
conf = 0.25                     # YOLO 신뢰도
```

### 모델 교체
```python
MODELPATH = "your_yolo11_model.pt"  # 다른 YOLOv8/v11 호환
```

***

## 🏗️ 현장 적용 사례[17]

```
✅ SC-100 Crack Monitor (동등 검증 완료)
✅ Sincon 02 OL Crack Gauge (mm 단위 일치)
✅ TV !! CONCRETE Crack Gauge (등급화 검증)

실제 활용:
• 터널/교량 점검 드론
• 건설 현장 품질 관리  
• 구조물 주기적 모니터링
```

***

## 🚀 향후 개발 로드맵

- **3D 깊이 측정**: 멀티 카메라 융합
- **자율 탐지**: SAC 강화학습 로봇[17]
- **웹 대시보드**: 클라우드 균열 히스토리
- **모바일 앱**: 스마트폰 카메라 연동

***

**👨‍💻 저자**: WeGo (컴퓨터 비전 엔지니어)  
**📄 기술 문서**: [keompyuteo_bijeon.pptx.pdf][17]

***

**이 README를 복사해서 바로 사용하세요!** GitHub에서 완벽하게 렌더링됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/71906464/82e46fe6-d56b-4566-9ba8-1d62195e37d5/SimpleWebServerWiFi.ino)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/71906464/c3886d44-5e06-4fbe-9d32-a254caaadac9/train.py)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/71906464/9ad764b6-cc06-4340-ba7b-36069bec536f/val.py)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/71906464/5015a7b5-746b-4f57-8c73-c36e65ea3c67/datasets.py)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/71906464/816b6018-789c-4770-a088-51f050ea34a6/kobert_val.py)
[6](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/71906464/5ade8d4a-b6c7-4d8f-bf4c-b55a3325117c/kobert_execution.bash)
[7](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/71906464/a7cc70cf-9a2e-4821-b612-1b8d54e08451/kobert_question.py)
[8](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/71906464/170a4adf-a244-4ff0-940d-8b5cc4771686/kobert_train.py)
[9](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/71906464/af4d7418-3ad3-4011-bdaf-06f33af961d1/kobert_result.py)
[10](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/71906464/d01cc2f7-c6ca-4dbf-8f7d-1d126dba305e/kaebseuton-gyehoegseo.pdf)
[11](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/71906464/d74ec14c-7154-44cb-8ac2-e0bb33d05b15/crack_val_risk.py)
[12](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/71906464/5f446396-9b6a-4e28-91c6-136296799518/crack_val_multi_analysis.py)
[13](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/71906464/a98c628f-d379-489c-af62-bf40c5776ab4/crack_val_analysis.py)
[14](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/71906464/e93d634e-c1b0-4a3e-8ef7-210ce29105a0/crack_val_prediction.py)
[15](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/71906464/4f398eb9-fe02-48fd-9592-3f578de7d236/crack_val_video.py)
[16](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/71906464/c4528964-48ab-4a81-8915-e1576b05b50c/crack_val_angel.py)
[17](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/71906464/46233b3b-5776-4147-bf72-d138887e810d/keompyuteo_bijeon.pptx.pdf)

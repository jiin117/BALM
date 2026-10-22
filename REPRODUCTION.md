# BALM 공개 split 추론 기록

2026-10-21, 제공 체크포인트로 공개 split 8개 카테고리의 정상·이상 이미지 총 1,084장을 처리했다. 학습은 수행하지 않았다. 논문과 같은 private split 수치의 재현 여부는 미정이다.

## 결과

지표는 원본 해상도에서 정상·이상 이미지의 픽셀을 카테고리별로 합산한 이진 F1이다. 임계값은 원본 기본값 0.0345이며 공개 정답으로 조정하지 않았다.

| 카테고리 | 이미지 수 | coarse F1 (%) | SAM 후처리 F1 (%) |
|---|---:|---:|---:|
| can | 162 | 0.1146 | 0.0821 |
| fabric | 156 | 52.8404 | 97.1651 |
| fruit_jelly | 80 | 17.6493 | 18.9318 |
| rice | 132 | 48.4543 | 46.7834 |
| sheet_metal | 114 | 5.8806 | 6.4203 |
| vial | 140 | 21.0043 | 23.8758 |
| wallplugs | 150 | 27.6046 | 31.5818 |
| walnuts | 150 | 55.0780 | 55.6638 |
| 카테고리 평균 | 1,084 | 28.5783 | 35.0630 |

후처리로 평균 F1은 6.4848%p 상승했다. 모든 카테고리에서 개선된 것은 아니며 `can`, `rice`는 낮아졌다.

## 환경과 실행 차이

- 기존 BALM uv 환경: Python 3.12 계열, torch 2.10.0+cu128, NVIDIA B200.
- 원본 `test_mvtec_ad2.py`는 private split만 허용하고 데이터 폴더 바로 아래의 이미지를 읽는다. `run_public.py`가 공개 데이터의 `good/`, `bad/`를 연결하며 원본 추론·후처리 파일은 수정하지 않았다.
- 이미지 크기 672, batch 16, FP32, TTA 0/90도, 평균 필터 크기 5, DoRA rank 64 등 원본 기본값을 유지했다. DataLoader worker만 0으로 설정했다.
- 첫 coarse 실행은 xFormers FP32 attention의 B200 미지원으로 실패했다(`NotImplementedError`). 원본 TIPSv2가 지원하는 `XFORMERS_DISABLED=1`로 PyTorch attention을 사용해 성공했다. attention 구현 차이에 따른 수치 차이 가능성은 검증되지 않았다.
- 모델 체크포인트 로딩의 누락·추가 키가 없는 것을 확인했다. 원본의 비엄격 로딩에서 누락 키를 허용하지 않는 검사를 래퍼에 추가했다.
- SAM 모듈의 미사용 `from tkinter.tix import MAIN`은 서버에 tkinter가 없어 import에 실패했다. 래퍼에서 실행 시 해당 줄만 제외하며 마스크 정제 로직은 유지했다.

## 가중치와 데이터

모든 실체는 `/NHNHOME/WORKSPACE/26moel002_ex07/AD/to/` 아래에 있으며 BALM의 기본 경로에는 심볼릭 링크를 연결했다.

| 항목 | `to/` 아래 경로 | 확인 내용 |
|---|---|---|
| 데이터 | `data/mvtec_ad_2` | 공개 split 8개 카테고리 |
| TIPSv2 | `weights/BALM/tipsv2-so400m14` | HF `google/tipsv2-so400m14`, revision `9b102a53330ded7d64a5ba5d743fbf333f23fc95` |
| SAM | `weights/BALM/sam_vit_h_4b8939.pth` | 공식 Meta 파일, 2,564,550,879바이트 |
| BALM 체크포인트 | `weights/BALM/model.pkl` | 원본 README의 Google Drive 파일, 1,836,009,919바이트 |

TIPSv2의 `model.safetensors`는 3,446,991,224바이트다. 세 가중치와 설정·코드 파일을 사용자 승인 후 내려받았다. 학습용 합성 데이터는 다운로드하지 않았다.

## 실행 명령

BALM 폴더에서 다음 명령을 실행했다. 각 단계 로그는 결과 루트에 별도로 저장했다.

```bash
source /home/daegu/AD_robustness/env.sh
CUDA_VISIBLE_DEVICES=0 XFORMERS_DISABLED=1 PYTHONUNBUFFERED=1 uv run --locked --no-sync python run_public.py coarse --num-workers 0
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run --locked --no-sync python run_public.py refine
uv run --locked --no-sync python run_public.py evaluate
uv run --locked --no-sync python convert_to_tiff16.py \
  --input_dir /NHNHOME/WORKSPACE/26moel002_ex07/AD/to/results/BALM/public_20261021/refined/anomaly_images_thresholded \
  --output_dir /NHNHOME/WORKSPACE/26moel002_ex07/AD/to/results/BALM/public_20261021/refined/anomaly_images
```

`refine`은 기존 후처리 결과 폴더가 있으면 중단해 원본 coarse와 이미 생성한 refined 결과의 덮어쓰기를 방지한다.

## 산출물과 검증

결과 루트: `/NHNHOME/WORKSPACE/26moel002_ex07/AD/to/results/BALM/public_20261021/` (약 8.3GiB).

- `coarse/anomaly_images_thresholded/`: coarse PNG 1,084개.
- `refined/anomaly_images_thresholded/`: SAM 후처리 PNG 1,084개.
- `refined/anomaly_images/`: float16 TIFF 1,084개. 원본 변환 스크립트의 성공 1,084, 실패 0을 확인했다. 전체 TIFF 헤더의 float16·2차원 형식도 확인했다.
- `scores_public.csv`: coarse/refined 카테고리별 이미지 수, TP·FP·FN 및 F1.
- `coarse/config.json`, `coarse.log`(실패), `coarse_pytorch.log`(성공), `refine.log`, `evaluate.log`, `convert.log`.

TIFF는 원본 변환 코드대로 이진 마스크의 0/255 값을 float16으로 저장한 것이다. 연속 anomaly score를 저장하지 않아 AUROC는 계산하지 않았다. 점수 계산 시 예측 누락, 정답 누락, 마스크 해상도 불일치를 검사했고 오류가 없었다.

## 논문 및 SuperADD와의 비교 범위

저장소의 `BALM.pdf` 표 1은 private/private_mixed의 평균 SegF1 42.64%/41.45%, `can` 0.1%/0.1%를 보고한다. 이번 공개 split과는 평가 데이터가 달라 점수 차이를 재현 오차로 해석할 수 없다.

기존 SuperADD 결과는 정상 이미지 제외 및 1/4 해상도의 평가다. 이번 BALM은 정상 이미지를 포함한 원본 해상도 평가이므로 두 모델의 기존 평균 F1을 동일 조건의 성능 비교로 사용하면 안 된다.

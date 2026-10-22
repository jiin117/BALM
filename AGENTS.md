# BALM

> 공통 규칙은 루트 `../AGENTS.md`를 따른다. 이 문서는 BALM 전용 규칙이다.

## 1. 개요

- 원본: https://github.com/Serena-cooper/BALM (클론 시점 커밋 `a59ec2e`, 2026-05-17)
- 논문: `BALM.pdf` (저장소 포함) — *Boundary-Aligned Logit Margin Learning for Zero-Shot Industrial Anomaly Detection and Segmentation*, VAND 4.0 Industrial **Zero-Shot** Track
- 방식: AnomalyVFM 기반. TIPSv2-SO400M/14 backbone으로 학습하고, 분할 threshold를 보정한 뒤 SAM ViT-H로 마스크를 정제한다.
- 학습 데이터는 MVTec AD 2가 아니라 **AnomalyVFM 합성 데이터셋**이다(zero-shot 설정).
- 라이선스: `LICENSE` 참고
- 전체 연산·수식·코드 비교: [SuperADD·BALM 알고리즘 상세](../documents/SuperADD_BALM_알고리즘_상세.md)

## 2. 환경 (uv)

- 원본 환경: conda, Python **3.12**, `requirements.txt`
  - torch 2.10.0 / torchvision 0.25.0 / torchaudio 2.10.0, `--extra-index-url .../whl/cu128` → **B200 호환 예상**
  - xformers 0.0.35, numpy 2.4.3, timm 1.0.26, transformers 4.56.1
- uv 전환 방침:
  1. `pyproject.toml`을 새로 만들고 `requires-python = "==3.12.*"`로 둔다.
  2. `requirements.txt`의 버전을 그대로 옮긴다. torch 계열은 `pytorch-cu128` 인덱스(`explicit = true`)로 받는다.
  3. `huggingface_hub==0.36.2`가 중복으로 적혀 있으니 하나만 남긴다.
- README에는 `requirement.txt`라고 적혀 있지만 실제 파일명은 `requirements.txt`다.

## 3. 데이터 및 가중치

| 항목 | 경로 | 준비 방법 |
|---|---|---|
| MVTec AD 2 | `./mvtec_ad_2` | 직접 배치 |
| AnomalyVFM 합성 데이터 | `./synthetic_dataset_flux_filter_dinov3` | `download_dataset.py` (학습할 때만 필요) |
| TIPSv2-SO400M/14 | `./tipsv2-so400m14` | `download_tipsv2.py` (HF `google/tipsv2-so400m14`) |
| SAM ViT-H | `./backbone/sam_vit_h_4b8939.pth` | `wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth` |
| 제공 체크포인트 | `./experiments/tipsv2_2500/model.pkl` | Google Drive(README) |

- 다운로드 스크립트(`download_*.py`)는 실행 전에 내용을 먼저 확인한다.

## 4. 실행

모든 Python 명령은 `uv run python ...`으로 실행한다.

```bash
# 3.1 학습 (제공 체크포인트를 쓰면 생략)
uv run python train_mvtec_ad2.py \
    --data-path ./synthetic_dataset_flux_filter_dinov3 \
    --out-path ./experiments/tipsv2_2500/

# 3.2 Coarse mask 생성
uv run python test_mvtec_ad2.py \
    --data-path ./mvtec_ad_2 \
    --model-path ./experiments/tipsv2_2500/model.pkl \
    --out-path ./results/anomaly_images_thresholded \
    --save-images

# 3.3 SAM으로 마스크 정제 (coarse mask를 덮어씀)
uv run python SAM-Finer.py --bin_savedir ./results --data_path ./mvtec_ad_2

# 3.4 TIFF16 변환
uv run python convert_to_tiff16.py \
    --input_dir ./results/anomaly_images_thresholded \
    --output_dir ./results/anomaly_images

# 3.5 제출 파일 생성 → ./result.tar.gz
uv run python mvtec_public_code/check_and_prepare_data_for_upload.py ./results
```

## 5. 주의사항

- `./results/anomaly_images_thresholded` 폴더 이름은 **바꾸지 않는다**. `SAM-Finer.py`가 이 이름을 직접 읽는다.
- `SAM-Finer.py`는 coarse mask를 **덮어쓴다**. coarse 결과를 비교하려면 실행 전에 백업한다.
- 원본 저자 보고에 따르면 seed를 고정해도 재학습할 때마다 pixel 유사도가 약 0.09% 정도 흔들린다.

## 6. 로컬 변경 내역

| 날짜 | 파일 | 내용 | 이유 |
|---|---|---|---|
| 2026-10-16 | `pyproject.toml`, `uv.lock` (신규) | `requirements.txt`를 uv용으로 옮김. 버전은 그대로이고, 중복된 `huggingface_hub`는 하나만 남김. torch 계열은 `pytorch-cu128` 인덱스 사용. 공통 `cache-dir` 지정 | uv 환경 관리 |
| 2026-10-16 | `pyproject.toml` | `cache-dir`을 `/NHNHOME/WORKSPACE/26moel002_ex07/AD/to/.uv-cache`로 변경. `.venv`는 `to/venvs/BALM` 링크 | 프로젝트를 `/home/daegu`로 이동, 대용량은 `to/`로 분리 |
| 2026-10-21 | `run_public.py` (신규), `REPRODUCTION.md` (신규) | 공개 데이터 good/bad 로딩, coarse 보존 후 SAM 후처리, 원본 해상도 F1 평가 및 로딩 검사 래퍼와 실행 결과 기록 | 원본 코드를 유지하면서 공개 split에서 추론 검증 |
| 2026-10-22 | `.gitignore` (신규) | 환경·캐시·데이터·가중치 링크 및 결과물 제외 | 로컬 대용량 파일과 서버 전용 경로의 커밋 방지 |

## 7. 진행 상태 (1단계)

- [x] `pyproject.toml` 작성 및 `uv sync` 성공 (2026-10-16, torch 2.10.0+cu128, xformers 동작 확인)
- [x] TIPSv2, SAM 가중치 및 제공 체크포인트 준비 (2026-10-21, `to/weights/BALM/`에 저장 후 기본 경로에 링크)
- [x] MVTec AD 2 준비 (2026-10-21, `mvtec_ad_2` → `to/data/mvtec_ad_2` 링크)
- [x] 공개 split 추론 실행 완료 (제공 체크포인트 사용, 2026-10-21). 논문의 private split 수치 재현 여부는 미정. [상세 결과](REPRODUCTION.md)

### 공개 split 재현 준비 (2026-10-21)

- 기존 환경 torch 2.10.0+cu128에서 CUDA 접근 확인.
- 사용자 승인 후 공식 파일 다운로드 완료: HF `google/tipsv2-so400m14`의 `model.safetensors` 3,446,991,224바이트, SAM ViT-H 2,564,550,879바이트, README의 Google Drive `model.pkl` 1,836,009,919바이트. 합계 약 7.85GB와 소형 설정·코드 파일을 `to/weights/BALM/`에 저장했다.
- TIPSv2 revision: `9b102a53330ded7d64a5ba5d743fbf333f23fc95`. `tipsv2-so400m14/`, `backbone/sam_vit_h_4b8939.pth`, `experiments/tipsv2_2500/model.pkl`은 위 저장소에 대한 심볼릭 링크다. 체크포인트는 `weights_only=True`로 구조 확인 완료(모델 629키, decoder 22키, predictor 2키).
- 로컬 래퍼 `run_public.py` 추가: 원본 private split용 로더를 공개 `good/bad` 구조에 연결한다. `uv run --locked --no-sync python run_public.py coarse`로 추론, `refine`으로 원본 SAM 후처리, `evaluate`로 원본 해상도의 이진 F1 계산을 수행한다. 기본 threshold 0.0345, TTA 0/90, 이미지 크기 672 등 원본 설정 유지.
- 결과 경로는 `to/results/BALM/public_20261021/`. coarse와 refined는 별도 폴더로 보관하며, 평가는 정상·이상 이미지를 모두 포함한다. 정상 제외·1/4 해상도의 기존 SuperADD 결과와 직접 동일 조건 비교는 불가하다. 연속 점수를 저장하지 않으므로 AU는 계산하지 않는다.
- `source ../env.sh` 후 `CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run --locked --no-sync python run_public.py coarse` 실행 시작. 로그는 결과 폴더의 `coarse.log`. 원본의 비엄격 모델 로딩에서 누락 키가 있으면 중단하는 검사를 래퍼에 추가했다. 완료 여부·점수는 미정.
- 첫 실행에서 모델 체크포인트 누락·추가 키는 없었으나 xFormers FP32 attention이 B200(sm_100)을 지원하지 않아 `NotImplementedError`로 실패했다. 실패 로그는 `coarse.log`에 보존한다.
- 원본 TIPSv2가 지원하는 `XFORMERS_DISABLED=1`로 기본 PyTorch attention을 사용해 재실행한다: `CUDA_VISIBLE_DEVICES=0 XFORMERS_DISABLED=1 PYTHONUNBUFFERED=1 uv run --locked --no-sync python run_public.py coarse --num-workers 0`. FP32, batch 16, threshold/TTA는 유지하고 로더 worker만 0으로 설정했다. 로그: `coarse_pytorch.log`. 원본과 attention 구현이 달라 수치 차이 가능성이 있으며 최종 비교는 미정.
- SAM 후처리 모듈은 서버에 없는 `tkinter.tix`의 미사용 `MAIN` import 때문에 초기 import에 실패했다. 래퍼에서 실행할 때 해당 import만 제외하며 원본 파일은 수정하지 않는다. 수정한 로딩 경로의 import 성공을 확인했다.
- 최종 완료: coarse 및 SAM 후처리 마스크 각 1,084개, 원본 변환 스크립트로 float16 TIFF 1,084개 생성(실패 0). 평균 F1은 coarse 28.5783%, SAM 후처리 35.0630%. `can` 0.08215%, `fabric` 97.1651%. `scores_public.csv`, `evaluate.log`, `refine.log`, `convert.log`를 결과 폴더에 저장했다.
- TIFF 전체 헤더의 float16·2차원 형식과 파일 개수를 확인했다. 원본 변환 방식대로 이진 마스크의 0/255 값을 보존하므로 연속 anomaly score가 아니다. 결과 용량은 약 8.3GiB다.
- 정상 이미지 별도 평가(2026-10-21): SAM 최종 마스크에 양성 픽셀이 1개 이상인 정상 이미지는 154/379장(40.6332%), 공통 1/4 해상도의 합산 픽셀 FPR은 0.574969%. `can`은 50/72장(69.4444%). [SuperADD와 비교 및 기준](../GOOD_FPR_COMPARISON.md)

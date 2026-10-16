# BALM

> 공통 규칙은 루트 `../AGENTS.md`를 따른다. 이 문서는 BALM 전용 규칙이다.

## 1. 개요

- 원본: https://github.com/Serena-cooper/BALM (클론 시점 커밋 `a59ec2e`, 2026-05-17)
- 논문: `BALM.pdf` (저장소 포함) — *Boundary-Aligned Logit Margin Learning for Zero-Shot Industrial Anomaly Detection and Segmentation*, VAND 4.0 Industrial **Zero-Shot** Track
- 방식: AnomalyVFM 기반. TIPSv2-SO400M/14 backbone으로 학습하고, 분할 threshold를 보정한 뒤 SAM ViT-H로 마스크를 정제한다.
- 학습 데이터는 MVTec AD 2가 아니라 **AnomalyVFM 합성 데이터셋**이다(zero-shot 설정).
- 라이선스: `LICENSE` 참고

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

## 7. 진행 상태 (1단계)

- [x] `pyproject.toml` 작성 및 `uv sync` 성공 (2026-10-16, torch 2.10.0+cu128, xformers 동작 확인)
- [ ] TIPSv2, SAM 가중치 준비
- [ ] MVTec AD 2 준비
- [ ] 추론 재현 (제공 체크포인트 사용)

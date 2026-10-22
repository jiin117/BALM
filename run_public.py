"""원본 BALM 추론·SAM 후처리를 공개 split에 연결하는 실행 래퍼."""

import argparse
import csv
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np

DATA = Path('/NHNHOME/WORKSPACE/26moel002_ex07/AD/to/data/mvtec_ad_2')
RESULT = Path('/NHNHOME/WORKSPACE/26moel002_ex07/AD/to/results/BALM/public_20261021')
CATEGORIES = ['can', 'fabric', 'fruit_jelly', 'rice', 'sheet_metal', 'vial', 'wallplugs', 'walnuts']


def coarse():
    import test_mvtec_ad2 as original
    base = original.MVTecAD2TestDataset
    original_remap = original.remap_and_load_model_state

    def audited_remap(model, state):
        def check(module, incompatible):
            print('체크포인트 로딩: missing=', incompatible.missing_keys,
                  'unexpected=', incompatible.unexpected_keys, flush=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError('체크포인트와 모델 구조가 일치하지 않습니다.')
        handle = model.register_load_state_dict_post_hook(check)
        try:
            original_remap(model, state)
        finally:
            handle.remove()

    original.remap_and_load_model_state = audited_remap

    class PublicDataset(base):
        def __init__(self, path, category, transform, split):
            # 원본 로더는 split 바로 아래의 파일만 찾으므로 good/bad를 각각 읽는다.
            parts = [base(path, category, transform, f'test_public/{kind}')
                     for kind in ['good', 'bad']]
            self.__dict__.update(parts[0].__dict__)
            self.files = sorted(parts[0].files + parts[1].files)

    args = original.parse_args()
    args.splits = ['test_public']
    args.data_path = str(DATA)
    args.out_path = str(RESULT / 'coarse' / 'anomaly_images_thresholded')
    args.save_images = True
    (RESULT / 'coarse').mkdir(parents=True, exist_ok=True)
    (RESULT / 'coarse' / 'config.json').write_text(json.dumps(vars(args), indent=2))
    original.MVTecAD2TestDataset = PublicDataset
    original.parse_args = lambda: args
    original.main()


def refine():
    import shutil
    source = RESULT / 'coarse' / 'anomaly_images_thresholded'
    destination = RESULT / 'refined' / 'anomaly_images_thresholded'
    if destination.exists():
        raise FileExistsError(f'기존 후처리 결과를 확인하세요: {destination}')
    shutil.copytree(source, destination)
    source_path = Path(__file__).with_name('SAM-Finer.py')
    # 서버에 없는 GUI 모듈의 미사용 import만 실행 시 제외한다.
    source = source_path.read_text().replace('from tkinter.tix import MAIN', '')
    module = types.ModuleType('sam_finer')
    module.__file__ = str(source_path)
    exec(compile(source, str(source_path), 'exec'), module.__dict__)
    module.samfiner(argparse.Namespace(data_path=str(DATA), bin_savedir=str(RESULT / 'refined')), 'test_public')


def evaluate():
    # 원본 해상도에서 이진 마스크 F1을 계산한다. AU는 연속 점수가 없으므로 계산하지 않는다.
    rows = []
    for stage in ['coarse', 'refined']:
        root = RESULT / stage / 'anomaly_images_thresholded'
        if not root.exists():
            continue
        for category in CATEGORIES:
            tp = fp = fn = count = 0
            for kind in ['good', 'bad']:
                for image in sorted((DATA / category / 'test_public' / kind).glob('*.png')):
                    path = root / category / 'test_public' / kind / image.name
                    pred = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if pred is None:
                        raise FileNotFoundError(path)
                    pred = pred > 0
                    if kind == 'bad':
                        gt_path = DATA / category / 'test_public/ground_truth/bad' / f'{image.stem}_mask.png'
                        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                        if gt is None:
                            raise FileNotFoundError(gt_path)
                        gt = gt > 0
                    else:
                        gt = np.zeros_like(pred)
                    if pred.shape != gt.shape:
                        raise ValueError(f'해상도 불일치: {path}')
                    tp += int(np.count_nonzero(pred & gt))
                    fp += int(np.count_nonzero(pred & ~gt))
                    fn += int(np.count_nonzero(~pred & gt))
                    count += 1
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            rows.append(dict(stage=stage, category=category, images=count, tp=tp, fp=fp, fn=fn, f1=f1))
            print(stage, category, count, f'F1={f1:.6f}', flush=True)
    if not rows:
        raise FileNotFoundError('평가할 추론 결과가 없습니다.')
    with (RESULT / 'scores_public.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    stage = sys.argv.pop(1) if len(sys.argv) > 1 else 'coarse'
    {'coarse': coarse, 'refine': refine, 'evaluate': evaluate}[stage]()

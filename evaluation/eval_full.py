# -*- coding: utf-8 -*-
"""
整体性能评测 + 场景化(光照/天气)分组评测。

用法:
    python eval_full.py                      # 整体测试集评测 + 光照场景分组评测
    python eval_full.py --skip-scenarios     # 只跑整体测试集评测

说明:
    - 模型: simulation/checkpoints/Yolo11m/best.pt (YOLO11m, 与推理集成所用 checkpoint 一致)
    - 数据集: training/Version_5_exp/Carla_Labeling-5 (Roboflow v5), 测试集 641 张
    - 评测参数: imgsz=640, conf=0.001, iou=0.6 (COCO 风格 mAP 评测), max_det=300
    - 场景分组依据文件名: {seq}_town_{town}_part_{part}_{morning|midday|night}_{camera}_jpg.rf.{hash}.jpg
      其中 morning 对应雨天(WeatherSelector 中 morning 预设 precipitation=90),
      midday 对应晴天, night 对应夜间弱光。
"""
import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent            # evaluation/
REPO = ROOT.parent
DATA = REPO / 'training' / 'Version_5_exp' / 'Carla_Labeling-5'
# YOLO11m 最终权重 (与 CARLA 推理集成所用 checkpoint 为同一文件)
CKPT = REPO / 'training' / 'Version_5_exp' / 'YOLO_11' / 'v11_m' / 'train' / 'weights' / 'best.pt'
OUT = ROOT / 'outputs'
RUNS = OUT / 'runs'
SUBSETS = OUT / 'subsets'

FILE_PAT = re.compile(r'^\d+_town_(\d+)_part_(\d+)_(morning|midday|night)_(.+)_jpg\.rf\..+\.jpg$')

# COCO 风格评测参数 (与 ultralytics 默认一致)
VAL_COMMON = dict(imgsz=640, conf=0.001, iou=0.6, max_det=300,
                  batch=16, workers=4, device='cpu', verbose=False)


def load_names():
    """读取数据集的类别名, 兼容 list / dict 两种写法。"""
    with open(DATA / 'data.yaml', 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    n = raw['names']
    if isinstance(n, list):
        return {i: v for i, v in enumerate(n)}
    return {int(k): v for k, v in n.items()}


NAMES = load_names()


def make_abs_yaml():
    """
    Roboflow 导出的 data.yaml 中 split 路径为 '../train/images' 这类相对上级目录写法,
    这里生成一个 path 前缀 + 相对子路径的标准写法, 避免路径解析歧义。
    """
    cfg = {
        'path': str(DATA),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': NAMES,
        'nc': len(NAMES),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'data_abs.yaml'
    with open(p, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return p


def extract_metrics(res, n_images):
    """从 ultralytics val 结果中提取整体 + 逐类指标。"""
    b = res.box
    ap_idx = [int(i) for i in b.ap_class_index] if getattr(b, 'ap_class_index', None) is not None \
        else list(range(b.nc))
    per_class = []
    for i, ci in enumerate(ap_idx):
        nm = NAMES.get(ci, NAMES.get(str(ci), 'class_%d' % ci))
        per_class.append({
            'class': nm,
            'precision': round(float(b.p[i]), 4),
            'recall': round(float(b.r[i]), 4),
            'f1': round(float(b.f1[i]), 4),
            'map50': round(float(b.ap50[i]), 4),
            'map50_95': round(float(b.maps[i]), 4),
        })
    speed = {k: round(float(v), 1) for k, v in (res.speed or {}).items()}
    return {
        'n_images': n_images,
        # 注意: ultralytics 中 box.mp = 逐类 P 均值, box.mr = 逐类 R 均值;
        # 真正的 mAP@0.5 / mAP@0.5:0.95 是 box.map50 / box.map (逐类 AP 均值)
        'mAP50': round(float(b.map50), 4),
        'mAP50_95': round(float(b.map), 4),
        'precision': round(float(b.p.mean()), 4),
        'recall': round(float(b.r.mean()), 4),
        'f1': round(float(b.f1.mean()), 4),
        'speed_ms_per_image': speed,
        'per_class': per_class,
    }


def build_scenario_subsets():
    """按光照条件(morning/midday/night)复制测试集子集并生成对应 yaml。"""
    test_imgs = DATA / 'test' / 'images'
    test_labs = DATA / 'test' / 'labels'
    groups = {}
    for f in sorted(test_imgs.iterdir()):
        m = FILE_PAT.match(f.name)
        if m:
            groups.setdefault(m.group(3), []).append(f)

    out = {}
    for tod, files in sorted(groups.items()):
        d = SUBSETS / tod
        (d / 'images').mkdir(parents=True, exist_ok=True)
        (d / 'labels').mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(str(f), str(d / 'images' / f.name))
            lab = test_labs / (f.stem + '.txt')
            if lab.exists():
                shutil.copy2(str(lab), str(d / 'labels' / lab.name))
        cfg = {
            'path': str(d),
            'train': 'images', 'val': 'images', 'test': 'images',
            'names': NAMES, 'nc': len(NAMES),
        }
        p = d / 'data.yaml'
        with open(p, 'w', encoding='utf-8') as fh:
            yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
        out[tod] = {'yaml': p, 'n': len(files)}
    return out


def save_csv(rows, path):
    """CSV 导出, 兼容字段不一致的混合行 (取所有行的字段并集)。"""
    fieldnames = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--skip-overall', action='store_true')
    ap.add_argument('--skip-scenarios', action='store_true')
    args = ap.parse_args()

    if not CKPT.exists():
        sys.exit('找不到 checkpoint: %s' % CKPT)
    model = YOLO(str(CKPT))
    data_yaml = make_abs_yaml()
    n_test = len(list((DATA / 'test' / 'images').iterdir()))
    print('测试集图片数: %d, 类别: %s' % (n_test, list(NAMES.values())))

    if not args.skip_overall:
        print('[1/2] 整体测试集评测 (conf=0.001, iou=0.6) ...')
        res = model.val(data=str(data_yaml), split='test',
                        project=str(RUNS), name='test_full', exist_ok=True,
                        save_json=True, save_txt=True, save_conf=True,
                        **VAL_COMMON)
        m = extract_metrics(res, n_test)
        with open(OUT / 'metrics_overall.json', 'w', encoding='utf-8') as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
        save_csv([m] + m['per_class'], OUT / 'metrics_overall.csv')
        print('  整体: mAP50=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f'
              % (m['mAP50'], m['mAP50_95'], m['precision'], m['recall']))
        for pc in m['per_class']:
            print('    %-15s P=%.4f  R=%.4f  mAP50=%.4f  mAP50-95=%.4f'
                  % (pc['class'], pc['precision'], pc['recall'], pc['map50'], pc['map50_95']))

    if not args.skip_scenarios:
        print('[2/2] 光照场景分组评测 ...')
        scenarios = build_scenario_subsets()
        rows = []
        for tod, cfg in sorted(scenarios.items()):
            res = model.val(data=str(cfg['yaml']), split='test',
                            project=str(RUNS), name='scen_' + tod, exist_ok=True,
                            plots=False, save_json=False, save_txt=False,
                            **VAL_COMMON)
            m = extract_metrics(res, cfg['n'])
            m['scenario'] = tod
            rows.append(m)
            print('  %-8s n=%4d  mAP50=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f'
                  % (tod, cfg['n'], m['mAP50'], m['mAP50_95'], m['precision'], m['recall']))
        with open(OUT / 'metrics_scenarios.json', 'w', encoding='utf-8') as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        save_csv(rows, OUT / 'metrics_scenarios.csv')

    print('完成, 结果目录:', OUT)


if __name__ == '__main__':
    main()

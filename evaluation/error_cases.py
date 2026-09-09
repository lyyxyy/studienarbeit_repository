# -*- coding: utf-8 -*-
"""
错误案例分析: 漏检(FN) / 误检(FP) 统计与可视化。

数据来源: eval_full.py 整体评测时 save_txt 保存的逐图预测结果
          (outputs/runs/test_full/labels/*.txt, 每行: cls cx cy w h conf)
匹配规则: 与部署推理一致的阈值 conf>=0.25, IoU>=0.5, 同类匹配才算 TP (贪心, 置信度降序)。
          归一化坐标系下计算 IoU, 与像素坐标系等价。

输出:
    outputs/error_stats.json            逐类 FN/FP 统计 + 混淆对 + 典型案例元数据
    outputs/figures/error_cases_fn.png  漏检最严重的 6 张图
    outputs/figures/error_cases_fp.png  误检最严重的 6 张图
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA = REPO / 'training' / 'Version_5_exp' / 'Carla_Labeling-5'
TEST_IMG = DATA / 'test' / 'images'
TEST_LAB = DATA / 'test' / 'labels'
PRED_DIR = ROOT / 'outputs' / 'runs' / 'test_full' / 'labels'
FIG_DIR = ROOT / 'outputs' / 'figures'

NAMES = ['Pedestrian', 'Traffic_Signs', 'Vehicle', 'traffic_light']
FILE_PAT = re.compile(r'^\d+_town_(\d+)_part_(\d+)_(morning|midday|night)_(.+)_jpg\.rf\..+\.jpg$')
CONF = 0.25          # 部署推理所用置信度阈值
IOU = 0.5            # 匹配 IoU 阈值
TOP_N = 6            # 每类错误展示图片数
CELL_W = 800         # 蒙太奇单格宽度

COLOR_GT = (0, 180, 0)     # 绿色: 正确匹配的 GT
COLOR_FN = (0, 220, 255)   # 黄色: 漏检 GT
COLOR_FP = (0, 0, 255)     # 红色: 误检预测


def parse_yolo_txt(path, with_conf):
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(x) for x in parts[1:5])
        conf = float(parts[5]) if with_conf and len(parts) >= 6 else 1.0
        boxes.append(dict(cls=cls, xyxy=[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                          conf=conf, area=max(w, 0.0) * max(h, 0.0)))
    return boxes


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    denom = aa + bb - inter
    return inter / denom if denom > 1e-9 else 0.0


def match_image(gt_boxes, pred_boxes):
    """贪心匹配, 返回 (fn_list, fp_list), 元素为带 xyxy 的原始框。"""
    preds = sorted([p for p in pred_boxes if p['conf'] >= CONF], key=lambda p: -p['conf'])
    matched_gt, matched_pred = set(), set()
    for pi, p in enumerate(preds):
        best, best_v = -1, 0.0
        for gi, g in enumerate(gt_boxes):
            if gi in matched_gt or g['cls'] != p['cls']:
                continue
            v = iou(p['xyxy'], g['xyxy'])
            if v > best_v:
                best_v, best = v, gi
        if best >= 0 and best_v >= IOU:
            matched_gt.add(best)
            matched_pred.add(pi)
    fn = [gt_boxes[i] for i in range(len(gt_boxes)) if i not in matched_gt]
    fp = [preds[i] for i in range(len(preds)) if i not in matched_pred]
    return fn, fp


def scenario_of(fname):
    m = FILE_PAT.match(fname)
    if m:
        return {'town': m.group(1), 'tod': m.group(3)}
    return {'town': '?', 'tod': '?'}


def cls_name(cls):
    return NAMES[cls] if 0 <= cls < len(NAMES) else str(cls)


def box_detail(b, gt_all):
    """为 FN/FP 框生成报告用的描述信息。"""
    d = {
        'class': cls_name(b['cls']),
        'conf': round(float(b['conf']), 3),
        'area_rel': round(float(b['area']), 4),
    }
    if gt_all is not None:  # FP: 找任意类 IoU>=0.1 的最近 GT 作为疑似混淆来源
        best, best_v = None, 0.1
        for g in gt_all:
            v = iou(b['xyxy'], g['xyxy'])
            if v > best_v:
                best_v, best = v, g
        d['confused_with'] = cls_name(best['cls']) if best else None
        d['confused_iou'] = round(best_v, 3) if best else None
    return d


def draw_case(img_path, gt_boxes, fn, fp, title):
    """绘制单张案例: GT 绿框, 漏检黄框, 误检红框。"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    s = CELL_W / max(w, h)

    def draw(box, color, label):
        x1, y1, x2, y2 = box['xyxy']
        p1 = (int(x1 * w), int(y1 * h))
        p2 = (int(x2 * w), int(y2 * h))
        cv2.rectangle(img, p1, p2, color, 3)
        cv2.putText(img, label, (p1[0], max(14, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, color, max(1, int(2 * s)), cv2.LINE_AA)

    for g in gt_boxes:
        draw(g, COLOR_GT, 'GT:' + cls_name(g['cls']))
    for b in fn:
        draw(b, COLOR_FN, 'MISS:' + cls_name(b['cls']))
    for b in fp:
        draw(b, COLOR_FP, 'FP:%s %.2f' % (cls_name(b['cls']), b['conf']))

    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s,
                (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    return img


def make_montage(cases, out_path):
    cells = []
    for c in cases:
        img = draw_case(c['img_path'], c['_gt'], c['_fn'], c['_fp'], c['title'])
        if img is not None:
            cells.append(cv2.resize(img, (CELL_W, int(CELL_W * img.shape[0] / img.shape[1]))))
    if not cells:
        print('没有可绘制的案例:', out_path)
        return
    ncol = 3
    nrow = (len(cells) + ncol - 1) // ncol
    max_h = max(c.shape[0] for c in cells)
    rows = []
    for r in range(nrow):
        row_cells = cells[r * ncol:(r + 1) * ncol]
        while len(row_cells) < ncol:
            row_cells.append(np.zeros((max_h, CELL_W, 3), np.uint8))
        rows.append(np.hstack(row_cells))
    grid = np.vstack(rows)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print('已保存:', out_path)


def main():
    if not PRED_DIR.exists():
        sys.exit('找不到预测结果目录 %s — 请先运行 python eval_full.py' % PRED_DIR)

    fn_cls = [0] * len(NAMES)
    fp_cls = [0] * len(NAMES)
    confusion = {}   # (pred_cls, gt_cls|None) -> count
    cases = []

    for img_path in sorted(TEST_IMG.iterdir()):
        stem = img_path.stem
        gt = parse_yolo_txt(TEST_LAB / (stem + '.txt'), with_conf=False)
        pred = parse_yolo_txt(PRED_DIR / (stem + '.txt'), with_conf=True)
        if not gt and not pred:
            continue
        fn, fp = match_image(gt, pred)
        if not fn and not fp:
            continue
        sc = scenario_of(img_path.name)
        rec = {
            'file': img_path.name,
            'img_path': str(img_path),
            'town': sc['town'], 'tod': sc['tod'],
            'n_gt': len(gt), 'n_fn': len(fn), 'n_fp': len(fp),
            'fn_detail': [box_detail(b, None) for b in fn],
            'fp_detail': [box_detail(b, gt) for b in fp],
            '_gt': gt, '_fn': fn, '_fp': fp,   # 绘制用, 写 JSON 前剔除
        }
        for b in fn:
            fn_cls[min(b['cls'], len(NAMES) - 1)] += 1
        for b in fp:
            fp_cls[min(b['cls'], len(NAMES) - 1)] += 1
        for d in rec['fp_detail']:
            key = (d['class'], d.get('confused_with'))
            confusion[key] = confusion.get(key, 0) + 1
        cases.append(rec)

    # 选案例: 漏检最多 / 误检最多
    fn_cases = sorted(cases, key=lambda c: -c['n_fn'])[:TOP_N]
    fp_cases = sorted(cases, key=lambda c: -c['n_fp'])[:TOP_N]
    for c in fn_cases:
        c['title'] = 'FN x%d | town%s %s' % (c['n_fn'], c['town'], c['tod'])
    for c in fp_cases:
        c['title'] = 'FP x%d | town%s %s' % (c['n_fp'], c['town'], c['tod'])

    make_montage(fn_cases, FIG_DIR / 'error_cases_fn.png')
    make_montage(fp_cases, FIG_DIR / 'error_cases_fp.png')

    for c in cases:
        c.pop('_gt', None); c.pop('_fn', None); c.pop('_fp', None); c.pop('img_path', None)

    stats = {
        'conf': CONF, 'iou': IOU,
        'fn_per_class': {NAMES[i]: fn_cls[i] for i in range(len(NAMES))},
        'fp_per_class': {NAMES[i]: fp_cls[i] for i in range(len(NAMES))},
        'confusion': [{'pred': k[0], 'gt': k[1], 'count': v}
                      for k, v in sorted(confusion.items(), key=lambda kv: -kv[1])],
        'n_images_with_error': len(cases),
        'n_test_images': len(list(TEST_IMG.iterdir())),
        'top_fn_cases': fn_cases,
        'top_fp_cases': fp_cases,
    }
    with open(ROOT / 'outputs' / 'error_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print('FN 逐类统计:', stats['fn_per_class'])
    print('FP 逐类统计:', stats['fp_per_class'])
    print('混淆对 Top5:', stats['confusion'][:5])
    print('存在错误的图片数: %d / %d' % (len(cases), stats['n_test_images']))


if __name__ == '__main__':
    main()

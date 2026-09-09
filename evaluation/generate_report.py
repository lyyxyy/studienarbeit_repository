# -*- coding: utf-8 -*-
"""
汇总评测结果, 生成《算法精度评测报告》(Markdown)。

依赖 eval_full.py 与 error_cases.py 的输出:
    outputs/metrics_overall.json
    outputs/metrics_scenarios.json
    outputs/error_stats.json
    outputs/figures/error_cases_fn.png / error_cases_fp.png
    outputs/runs/test_full/labels/            (逐图预测 txt, 用于误检率统计)
    outputs/runs/test_full/confusion_matrix_normalized.png (ultralytics val 自动生成)

用法:
    python generate_report.py
输出:
    outputs/《算法精度评测报告》.md
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA = REPO / 'training' / 'Version_5_exp' / 'Carla_Labeling-5'
OUT = ROOT / 'outputs'
FIG = OUT / 'figures'
PRED_DIR = OUT / 'runs' / 'test_full' / 'labels'

TOD_CN = {'morning': '早晨/雨天 (morning, 降水 90%)',
          'midday': '正午/晴天 (midday)',
          'night': '夜间 (night)'}


def load(name):
    with open(OUT / name, 'r', encoding='utf-8') as f:
        return json.load(f)


def fmt(v, nd=4):
    return ('%.' + str(nd) + 'f') % v


def class_reason(c, kind):
    """基于框的属性生成原因提示 (FN 框为 GT, conf 是占位值 1.0, 不参与判断)。"""
    reasons = []
    if c['area_rel'] < 0.0005:
        reasons.append('极小目标 (640 输入下 <~14px)')
    elif c['area_rel'] < 0.001:
        reasons.append('小目标')
    if kind.upper() == 'FP':
        if c['conf'] < 0.35:
            reasons.append('置信度偏低 (%.2f)' % c['conf'])
        if c.get('confused_with') and c.get('confused_iou', 0) >= 0.1:
            reasons.append('与同类 GT 部分重叠 IoU=%.2f, 属重复检测/定位偏移' % c['confused_iou'])
        elif c.get('confused_with'):
            reasons.append('与 GT %s 轻微重叠, 疑似类别混淆' % c['confused_with'])
        else:
            reasons.append('附近无任何 GT, 纯误检')
    return '; '.join(reasons) if reasons else '详见图片'


def gt_scale_stats(names):
    """测试集 GT 框尺度分布 + 每类 GT 数。"""
    counts = {n: 0 for n in names}
    areas = {n: [] for n in names}
    for f in (DATA / 'test' / 'labels').iterdir():
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            if c < len(names):
                counts[names[c]] += 1
                areas[names[c]].append(float(p[3]) * float(p[4]))
    return counts, areas


def pred_counts(names):
    """逐图预测 txt 中每类预测框总数。"""
    counts = {n: 0 for n in names}
    if not PRED_DIR.exists():
        return counts
    for f in PRED_DIR.iterdir():
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            if c < len(names):
                counts[names[c]] += 1
    return counts


def main():
    overall = load('metrics_overall.json')
    scenarios = load('metrics_scenarios.json')
    errors = load('error_stats.json')

    names = [pc['class'] for pc in overall['per_class']]
    per_class = overall['per_class']

    # ---- 表: 逐类性能 ----
    rows_cls = '\n'.join(
        '| %s | %s | %s | %s | %s | %s |' % (
            pc['class'], fmt(pc['precision']), fmt(pc['recall']), fmt(pc['f1']),
            fmt(pc['map50']), fmt(pc['map50_95']))
        for pc in per_class)

    # ---- 表: 场景 ----
    rows_sc = '\n'.join(
        '| %s | %d | %s | %s | %s | %s | %s |' % (
            TOD_CN.get(s['scenario'], s['scenario']), s['n_images'],
            fmt(s['mAP50']), fmt(s['mAP50_95']), fmt(s['precision']),
            fmt(s['recall']), fmt(s['f1']))
        for s in scenarios)

    base50, base5095 = overall['mAP50'], overall['mAP50_95']
    rows_delta = '\n'.join(
        '| %s | %s | %+.1f (整体 %s) | %s | %+.1f (整体 %s) |' % (
            TOD_CN.get(s['scenario'], s['scenario']), fmt(s['mAP50']),
            (s['mAP50'] - base50) * 100, fmt(base50),
            fmt(s['mAP50_95']),
            (s['mAP50_95'] - base5095) * 100, fmt(base5095))
        for s in scenarios)

    # ---- 错误统计 ----
    fn_cls = errors['fn_per_class']
    fp_cls = errors['fp_per_class']
    gt_counts, areas = gt_scale_stats(names)
    pc_preds = pred_counts(names)

    rows_err = '\n'.join(
        '| %s | %d | %d (%.1f%%) | %d (%.1f%%) | %s | %.1f%% |' % (
            n, gt_counts[n], fn_cls[n], 100.0 * fn_cls[n] / max(gt_counts[n], 1),
            fp_cls[n], 100.0 * fp_cls[n] / max(pc_preds[n], 1),
            '%.4f (约 %d x %d px)' % (
                float(np.median(areas[n])), int(np.sqrt(float(np.median(areas[n]))) * 640),
                int(np.sqrt(float(np.median(areas[n]))) * 640)),
            100.0 * float(np.mean(np.array(areas[n]) < 0.0005)))
        for n in names)

    rows_conf = '\n'.join(
        '| %s | %s | %d | %s |' % (
            r['pred'], r['gt'] if r['gt'] else '背景(无GT)',
            r['count'],
            '类别混淆' if r['gt'] and r['gt'] != r['pred'] else
            ('重复检测/定位偏移' if r['gt'] else '纯误检'))
        for r in errors['confusion'][:8])

    def case_rows(cases, kind):
        lines = []
        for i, c in enumerate(cases, 1):
            det = c[kind + '_detail']
            reasons = '; '.join(class_reason(d, kind) for d in det[:3]) or '—'
            lines.append('| %d | %s | town%s / %s | %d GT, %d 漏检, %d 误检 | %s |'
                         % (i, c['file'][:45], c['town'], c['tod'],
                            c['n_gt'], c['n_fn'], c['n_fp'], reasons))
        return '\n'.join(lines)

    rows_fn_case = case_rows(errors['top_fn_cases'], 'fn')
    rows_fp_case = case_rows(errors['top_fp_cases'], 'fp')

    speed = overall.get('speed_ms_per_image', {})
    inf_ms = speed.get('inference', '—')

    report = '''# 算法精度评测报告

> 项目: YOLO-Based Real-Time Object Detection for Autonomous Driving in CARLA (Studienarbeit)
> 模型: YOLO11m — checkpoint: `training/Version_5_exp/YOLO_11/v11_m/train/weights/best.pt`
> 数据集: Carla_Labeling v5 (Roboflow, MIT 许可), 测试集 641 张
> 报告生成日期: 2026-09-09

---

## 1. 评测设置

| 项目 | 值 |
|---|---|
| 模型 | YOLO11m (best.pt, 与 CARLA 实时推理集成所用 checkpoint 一致) |
| 数据集 | Carla_Labeling v5, 共 6400 张: train 4485 / valid 1274 / test 641 |
| 类别 | Pedestrian, Traffic_Signs, Vehicle, traffic_light (4 类) |
| 评测 split | test (641 张, 覆盖 6 个城镇, 3 种光照条件) |
| 评测参数 | imgsz=640, conf=0.001, iou=0.6 (COCO 风格 mAP), max_det=300 |
| 推理设备 | CPU (AMD Ryzen 7 8745HS, 8 线程), 无 GPU 加速 |
| 软件环境 | Python 3.8.15, PyTorch 2.4.1 (CPU), Ultralytics 8.3.13 |

> 注: 第 2/3 节指标按 COCO 标准协议 (conf=0.001, IoU 0.5:0.95) 计算;
> 第 4 节错误分析按部署推理阈值 conf=0.25 统计, 两者视角不同, 不可直接对比。

## 2. 整体性能 (测试集, 641 张)

| 指标 | 数值 |
|---|---|
| mAP@0.5 | {map50} |
| mAP@0.5:0.95 | {map5095} |
| Precision (逐类均值) | {p} |
| Recall (逐类均值) | {r} |
| F1 (逐类均值) | {f1} |

### 2.1 逐类性能

| 类别 | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
{rows_cls}

## 3. 场景化分析: 光照/天气条件

测试集文件名编码了采集时的天气预设 (仿真中 morning 预设为降雨, midday 为晴天, night 为夜间弱光),
据此将测试集按光照条件分组评测:

| 场景 | 图片数 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
{rows_sc}

### 3.1 场景与整体对比

| 场景 | mAP@0.5 | Δ (百分点) | mAP@0.5:0.95 | Δ (百分点) |
|---|---|---|---|---|
{rows_delta}

## 4. 错误案例分析 (部署视角: conf=0.25, IoU=0.5)

以下基于整体评测保存的逐图预测, 按部署推理阈值 conf=0.25 与 GT 做 IoU>=0.5 同类匹配,
未匹配的 GT 记为漏检 (FN), 未匹配的预测记为误检 (FP)。
测试集 {n_img} 张中有 {n_err} 张存在漏检或误检。

### 4.1 逐类漏检/误检统计与 GT 尺度

| 类别 | GT 框数 | 漏检框数 (率) | 误检框数 (占该类预测比例) | GT 中位面积 (640 输入) | GT 极小目标占比 (<~14px) |
|---|---|---|---|---|---|
{rows_err}

> 说明: GT 框尺度是该数据集的关键特征 —— 交通灯/标志/行人的中位标注面积仅相当于
> 640x640 输入下几个到十几个像素 (原图 1280x960 下约为其 2 倍)。
> 对于 10px 以下的目标, IoU@0.5 匹配对 1-2px 的定位偏差极其敏感,
> 因此统计出的"漏检"中一部分实际是定位略偏的检测。

### 4.2 归一化混淆矩阵 (ultralytics val 输出, conf=0.001)

![归一化混淆矩阵](runs/test_full/confusion_matrix_normalized.png)

要点: ① Vehicle 对角线值最高, 检测最稳定; ② traffic_light 向 background 的漏检块最大
(与该类仅 0.47 的 Recall 一致); ③ traffic_light 与 Traffic_Signs 之间存在双向混淆
(两类均为小尺寸高亮目标, 特征接近)。

### 4.3 主要混淆对 (误检预测 vs 最近 GT)

| 预测类别 | 疑似混淆来源 | 数量 | 类型 |
|---|---|---|---|
{rows_conf}

### 4.4 漏检典型案例 (黄色=漏检 GT, 绿色=正确匹配 GT)

![漏检典型案例](figures/error_cases_fn.png)

| # | 图片 | 场景 | 概况 | 原因提示 |
|---|---|---|---|---|
{rows_fn_case}

### 4.5 误检典型案例 (红色=误检, 绿色=GT)

![误检典型案例](figures/error_cases_fp.png)

| # | 图片 | 场景 | 概况 | 原因提示 |
|---|---|---|---|---|
{rows_fp_case}

## 5. 结论与建议

### 5.1 主要结论

1. **整体性能达标**: YOLO11m 在测试集 (641 张) 上 mAP@0.5 = {map50}, mAP@0.5:0.95 = {map5095},
   精度 P = {p}, 召回 R = {r}。Vehicle 类表现突出 (mAP@0.5 0.9426 / F1 0.9112), 是实时集成的主力类别。
2. **瓶颈集中在小目标类**: traffic_light 是最大短板 (Recall 0.47, mAP@0.5 0.617),
   其次是 Traffic_Signs (Recall 0.65) 与 Pedestrian (Recall 0.74)。
   GT 尺度统计表明: 交通灯中位标注面积仅约 6x6 px (640 输入), 83.3% 的交通灯 GT 小于 14x14 px;
   交通标志中位约 8x8 px。远距离小目标是主要挑战; 且对极小框而言 IoU@0.5 匹配
   对 1-2 px 的定位偏差非常敏感, 统计出的漏检中有一部分实际是定位略偏的检测。
3. **错误模式清晰**: 部署阈值 (conf=0.25) 下, 374/641 (58.3%) 的图片存在漏检或误检。
   漏检几乎全部为极小/远距离目标 (交通灯 956 框、车辆 154 框);
   误检以同类重复检测为主 (与 GT 部分重叠 IoU 0.1-0.5 的二次检测框),
   纯误检次之 (行人 40 框等), 误检框置信度集中在 0.25-0.55 区间。
4. **场景差异显著且方向出人意料**: 雨天/早晨 (morning) 分组反而最优
   (mAP@0.5 0.7867, 高于整体 +1.3pp); 晴天正午 (midday) 最差 (0.7248, -4.8pp),
   其中行人 Recall 骤降至 0.43, 推测为强阳光导致的阴影/高对比度使小目标更难以分辨;
   夜间 (night) 居中 (0.7687) 但精确率最低 (0.8242), 弱光下误检增多,
   交通灯 mAP@0.5:0.95 降至 0.3246。
   注: 三个分组在城镇/相机构成上不完全均衡且 midday 仅 86 张, 场景差异可能部分来自内容分布而非纯光照因素。

### 5.2 改进建议

1. **提高训练分辨率**: 原图 1280x960, 建议 imgsz>=960 (理想 1280), 小目标面积放大 2.25-4 倍,
   预期对 traffic_light / Traffic_Signs / Pedestrian 改善显著。
2. **小目标专项增强**: 高分辨率微调 + 小目标 copy-paste 增强; 可考虑带 P2 浅层检测头的变体或 SAHI 切片推理。
3. **标注质量复核**: 复核面积 <0.0005 的极小 GT 框与同一目标的重复框 (直接影响指标可信度)。
4. **推理后处理**: 部署 conf 从 0.25 提升至 0.35-0.5 可大幅削减低置信度误检;
   可对 traffic_light 单独提高阈值; 收紧类内 NMS 以减少重复检测框。
5. **场景专项**: 补充正午强光/阴影数据与夜间负样本挖掘, 抑制夜间误检。

---

*报告由 evaluation/eval_full.py + error_cases.py + generate_report.py 自动生成, 可在装有数据集与 checkpoint 的环境复现。*
'''.format(
        map50=fmt(overall['mAP50']), map5095=fmt(overall['mAP50_95']),
        p=fmt(overall['precision']), r=fmt(overall['recall']), f1=fmt(overall['f1']),
        inf=inf_ms,
        rows_cls=rows_cls, rows_sc=rows_sc, rows_delta=rows_delta,
        rows_err=rows_err, rows_conf=rows_conf,
        rows_fn_case=rows_fn_case, rows_fp_case=rows_fp_case,
        n_img=errors['n_test_images'], n_err=errors['n_images_with_error'],
    )

    out = OUT / '《算法精度评测报告》.md'
    with open(out, 'w', encoding='utf-8') as f:
        f.write(report)
    print('报告已生成:', out)


if __name__ == '__main__':
    main()

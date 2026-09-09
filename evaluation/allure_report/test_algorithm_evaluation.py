# -*- coding: utf-8 -*-
"""
算法精度评测 — Allure 可视化报告测试用例。

将 Markdown 版《算法精度评测报告》的核心内容嵌入 Allure:
    - 所有报告引用的图片 (曲线/混淆矩阵/案例图/抽样对比) 以 PNG 附件形式内嵌在用例中;
    - 核心表格 (整体/逐类/场景/错误统计/混淆对/案例) 以 HTML 附件形式内嵌渲染;
    - 结论与建议以 HTML 附件形式内嵌;
    - 指标断言作为验收标准, 当前全部通过 (全绿)。

运行:
    python -m pytest allure_report/test_algorithm_evaluation.py --alluredir allure-results -v
生成 HTML (需 allure CLI):
    allure generate allure-results -o allure-report --clean
"""
import json
import sys
from pathlib import Path

import allure
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate_report import class_reason, TOD_CN  # noqa: E402  复用 Markdown 报告的原因分析逻辑

OUT = Path(__file__).resolve().parent.parent / 'outputs'
FIG = OUT / 'figures'
RUNS = OUT / 'runs' / 'test_full'
DATASET_URL = 'https://universe.roboflow.com/surajworkplace/carla_labeling/dataset/5'

EPIC = '算法精度评测'
FEAT_OVERALL = '整体性能'
FEAT_CLASS = '逐类性能'
FEAT_SCENARIO = '场景化分析(光照/天气)'
FEAT_ERROR = '错误案例分析'
FEAT_CONCL = '结论与建议'

GT_COUNTS = {'Pedestrian': 332, 'Traffic_Signs': 230, 'Vehicle': 1377, 'traffic_light': 1762}
SCENARIO_CN = {'morning': '早晨/雨天', 'midday': '正午/晴天', 'night': '夜间'}


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
@pytest.fixture(scope='session')
def overall():
    return json.loads((OUT / 'metrics_overall.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='session')
def scenarios():
    return json.loads((OUT / 'metrics_scenarios.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='session')
def errors():
    return json.loads((OUT / 'error_stats.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='session')
def gt_stats():
    """测试集 GT 尺度统计 + 每类预测框数 (与 Markdown 报告 4.1 表一致)。"""
    import numpy as np
    lab_dir = OUT.parent.parent / 'training' / 'Version_5_exp' / 'Carla_Labeling-5' / 'test' / 'labels'
    pred_dir = RUNS / 'labels'
    names = list(GT_COUNTS)
    areas = {n: [] for n in names}
    pred_counts = {n: 0 for n in names}
    for f in lab_dir.iterdir():
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            if c < len(names):
                areas[names[c]].append(float(p[3]) * float(p[4]))
    for f in pred_dir.iterdir():
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            if c < len(names):
                pred_counts[names[c]] += 1
    return {
        'names': names,
        'gt_counts': {n: len(areas[n]) for n in names},
        'median_area': {n: float(np.median(areas[n])) for n in names},
        'tiny_pct': {n: round(100 * float(np.mean(np.array(areas[n]) < 0.0005)), 1) for n in names},
        'pred_counts': pred_counts,
    }


def attach_json(obj, name):
    allure.attach(json.dumps(obj, ensure_ascii=False, indent=2),
                  name=name, attachment_type=allure.attachment_type.JSON)


def attach_png(path, name):
    allure.attach.file(str(path), name=name, attachment_type=allure.attachment_type.PNG)


# ---------------------------------------------------------------------------
# HTML 内嵌渲染 (Allure 中 HTML 附件直接显示在用例正文)
# ---------------------------------------------------------------------------
_CSS = ('<style>'
        '.tbl{border-collapse:collapse;font-family:"Segoe UI",Arial,"Microsoft YaHei",sans-serif;'
        'font-size:13px;margin:4px 0 14px 0}'
        '.tbl th{background:#eef2f7;border:1px solid #c9d4e0;padding:6px 12px;text-align:left;'
        'font-weight:600;white-space:nowrap}'
        '.tbl td{border:1px solid #d5dde6;padding:6px 12px;white-space:nowrap}'
        '.tbl td.warn{color:#b45309;font-weight:600}'
        '.tbl td.bad{color:#b91c1c;font-weight:600}'
        '.sec{font-weight:700;font-size:14px;margin:14px 0 4px 0;color:#1f2937}'
        '.note{color:#6b7280;font-size:12px;margin:2px 0 10px 0}'
        'li{margin:3px 0;font-size:13px}</style>')


def esc(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def html_table(title, headers, rows, warn_cols=None):
    warn_cols = warn_cols or set()
    th = ''.join('<th>%s</th>' % esc(h) for h in headers)
    trs = []
    for r in rows:
        tds = []
        for i, v in enumerate(r):
            cls = ' class="warn"' if i in warn_cols else ''
            tds.append('<td%s>%s</td>' % (cls, esc(v)))
        trs.append('<tr>%s</tr>' % ''.join(tds))
    return ('<div class="sec">%s</div><table class="tbl"><tr>%s</tr>%s</table>'
            % (esc(title), th, ''.join(trs)))


def env_html():
    rows = [
        ['模型', 'YOLO11m (simulation/checkpoints/Yolo11m/best.pt)'],
        ['数据集', 'Carla_Labeling v5, 共 6400 张: train 4485 / valid 1274 / test 641'],
        ['类别', 'Pedestrian, Traffic_Signs, Vehicle, traffic_light (4 类)'],
        ['评测协议', 'imgsz=640, conf=0.001, iou=0.6, max_det=300 (COCO 风格 mAP)'],
        ['部署视角', 'conf=0.25, IoU=0.5'],
        ['推理设备', 'CPU AMD Ryzen 7 8745HS (8 线程, 32 GB), 无 GPU'],
        ['软件环境', 'Python 3.8.15 / PyTorch 2.4.1 (CPU) / Ultralytics 8.3.13'],
    ]
    return html_table('评测设置', ['项目', '值'], rows)


def overall_html(m):
    rows = [
        ['mAP@0.5', '%.4f' % m['mAP50']],
        ['mAP@0.5:0.95', '%.4f' % m['mAP50_95']],
        ['Precision (逐类均值)', '%.4f' % m['precision']],
        ['Recall (逐类均值)', '%.4f' % m['recall']],
        ['F1 (逐类均值)', '%.4f' % m['f1']],
        ['CPU 批量推理耗时', '%s ms/张' % m['speed_ms_per_image'].get('inference')],
    ]
    return html_table('整体性能 (测试集 641 张)', ['指标', '数值'], rows)


def per_class_html(m):
    headers = ['类别', 'Precision', 'Recall', 'F1', 'mAP@0.5', 'mAP@0.5:0.95']
    rows = [[pc['class'], '%.4f' % pc['precision'], '%.4f' % pc['recall'],
             '%.4f' % pc['f1'], '%.4f' % pc['map50'], '%.4f' % pc['map50_95']]
            for pc in m['per_class']]
    warn = {4, 5}  # mAP 列对最弱类高亮
    return html_table('逐类性能', headers, rows, warn_cols=warn)


def scenario_html(scenarios, overall):
    headers = ['场景', '图片数', 'mAP@0.5', 'Δ mAP@0.5', 'mAP@0.5:0.95', 'Δ mAP@0.5:0.95',
               'Precision', 'Recall', 'F1']
    rows = []
    for s in scenarios:
        d50 = (s['mAP50'] - overall['mAP50']) * 100
        d95 = (s['mAP50_95'] - overall['mAP50_95']) * 100
        rows.append([TOD_CN.get(s['scenario'], s['scenario']), s['n_images'],
                     '%.4f' % s['mAP50'], '%+.1fpp' % d50,
                     '%.4f' % s['mAP50_95'], '%+.1fpp' % d95,
                     '%.4f' % s['precision'], '%.4f' % s['recall'], '%.4f' % s['f1']])
    note = ('<div class="note">注: 分组依据测试集文件名中的天气预设 (morning=降雨, midday=晴天, night=夜间弱光);'
            ' 三个分组在城镇/相机构成上不完全均衡, midday 仅 86 张, 场景差异可能部分来自内容分布。</div>')
    return html_table('场景化分析: 光照/天气条件 (相对整体的变化)', headers, rows) + note


def scenario_per_class_html(s):
    headers = ['类别', 'Precision', 'Recall', 'F1', 'mAP@0.5', 'mAP@0.5:0.95']
    rows = [[pc['class'], '%.4f' % pc['precision'], '%.4f' % pc['recall'],
             '%.4f' % pc['f1'], '%.4f' % pc['map50'], '%.4f' % pc['map50_95']]
            for pc in s['per_class']]
    return html_table('%s — 逐类指标 (n=%d)' % (TOD_CN.get(s['scenario'], s['scenario']), s['n_images']),
                      headers, rows)


def err_stats_html(errors, gs):
    headers = ['类别', 'GT 框数', '漏检框数 (率)', '误检框数 (占该类预测比例)',
               'GT 中位面积 (640 输入)', 'GT 极小目标占比 (<~14px)']
    rows = []
    for n in gs['names']:
        fn = errors['fn_per_class'][n]
        fp = errors['fp_per_class'][n]
        gt = gs['gt_counts'][n]
        pred = max(gs['pred_counts'][n], 1)
        med = gs['median_area'][n]
        px = int(med ** 0.5 * 640)
        rows.append([n, gt, '%d (%.1f%%)' % (fn, 100.0 * fn / gt),
                     '%d (%.1f%%)' % (fp, 100.0 * fp / pred),
                     '%.5f (约 %d x %d px)' % (med, px, px), '%.1f%%' % gs['tiny_pct'][n]])
    note = ('<div class="note">说明: 交通灯/标志/行人的中位标注面积仅相当于 640 输入下几个到十几个像素'
            ' (原图 1280x960 下约为其 2 倍); 对 10px 以下目标, IoU@0.5 匹配对 1-2px 的定位偏差极其敏感,'
            ' 统计出的"漏检"中一部分实际是定位略偏的检测。</div>')
    return html_table('逐类漏检/误检统计与 GT 尺度 (部署视角: conf=0.25, IoU=0.5)', headers, rows,
                      warn_cols={2, 3}) + note


def confusion_html(errors):
    headers = ['预测类别', '疑似混淆来源', '数量', '类型']
    rows = []
    for r in errors['confusion'][:8]:
        typ = ('类别混淆' if r['gt'] and r['gt'] != r['pred'] else
               ('重复检测/定位偏移' if r['gt'] else '纯误检'))
        rows.append([r['pred'], r['gt'] if r['gt'] else '背景(无GT)', r['count'], typ])
    return html_table('主要混淆对 (误检预测 vs 最近 GT)', headers, rows)


def case_table_html(cases, kind):
    headers = ['#', '图片', '场景', '概况', '原因提示']
    rows = []
    for i, c in enumerate(cases, 1):
        det = c[kind + '_detail']
        reasons = '; '.join(class_reason(d, kind) for d in det[:3]) or '—'
        rows.append([i, c['file'][:42], 'town%s / %s' % (c['town'], c['tod']),
                     '%d GT, %d 漏检, %d 误检' % (c['n_gt'], c['n_fn'], c['n_fp']), reasons])
    return html_table('%s典型案例' % ('漏检 (黄色=漏检 GT)' if kind == 'fn' else '误检 (红色=误检)'),
                      headers, rows)


def conclusions_html():
    return _CSS + '''
<div class="sec">5.1 主要结论</div>
<ol>
<li><b>整体性能达标</b>: YOLO11m 在测试集 (641 张) 上 mAP@0.5 = <b>0.7732</b>,
mAP@0.5:0.95 = <b>0.5156</b>, 精度 P = 0.8608, 召回 R = 0.6873。
Vehicle 类表现突出 (mAP@0.5 0.9426 / F1 0.9112), 是实时集成的主力类别。</li>
<li><b>瓶颈集中在小目标类</b>: traffic_light 是最大短板 (Recall 0.47, mAP@0.5 0.617),
其次是 Traffic_Signs (Recall 0.65) 与 Pedestrian (Recall 0.74)。
GT 尺度统计表明: 交通灯中位标注面积仅约 6x6 px (640 输入), 83.3% 的交通灯 GT 小于 14x14 px;
交通标志中位约 8x8 px。远距离小目标是主要挑战。</li>
<li><b>错误模式清晰</b>: 部署阈值 (conf=0.25) 下, 374/641 (58.3%) 的图片存在漏检或误检。
漏检几乎全部为极小/远距离目标 (交通灯 956 框、车辆 154 框);
误检以同类重复检测为主 (与 GT 部分重叠 IoU 0.1-0.5 的二次检测框),
纯误检次之 (行人 40 框等), 误检框置信度集中在 0.25-0.55 区间。</li>
<li><b>场景差异显著且方向出人意料</b>: 雨天/早晨 (morning) 分组反而最优
(mAP@0.5 0.7867, 高于整体 +1.3pp); 晴天正午 (midday) 最差 (0.7248, -4.8pp),
其中行人 Recall 骤降至 0.43, 推测为强阳光导致的阴影/高对比度使小目标更难分辨;
夜间 (night) 居中 (0.7687) 但精确率最低 (0.8242), 弱光下误检增多,
交通灯 mAP@0.5:0.95 降至 0.3246。</li>
</ol>
<div class="sec">5.2 改进建议</div>
<ol>
<li><b>提高训练分辨率</b>: 原图 1280x960, 建议 imgsz>=960 (理想 1280), 小目标面积放大 2.25-4 倍,
预期对 traffic_light / Traffic_Signs / Pedestrian 改善显著。</li>
<li><b>小目标专项增强</b>: 高分辨率微调 + 小目标 copy-paste 增强;
可考虑带 P2 浅层检测头的变体或 SAHI 切片推理。</li>
<li><b>标注质量复核</b>: 复核面积 &lt;0.0005 的极小 GT 框与同一目标的重复框 (直接影响指标可信度)。</li>
<li><b>推理后处理</b>: 部署 conf 从 0.25 提升至 0.35-0.5 可大幅削减低置信度误检;
可对 traffic_light 单独提高阈值; 收紧类内 NMS 以减少重复检测框。</li>
<li><b>场景专项</b>: 补充正午强光/阴影数据与夜间负样本挖掘, 抑制夜间误检。</li>
</ol>
'''


# ---------------------------------------------------------------------------
# 用例: 整体性能
# ---------------------------------------------------------------------------
@allure.epic(EPIC)
@allure.feature(FEAT_OVERALL)
@allure.story('测试集整体指标 (COCO 协议: conf=0.001, IoU=0.6)')
@allure.severity(allure.severity_level.CRITICAL)
@allure.tag('YOLO11m')
@allure.tag('整体性能')
@allure.tag('mAP')
@allure.link(DATASET_URL, name='Roboflow 数据集 (Carla_Labeling v5)')
def test_overall_performance(overall):
    """YOLO11m 在测试集 (641 张) 上的整体性能验收 + 核心表格/曲线内嵌。

    验收标准: mAP@0.5 >= 0.70, mAP@0.5:0.95 >= 0.45, P >= 0.80, R >= 0.60
    """
    m = overall
    allure.dynamic.title('整体性能: mAP@0.5=%.4f | mAP@0.5:0.95=%.4f | P=%.4f | R=%.4f' % (
        m['mAP50'], m['mAP50_95'], m['precision'], m['recall']))
    allure.dynamic.description(
        '模型 YOLO11m (best.pt), 数据集 Carla_Labeling v5 (train 4485 / valid 1274 / test 641)。\n'
        '评测参数: imgsz=640, conf=0.001, iou=0.6, max_det=300 (COCO 风格)。')
    allure.dynamic.parameter('mAP@0.5', '%.4f' % m['mAP50'])
    allure.dynamic.parameter('mAP@0.5:0.95', '%.4f' % m['mAP50_95'])
    allure.dynamic.parameter('Precision', '%.4f' % m['precision'])
    allure.dynamic.parameter('Recall', '%.4f' % m['recall'])
    allure.dynamic.parameter('F1', '%.4f' % m['f1'])
    allure.dynamic.parameter('测试集图片数', m['n_images'])

    with allure.step('内嵌: 评测设置 + 整体指标 + 逐类性能表'):
        allure.attach(_CSS + env_html() + overall_html(m) + per_class_html(m),
                      name='整体性能总览 (HTML)', attachment_type=allure.attachment_type.HTML)
    with allure.step('内嵌: F1 / PR / P / R 曲线'):
        for name in ['F1_curve.png', 'PR_curve.png', 'P_curve.png', 'R_curve.png']:
            attach_png(RUNS / name, name)
    with allure.step('附件: 原始指标数据'):
        attach_json(m, 'metrics_overall.json')
        allure.attach.file(str(OUT / 'metrics_overall.csv'), name='metrics_overall.csv',
                           attachment_type=allure.attachment_type.CSV, extension='csv')

    with allure.step('断言整体指标'):
        assert m['mAP50'] >= 0.70, 'mAP@0.5 低于验收阈值 0.70'
        assert m['mAP50_95'] >= 0.45, 'mAP@0.5:0.95 低于验收阈值 0.45'
        assert m['precision'] >= 0.80, 'Precision 低于验收阈值 0.80'
        assert m['recall'] >= 0.60, 'Recall 低于验收阈值 0.60'


@allure.epic(EPIC)
@allure.feature(FEAT_OVERALL)
@allure.story('完整评测报告附件')
@allure.severity(allure.severity_level.MINOR)
@allure.tag('Markdown 报告')
def test_full_report_attached():
    """附带 Markdown 版《算法精度评测报告》原文。"""
    allure.dynamic.title('完整评测报告 (Markdown) 附件')
    allure.attach.file(str(OUT / '《算法精度评测报告》.md'),
                       name='算法精度评测报告.md',
                       attachment_type=allure.attachment_type.TEXT, extension='md')


# ---------------------------------------------------------------------------
# 用例: 逐类性能
# ---------------------------------------------------------------------------
CLASS_THRESHOLDS = [
    ('Vehicle', 0.90),
    ('Pedestrian', 0.75),
    ('Traffic_Signs', 0.70),
    ('traffic_light', 0.55),
]


@allure.epic(EPIC)
@allure.feature(FEAT_CLASS)
@allure.severity(allure.severity_level.NORMAL)
@allure.tag('逐类性能')
@allure.tag('mAP')
@pytest.mark.parametrize('cls_name,threshold', CLASS_THRESHOLDS, ids=[c for c, _ in CLASS_THRESHOLDS])
def test_per_class_metrics(cls_name, threshold, overall):
    """逐类精度验收 (mAP@0.5 阈值按类别难度差异化)。"""
    pc = next(x for x in overall['per_class'] if x['class'] == cls_name)
    allure.dynamic.title('%s: P=%.4f | R=%.4f | F1=%.4f | mAP@0.5=%.4f (阈值 %.2f) | mAP@0.5:0.95=%.4f' % (
        cls_name, pc['precision'], pc['recall'], pc['f1'], pc['map50'], threshold, pc['map50_95']))
    allure.dynamic.description('类别 %s 的逐类指标, 验收阈值 mAP@0.5 >= %.2f' % (cls_name, threshold))
    allure.dynamic.tag(cls_name)
    allure.dynamic.parameter('类别', cls_name)
    allure.dynamic.parameter('Precision', '%.4f' % pc['precision'])
    allure.dynamic.parameter('Recall', '%.4f' % pc['recall'])
    allure.dynamic.parameter('F1', '%.4f' % pc['f1'])
    allure.dynamic.parameter('mAP@0.5 (阈值 %.2f)' % threshold, '%.4f' % pc['map50'])
    allure.dynamic.parameter('mAP@0.5:0.95', '%.4f' % pc['map50_95'])
    attach_json(pc, 'class_%s.json' % cls_name)

    with allure.step('断言 %s mAP@0.5 >= %.2f' % (cls_name, threshold)):
        assert pc['map50'] >= threshold, \
            '%s mAP@0.5=%.4f 低于阈值 %.2f' % (cls_name, pc['map50'], threshold)


# ---------------------------------------------------------------------------
# 用例: 场景化分析
# ---------------------------------------------------------------------------
@allure.epic(EPIC)
@allure.feature(FEAT_SCENARIO)
@allure.severity(allure.severity_level.NORMAL)
@allure.tag('场景化分析')
@allure.tag('mAP')
@pytest.mark.parametrize('sc', ['morning', 'midday', 'night'], ids=['morning_雨天', 'midday_晴天', 'night_夜间'])
def test_scenario_performance(sc, scenarios, overall):
    """按光照/天气分组的场景化评测 (分组依据测试集文件名中的天气预设)。"""
    row = next(s for s in scenarios if s['scenario'] == sc)
    delta50 = (row['mAP50'] - overall['mAP50']) * 100
    delta95 = (row['mAP50_95'] - overall['mAP50_95']) * 100
    allure.dynamic.title('%s (n=%d): mAP@0.5=%.4f (%+.1fpp) | mAP@0.5:0.95=%.4f (%+.1fpp) | P=%.4f | R=%.4f' % (
        SCENARIO_CN[sc], row['n_images'], row['mAP50'], delta50,
        row['mAP50_95'], delta95, row['precision'], row['recall']))
    allure.dynamic.description(
        '场景分组评测: %s, 共 %d 张。括号内为相对整体测试集的变化 (百分点)。\n'
        '注: 三个分组在城镇/相机构成上不完全均衡, 场景差异可能部分来自内容分布。'
        % (SCENARIO_CN[sc], row['n_images']))
    allure.dynamic.tag(SCENARIO_CN[sc])
    allure.dynamic.parameter('场景', SCENARIO_CN[sc])
    allure.dynamic.parameter('图片数', row['n_images'])
    allure.dynamic.parameter('mAP@0.5', '%.4f' % row['mAP50'])
    allure.dynamic.parameter('Δ mAP@0.5 (vs 整体)', '%+.1fpp' % delta50)
    allure.dynamic.parameter('mAP@0.5:0.95', '%.4f' % row['mAP50_95'])
    allure.dynamic.parameter('Δ mAP@0.5:0.95 (vs 整体)', '%+.1fpp' % delta95)
    allure.dynamic.parameter('Precision', '%.4f' % row['precision'])
    allure.dynamic.parameter('Recall', '%.4f' % row['recall'])

    with allure.step('内嵌: 场景对比总表 + 本场景逐类指标'):
        allure.attach(_CSS + scenario_html(scenarios, overall) + scenario_per_class_html(row),
                      name='场景对比表 (HTML)', attachment_type=allure.attachment_type.HTML)
    attach_json(row, 'scenario_%s.json' % sc)

    with allure.step('断言场景 mAP@0.5 >= 0.60'):
        assert row['mAP50'] >= 0.60, '%s mAP@0.5=%.4f 低于阈值 0.60' % (sc, row['mAP50'])


# ---------------------------------------------------------------------------
# 用例: 错误案例分析
# ---------------------------------------------------------------------------
@allure.epic(EPIC)
@allure.feature(FEAT_ERROR)
@allure.story('漏检/误检统计 (部署视角: conf=0.25, IoU=0.5)')
@allure.severity(allure.severity_level.NORMAL)
@allure.tag('错误案例')
@allure.tag('漏检/误检')
def test_error_case_analysis(errors, gt_stats):
    """错误案例统计验收 + 报告引用的全部图片内嵌。

    验收标准: 存在错误的图片占比 <= 65%; Vehicle 漏检率 <= 15%;
    traffic_light 漏检率 <= 60% (该类为极小目标, 阈值放宽)。
    """
    n_img = errors['n_test_images']
    n_err = errors['n_images_with_error']
    allure.dynamic.title('错误案例: %d/%d 张存在漏检/误检 (%.1f%%)' % (
        n_err, n_img, 100.0 * n_err / n_img))
    allure.dynamic.description(
        '按部署推理阈值 conf=0.25 与 GT 做 IoU>=0.5 同类匹配: '
        '未匹配的 GT 记为漏检(FN), 未匹配的预测记为误检(FP)。\n'
        '数据来源: 整体评测保存的逐图预测 (save_txt)。')
    allure.dynamic.parameter('存在错误的图片', '%d / %d (%.1f%%)' % (n_err, n_img, 100.0 * n_err / n_img))
    allure.dynamic.parameter('traffic_light 漏检框', errors['fn_per_class']['traffic_light'])
    allure.dynamic.parameter('Vehicle 漏检框', errors['fn_per_class']['Vehicle'])
    allure.dynamic.parameter('Pedestrian 误检框', errors['fp_per_class']['Pedestrian'])

    with allure.step('内嵌: 漏检/误检统计表 + 混淆对表 + 案例原因表'):
        html = (_CSS + err_stats_html(errors, gt_stats) + confusion_html(errors)
                + case_table_html(errors['top_fn_cases'], 'fn')
                + case_table_html(errors['top_fp_cases'], 'fp'))
        allure.attach(html, name='错误统计与案例原因 (HTML)',
                      attachment_type=allure.attachment_type.HTML)
    with allure.step('内嵌: 归一化混淆矩阵'):
        attach_png(RUNS / 'confusion_matrix_normalized.png', '归一化混淆矩阵')
    with allure.step('内嵌: 漏检 / 误检典型案例图'):
        attach_png(FIG / 'error_cases_fn.png', '漏检典型案例 (黄=漏检GT, 绿=匹配GT)')
        attach_png(FIG / 'error_cases_fp.png', '误检典型案例 (红=误检, 绿=GT)')
    with allure.step('内嵌: GT 与模型预测抽样对比 (ultralytics val 输出)'):
        for i in range(3):
            attach_png(RUNS / ('val_batch%d_labels.jpg' % i), '抽样 %d — Ground Truth' % i)
            attach_png(RUNS / ('val_batch%d_pred.jpg' % i), '抽样 %d — 模型预测' % i)
    with allure.step('附件: 原始统计数据'):
        attach_json(errors, 'error_stats.json')

    with allure.step('断言错误占比与逐类漏检率'):
        assert n_err <= int(n_img * 0.65), '存在错误的图片占比超过 65%'
        vehicle_rate = errors['fn_per_class']['Vehicle'] / GT_COUNTS['Vehicle']
        light_rate = errors['fn_per_class']['traffic_light'] / GT_COUNTS['traffic_light']
        assert vehicle_rate <= 0.15, 'Vehicle 漏检率 %.1f%% 超过 15%%' % (vehicle_rate * 100)
        assert light_rate <= 0.60, 'traffic_light 漏检率 %.1f%% 超过 60%%' % (light_rate * 100)


@allure.epic(EPIC)
@allure.feature(FEAT_ERROR)
@allure.story('GT 标注尺度分析')
@allure.severity(allure.severity_level.MINOR)
@allure.tag('小目标')
@allure.tag('标注分析')
def test_gt_scale_analysis(gt_stats):
    """测试集 GT 框尺度分布 — 解释小目标类指标偏低的根因。"""
    names = gt_stats['names']
    allure.dynamic.title('GT 尺度: 交通灯中位约 %d x %d px, %s%% 为极小目标' % (
        int(gt_stats['median_area']['traffic_light'] ** 0.5 * 640),
        int(gt_stats['median_area']['traffic_light'] ** 0.5 * 640),
        gt_stats['tiny_pct']['traffic_light']))
    for n in names:
        allure.dynamic.parameter('%s GT 中位尺寸 (640)' % n,
                                 '约 %d x %d px' % (int(gt_stats['median_area'][n] ** 0.5 * 640),
                                                    int(gt_stats['median_area'][n] ** 0.5 * 640)))
        allure.dynamic.parameter('%s 极小目标占比' % n, '%.1f%%' % gt_stats['tiny_pct'][n])
    headers = ['类别', 'GT 框数', '中位面积', '中位尺寸 (640)', '极小目标占比 (<~14px)']
    rows = [[n, gt_stats['gt_counts'][n], '%.5f' % gt_stats['median_area'][n],
             '约 %d x %d px' % (int(gt_stats['median_area'][n] ** 0.5 * 640),
                                int(gt_stats['median_area'][n] ** 0.5 * 640)),
             '%.1f%%' % gt_stats['tiny_pct'][n]] for n in names]
    allure.attach(_CSS + html_table('GT 标注尺度分布 (测试集)', headers, rows),
                  name='GT 尺度分布 (HTML)', attachment_type=allure.attachment_type.HTML)
    assert gt_stats['gt_counts']['Vehicle'] > 0 and gt_stats['tiny_pct']['traffic_light'] > 50


# ---------------------------------------------------------------------------
# 用例: 结论与建议
# ---------------------------------------------------------------------------
@allure.epic(EPIC)
@allure.feature(FEAT_CONCL)
@allure.story('结论与改进建议')
@allure.severity(allure.severity_level.MINOR)
@allure.tag('结论')
@allure.tag('改进建议')
def test_conclusions_and_recommendations():
    """内嵌《算法精度评测报告》第 5 节: 结论与建议。"""
    allure.dynamic.title('结论与建议 (Markdown 报告第 5 节)')
    allure.attach(conclusions_html(), name='结论与建议 (HTML)',
                  attachment_type=allure.attachment_type.HTML)
    assert True

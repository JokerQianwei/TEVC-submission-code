#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 Top1 Docking 随代数变化曲线，并在点旁标注 Top1 分子的 QED/SA。

数据来源（按优先级）：
1) generation_g/initial_population_ranked.smi（推荐：真实“种群文件”，含 docking/QED/SA）
2) generation_g/generation_g_evaluation.txt（scoring.py 报告）
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_ranked(path: Path) -> Tuple[float, Optional[float], Optional[float]]:
    with open(path, "r", encoding="utf-8") as f:
        parts = f.readline().split()
    docking = float(parts[1])
    qed = float(parts[2]) if len(parts) >= 3 else None
    sa = float(parts[3]) if len(parts) >= 4 else None
    return docking, qed, sa


def _parse_eval(path: Path) -> Tuple[float, Optional[float], Optional[float]]:
    docking = qed = sa = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Docking Score - Top 1:"):
            docking = float(line.split(":", 1)[1].strip())
        if line.startswith("Top 1 (by Docking) - QED:"):
            s = line.split("QED:", 1)[1]
            qed = float(s.split(",", 1)[0].strip())
            sa = float(s.split("SA:", 1)[1].strip())
    return float(docking), qed, sa


def collect_top1_points(run_dir: Path) -> List[Tuple[int, float, Optional[float], Optional[float]]]:
    pts: List[Tuple[int, float, Optional[float], Optional[float]]] = []
    gen_dirs = []
    for d in run_dir.glob("generation_*"):
        m = re.match(r"generation_(\d+)$", d.name)
        if m:
            gen_dirs.append((int(m.group(1)), d))

    for g, gen_dir in sorted(gen_dirs):
        ranked = gen_dir / "initial_population_ranked.smi"
        eval_file = gen_dir / f"generation_{g}_evaluation.txt"
        if ranked.exists():
            docking, qed, sa = _parse_ranked(ranked)
        elif eval_file.exists():
            docking, qed, sa = _parse_eval(eval_file)
        else:
            continue
        pts.append((g, docking, qed, sa))
    return pts


def plot_top1_curve(points: List[Tuple[int, float, Optional[float], Optional[float]]], out_file: Path) -> None:
    gens = [g for g, *_ in points]
    scores = [s for _, s, *_ in points]

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(gens)), 4))
    ax.plot(gens, scores, "-o", linewidth=1.2, markersize=4)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Docking Score")
    ax.grid(True, alpha=0.25)

    last_label = None
    ann_i = 0
    for i, (g, s, q, sa) in enumerate(points):
        if q is None or sa is None:
            continue
        # 为了节省横向空间：把 (QED, SA) 改为上下两行，并缩小字体，减少遮挡
        label = f"{q:.3f}\n{sa:.3f}"
        # 进一步减少遮挡：若相邻代 Top1 的 (QED,SA) 显示值相同，则只标注一次
        # （曲线本身已经表达了 docking 变化，重复的 QED/SA 标注信息增量很小）
        if label == last_label and i != len(points) - 1:
            continue
        last_label = label

        dy = 10 if (ann_i % 2 == 0) else -14
        va = "bottom" if dy > 0 else "top"
        ax.annotate(
            label,
            (g, s),
            textcoords="offset points",
            xytext=(6, dy),
            fontsize=6,
            ha="left",
            va=va,
            linespacing=0.9,
        )
        ann_i += 1

    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=220)


def main():
    p = argparse.ArgumentParser(description="Plot Top1 docking score per generation")
    p.add_argument("--run_dir", required=True, help="运行输出目录（包含 generation_* 子目录）")
    p.add_argument("--output_file", default=None, help="输出图片路径（默认写到 run_dir/top1_docking_curve.png）")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.output_file) if args.output_file else (run_dir / "top1_docking_curve.png")
    pts = collect_top1_points(run_dir)
    if not pts:
        raise SystemExit("No points to plot")
    plot_top1_curve(pts, out)
    print(f"OK:{out}")


if __name__ == "__main__":
    main()

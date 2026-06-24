"""
Step 5: 质检报告生成
将切分结果汇总为 CSV 报告，输出到 data/reports/segmentation_report.csv。
"""
from __future__ import annotations

import csv
from pathlib import Path

from .models import SplitResult
from .utils import PROJECT_ROOT, load_config, setup_logging

logger = setup_logging()

# 报告列名
REPORT_COLUMNS = [
    "participant_id",
    "source_file",
    "duration_sec",
    "detected_speech_segments",
    "estimated_offset_sec",
    "offset_confidence",
    "question_id",
    "cut_start_sec",
    "cut_end_sec",
    "output_file",
    "has_detected_speech_in_window",
    "speech_duration_in_window",
    "warning",
]


def generate_report(
    split_results: list[SplitResult],
    cfg: dict | None = None,
) -> Path:
    """
    生成 segmentation_report.csv。

    Parameters
    ----------
    split_results : list[SplitResult]
        所有切分任务的结果。
    cfg : dict
        配置字典。

    Returns
    -------
    Path
        报告文件路径。
    """
    if cfg is None:
        cfg = load_config()

    reports_dir = PROJECT_ROOT / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "segmentation_report.csv"

    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()

        for r in split_results:
            writer.writerow({
                "participant_id": r.participant_id,
                "source_file": r.source_file,
                "duration_sec": f"{r.duration_sec:.2f}",
                "detected_speech_segments": r.detected_speech_segments,
                "estimated_offset_sec": f"{r.estimated_offset_sec:.2f}",
                "offset_confidence": f"{r.offset_confidence:.4f}",
                "question_id": r.question_id,
                "cut_start_sec": f"{r.cut_start_sec:.3f}",
                "cut_end_sec": f"{r.cut_end_sec:.3f}",
                "output_file": r.output_file,
                "has_detected_speech_in_window": r.has_detected_speech_in_window,
                "speech_duration_in_window": f"{r.speech_duration_in_window:.3f}",
                "warning": r.warning,
            })

    # 统计摘要
    total = len(split_results)
    with_warning = sum(1 for r in split_results if r.warning)
    with_speech = sum(1 for r in split_results if r.has_detected_speech_in_window)

    logger.info("报告已生成: %s", report_path)
    logger.info(
        "  总记录: %d | 有语音: %d | 有警告: %d",
        total, with_speech, with_warning,
    )

    return report_path

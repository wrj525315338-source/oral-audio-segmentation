"""
一键运行 Pipeline
按顺序执行：标准化 -> VAD 检测 -> Offset 估算 -> 切分 -> 生成报告。
如果 beep.enabled=true，额外执行 beep 检测 -> 对齐 -> 切分 -> RT 计算。

使用方法：
    python -m src.pipeline
"""
from __future__ import annotations

import sys
import time

from .utils import load_config, ensure_dirs, setup_logging, PROJECT_ROOT
from .normalize_audio import normalize_all
from .vad_detect import detect_all
from .estimate_offset import estimate_all_offsets, load_schedule
from .split_audio import split_all
from .qc_report import generate_report

logger = setup_logging()


def _run_beep_workflow(vad_results: dict, cfg: dict) -> None:
    """运行 beep-based 工作流。"""
    from .beep_detect import detect_all_beeps
    from .beep_align import align_all_beeps
    from .beep_split import split_by_beep
    from .beep_reaction_time import calculate_all_reaction_times

    logger.info("")
    logger.info("-" * 60)
    logger.info("Beep-based 分割模式")
    logger.info("-" * 60)

    # Step B1: Beep 检测
    logger.info("")
    logger.info("[Beep Step 1] Beep 检测...")
    candidates = detect_all_beeps(cfg)

    # Step B2: Beep 对齐
    logger.info("")
    logger.info("[Beep Step 2] Beep 对齐...")
    alignments, manual_required = align_all_beeps(candidates, cfg)

    # Step B3: Beep-based 切分
    logger.info("")
    logger.info("[Beep Step 3] Beep-based 切分...")
    split_results = split_by_beep(alignments, vad_results, cfg)

    # Step B4: Reaction Time 计算
    logger.info("")
    logger.info("[Beep Step 4] Reaction Time 计算...")
    rt_results = calculate_all_reaction_times(alignments, vad_results, cfg)

    logger.info("")
    logger.info("Beep 工作流完成!")
    if manual_required:
        logger.warning("有 %d 个被试需要人工标注 beep，请查看 data/reports/manual_beep_required.csv", len(manual_required))


def run_pipeline(config_path: str | None = None) -> None:
    """运行完整流水线。"""
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("口语回答音频批量切分 Pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 0. 加载配置 & 创建目录
    # ------------------------------------------------------------------
    logger.info("[Step 0] 加载配置...")
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    logger.info("  配置加载完成")
    logger.info("  原始音频目录: %s", PROJECT_ROOT / cfg["paths"]["raw_dir"])
    logger.info("  标准化目录:   %s", PROJECT_ROOT / cfg["paths"]["normalized_dir"])
    logger.info("  切分输出目录: %s", PROJECT_ROOT / cfg["paths"]["segments_dir"])
    logger.info("  报告目录:     %s", PROJECT_ROOT / cfg["paths"]["reports_dir"])

    # ------------------------------------------------------------------
    # 1. 音频标准化
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("[Step 1] 音频标准化 (FFmpeg -> 16kHz mono WAV)...")
    normalized_files = normalize_all(cfg)
    if not normalized_files:
        logger.error("没有成功标准化的音频，流水线终止。")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. VAD 检测
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("[Step 2] Silero VAD 语音活动检测...")
    vad_results = detect_all(cfg)
    if not vad_results:
        logger.error("VAD 检测结果为空，流水线终止。")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Offset 估算
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("[Step 3] Offset 估算...")
    schedule = load_schedule(cfg)
    logger.info("  加载 schedule.csv: %d 个问题", len(schedule))
    offset_estimates = estimate_all_offsets(vad_results, cfg)

    # ------------------------------------------------------------------
    # 4. 音频切分
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("[Step 4] 音频切分...")
    split_results = split_all(vad_results, offset_estimates, cfg)

    # ------------------------------------------------------------------
    # 5. 生成报告
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("[Step 5] 生成质检报告...")
    report_path = generate_report(split_results, cfg)

    # ------------------------------------------------------------------
    # 6. Beep-based 工作流（可选）
    # ------------------------------------------------------------------
    if cfg.get("beep", {}).get("enabled", False):
        _run_beep_workflow(vad_results, cfg)

    # ------------------------------------------------------------------
    # 完成
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline 完成! 耗时 %.1f 秒", elapsed)
    logger.info("报告路径: %s", report_path)
    if cfg.get("beep", {}).get("enabled", False):
        logger.info("Beep 切分目录: data/segments_beep/")
        logger.info("Beep 报告: data/reports/beep_*.csv")
    logger.info("=" * 60)


def main():
    """命令行入口。"""
    import argparse
    parser = argparse.ArgumentParser(
        description="口语回答音频批量切分 Pipeline"
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="配置文件路径 (默认: config/settings.yaml)",
    )
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()

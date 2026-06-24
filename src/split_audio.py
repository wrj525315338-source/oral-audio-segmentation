"""
Step 4: 音频切分
根据 offset + schedule.csv + 缓冲区，使用 FFmpeg 批量切出每个被试每道题的回答音频。
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import (
    OffsetEstimate, ScheduleEntry, VADResult, SpeechSegment, SplitResult,
)
from .utils import PROJECT_ROOT, load_config, setup_logging

logger = setup_logging()


@dataclass
class SplitTask:
    """一个切分任务。"""
    participant_id: str
    question_id: str
    source_file: Path
    cut_start: float   # 切分起点（秒，含 pre_buffer）
    cut_end: float     # 切分终点（秒，含 post_buffer）
    output_file: Path


def _has_speech_in_window(
    window_start: float,
    window_end: float,
    segments: list[SpeechSegment],
) -> tuple[bool, float]:
    """
    检查给定窗口内是否有 VAD 检测到的语音。

    Returns
    -------
    tuple[bool, float]
        (是否有语音, 窗口内语音总时长)
    """
    total = 0.0
    for seg in segments:
        overlap_start = max(window_start, seg.start_sec)
        overlap_end = min(window_end, seg.end_sec)
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
    return total > 0, total


def build_split_tasks(
    participant_id: str,
    source_file: Path,
    offset_estimate: OffsetEstimate,
    schedule: list[ScheduleEntry],
    vad_result: VADResult,
    cfg: dict,
) -> list[SplitTask]:
    """
    为单个被试构建所有切分任务。

    根据 offset + schedule + buffer 计算每道题的切分边界。
    """
    split_cfg = cfg["split"]
    pre_buf = split_cfg["pre_buffer"]
    post_buf = split_cfg["post_buffer"]
    segments_dir = PROJECT_ROOT / cfg["paths"]["segments_dir"]

    offset = offset_estimate.estimated_offset_sec
    tasks: list[SplitTask] = []

    for entry in schedule:
        # 计算切分边界（标准窗口 + offset + buffer）
        cut_start = max(0.0, entry.answer_start + offset - pre_buf)
        cut_end = entry.answer_end + offset + post_buf

        output_name = f"{participant_id}_{entry.question_id}.wav"
        output_file = segments_dir / output_name

        tasks.append(SplitTask(
            participant_id=participant_id,
            question_id=entry.question_id,
            source_file=source_file,
            cut_start=cut_start,
            cut_end=cut_end,
            output_file=output_file,
        ))

    return tasks


def execute_split(task: SplitTask, duration_sec: float) -> bool:
    """使用 FFmpeg 执行单个切分任务。返回是否成功。"""
    task.output_file.parent.mkdir(parents=True, exist_ok=True)

    # 确保切分终点不超过音频时长
    actual_end = min(task.cut_end, duration_sec)
    if actual_end <= task.cut_start:
        logger.warning(
            "切分区间无效: %s [%s] start=%.2f, end=%.2f",
            task.participant_id, task.question_id,
            task.cut_start, actual_end,
        )
        return False

    duration = actual_end - task.cut_start
    cmd = [
        "ffmpeg", "-y",
        "-i", str(task.source_file),
        "-ss", f"{task.cut_start:.3f}",
        "-t", f"{duration:.3f}",
        "-acodec", "copy",   # 不重新编码（已经是标准 WAV）
        "-loglevel", "error",
        str(task.output_file),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.error(
                "FFmpeg 切分失败: %s [%s]\n%s",
                task.participant_id, task.question_id, result.stderr.strip(),
            )
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg 切分超时: %s [%s]", task.participant_id, task.question_id)
        return False


def split_all(
    vad_results: dict[str, VADResult],
    offset_estimates: dict[str, OffsetEstimate],
    cfg: dict | None = None,
) -> list[SplitResult]:
    """
    批量切分所有被试的所有题目。

    Parameters
    ----------
    vad_results : dict[str, VADResult]
        participant_id -> VADResult。
    offset_estimates : dict[str, OffsetEstimate]
        participant_id -> OffsetEstimate。

    Returns
    -------
    list[SplitResult]
        所有切分任务的结果，用于报告生成。
    """
    if cfg is None:
        cfg = load_config()

    from .estimate_offset import load_schedule
    schedule = load_schedule(cfg)

    skip_on_failure = cfg["split"]["skip_on_offset_failure"]

    all_results: list[SplitResult] = []
    total_tasks = 0
    success_tasks = 0

    for pid, vad_result in vad_results.items():
        if pid not in offset_estimates:
            logger.warning("缺少 offset 估算: %s，跳过", pid)
            continue

        offset_est = offset_estimates[pid]

        # 如果 offset 不可靠且配置要求跳过
        if not offset_est.is_reliable and skip_on_failure:
            logger.warning(
                "Offset 不可靠: %s (%s)，跳过切分",
                pid, offset_est.warning,
            )
            # 仍然记录报告
            for entry in schedule:
                all_results.append(SplitResult(
                    participant_id=pid,
                    source_file=str(vad_result.file_path.name),
                    duration_sec=vad_result.duration_sec,
                    detected_speech_segments=len(vad_result.segments),
                    estimated_offset_sec=offset_est.estimated_offset_sec,
                    offset_confidence=offset_est.confidence,
                    question_id=entry.question_id,
                    cut_start_sec=0.0,
                    cut_end_sec=0.0,
                    output_file="",
                    has_detected_speech_in_window=False,
                    speech_duration_in_window=0.0,
                    warning=f"跳过: offset 不可靠 ({offset_est.warning})",
                ))
            continue

        # 构建切分任务
        tasks = build_split_tasks(
            pid, vad_result.file_path, offset_est, schedule, vad_result, cfg,
        )

        for task in tasks:
            total_tasks += 1
            # 检查窗口内是否有语音
            has_speech, speech_dur = _has_speech_in_window(
                task.cut_start, task.cut_end, vad_result.segments,
            )

            # 执行切分
            ok = execute_split(task, vad_result.duration_sec)
            if ok:
                success_tasks += 1

            warning = ""
            if not ok:
                warning = "切分失败"
            elif not has_speech:
                warning = "窗口内未检测到语音"

            all_results.append(SplitResult(
                participant_id=pid,
                source_file=str(vad_result.file_path.name),
                duration_sec=vad_result.duration_sec,
                detected_speech_segments=len(vad_result.segments),
                estimated_offset_sec=offset_est.estimated_offset_sec,
                offset_confidence=offset_est.confidence,
                question_id=task.question_id,
                cut_start_sec=task.cut_start,
                cut_end_sec=task.cut_end,
                output_file=str(task.output_file.name) if ok else "",
                has_detected_speech_in_window=has_speech,
                speech_duration_in_window=speech_dur,
                warning=warning,
            ))

    logger.info("切分完成: %d/%d 成功", success_tasks, total_tasks)
    return all_results

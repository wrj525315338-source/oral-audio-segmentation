"""
Step 3: Offset 估算
基于 VAD 检测结果和 schedule.csv，估算每个学生音频相对于标准视频的时间偏移量。

核心算法：
1. 读取 schedule.csv 中每个问题的标准回答窗口 [answer_start, answer_end]。
2. 对于每个候选 offset（由 detected_speech_start - answer_start 生成），
   将所有标准窗口平移 offset，计算与 VAD 语音片段的总重叠时长。
3. 选择总重叠时长最高的 offset 作为最佳估计。
"""
from __future__ import annotations

import csv
from pathlib import Path

from .models import ScheduleEntry, OffsetEstimate, VADResult, SpeechSegment
from .utils import PROJECT_ROOT, load_config, setup_logging

logger = setup_logging()


def load_schedule(cfg: dict | None = None) -> list[ScheduleEntry]:
    """读取 config/schedule.csv。"""
    if cfg is None:
        cfg = load_config()
    schedule_path = PROJECT_ROOT / cfg["paths"]["schedule_csv"]
    entries: list[ScheduleEntry] = []
    with open(schedule_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(ScheduleEntry(
                question_id=row["question_id"].strip(),
                answer_start=float(row["answer_start"]),
                answer_end=float(row["answer_end"]),
            ))
    return entries


def _compute_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """计算两个区间的重叠时长（秒）。"""
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    return max(0.0, overlap_end - overlap_start)


def _total_overlap_for_offset(
    offset: float,
    schedule: list[ScheduleEntry],
    segments: list[SpeechSegment],
) -> tuple[float, int]:
    """
    给定一个 offset，计算所有标准窗口平移后与 VAD 片段的总重叠时长。

    Parameters
    ----------
    offset : float
        候选偏移量（秒）。标准窗口 [start, end] 平移为 [start+offset, end+offset]。
    schedule : list[ScheduleEntry]
        标准实验时间表。
    segments : list[SpeechSegment]
        VAD 检测到的语音片段。

    Returns
    -------
    tuple[float, int]
        (总重叠时长, 检测到语音的窗口数)
    """
    total_overlap = 0.0
    windows_with_speech = 0

    for entry in schedule:
        # 平移后的回答窗口
        win_start = entry.answer_start + offset
        win_end = entry.answer_end + offset

        # 计算该窗口与所有 VAD 片段的重叠
        window_overlap = 0.0
        for seg in segments:
            window_overlap += _compute_overlap(
                win_start, win_end, seg.start_sec, seg.end_sec
            )

        if window_overlap > 0:
            windows_with_speech += 1
        total_overlap += window_overlap

    return total_overlap, windows_with_speech


def _generate_candidate_offsets(
    segments: list[SpeechSegment],
    schedule: list[ScheduleEntry],
    cfg: dict,
) -> list[float]:
    """
    生成候选 offset 列表。

    方法：对于每个 VAD 语音段的起始时间，减去每个标准 answer_start，
    得到候选 offset。加上搜索范围内的均匀采样。
    """
    offset_cfg = cfg["offset"]
    search_min = offset_cfg["search_range_min"]
    search_max = offset_cfg["search_range_max"]
    step = offset_cfg["step"]

    candidates: set[float] = set()

    # 方法1：由 VAD 段起始 - 标准 answer_start 生成
    for seg in segments:
        for entry in schedule:
            offset = round(seg.start_sec - entry.answer_start, 2)
            if search_min <= offset <= search_max:
                candidates.add(offset)

    # 方法2：均匀采样搜索范围
    t = search_min
    while t <= search_max:
        candidates.add(round(t, 2))
        t += step

    return sorted(candidates)


def estimate_offset(
    participant_id: str,
    vad_result: VADResult,
    schedule: list[ScheduleEntry],
    cfg: dict | None = None,
) -> OffsetEstimate:
    """
    估算单个被试的 offset。

    算法：
    1. 生成候选 offset 列表。
    2. 对每个候选 offset，计算所有标准窗口平移后的总重叠时长。
    3. 选择总重叠时长最高的 offset。
    4. 根据置信度阈值判断是否可靠。
    """
    if cfg is None:
        cfg = load_config()

    offset_cfg = cfg["offset"]
    min_confidence = offset_cfg["min_confidence"]
    min_windows = offset_cfg["min_windows_with_speech"]

    segments = vad_result.segments
    total_window_duration = sum(e.answer_end - e.answer_start for e in schedule)

    # 如果没有检测到语音段，直接返回失败
    if not segments:
        return OffsetEstimate(
            participant_id=participant_id,
            estimated_offset_sec=0.0,
            confidence=0.0,
            windows_with_speech=0,
            total_windows=len(schedule),
            is_reliable=False,
            warning="未检测到任何语音段",
        )

    # 生成候选 offset
    candidates = _generate_candidate_offsets(segments, schedule, cfg)
    if not candidates:
        return OffsetEstimate(
            participant_id=participant_id,
            estimated_offset_sec=0.0,
            confidence=0.0,
            windows_with_speech=0,
            total_windows=len(schedule),
            is_reliable=False,
            warning="无法生成候选 offset",
        )

    # 搜索最佳 offset
    best_offset = 0.0
    best_score = -1.0
    best_windows = 0

    for offset in candidates:
        score, n_windows = _total_overlap_for_offset(offset, schedule, segments)
        if score > best_score:
            best_score = score
            best_offset = offset
            best_windows = n_windows

    # 计算置信度 = 总重叠时长 / 总标准窗口时长
    confidence = best_score / total_window_duration if total_window_duration > 0 else 0.0

    # 判断是否可靠
    is_reliable = (confidence >= min_confidence) and (best_windows >= min_windows)

    warning = ""
    if not is_reliable:
        reasons = []
        if confidence < min_confidence:
            reasons.append(f"置信度 {confidence:.3f} < 阈值 {min_confidence}")
        if best_windows < min_windows:
            reasons.append(f"检测到语音的窗口 {best_windows} < 阈值 {min_windows}")
        warning = "; ".join(reasons)

    estimate = OffsetEstimate(
        participant_id=participant_id,
        estimated_offset_sec=best_offset,
        confidence=confidence,
        windows_with_speech=best_windows,
        total_windows=len(schedule),
        is_reliable=is_reliable,
        warning=warning,
    )

    logger.info(
        "Offset: %s | %.2fs | 置信度 %.3f | 窗口 %d/%d | %s",
        participant_id, best_offset, confidence,
        best_windows, len(schedule),
        "可靠" if is_reliable else f"不可靠: {warning}",
    )
    return estimate


def estimate_all_offsets(
    vad_results: dict[str, VADResult],
    cfg: dict | None = None,
) -> dict[str, OffsetEstimate]:
    """
    为所有被试估算 offset。

    Parameters
    ----------
    vad_results : dict[str, VADResult]
        participant_id -> VADResult。

    Returns
    -------
    dict[str, OffsetEstimate]
        participant_id -> OffsetEstimate。
    """
    if cfg is None:
        cfg = load_config()

    schedule = load_schedule(cfg)
    logger.info("开始估算 offset，共 %d 个被试...", len(vad_results))

    estimates: dict[str, OffsetEstimate] = {}
    for pid, vad_result in vad_results.items():
        est = estimate_offset(pid, vad_result, schedule, cfg)
        estimates[pid] = est

    reliable = sum(1 for e in estimates.values() if e.is_reliable)
    logger.info("Offset 估算完成: %d/%d 可靠", reliable, len(estimates))
    return estimates


if __name__ == "__main__":
    from .vad_detect import detect_all
    cfg = load_config()
    vad_results = detect_all(cfg)
    estimates = estimate_all_offsets(vad_results, cfg)
    for pid, est in estimates.items():
        print(
            f"{pid}: offset={est.estimated_offset_sec:.2f}s, "
            f"confidence={est.confidence:.3f}, "
            f"windows={est.windows_with_speech}/{est.total_windows}, "
            f"reliable={est.is_reliable}"
        )

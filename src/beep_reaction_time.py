"""
Reaction Time 计算模块
基于 beep 时间和 VAD 检测结果，计算学生从 beep 到首次开口的反应时间。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .models import BeepAlignment, ReactionTimeResult
from .utils import PROJECT_ROOT, load_config, setup_logging

logger = setup_logging()


def _find_first_speech_in_window(
    window_start: float,
    window_end: float,
    vad_segments: list,
) -> float | None:
    """
    在给定窗口内找到第一个语音段的起始时间。

    Returns
    -------
    float | None
        第一个语音段的起始时间，如果没有语音则返回 None。
        如果语音段在 beep 之前就开始并延续到窗口内，返回原始
        起始时间（可能 < window_start），以便调用方正确标记为
        onset_before_beep。
    """
    for seg in vad_segments:
        # 语音段在窗口内
        if seg.start_sec >= window_start and seg.start_sec <= window_end:
            return seg.start_sec
        # 语音段跨越窗口起始边界（beep 前就开始的语音）
        if seg.start_sec < window_start and seg.end_sec > window_start:
            return seg.start_sec  # 返回实际起始时间，由调用方判断是否 onset_before_beep

    return None


def calculate_reaction_time_for_participant(
    participant_id: str,
    alignments: dict[str, BeepAlignment],
    vad_segments: list,
    beep_template_duration: float,
    cfg: dict,
) -> list[ReactionTimeResult]:
    """
    为单个被试计算所有问题的 reaction time。

    Parameters
    ----------
    participant_id : str
        被试 ID。
    alignments : dict[str, BeepAlignment]
        question_id -> BeepAlignment。
    vad_segments : list
        VAD 检测到的语音片段列表。
    beep_template_duration : float
        beep 模板的时长（秒）。
    cfg : dict
        配置字典。

    Returns
    -------
    list[ReactionTimeResult]
        每道题的 reaction time 结果。
    """
    beep_cfg = cfg["beep"]
    rt_reference = beep_cfg["rt_reference"]  # "beep_offset" 或 "beep_onset"
    max_rt = 15.0  # 最大合理 RT（秒）

    results: list[ReactionTimeResult] = []

    for qid in sorted(alignments.keys()):
        alignment = alignments[qid]

        # 跳过缺失的 beep
        if alignment.beep_source in ("manual_required", "missing"):
            results.append(ReactionTimeResult(
                participant_id=participant_id,
                question_id=qid,
                beep_time_sec=alignment.beep_time_sec,
                beep_source=alignment.beep_source,
                first_speech_start_sec=0.0,
                rt_from_beep_onset_sec=0.0,
                rt_from_beep_offset_sec=0.0,
                reaction_time_sec=0.0,
                rt_status="missing_beep",
                warning="beep 缺失，无法计算 RT",
            ))
            continue

        beep_onset = alignment.beep_time_sec
        beep_offset = beep_onset + beep_template_duration
        answer_duration = alignment.answer_duration_sec

        # 搜索窗口：beep 后到回答结束
        window_start = beep_onset
        window_end = beep_onset + answer_duration

        # 在窗口内找第一个语音
        first_speech = _find_first_speech_in_window(
            window_start, window_end, vad_segments,
        )

        if first_speech is None:
            results.append(ReactionTimeResult(
                participant_id=participant_id,
                question_id=qid,
                beep_time_sec=beep_onset,
                beep_source=alignment.beep_source,
                first_speech_start_sec=0.0,
                rt_from_beep_onset_sec=0.0,
                rt_from_beep_offset_sec=0.0,
                reaction_time_sec=0.0,
                rt_status="no_speech_detected",
                warning="窗口内未检测到语音",
            ))
            continue

        # 计算 RT
        rt_from_onset = first_speech - beep_onset
        rt_from_offset = first_speech - beep_offset

        # 选择主 RT
        if rt_reference == "beep_offset":
            reaction_time = rt_from_offset
        else:
            reaction_time = rt_from_onset

        # 判断 RT 状态
        rt_status = "valid"
        warning = ""

        if rt_from_onset < 0:
            rt_status = "onset_before_beep"
            warning = f"语音在 beep 之前开始 ({rt_from_onset:.3f}s)"
        elif reaction_time > max_rt:
            rt_status = "rt_too_long"
            warning = f"RT 过长 ({reaction_time:.3f}s > {max_rt}s)"

        results.append(ReactionTimeResult(
            participant_id=participant_id,
            question_id=qid,
            beep_time_sec=round(beep_onset, 4),
            beep_source=alignment.beep_source,
            first_speech_start_sec=round(first_speech, 4),
            rt_from_beep_onset_sec=round(rt_from_onset, 4),
            rt_from_beep_offset_sec=round(rt_from_offset, 4),
            reaction_time_sec=round(reaction_time, 4),
            rt_status=rt_status,
            warning=warning,
        ))

    return results


def calculate_all_reaction_times(
    all_alignments: dict[str, dict[str, BeepAlignment]],
    vad_results: dict,
    cfg: dict | None = None,
) -> list[ReactionTimeResult]:
    """
    为所有被试计算 reaction time。

    Parameters
    ----------
    all_alignments : dict
        participant_id -> {question_id: BeepAlignment}。
    vad_results : dict
        participant_id -> VADResult。
    cfg : dict
        配置字典。

    Returns
    -------
    list[ReactionTimeResult]
        所有结果列表。
    """
    if cfg is None:
        cfg = load_config()

    # 获取 beep 模板时长
    template_path = PROJECT_ROOT / cfg["beep"]["template_path"]
    import soundfile as sf
    template_info = sf.info(str(template_path))
    beep_template_duration = template_info.duration

    logger.info("开始计算 reaction time，共 %d 个被试...", len(all_alignments))

    all_results: list[ReactionTimeResult] = []

    for pid in sorted(all_alignments.keys()):
        alignments = all_alignments[pid]
        vad_result = vad_results.get(pid)
        vad_segments = vad_result.segments if vad_result else []

        results = calculate_reaction_time_for_participant(
            pid, alignments, vad_segments, beep_template_duration, cfg,
        )
        all_results.extend(results)

    # 输出报告
    _write_rt_report(all_results, cfg)

    valid_count = sum(1 for r in all_results if r.rt_status == "valid")
    logger.info(
        "RT 计算完成: %d/%d 有效",
        valid_count, len(all_results),
    )

    return all_results


def _write_rt_report(
    results: list[ReactionTimeResult],
    cfg: dict,
) -> Path:
    """输出 beep_reaction_time_report.csv。"""
    reports_dir = PROJECT_ROOT / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "beep_reaction_time_report.csv"

    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "participant_id", "question_id", "beep_time_sec", "beep_source",
            "first_speech_start_sec", "rt_from_beep_onset_sec",
            "rt_from_beep_offset_sec", "reaction_time_sec", "rt_status", "warning",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "participant_id": r.participant_id,
                "question_id": r.question_id,
                "beep_time_sec": f"{r.beep_time_sec:.4f}",
                "beep_source": r.beep_source,
                "first_speech_start_sec": f"{r.first_speech_start_sec:.4f}",
                "rt_from_beep_onset_sec": f"{r.rt_from_beep_onset_sec:.4f}",
                "rt_from_beep_offset_sec": f"{r.rt_from_beep_offset_sec:.4f}",
                "reaction_time_sec": f"{r.reaction_time_sec:.4f}",
                "rt_status": r.rt_status,
                "warning": r.warning,
            })

    logger.info("RT 报告: %s", report_path)
    return report_path


if __name__ == "__main__":
    from .beep_detect import detect_all_beeps
    from .beep_align import align_all_beeps
    from .vad_detect import detect_all

    cfg = load_config()
    vad_results = detect_all(cfg)
    candidates = detect_all_beeps(cfg)
    alignments, _ = align_all_beeps(candidates, cfg)
    results = calculate_all_reaction_times(alignments, vad_results, cfg)

    for r in results:
        print(
            f"{r.participant_id}_{r.question_id}: "
            f"RT={r.reaction_time_sec:.3f}s ({r.rt_status})"
        )

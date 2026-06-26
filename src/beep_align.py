"""
Beep 对齐模块
将检测到的 beep 候选与 beep_schedule.csv 中的问题对齐。
支持人工标注覆盖和推断缺失 beep。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .beep_detect import BeepCandidate
from .models import BeepScheduleEntry, ManualBeepTime, BeepAlignment
from .utils import PROJECT_ROOT, load_config, setup_logging

logger = setup_logging()


def load_beep_schedule(cfg: dict | None = None) -> list[BeepScheduleEntry]:
    """读取 config/beep_schedule.csv。"""
    if cfg is None:
        cfg = load_config()
    schedule_path = PROJECT_ROOT / cfg["beep"]["beep_schedule_csv"]
    entries: list[BeepScheduleEntry] = []
    with open(schedule_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(BeepScheduleEntry(
                question_id=row["question_id"].strip(),
                beep_offset_from_q01=float(row["beep_offset_from_q01"]),
                answer_duration_sec=float(row["answer_duration_sec"]),
            ))
    return entries


def load_manual_beep_times(cfg: dict | None = None) -> dict[str, list[ManualBeepTime]]:
    """
    读取 config/manual_beep_times.csv。

    Returns
    -------
    dict[str, list[ManualBeepTime]]
        participant_id -> 人工标注列表。
    """
    if cfg is None:
        cfg = load_config()
    manual_path = PROJECT_ROOT / cfg["beep"]["manual_beep_csv"]
    result: dict[str, list[ManualBeepTime]] = {}

    if not manual_path.exists():
        return result

    with open(manual_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["participant_id"].strip()
            qid = row["question_id"].strip() if row.get("question_id") else ""
            beep_time = float(row["beep_time_sec"])
            notes = row.get("notes", "").strip()

            entry = ManualBeepTime(
                participant_id=pid,
                question_id=qid,
                beep_time_sec=beep_time,
                notes=notes,
            )
            if pid not in result:
                result[pid] = []
            result[pid].append(entry)

    return result


def _find_q01_beep_from_candidates(
    candidates: list[BeepCandidate],
    beep_schedule: list[BeepScheduleEntry],
    tolerance: float,
) -> BeepCandidate | None:
    """
    从候选列表中找到最可能是 Q01 beep 的候选。

    策略：对每个候选，假设它是 Q01，计算有多少其他候选能匹配到
    beep_schedule 中的后续题目。选择匹配数最多的候选；平局时
    优先选最早出现的（避免将后续 beep 误判为 Q01）。
    """
    if not candidates:
        return None

    best_candidate: BeepCandidate | None = None
    best_match_count = -1
    best_time = float('inf')

    for candidate in candidates:
        q01_time = candidate.detected_beep_time_sec
        match_count = 0

        for entry in beep_schedule:
            if entry.beep_offset_from_q01 == 0.0:
                continue  # 跳过 Q01 自身
            expected_time = q01_time + entry.beep_offset_from_q01

            for other in candidates:
                if abs(other.detected_beep_time_sec - expected_time) <= tolerance:
                    match_count += 1
                    break

        # 优先选匹配数最多的；平局选最早出现的
        if (match_count > best_match_count or
                (match_count == best_match_count and q01_time < best_time)):
            best_match_count = match_count
            best_candidate = candidate
            best_time = q01_time

    return best_candidate


def _match_beeps_to_questions(
    q01_beep: BeepCandidate,
    all_candidates: list[BeepCandidate],
    beep_schedule: list[BeepScheduleEntry],
    tolerance: float,
) -> dict[str, BeepAlignment]:
    """
    根据 Q01 beep 和固定时间间隔，将候选匹配到各问题。

    Parameters
    ----------
    q01_beep : BeepCandidate
        Q01 的 beep 候选。
    all_candidates : list[BeepCandidate]
        所有检测到的 beep 候选。
    beep_schedule : list[BeepScheduleEntry]
        beep 时间表。
    tolerance : float
        匹配容差（秒）。

    Returns
    -------
    dict[str, BeepAlignment]
        question_id -> BeepAlignment。
    """
    q01_time = q01_beep.detected_beep_time_sec
    alignments: dict[str, BeepAlignment] = {}

    for entry in beep_schedule:
        # 预期的 beep 时间 = Q01 beep 时间 + 相对偏移
        expected_time = q01_time + entry.beep_offset_from_q01

        # 在候选中查找最接近的
        best_match: BeepCandidate | None = None
        best_dist = float('inf')

        for candidate in all_candidates:
            dist = abs(candidate.detected_beep_time_sec - expected_time)
            if dist < tolerance and dist < best_dist:
                best_dist = dist
                best_match = candidate

        if best_match is not None:
            # 检测到了该题的 beep
            alignments[entry.question_id] = BeepAlignment(
                question_id=entry.question_id,
                beep_time_sec=best_match.detected_beep_time_sec,
                beep_source="detected",
                beep_confidence=best_match.confidence,
                answer_duration_sec=entry.answer_duration_sec,
                warning="",
            )
        else:
            # 没有检测到，根据间隔推断
            alignments[entry.question_id] = BeepAlignment(
                question_id=entry.question_id,
                beep_time_sec=round(expected_time, 4),
                beep_source="inferred_from_detected_beeps",
                beep_confidence=q01_beep.confidence,  # 使用 Q01 的置信度
                answer_duration_sec=entry.answer_duration_sec,
                warning="beep 未直接检测到，由 Q01 间隔推断",
            )

    return alignments


def align_beeps_for_participant(
    participant_id: str,
    candidates: list[BeepCandidate],
    beep_schedule: list[BeepScheduleEntry],
    manual_times: list[ManualBeepTime] | None,
    cfg: dict,
) -> dict[str, BeepAlignment]:
    """
    为单个被试对齐 beep 到各问题。

    优先级：
    1. manual_beep_times.csv 中的人工标注（如果 manual_override=true）
    2. 自动检测结果
    3. 根据间隔推断
    """
    beep_cfg = cfg["beep"]
    tolerance = beep_cfg["search_tolerance_sec"]
    manual_override = beep_cfg["manual_override"]

    alignments: dict[str, BeepAlignment] = {}

    # 创建 manual lookup: question_id -> ManualBeepTime
    manual_lookup: dict[str, ManualBeepTime] = {}
    manual_q01: ManualBeepTime | None = None
    if manual_times:
        for mt in manual_times:
            if mt.question_id:
                manual_lookup[mt.question_id] = mt
            if mt.question_id == "Q01" or not mt.question_id:
                manual_q01 = mt

    # 情况1: manual_beep_times.csv 中有 Q01 人工标注
    if manual_override and manual_q01 is not None:
        q01_time = manual_q01.beep_time_sec
        for entry in beep_schedule:
            expected_time = q01_time + entry.beep_offset_from_q01

            # 检查该题是否也有人工标注
            if entry.question_id in manual_lookup:
                # 使用该题的人工标注
                mt = manual_lookup[entry.question_id]
                alignments[entry.question_id] = BeepAlignment(
                    question_id=entry.question_id,
                    beep_time_sec=mt.beep_time_sec,
                    beep_source="manual",
                    beep_confidence=1.0,
                    answer_duration_sec=entry.answer_duration_sec,
                    warning="",
                )
            else:
                # 由 Q01 人工标注推断
                alignments[entry.question_id] = BeepAlignment(
                    question_id=entry.question_id,
                    beep_time_sec=round(expected_time, 4),
                    beep_source="manual_anchor_inferred",
                    beep_confidence=1.0,
                    answer_duration_sec=entry.answer_duration_sec,
                    warning="由人工标注的 Q01 beep 推断",
                )

        logger.info("使用人工标注: %s Q01=%.3fs", participant_id, q01_time)
        return alignments

    # 情况2: 自动检测
    if not candidates:
        # 没有检测到任何 beep
        for entry in beep_schedule:
            alignments[entry.question_id] = BeepAlignment(
                question_id=entry.question_id,
                beep_time_sec=0.0,
                beep_source="manual_required",
                beep_confidence=0.0,
                answer_duration_sec=entry.answer_duration_sec,
                warning="未检测到 beep，需要人工标注",
            )
        logger.warning("未检测到 beep: %s，需要人工标注", participant_id)
        return alignments

    # 找到 Q01 beep
    q01_beep = _find_q01_beep_from_candidates(candidates, beep_schedule, tolerance)

    if q01_beep is None:
        for entry in beep_schedule:
            alignments[entry.question_id] = BeepAlignment(
                question_id=entry.question_id,
                beep_time_sec=0.0,
                beep_source="manual_required",
                beep_confidence=0.0,
                answer_duration_sec=entry.answer_duration_sec,
                warning="无法确定 Q01 beep，需要人工标注",
            )
        return alignments

    # 根据 Q01 和间隔匹配所有问题
    alignments = _match_beeps_to_questions(
        q01_beep, candidates, beep_schedule, tolerance,
    )

    # 如果某题有人工标注，覆盖自动检测结果
    if manual_override and manual_times:
        for entry in beep_schedule:
            if entry.question_id in manual_lookup:
                mt = manual_lookup[entry.question_id]
                alignments[entry.question_id] = BeepAlignment(
                    question_id=entry.question_id,
                    beep_time_sec=mt.beep_time_sec,
                    beep_source="manual",
                    beep_confidence=1.0,
                    answer_duration_sec=entry.answer_duration_sec,
                    warning="人工标注覆盖自动检测",
                )

    return alignments


def align_all_beeps(
    all_candidates: dict[str, list[BeepCandidate]],
    cfg: dict | None = None,
) -> tuple[dict[str, dict[str, BeepAlignment]], list[dict[str, str]]]:
    """
    为所有被试对齐 beep。

    Returns
    -------
    tuple[dict, list]
        (对齐结果, 人工介入列表)
    """
    if cfg is None:
        cfg = load_config()

    beep_schedule = load_beep_schedule(cfg)
    manual_times = load_manual_beep_times(cfg)

    # 获取所有被试 ID
    norm_dir = PROJECT_ROOT / cfg["paths"]["normalized_dir"]
    from .utils import participant_id_from_filename
    all_pids = [participant_id_from_filename(f.name) for f in sorted(norm_dir.glob("*.wav"))]

    logger.info("开始 beep 对齐，共 %d 个被试...", len(all_pids))

    all_alignments: dict[str, dict[str, BeepAlignment]] = {}
    manual_required: list[dict[str, str]] = []

    for pid in all_pids:
        candidates = all_candidates.get(pid, [])
        manual = manual_times.get(pid, None)

        alignments = align_beeps_for_participant(
            pid, candidates, beep_schedule, manual, cfg,
        )
        all_alignments[pid] = alignments

        # 检查是否需要人工介入
        q01_alignment = alignments.get("Q01")
        if q01_alignment and q01_alignment.beep_source == "manual_required":
            manual_required.append({
                "participant_id": pid,
                "normalized_audio_file": f"{pid}.wav",
                "reason": q01_alignment.warning,
                "suggested_action": (
                    f"Please open data/normalized/{pid}.wav, find the Q01 beep "
                    "time in seconds, and add it to config/manual_beep_times.csv."
                ),
            })

    # 输出报告
    _write_alignment_report(all_alignments, cfg)
    if cfg["beep"]["generate_manual_required_csv"]:
        _write_manual_required_report(manual_required, cfg)

    reliable = sum(
        1 for aligns in all_alignments.values()
        if any(a.beep_source in ("detected", "manual", "manual_anchor_inferred")
               for a in aligns.values())
    )
    logger.info("Beep 对齐完成: %d/%d 可靠", reliable, len(all_pids))

    return all_alignments, manual_required


def _write_alignment_report(
    all_alignments: dict[str, dict[str, BeepAlignment]],
    cfg: dict,
) -> Path:
    """输出 beep_alignment_report.csv。"""
    reports_dir = PROJECT_ROOT / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "beep_alignment_report.csv"

    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "participant_id", "question_id", "beep_time_sec",
            "beep_source", "beep_confidence", "warning",
        ])
        writer.writeheader()
        for pid in sorted(all_alignments.keys()):
            for qid in sorted(all_alignments[pid].keys()):
                a = all_alignments[pid][qid]
                writer.writerow({
                    "participant_id": pid,
                    "question_id": a.question_id,
                    "beep_time_sec": f"{a.beep_time_sec:.4f}",
                    "beep_source": a.beep_source,
                    "beep_confidence": f"{a.beep_confidence:.4f}",
                    "warning": a.warning,
                })

    logger.info("beep 对齐报告: %s", report_path)
    return report_path


def _write_manual_required_report(
    manual_required: list[dict[str, str]],
    cfg: dict,
) -> Path:
    """输出 manual_beep_required.csv。"""
    reports_dir = PROJECT_ROOT / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "manual_beep_required.csv"

    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "participant_id", "normalized_audio_file", "reason", "suggested_action",
        ])
        writer.writeheader()
        for row in manual_required:
            writer.writerow(row)

    if manual_required:
        logger.warning("有 %d 个被试需要人工标注 beep，见: %s", len(manual_required), report_path)
    else:
        logger.info("所有被试 beep 对齐成功，无需人工介入")

    return report_path


if __name__ == "__main__":
    from .beep_detect import detect_all_beeps
    cfg = load_config()
    candidates = detect_all_beeps(cfg)
    alignments, manual_req = align_all_beeps(candidates, cfg)
    for pid, aligns in alignments.items():
        print(f"\n{pid}:")
        for qid, a in aligns.items():
            print(f"  {qid}: {a.beep_time_sec:.3f}s ({a.beep_source}, conf={a.beep_confidence:.3f})")

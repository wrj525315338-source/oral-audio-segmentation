"""
Beep-based 音频切分模块
根据 beep_alignment_report.csv 中的 beep_time_sec 切分音频。
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import BeepAlignment, BeepSplitResult
from .utils import PROJECT_ROOT, load_config, setup_logging

logger = setup_logging()


def _has_vad_speech_in_window(
    window_start: float,
    window_end: float,
    vad_segments: list,
) -> tuple[bool, float]:
    """检查窗口内是否有 VAD 检测到的语音。"""
    total = 0.0
    for seg in vad_segments:
        overlap_start = max(window_start, seg.start_sec)
        overlap_end = min(window_end, seg.end_sec)
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
    return total > 0, total


def _execute_beep_split(
    source_file: Path,
    output_file: Path,
    cut_start: float,
    cut_end: float,
    duration_sec: float,
) -> bool:
    """使用 FFmpeg 执行单个切分任务。"""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 确保切分边界有效
    actual_start = max(0.0, cut_start)
    actual_end = min(cut_end, duration_sec)
    if actual_end <= actual_start:
        return False

    ffmpeg_duration = actual_end - actual_start
    cmd = [
        "ffmpeg", "-y",
        "-i", str(source_file),
        "-ss", f"{actual_start:.3f}",
        "-t", f"{ffmpeg_duration:.3f}",
        "-acodec", "copy",
        "-loglevel", "error",
        str(output_file),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def split_by_beep(
    all_alignments: dict[str, dict[str, BeepAlignment]],
    vad_results: dict,
    cfg: dict | None = None,
) -> list[BeepSplitResult]:
    """
    根据 beep 对齐结果切分所有音频。

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
    list[BeepSplitResult]
        所有切分任务的结果。
    """
    if cfg is None:
        cfg = load_config()

    beep_cfg = cfg["beep"]
    pre_buffer = beep_cfg["pre_buffer_sec"]
    post_buffer = beep_cfg["post_buffer_sec"]

    segments_beep_dir = PROJECT_ROOT / "data" / "segments_beep"
    segments_beep_dir.mkdir(parents=True, exist_ok=True)

    norm_dir = PROJECT_ROOT / cfg["paths"]["normalized_dir"]

    all_results: list[BeepSplitResult] = []
    total_tasks = 0
    success_tasks = 0

    for pid in sorted(all_alignments.keys()):
        alignments = all_alignments[pid]

        # 从 VAD 结果获取实际的标准化音频路径（避免 participant_id 与文件名不一致）
        vad_result = vad_results.get(pid)
        if vad_result:
            source_file = vad_result.file_path
        else:
            # 回退：尝试在 normalized 目录中查找
            from .utils import participant_id_from_filename
            matches = [
                f for f in norm_dir.glob("*.wav")
                if participant_id_from_filename(f.name) == pid
            ]
            if matches:
                source_file = matches[0]
            else:
                logger.warning("标准化音频不存在: %s", pid)
                continue

        if not source_file.exists():
            logger.warning("标准化音频不存在: %s", source_file)
            continue

        # 获取音频时长
        import soundfile as sf
        info = sf.info(str(source_file))
        duration_sec = info.duration

        vad_segments = vad_result.segments if vad_result else []

        for qid in sorted(alignments.keys()):
            alignment = alignments[qid]
            total_tasks += 1

            # 跳过需要人工标注或缺失的 beep
            if alignment.beep_source in ("manual_required", "missing"):
                all_results.append(BeepSplitResult(
                    participant_id=pid,
                    question_id=qid,
                    beep_time_sec=alignment.beep_time_sec,
                    beep_source=alignment.beep_source,
                    cut_start_sec=0.0,
                    cut_end_sec=0.0,
                    output_file="",
                    has_speech_in_window=False,
                    speech_duration_in_window=0.0,
                    warning=f"跳过: {alignment.beep_source}",
                ))
                continue

            # 计算切分边界
            beep_time = alignment.beep_time_sec
            answer_duration = alignment.answer_duration_sec
            cut_start = beep_time - pre_buffer
            cut_end = beep_time + answer_duration + post_buffer

            # 执行切分
            output_name = f"{pid}_{qid}.wav"
            output_file = segments_beep_dir / output_name

            ok = _execute_beep_split(source_file, output_file, cut_start, cut_end, duration_sec)

            if ok:
                success_tasks += 1

            # 检查窗口内是否有语音
            has_speech, speech_dur = _has_vad_speech_in_window(
                cut_start, cut_end, vad_segments,
            )

            warning = ""
            if not ok:
                warning = "切分失败"
            elif alignment.beep_source == "inferred_from_detected_beeps":
                warning = "beep 由间隔推断"
            elif not has_speech:
                warning = "窗口内未检测到语音"

            all_results.append(BeepSplitResult(
                participant_id=pid,
                question_id=qid,
                beep_time_sec=beep_time,
                beep_source=alignment.beep_source,
                cut_start_sec=cut_start,
                cut_end_sec=cut_end,
                output_file=output_name if ok else "",
                has_speech_in_window=has_speech,
                speech_duration_in_window=speech_dur,
                warning=warning,
            ))

    logger.info("Beep 切分完成: %d/%d 成功", success_tasks, total_tasks)
    return all_results


if __name__ == "__main__":
    from .beep_detect import detect_all_beeps
    from .beep_align import align_all_beeps
    from .vad_detect import detect_all

    cfg = load_config()
    vad_results = detect_all(cfg)
    candidates = detect_all_beeps(cfg)
    alignments, _ = align_all_beeps(candidates, cfg)
    results = split_by_beep(alignments, vad_results, cfg)

    for r in results:
        print(f"{r.participant_id}_{r.question_id}: {r.beep_source} | {r.warning}")

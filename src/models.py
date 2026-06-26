"""
数据模型定义：所有 dataclass 集中在此，避免跨模块的 torch 循环依赖。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SpeechSegment:
    """一个 VAD 检测到的语音片段。"""
    start_sec: float
    end_sec: float
    confidence: float = 1.0


@dataclass
class VADResult:
    """单个文件的 VAD 检测结果。"""
    file_path: Path
    duration_sec: float
    segments: list[SpeechSegment] = field(default_factory=list)


@dataclass
class ScheduleEntry:
    """schedule.csv 中的一行。"""
    question_id: str
    answer_start: float
    answer_end: float


@dataclass
class OffsetEstimate:
    """Offset 估算结果。"""
    participant_id: str
    estimated_offset_sec: float
    confidence: float
    windows_with_speech: int
    total_windows: int
    is_reliable: bool
    warning: str = ""


@dataclass
class SplitResult:
    """单个切分任务的结果，用于报告生成。"""
    participant_id: str
    source_file: str
    duration_sec: float
    detected_speech_segments: int
    estimated_offset_sec: float
    offset_confidence: float
    question_id: str
    cut_start_sec: float
    cut_end_sec: float
    output_file: str
    has_detected_speech_in_window: bool
    speech_duration_in_window: float
    warning: str


# ============================================================
# Beep-based 模式数据模型
# ============================================================


@dataclass
class BeepScheduleEntry:
    """beep_schedule.csv 中的一行。"""
    question_id: str
    beep_offset_from_q01: float
    answer_duration_sec: float


@dataclass
class ManualBeepTime:
    """manual_beep_times.csv 中的一行。"""
    participant_id: str
    question_id: str
    beep_time_sec: float
    notes: str = ""


@dataclass
class BeepAlignment:
    """beep 对齐结果。"""
    question_id: str
    beep_time_sec: float
    beep_source: str  # detected, manual, manual_anchor_inferred, inferred_from_detected_beeps, manual_required, missing
    beep_confidence: float
    answer_duration_sec: float
    warning: str = ""


@dataclass
class BeepSplitResult:
    """beep-based 切分结果。"""
    participant_id: str
    question_id: str
    beep_time_sec: float
    beep_source: str
    cut_start_sec: float
    cut_end_sec: float
    output_file: str
    has_speech_in_window: bool
    speech_duration_in_window: float
    warning: str


@dataclass
class ReactionTimeResult:
    """Reaction time 计算结果。"""
    participant_id: str
    question_id: str
    beep_time_sec: float
    beep_source: str
    first_speech_start_sec: float
    rt_from_beep_onset_sec: float
    rt_from_beep_offset_sec: float
    reaction_time_sec: float
    rt_status: str  # valid, no_speech_detected, onset_before_beep, rt_too_long, manual_required, missing_beep
    warning: str = ""

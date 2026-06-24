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

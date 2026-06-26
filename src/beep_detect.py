"""
Beep 检测模块
使用模板匹配（normalized cross-correlation）检测音频中的 beep 提示音。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags

from .utils import PROJECT_ROOT, load_config, setup_logging

logger = setup_logging()


@dataclass
class BeepCandidate:
    """单个 beep 候选检测结果。"""
    participant_id: str
    detected_beep_time_sec: float
    beep_duration_sec: float
    confidence: float


def _normalized_cross_correlation(
    signal: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    """
    计算归一化互相关 (normalized cross-correlation)。

    Parameters
    ----------
    signal : np.ndarray
        输入信号。
    template : np.ndarray
        模板信号。

    Returns
    -------
    np.ndarray
        归一化互相关值，范围 [-1, 1]。
    """
    # 标准化信号和模板
    signal = signal.astype(np.float64)
    template = template.astype(np.float64)

    # 去除直流分量
    signal = signal - np.mean(signal)
    template = template - np.mean(template)

    # 计算互相关
    correlation = correlate(signal, template, mode='valid')

    # 归一化：除以信号和模板的能量的几何平均
    template_energy = np.sum(template ** 2)
    if template_energy == 0:
        return np.zeros_like(correlation)

    # 滑动窗口计算信号能量
    signal_energy = np.convolve(signal ** 2, np.ones(len(template)), mode='valid')
    normalization = np.sqrt(signal_energy * template_energy)

    # 避免除零
    normalization = np.where(normalization == 0, 1.0, normalization)
    normalized_corr = correlation / normalization

    return normalized_corr


def detect_beeps_in_audio(
    audio_path: Path,
    template_path: Path,
    cfg: dict,
) -> list[BeepCandidate]:
    """
    在单个音频文件中检测 beep 提示音。

    Parameters
    ----------
    audio_path : Path
        标准化后的 WAV 文件路径。
    template_path : Path
        beep 模板音频路径。
    cfg : dict
        配置字典。

    Returns
    -------
    list[BeepCandidate]
        检测到的 beep 候选列表，按时间排序。
    """
    beep_cfg = cfg["beep"]
    min_confidence = beep_cfg["min_confidence"]
    sr_expected = cfg["audio"]["sample_rate"]

    # 读取音频和模板
    audio_data, sr_audio = sf.read(str(audio_path), dtype="float32")
    template_data, sr_template = sf.read(str(template_path), dtype="float32")

    # 确保采样率一致
    if sr_audio != sr_expected:
        raise ValueError(
            f"{audio_path.name}: 采样率 {sr_audio}Hz != {sr_expected}Hz，"
            "请先运行 normalize_audio.py"
        )
    if sr_template != sr_expected:
        raise ValueError(
            f"beep 模板: 采样率 {sr_template}Hz != {sr_expected}Hz，"
            "请重新生成 16kHz mono WAV 格式的模板"
        )

    # 转 mono
    if audio_data.ndim == 2:
        audio_data = audio_data.mean(axis=1)
    if template_data.ndim == 2:
        template_data = template_data.mean(axis=1)

    template_len = len(template_data)
    audio_len = len(audio_data)

    if template_len > audio_len:
        logger.warning("模板长度超过音频长度: %s", audio_path.name)
        return []

    # 计算归一化互相关
    ncc = _normalized_cross_correlation(audio_data, template_data)

    # 计算对应的时间轴（lag 对应的起始位置）
    # correlate(mode='valid') 的输出长度 = audio_len - template_len + 1
    # 第 i 个值对应模板在信号中的起始位置为 i
    lags = correlation_lags(audio_len, template_len, mode='valid')
    times_sec = lags / sr_audio

    # 找到超过阈值的峰值
    candidates: list[BeepCandidate] = []

    # 使用滑动窗口找局部最大值，避免重复检测
    # 最小间隔设为模板长度的一半
    min_gap_samples = template_len // 2

    # 找到所有超过阈值的位置
    above_threshold = np.where(ncc >= min_confidence)[0]

    if len(above_threshold) == 0:
        return []

    # 聚类：将相邻的超过阈值的位置分组，取每组最大值
    clusters: list[list[int]] = []
    current_cluster: list[int] = [above_threshold[0]]

    for idx in above_threshold[1:]:
        if idx - current_cluster[-1] <= min_gap_samples:
            current_cluster.append(idx)
        else:
            clusters.append(current_cluster)
            current_cluster = [idx]
    clusters.append(current_cluster)

    # 从每个聚类中取最大值
    for cluster in clusters:
        cluster_values = ncc[cluster]
        max_idx_in_cluster = cluster[np.argmax(cluster_values)]
        confidence = float(ncc[max_idx_in_cluster])

        if confidence >= min_confidence:
            beep_time = float(times_sec[max_idx_in_cluster])
            beep_duration = template_len / sr_audio

            candidates.append(BeepCandidate(
                participant_id="",  # 将在调用时填充
                detected_beep_time_sec=round(beep_time, 4),
                beep_duration_sec=round(beep_duration, 4),
                confidence=round(confidence, 4),
            ))

    # 按时间排序
    candidates.sort(key=lambda c: c.detected_beep_time_sec)

    return candidates


def detect_all_beeps(cfg: dict | None = None) -> dict[str, list[BeepCandidate]]:
    """
    对所有标准化音频进行 beep 检测。

    Returns
    -------
    dict[str, list[BeepCandidate]]
        participant_id -> beep 候选列表。
    """
    if cfg is None:
        cfg = load_config()

    beep_cfg = cfg["beep"]
    template_path = PROJECT_ROOT / beep_cfg["template_path"]

    if not template_path.exists():
        logger.error("beep 模板文件不存在: %s", template_path)
        return {}

    norm_dir = PROJECT_ROOT / cfg["paths"]["normalized_dir"]
    if not norm_dir.exists():
        logger.warning("标准化目录不存在: %s", norm_dir)
        return {}

    wav_files = sorted(norm_dir.glob("*.wav"))
    if not wav_files:
        logger.warning("在 %s 中未找到 WAV 文件", norm_dir)
        return {}

    from .utils import participant_id_from_filename

    logger.info("开始 beep 检测，共 %d 个文件...", len(wav_files))

    all_candidates: dict[str, list[BeepCandidate]] = {}
    for wav in wav_files:
        pid = participant_id_from_filename(wav.name)
        try:
            candidates = detect_beeps_in_audio(wav, template_path, cfg)
            # 填充 participant_id
            for c in candidates:
                c.participant_id = pid
            all_candidates[pid] = candidates
            logger.info(
                "Beep 检测: %s | 检测到 %d 个候选",
                wav.name, len(candidates),
            )
        except Exception as e:
            logger.error("Beep 检测失败: %s - %s", wav.name, e)
            all_candidates[pid] = []

    # 生成报告
    _write_candidates_report(all_candidates, cfg)

    total = sum(len(v) for v in all_candidates.values())
    logger.info("Beep 检测完成: %d 个文件，共 %d 个候选", len(all_candidates), total)

    return all_candidates


def _write_candidates_report(
    all_candidates: dict[str, list[BeepCandidate]],
    cfg: dict,
) -> Path:
    """输出 beep_candidates_report.csv。"""
    reports_dir = PROJECT_ROOT / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "beep_candidates_report.csv"

    with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "participant_id", "detected_beep_time_sec",
            "beep_duration_sec", "confidence",
        ])
        writer.writeheader()
        for pid, candidates in all_candidates.items():
            for c in candidates:
                writer.writerow({
                    "participant_id": c.participant_id,
                    "detected_beep_time_sec": f"{c.detected_beep_time_sec:.4f}",
                    "beep_duration_sec": f"{c.beep_duration_sec:.4f}",
                    "confidence": f"{c.confidence:.4f}",
                })

    logger.info("beep 候选报告: %s", report_path)
    return report_path


if __name__ == "__main__":
    candidates = detect_all_beeps()
    for pid, cands in candidates.items():
        print(f"{pid}: {len(cands)} beep candidates")
        for c in cands:
            print(f"  {c.detected_beep_time_sec:.3f}s (conf={c.confidence:.3f})")

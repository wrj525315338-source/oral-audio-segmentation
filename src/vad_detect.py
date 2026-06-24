"""
Step 2: Silero VAD 语音活动检测
对标准化后的 WAV 文件进行 VAD 检测，返回每个文件的语音片段列表。

注意：torch/torchaudio 仅在函数内部导入（lazy import），
避免在未安装 torch 的环境中 import 本模块就报错。
"""
from __future__ import annotations

from pathlib import Path

from .models import SpeechSegment, VADResult
from .utils import PROJECT_ROOT, load_config, setup_logging

logger = setup_logging()


def _load_model():
    """加载 Silero VAD 模型（首次调用会自动下载）。"""
    import torch
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    return model, utils


def detect_speech(
    audio_path: Path,
    cfg: dict | None = None,
) -> VADResult:
    """
    对单个音频文件进行 VAD 检测。

    Parameters
    ----------
    audio_path : Path
        标准化后的 WAV 文件路径。
    cfg : dict
        配置字典，如果为 None 则自动加载。

    Returns
    -------
    VADResult
        包含文件路径、总时长、语音片段列表。
    """
    import torch
    import torchaudio

    if cfg is None:
        cfg = load_config()

    vad_cfg = cfg["vad"]
    threshold = vad_cfg["threshold"]
    min_speech_ms = vad_cfg["min_speech_duration_ms"]
    min_silence_ms = vad_cfg["min_silence_duration_ms"]
    speech_pad_ms = vad_cfg["speech_pad_ms"]
    window_size = vad_cfg["window_size_samples"]

    # 加载音频
    waveform, sr = torchaudio.load(str(audio_path))
    if sr != 16000:
        # 重采样到 16kHz（Silero VAD 要求）
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000

    # 如果是多声道，取均值
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.squeeze(0)  # -> (samples,)
    duration_sec = len(waveform) / sr

    # 加载模型
    model, utils = _load_model()
    get_speech_timestamps = utils[0]

    # 获取语音时间戳
    speech_timestamps = get_speech_timestamps(
        waveform,
        model,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        window_size_samples=window_size,
        return_seconds=False,  # 返回采样点索引
    )

    # 转换为秒
    segments: list[SpeechSegment] = []
    for ts in speech_timestamps:
        start_sec = ts["start"] / sr
        end_sec = ts["end"] / sr
        segments.append(SpeechSegment(start_sec=start_sec, end_sec=end_sec))

    result = VADResult(
        file_path=audio_path,
        duration_sec=duration_sec,
        segments=segments,
    )
    logger.info(
        "VAD: %s | 时长 %.1fs | 检测到 %d 个语音段",
        audio_path.name, duration_sec, len(segments),
    )
    return result


def detect_all(cfg: dict | None = None) -> dict[str, VADResult]:
    """
    对 data/normalized/ 中所有 WAV 文件进行 VAD 检测。

    Returns
    -------
    dict[str, VADResult]
        键为 participant_id，值为 VADResult。
    """
    if cfg is None:
        cfg = load_config()

    norm_dir = PROJECT_ROOT / cfg["paths"]["normalized_dir"]
    if not norm_dir.exists():
        logger.warning("标准化目录不存在: %s", norm_dir)
        return {}

    wav_files = sorted(norm_dir.glob("*.wav"))
    if not wav_files:
        logger.warning("在 %s 中未找到 WAV 文件", norm_dir)
        return {}

    logger.info("开始 VAD 检测，共 %d 个文件...", len(wav_files))

    from .utils import participant_id_from_filename

    results: dict[str, VADResult] = {}
    for wav in wav_files:
        pid = participant_id_from_filename(wav.name)
        try:
            result = detect_speech(wav, cfg)
            results[pid] = result
        except Exception as e:
            logger.error("VAD 检测失败: %s - %s", wav.name, e)

    logger.info("VAD 检测完成: %d/%d 成功", len(results), len(wav_files))
    return results


if __name__ == "__main__":
    results = detect_all()
    for pid, r in results.items():
        print(f"{pid}: {len(r.segments)} speech segments in {r.duration_sec:.1f}s")

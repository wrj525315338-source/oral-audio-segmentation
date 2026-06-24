"""
Step 1: 音频标准化
将 data/raw/ 中的所有音频通过 FFmpeg 转换为 16kHz mono WAV。
输出到 data/normalized/。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .utils import PROJECT_ROOT, load_config, setup_logging

logger = setup_logging()

# 支持的音频扩展名
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".wma", ".aac", ".opus"}


def normalize_one(
    src: Path,
    dst: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    codec: str = "pcm_s16le",
) -> bool:
    """调用 FFmpeg 将单个音频转为标准 WAV。返回是否成功。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",           # 覆盖已有文件
        "-i", str(src),           # 输入
        "-ar", str(sample_rate),  # 采样率
        "-ac", str(channels),     # 声道数
        "-acodec", codec,         # 编码
        "-loglevel", "error",     # 只输出错误
        str(dst),                 # 输出
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error("FFmpeg 失败: %s\n%s", src.name, result.stderr.strip())
            return False
        return True
    except FileNotFoundError:
        logger.error("FFmpeg 未安装或不在 PATH 中，请先安装 FFmpeg")
        raise
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg 超时: %s", src.name)
        return False


def normalize_all(cfg: dict | None = None) -> list[Path]:
    """
    批量标准化 data/raw/ 中所有音频文件。
    返回成功标准化的文件路径列表。
    """
    if cfg is None:
        cfg = load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    norm_dir = PROJECT_ROOT / cfg["paths"]["normalized_dir"]

    audio_cfg = cfg["audio"]
    sr = audio_cfg["sample_rate"]
    ch = audio_cfg["channels"]
    codec = audio_cfg["codec"]

    if not raw_dir.exists():
        logger.warning("原始音频目录不存在: %s", raw_dir)
        return []

    # 收集所有音频文件
    audio_files = sorted(
        f for f in raw_dir.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        logger.warning("在 %s 中未找到音频文件", raw_dir)
        return []

    logger.info("找到 %d 个原始音频文件，开始标准化...", len(audio_files))

    success_list: list[Path] = []
    for src in audio_files:
        # 输出文件名统一为 .wav
        dst = norm_dir / (src.stem + ".wav")
        if dst.exists():
            logger.info("已存在，跳过: %s", dst.name)
            success_list.append(dst)
            continue
        ok = normalize_one(src, dst, sr, ch, codec)
        if ok:
            logger.info("标准化完成: %s -> %s", src.name, dst.name)
            success_list.append(dst)
        else:
            logger.error("标准化失败: %s", src.name)

    logger.info("标准化完成: %d/%d 成功", len(success_list), len(audio_files))
    return success_list


if __name__ == "__main__":
    normalize_all()

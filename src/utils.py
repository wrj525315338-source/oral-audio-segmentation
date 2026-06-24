"""
公共工具函数：配置加载、目录初始化、日志等。
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# 项目根目录：oral-audio-segmentation/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """加载 settings.yaml，返回字典。"""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """根据配置创建所有输出目录。"""
    for key in ("normalized_dir", "segments_dir", "reports_dir"):
        d = PROJECT_ROOT / cfg["paths"][key]
        d.mkdir(parents=True, exist_ok=True)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """配置并返回项目 logger。"""
    logger = logging.getLogger("oral_seg")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def participant_id_from_filename(filename: str) -> str:
    """
    从文件名提取 participant_id。
    支持格式：P001.wav / P001_xxx.wav / P001-xxx.mp3 等。
    如果文件名不匹配，返回去掉后缀的原始名。
    """
    stem = Path(filename).stem
    # 尝试匹配 P + 数字
    import re
    m = re.match(r"^(P\d+)", stem, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return stem

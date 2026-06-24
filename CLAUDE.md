# CLAUDE.md — 项目开发指引

## 项目概述

口语回答音频批量切分工具。将每位学生约 5 分钟的口语考试录音，根据标准实验视频的时间表自动切分为每道题的回答音频片段。

## 技术栈

- Python 3.9+
- FFmpeg（音频格式转换和切分）
- Silero VAD（语音活动检测）
- PyTorch / torchaudio
- PyYAML

## 架构

Pipeline 模式，5 个步骤顺序执行：

```
normalize_audio.py -> vad_detect.py -> estimate_offset.py -> split_audio.py -> qc_report.py
```

所有步骤由 `pipeline.py` 串联，可通过 `python -m src.pipeline` 一键运行。

## 关键设计决策

1. **VAD 不直接决定切分边界**：VAD 仅用于 offset 估算和质检。最终切分边界来自 `schedule.csv + offset + buffer`。
2. **Offset 估算基于全局重叠度**：不依赖第一段语音，而是计算所有候选 offset 下标准窗口与 VAD 片段的总重叠时长，选择最高分。
3. **安全降级**：offset 不可靠时标记 warning，不崩溃。可在配置中选择跳过或用 offset=0 切分。
4. **原始数据不可变**：`data/raw/` 中的文件绝不修改。

## 常用命令

```bash
# 运行完整流水线
python -m src.pipeline

# 运行单个模块（测试用）
python -m src.normalize_audio
python -m src.vad_detect
python -m src.estimate_offset
```

## 配置文件

- `config/settings.yaml`：所有可调参数（路径、VAD 参数、offset 搜索范围、切分缓冲）
- `config/schedule.csv`：标准实验视频的时间表

## 数据目录

- `data/raw/`：原始音频（只读）
- `data/normalized/`：标准化 WAV（可重新生成）
- `data/segments/`：切分结果
- `data/reports/`：质检报告

## 代码风格

- 使用 type hints
- 函数/类有 docstring
- 日志使用统一的 `setup_logging()` logger
- 路径操作使用 `pathlib.Path`

# 口语回答音频批量切分工具

批量将口语考试中每位学生的约 5 分钟录音，按标准实验视频的时间表自动切分为每道题的回答音频片段。

## 项目背景

- 实验流程：考官提问 → 学生准备 10 秒 → 学生回答 10 秒
- 每位学生自行开始录音，因此相对于标准视频存在未知时间偏移量 (offset)
- 本工具通过 VAD 检测 + 时间对齐算法自动估算 offset 并切分

## 目录结构

```
oral-audio-segmentation/
├── config/
│   ├── schedule.csv        # 标准实验视频的时间表
│   └── settings.yaml       # 全局配置（路径、VAD 参数、切分缓冲等）
├── data/
│   ├── raw/                # 原始音频（只读，不会被修改）
│   ├── normalized/         # 标准化后的 16kHz mono WAV
│   ├── segments/           # 切分后的回答片段
│   └── reports/            # 质检报告
├── src/
│   ├── __init__.py
│   ├── utils.py            # 公共工具函数
│   ├── normalize_audio.py  # Step 1: FFmpeg 音频标准化
│   ├── vad_detect.py       # Step 2: Silero VAD 语音检测
│   ├── estimate_offset.py  # Step 3: Offset 估算
│   ├── split_audio.py      # Step 4: FFmpeg 音频切分
│   ├── qc_report.py        # Step 5: 质检报告生成
│   └── pipeline.py         # 一键运行入口
├── requirements.txt
├── README.md
├── CLAUDE.md
└── .gitignore
```

## 安装

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 安装 FFmpeg

FFmpeg 用于音频格式转换和切分，必须在系统 PATH 中可用。

**Windows:**
- 下载: https://ffmpeg.org/download.html
- 解压后将 `bin` 目录添加到系统 PATH
- 或使用: `winget install FFmpeg` / `choco install ffmpeg`

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install ffmpeg
```

验证安装：
```bash
ffmpeg -version
```

## 使用方法

### Step 1: 准备原始音频

将所有学生的音频文件放入 `data/raw/` 目录。支持格式：`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.wma`, `.aac`, `.opus`

文件命名建议：`P001.mp3`, `P002.mp3`, ... （以 `P` + 数字开头，便于自动识别 participant_id）

```
data/raw/
├── P001.mp3
├── P002.mp3
├── P003.mp3
└── ...
```

### Step 2: 检查/修改配置

**config/schedule.csv** — 标准实验视频的时间表：

```csv
question_id,answer_start,answer_end
Q01,114.0,125.0
Q02,138.0,149.0
...
```

- `answer_start` / `answer_end`: 每道题回答阶段在标准视频中的起止时间（秒）

**config/settings.yaml** — 全局配置：

- VAD 灵敏度、搜索范围、切分缓冲等参数均可在此调整
- 详见文件内注释

### Step 3: 一键运行

```bash
python -m src.pipeline
```

流水线会依次执行：
1. 将 `data/raw/` 中的音频标准化为 16kHz mono WAV → `data/normalized/`
2. 使用 Silero VAD 检测每段音频中的语音片段
3. 估算每个学生相对于标准视频的时间偏移量
4. 根据 offset + schedule 批量切分 → `data/segments/`
5. 生成质检报告 → `data/reports/segmentation_report.csv`

### Step 4: 查看结果

**切分后的音频：**
```
data/segments/
├── P001_Q01.wav
├── P001_Q02.wav
├── P002_Q01.wav
└── ...
```

**质检报告：** `data/reports/segmentation_report.csv`

| 列名 | 说明 |
|------|------|
| participant_id | 被试编号 |
| source_file | 源文件名 |
| duration_sec | 源音频总时长 |
| detected_speech_segments | VAD 检测到的语音段数 |
| estimated_offset_sec | 估算的时间偏移量 |
| offset_confidence | offset 置信度 (0~1) |
| question_id | 题目编号 |
| cut_start_sec | 切分起点（秒）|
| cut_end_sec | 切分终点（秒）|
| output_file | 输出文件名 |
| has_detected_speech_in_window | 窗口内是否检测到语音 |
| speech_duration_in_window | 窗口内语音总时长（秒）|
| warning | 警告信息 |

## 算法说明

### Offset 估算算法

1. VAD 检测每个学生音频中的所有语音片段
2. 生成候选 offset：每个 `detected_speech_start - answer_start` 都是一个候选
3. 对每个候选 offset，将所有标准回答窗口平移，计算与 VAD 片段的总重叠时长
4. 选择总重叠时长最高的 offset 作为最佳估计
5. 如果置信度过低或检测到语音的窗口数不足，标记 warning

### 切分边界

最终切分边界 = `schedule.csv` 中的标准窗口 + `estimated_offset` + `pre_buffer/post_buffer`

VAD **不直接**决定切分边界，仅用于 offset 估算和质检。

## 常见问题

**Q: FFmpeg 报错 "not found"**
A: 确保 FFmpeg 已安装并在系统 PATH 中。运行 `ffmpeg -version` 验证。

**Q: Silero VAD 首次运行很慢**
A: 首次运行会自动下载模型（约 2MB），之后会缓存。

**Q: offset 置信度很低怎么办**
A: 检查原始音频质量，确认 schedule.csv 时间表正确。可以在 `config/settings.yaml` 中调整 VAD 阈值或搜索范围。

**Q: 如何只重新切分（跳过标准化和 VAD）**
A: `data/normalized/` 中已有的文件会自动跳过。VAD 结果目前不缓存，如需优化可自行扩展。

---

## Beep-based 分割模式（Reaction Time 分析）

### 概述

Beep-based 模式使用实验视频中的 beep 提示音作为回答窗口的外部时间锚点，比 VAD-offset 估算更精确，适合正式的 reaction time 分析。

**工作原理：**
1. 实验视频中每道题正式回答前有 beep 提示音
2. 大多数学生录音中能听到 beep
3. 使用模板匹配自动检测 beep 时间
4. 以 beep 为锚点切分音频，计算 reaction time

### 配置

**config/beep_template.wav** — beep 模板音频（用户提供）

**config/beep_schedule.csv** — beep 时间表：

```csv
question_id,beep_offset_from_q01,answer_duration_sec
Q01,0.0,10.0
Q02,24.0,10.0
...
```

- `beep_offset_from_q01`: 每题 beep 相对于 Q01 beep 的时间差
- `answer_duration_sec`: beep 后允许回答的时长

**config/manual_beep_times.csv** — 人工标注（可选）：

```csv
participant_id,question_id,beep_time_sec,notes
P001,Q01,30.245,
P008,Q01,28.912,manual checked
```

**config/settings.yaml** — beep 配置：

```yaml
beep:
  enabled: true                    # 启用 beep 模式
  template_path: "config/beep_template.wav"
  min_confidence: 0.4              # 最低置信度阈值
  search_tolerance_sec: 1.0        # 搜索容差
  pre_buffer_sec: 0.3              # 切分起点前缓冲
  post_buffer_sec: 0.5             # 切分终点后缓冲
  rt_reference: "beep_offset"      # RT 参考点
  manual_override: true            # 人工标注覆盖自动检测
```

### 运行

```bash
python -m src.pipeline
```

Pipeline 会自动执行：
1. 标准化 + VAD 检测（同上）
2. **Beep 检测**：模板匹配检测每个被试的 beep
3. **Beep 对齐**：将 beep 匹配到 Q01-Q10
4. **Beep 切分**：以 beep 为锚点切分 → `data/segments_beep/`
5. **RT 计算**：计算 reaction time → `data/reports/beep_reaction_time_report.csv`

### 输出

**切分后的音频：**
```
data/segments_beep/
├── P001_Q01.wav
├── P001_Q02.wav
└── ...
```

**报告文件：**

| 文件 | 说明 |
|------|------|
| `beep_candidates_report.csv` | 所有 beep 检测候选 |
| `beep_alignment_report.csv` | beep 对齐结果 |
| `manual_beep_required.csv` | 需要人工标注的被试 |
| `beep_reaction_time_report.csv` | Reaction time 分析结果 |

### Reaction Time 报告字段

| 字段 | 说明 |
|------|------|
| participant_id | 被试编号 |
| question_id | 题目编号 |
| beep_time_sec | beep 时间（秒）|
| beep_source | beep 来源（detected/manual/inferred）|
| first_speech_start_sec | 首次语音起始时间 |
| rt_from_beep_onset_sec | 从 beep 开始到首次语音的 RT |
| rt_from_beep_offset_sec | 从 beep 结束到首次语音的 RT |
| reaction_time_sec | 主 RT（根据 rt_reference 配置）|
| rt_status | 状态（valid/no_speech_detected/onset_before_beep/rt_too_long）|
| warning | 警告信息 |

### 人工介入流程

如果自动检测不到 beep，会生成 `data/reports/manual_beep_required.csv`：

1. 用音频播放器打开对应的 `data/normalized/PXXX.wav`
2. 找到 Q01 beep 的时间（秒）
3. 填入 `config/manual_beep_times.csv`
4. 重新运行 `python -m src.pipeline`

人工标注会优先覆盖自动检测结果。

### 两种模式对比

| 特性 | VAD-offset 模式 | Beep-based 模式 |
|------|-----------------|-----------------|
| 时间锚点 | 学生语音（推断） | beep 提示音（外部） |
| 精度 | 中等 | 高 |
| 适用场景 | 快速切分 | 正式 RT 分析 |
| 输出目录 | `data/segments/` | `data/segments_beep/` |
| 人工介入 | 通常不需要 | 可能需要标注 beep |

**建议：** 正式分析使用 beep-based 模式，VAD-offset 模式作为备选。

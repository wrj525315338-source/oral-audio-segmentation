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

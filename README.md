# AimBench

**AimBench** 是一个在 Aimlabs **Gridshot** 场景下对比视觉检测算法的基准框架：`color`、`template`、`cnn`、`yolo` 四种检测器共用同一条 WGC 屏幕采集、相机标定、目标选择和鼠标输入管线，用来隔离"检测算法本身"的差异。

*Compare color / template / CNN / YOLO target detectors on Aimlabs Gridshot with a shared capture-and-control pipeline.*

## 特性

- **公平对比**：四种检测器实现同一 `VisionBackend` 契约，采集、控制与输入路径完全一致
- **WGC 屏幕采集**：基于 DXcam 的窗口 ROI 采集，带源时间戳、重复帧与发布延迟追踪
- **新鲜度门控**：过期检测结果不触发输入；大位移后等待画面反馈再进行下一次动作
- **相机模型控制**：按标定参数把目标像素换算为鼠标位移，中心死区内直接点击
- **HUD 观测**：识别计时/分数，自动判定开局、暂停与正常结束，全部检测器共享
- **有界记录**：每秒聚合阶段耗时，不落盘逐帧坐标、输入明细或画面序列
- **可复现元数据**：记录配置、源码指纹、模型哈希与环境版本

## 工作流程

主循环每帧依次执行：

1. **采集** —— 从 WGC 缓冲读取最新已提交帧（BGRA），附带发布时间戳
2. **观测** —— HUD 识别与场景状态机（active / paused / hud_unrecognized / timer_stalled）
3. **检测** —— 当前算法输出目标列表，runner 校验时效与数值有效性
4. **控制** —— 选择离准星最近的目标，换算并提交鼠标位移与点击
5. **记录** —— 采集/视觉/控制/观测四个阶段的耗时按秒聚合

## 环境要求

| 组件 | 要求 |
| --- | --- |
| 操作系统 | Windows 10/11（采集与输入依赖 Win32 / WGC） |
| Python | 3.12+（YOLO 运行库目前仅提供 CPython 3.12 x64 binding） |
| 基础依赖 | numpy、opencv-python、dxcam、pywin32（版本见 `pyproject.toml`） |
| `cnn` | NVIDIA GPU + Torch + TensorRT Python 包 |
| `yolo` | NVIDIA GPU + TensorRT-YOLO 原生库（TensorRT SDK / CUDA） |
| `color` / `template` | 无 GPU 需求，纯 OpenCV |

## 安装

### 1. 获取代码并安装依赖

```bash
git clone <本仓库地址>
cd AimBench
python -m pip install -e .
```

不安装也可以：在 IDE 中打开项目、选择 Python 3.12 解释器，直接运行根目录 `main.py`（入口会把 `src/` 加入 `sys.path`）。

### 2. 构建 TensorRT 引擎（仅 cnn / yolo 需要）

仓库内含 ONNX 模型、输入约定和校验值（见 `models/manifest.json`）。TensorRT engine 与 GPU 和库版本绑定，需要在本机构建到 `.local/models/`：

```bash
# CNN：UINT8 NHWC 输入、FP16 热图输出（ONNX 内含类型转换与归一化）
# 引擎必须与解释器内的 TensorRT Python 包版本匹配
trtexec --onnx=models/cnn/model.onnx --saveEngine=.local/models/cnn.engine --skipInference

# YOLO：FP16，引擎必须与原生 SDK 版本匹配（参考环境为 TensorRT 10.7）
trtexec --onnx=models/yolo/model.onnx --saveEngine=.local/models/yolo.engine --fp16 --skipInference --builderOptimizationLevel=5 --memPoolSize=workspace:4G
```

YOLO 还需要 TensorRT-YOLO 6.4.0 的 Python binding 与 DLL（应用 `models/yolo/xyxy_float.patch` 后自行构建），放置到 `.local/runtime/yolo/`，并在 `.local/runtime.json` 写入 `tensorrt_root` / `cuda_root`。完整步骤见 [docs/development.md](docs/development.md)。

### 3. 自检

```bash
python main.py check --algorithm cnn   # 或 yolo / color / template
```

## 快速开始

1. 编辑 `configs/default.json`，把 `algorithm` 设为 `color` / `template` / `cnn` / `yolo` 之一。
2. 启动 Aimlabs，进入 Gridshot 准备界面，保持窗口完整可见。
3. 运行 `main.py`，看到提示后在 5 秒内切回游戏。
4. 程序自动点击开始并校验 01:00 / 0 分开局，然后接管瞄准；运行中按 **F8 / Esc** 或切出游戏可随时停止。
5. 一局结束后按终端提示录入结算成绩（回车可跳过，稍后用 `result` 命令补填）。
6. `python main.py compare` 查看已录入成绩。

程序会在每帧检查前台窗口、画面尺寸、检测时效和移动反馈，任何异常都会安全停止并保留现场数据。

## 测试条件与标定

基准条件记录在配置的 `conditions` 字段中，程序只记录、不修改游戏设置，需要手动保持一致：

- **1280×720 客户区、灵敏度 1.8、FOV 103、游戏帧率上限 300、隐藏武器**
- `calibration` 中的 `fx` / `fy` / `yaw_per_count` / `pitch_per_count` 是上述条件下的实测标定值；仓库不包含标定工具，若更改分辨率或灵敏度需自行重新标定
- 游戏窗口需完整位于一个屏幕内；采集输出由窗口位置自动选择，也可在 `capture_params` 中手动指定
- 修改窗口大小或灵敏度后必须重新标定，否则位移换算不准

## 配置

配置为单个 JSON（默认 `configs/default.json`，可用 `run --config` 指定其他文件）：

| 字段 | 说明 |
| --- | --- |
| `algorithm` | 检测器名或 `module:factory` 形式的外部实现 |
| `detector_params` | 覆盖检测器默认参数（默认值见 `src/aimbench/registry.py`） |
| `controller_params` | 目标选择与移动反馈阈值（如 `center_deadzone_px`） |
| `gate_params` | 预测式新鲜度门控参数 |
| `capture_params` | `device_idx` / `output_idx` / `max_buffer_len` / `poll_timeout_s` |
| `calibration` | 相机标定（见上节） |
| `conditions` | 测试条件记录（不作用于游戏） |
| `run` | 运行选项：窗口关键词、各类超时、输出目录等（见 `src/aimbench/config.py` 的 `RunOptions`） |

控制阈值、算法参数和运行选项分别放在 `controller_params`、`detector_params`、`run` 中。**对比检测器时不要同时改动公共控制参数**（`controller_params` / `gate_params` / `calibration` / `conditions`），否则成绩不可比。

## 命令行

日常使用只需运行 `main.py`；执行过 `pip install -e .` 后也可用 `aimbench` 命令。

| 用途 | 命令 |
| --- | --- |
| 运行（默认） | `python main.py [--config configs/default.json] [--algorithm yolo] [--label xxx] [--dry-run] [--no-prompt]` |
| 只读检查依赖与路径 | `python main.py check --algorithm cnn` |
| 补填最近一局成绩 | `python main.py result latest --score 226752 --accuracy 100 --shots 588` |
| 查看已录入成绩 | `python main.py compare` |
| 离线单元测试 | `python -m unittest discover -v` |

`--dry-run` 只检测不发送鼠标输入，可用于验证检测效果。

## 输出

每局在 `runs/<时间戳>_<算法>_<短ID>/` 新建一个目录：

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 配置、代码与模型指纹、环境版本、整局统计、结束原因、人工结算成绩 |
| `metrics.csv` | 每秒帧数、累计移动/点击数、采集/视觉/控制/HUD 平均耗时 |
| `result.png` | 正常结束后的结算界面截图 |
| `failure.png` / `error.log` | 异常结束时的最后画面与 traceback |

数据边界：

- 游戏期间只在内存汇总，停止输入后才写盘；强制结束进程可能只留下初始化摘要
- 每阶段最多保留 100,000 个浮点耗时样本（超出时摘要标注百分位已截断），生命周期事件最多 64 条
- 不记录逐帧坐标、输入明细或画面序列
- `summary.json` 的 `eligible_for_comparison` 标记该局是否满足对比条件（正常结束、真实输入、通过开局校验、无清理错误）

读数注意：结算分数与射击次数以游戏界面为准，控制器的点击次数是提交尝试、不等同命中；耗时是主机端阶段测量，不同屏幕、游戏条件或公共配置下的成绩应分开看。

## 实测数据

**数据集**：单机、单日（2026-09-05）的 52 局有效对局 = 13 轮 × 4 算法，每轮按 `color → template → cnn → yolo` 顺序交替运行，以平衡时间趋势。纳入标准：`eligible_for_comparison` 为真且含人工结算成绩；同日更早的试跑不计入。测试条件与公共配置完全一致（见上文），硬件见 [docs/development.md](docs/development.md) 的参考验证环境。

**游戏成绩**（每组 n = 13，分数为游戏结算界面读数）：

| 算法 | 分数均值 ± SD | 中位数 | 最小值 | 最大值 | 射击数均值 ± SD | 准确率 | 消费帧率均值 ± SD（帧/秒） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| color | 227466 ± 682 | 227520 | 226346 | 228316 | 589.9 ± 1.7 | 100% | 234.4 ± 3.3 |
| template | 226551 ± 886 | 226723 | 224736 | 227540 | 587.6 ± 2.1 | 100% | 226.9 ± 5.3 |
| cnn | 227592 ± 638 | 227530 | 226349 | 228321 | 590.2 ± 1.6 | 100% | 236.3 ± 2.9 |
| yolo | 227563 ± 589 | 227534 | 226353 | 228318 | 590.3 ± 1.8 | 100% | 236.6 ± 2.6 |

统计结论：Kruskal–Wallis 检验显示四组分数总体差异显著（p ≈ 0.008）；经 Bonferroni 校正的两两 Mann–Whitney 检验中，template 显著低于 cnn（校正后 p ≈ 0.034）与 yolo（校正后 p ≈ 0.016），其余两两差异不显著；射击数呈现同样模式（Kruskal–Wallis p ≈ 0.005）。结合分阶段耗时看，机制一致：template 的视觉检测最慢，压低了主机端消费帧率（226.9 对 234.4–236.6 帧/秒），60 秒窗口内的有效动作机会随之减少。除 template 外，其余三种检测器在本样本下统计上不可区分。

**主机端每帧分阶段耗时**（ms，每组 n = 13；格式为帧数加权均值（各局均值最小–最大））：

| 算法 | 采集（截图） | 视觉检测 | 控制 | HUD 观测 | 单帧合计 |
| --- | --- | --- | --- | --- | --- |
| color | 3.03（2.75–3.15） | 0.78（0.72–1.01） | 0.26（0.25–0.27） | 0.14（0.14–0.14） | 4.21 |
| template | 0.83（0.64–1.10） | 3.13（3.07–3.17） | 0.26（0.25–0.26） | 0.15（0.14–0.15） | 4.36 |
| cnn | 2.92（2.78–3.03） | 0.90（0.86–0.93） | 0.23（0.22–0.24） | 0.14（0.13–0.14） | 4.18 |
| yolo | 2.38（2.28–2.46） | 1.40（1.39–1.41） | 0.25（0.24–0.26） | 0.14（0.14–0.14） | 4.17 |

计时口径说明：

- 四个阶段均在局内测量：**采集** = 从发起轮询到取得新帧，**包含对新帧的有界等待**，因此视觉越快的算法在采集阶段等待越久——比较检测器时应看"采集 + 视觉"之和（四种算法均为 3.8–4.0 ms）；**视觉检测** = 检测器 `process()`；**控制** = 目标选择与输入决策；**HUD 观测** = 场景观测。结果写盘与截图编码发生在停止输入之后，不计入驻留耗时。
- 尾部延迟（各局 P95 的组均值）：视觉检测 1.01 / 3.85 / 1.34 / 1.68 ms（color / template / cnn / yolo）；采集阶段因含等待帧时间，P95 明显更高（4.8–6.1 ms）。
- 消费帧率 = `frames / elapsed_s`，为主机端处理节奏，非游戏渲染帧率；各局实际时长一致（约 59.8 s）。
- 该样本为单机单日的小样本（n = 13/组），分数含人工操作与环境波动因素，跨机器或跨配置比较无意义。

## 项目结构

```text
AimBench/
├── main.py                  # 入口（IDE 或命令行）
├── configs/default.json     # 运行配置
├── src/aimbench/
│   ├── runner.py            # 会话生命周期与主循环
│   ├── capture.py           # WGC 采集适配
│   ├── capture_probe.py     # 采集缓冲、发布时间戳
│   ├── game_observer.py     # HUD 识别与场景状态
│   ├── control/             # 相机模型、控制器、输入后端
│   ├── vision/              # 四种检测器
│   └── assets/              # 模板、HUD 字形、模型元信息
├── models/                  # ONNX、清单、YOLO 兼容补丁
├── tests/                   # 离线回归测试
├── docs/development.md      # 开发与部署说明
├── licenses/                # 第三方许可文本
├── .local/                  # 本机构建的引擎与运行库（不入 Git）
└── runs/                    # 每局结果（不入 Git）
```

## 开发

代码风格与测试由 GitHub Actions（Windows + Python 3.12）执行，本地可用相同命令：

```bash
python -m ruff check .
python -m ruff format --check .
python -m unittest discover -v
```

架构说明、检测器契约、模型与运行库构建步骤见 [docs/development.md](docs/development.md)。

## 适用范围与说明

- 本项目用于本地单机训练场景的算法评测，会向当前前台窗口发送真实鼠标输入；运行前请确认了解程序行为，并自行确认符合游戏服务条款。
- 目标模板与 HUD 字形取自 Gridshot 1280×720 画面，属于游戏特定资源（见 `src/aimbench/assets/`）。
- 第三方模型与运行库的来源和许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- 本仓库不包含训练数据、训练中间件或历史实验记录。

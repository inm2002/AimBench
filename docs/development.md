# 开发说明

面向开发与部署：架构总览、检测器契约、模型与运行库构建、记录边界与检查流程。

## 架构总览

一局游戏对应一个 `RunSession`（`src/aimbench/runner.py`）。主循环每帧依次执行：

```text
WGC 采集 (capture.py / capture_probe.py)
  → 场景观测 (game_observer.py)
  → 输入前置检查 (input_guard.py)
  → 检测器 (vision/)
  → 控制决策 (control/aim_controller.py + control/freshness.py)
  → 鼠标输入 (control/input_backend.py)
  → 按秒聚合记录 (recording.py)
```

| 模块 | 职责 |
| --- | --- |
| `runner.py` | 会话生命周期：组件准备、开局同步、主循环、结束分类与落盘 |
| `capture.py` / `capture_probe.py` | DXcam（WGC）窗口 ROI 采集、最新帧读取、发布时间戳与重复帧统计 |
| `game_observer.py` | HUD 数字识别、暂停标记、计时器滤波、场景状态机 |
| `lifecycle.py` | 开局同步（等待 01:00 / 0 分的新画面）与结束原因分类 |
| `input_guard.py` | 每次原生输入前的窗口/焦点/场景/截止时间检查 |
| `control/camera_model.py` | 像素与鼠标计数之间的相机投影 |
| `control/aim_controller.py` | 最近目标选择、中心死区、硬限幅、移动计划提交 |
| `control/freshness.py` | 预测式新鲜度门控与大位移反馈等待 |
| `control/input_backend.py` | SendInput 复用缓冲，含按钮释放补偿 |
| `vision/` | 四种检测器与热图解码 |
| `recording.py` | 有界指标、每秒 CSV、截图延迟编码、原子 JSON 写入 |
| `registry.py` | 检测器注册与参数解析 |
| `config.py` | 配置模型与校验（不加载 GPU / Win32 库） |

设计约束：检测器只负责识别目标；选目标、移动、点击、等待画面更新的策略全部在公共控制层，保证四种算法行为可比。

## 开发环境

- Windows，Python 3.12+；YOLO binding 目前仅提供 CPython 3.12 x64
- 依赖见 `pyproject.toml`；`python -m pip install -e ".[dev]"` 额外安装 ruff（`--no-deps` 只注册 `aimbench` 命令）
- Torch、TensorRT、CUDA 不由 pip 自动安装，需按 [模型与引擎构建](#模型与引擎构建) 一节准备

参考验证环境（README 中的成绩与耗时样本来自该环境）：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.7 |
| NumPy / OpenCV | 2.2.6 / 4.10.0.84 |
| DXcam / pywin32 | 0.3.0 / 312 |
| GPU | NVIDIA GeForce RTX 4080 SUPER |
| Torch / TensorRT Python 包（CNN 路径） | 2.11.0+cu128 / 11.2.1.2 |
| TensorRT-YOLO / TensorRT SDK / CUDA（YOLO 路径） | 6.4.0 / 10.7 / 12.6 |

CNN 走解释器内的 TensorRT Python 包，YOLO 走原生 DLL；两条路径独立记录，不要求统一升级。

## `.local/` 目录约定

引擎、运行库和 SDK 路径不入 Git，每台机器按以下布局自行重建：

```text
.local/
├── models/cnn.engine                    # trtexec 构建产物
├── models/yolo.engine
├── runtime/yolo/libs/py_trtyolo*.pyd    # 恰好一个
├── runtime/yolo/bin/trtyolo.dll
├── runtime/yolo/bin/custom_plugins.dll
└── runtime.json                         # {"tensorrt_root": "...", "cuda_root": "..."}
```

`runtime.json` 只服务 YOLO 路径；也可用环境变量 `TENSORRT_ROOT` / `CUDA_PATH` 覆盖。Python 解释器由 IDE 或 `pip` 选择，项目不管理虚拟环境。

## 模型与引擎构建

`models/manifest.json` 记录两个 ONNX 的输入约定、大小和 SHA256，构建前可据此校验文件完整性。

### CNN

- 输入 `uint8` NHWC RGB `192×320×3`，输出 `float16` 热图 `48×80`；ONNX 内含类型转换与归一化
- 用与解释器内 TensorRT Python 包匹配的版本构建，保留输入/输出类型：

```bash
trtexec --onnx=models/cnn/model.onnx --saveEngine=.local/models/cnn.engine --skipInference
```

### YOLO

- 输入 `float32` NCHW RGB `1×3×384×640`；letterbox 114 与 RGB/255 预处理在原生运行库内完成
- 用与原生 SDK 匹配的 TensorRT 构建 FP16：

```bash
trtexec --onnx=models/yolo/model.onnx --saveEngine=.local/models/yolo.engine --fp16 --skipInference --builderOptimizationLevel=5 --memPoolSize=workspace:4G
```

- **Binding**：基于 [TensorRT-YOLO 6.4.0](https://github.com/laugh12321/TensorRT-YOLO) 源码，应用 `models/yolo/xyxy_float.patch`（为 `DetectRes` 增加未取整框坐标 `xyxy_float`，避免中心位置被整数框截断），再按该项目的 CMake 说明构建 Python binding 与 DLL；`py_trtyolo*.pyd` 放入 `libs/`，`trtyolo.dll`、`custom_plugins.dll` 放入 `bin/`
- 运行时若 binding 缺少 `xyxy_float` 扩展会直接报错；原生 profiling 保持关闭

### 校验

```bash
python main.py check --algorithm cnn
python main.py check --algorithm yolo
```

## 检测器契约

实现 `VisionBackend`（`src/aimbench/vision/base.py`）的 `name` 与 `process(frame, frame_timestamp)`：

- 输入是当前帧的 `uint8` BGRA 图像，所有算法收到相同内容
- 输出 `Detection.x/y` 使用原始客户端像素坐标，必须是有限数值
- 结果必须带回当前帧的时间戳；runner 会拒绝旧帧结果与非有限坐标
- 检测器只识别目标，鼠标输入由公共控制器完成
- 可选 `metadata()` 返回初始化元信息，`close()` 释放资源；建议构造时自带 warmup，避免首帧计入耗时

注册方式：加入 `registry.py` 的 `DETECTORS`（含默认参数与路径），或直接配置 `algorithm: "your_package.detector:Factory"` 配合 `detector_params`。

测试：`RunSession` 支持注入 `capture` / `detector` / `input_device`（参考 `tests/support.py` 的 `ReplayCapture` / `ReplayDetector`），不操作游戏即可回归。

## 采集适配器细节

`capture_probe.DXcamProbe` 在 DXcam 0.3.0 实例上包装 `commit_write`，从而：

- 在生产者锁内记录 `producer_publish_ns`、源时间戳与唯一帧序号
- `get_frame()` 在同一锁内把最新已提交槽位复制到复用的消费端缓冲，防止后台覆写正在处理的像素
- 只包装当前实例，不修改 site-packages

此处依赖 DXcam 0.3.0 的内部接口（`_DXCamera__lock`、`_DXCamera__capture_runtime`、`commit_write`）；更换 DXcam 版本时必须复核这些接口并运行 `tests/test_runtime.py` 中的采集缓冲测试。过期判断使用主机接收时间和生产端发布时间；WGC 源时间戳只用于识别新帧，不作为已验证的游戏端到端延迟。

## 记录与统计边界

- 每秒汇总一次主机阶段耗时，整局统计均值、P50、P95、最大值和计数；四个阶段的原始数值缓冲上限合计约 1.53 MiB，超过样本上限时摘要标注百分位已截断，计数和均值仍覆盖全程。默认 60 秒运行只产生约 60 行 CSV。
- 截图在停止输入之后编码；每局至多一张结算图和一张失败现场图。
- `gc_policy=defer` 在准备阶段收集循环垃圾、局内暂停自动循环 GC、结束时恢复原状态；引用计数正常工作。内存每秒检查一次，超过配置的增长限额会停止。
- 数据精简后无法重建每次射击的画面因果，后续分析应以整局表现和阶段指标为基础。

## 代码风格与检查

4 空格、100 列、UTF-8、LF；ruff 规则统一在 `pyproject.toml` 管理。GitHub Actions（`.github/workflows/checks.yml`）在 windows-latest + Python 3.12 上执行相同检查：

```bash
python -m ruff check .
python -m ruff format --check .
python -m unittest discover -v
```

测试覆盖：亚像素热图解码、目标匹配（与穷举分配对拍）、移动反馈等待与超时、过期结果拒绝、开局迟到拒绝、采集缓冲所有权、部分点击释放、GC 恢复、指标有界性、成绩补填与结算截图稳定性。全部离线运行，不发送真实输入。

新增检测器时建议补充：合成帧上的行为测试（合法/非法输出）、参数校验测试，以及注入 `RunSession` 的端到端回归。

## 迁移到新机器清单

1. 安装 Python 3.12 与基础依赖（`python -m pip install -e .`）
2. 安装 GPU 驱动、CUDA、TensorRT；CNN 路径另需 Torch 与 TensorRT Python 包
3. 校验 ONNX 完整性（对照 `models/manifest.json`），构建两个 engine 到 `.local/models/`
4. 应用 `models/yolo/xyxy_float.patch` 构建 YOLO binding 与 DLL，放到 `.local/runtime/yolo/`
5. 写 `.local/runtime.json`（或设置 `TENSORRT_ROOT` / `CUDA_PATH`）
6. `python main.py check --algorithm cnn` 与 `--algorithm yolo` 全部 OK
7. 复核目标分辨率/灵敏度对应的 `calibration`；用 `--dry-run` 先验证检测效果
8. 首局人工确认开局校验与结束判定符合预期，再开始正式对比

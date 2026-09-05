# 第三方来源

本工程保留必要的推理模型和运行素材，不包含训练实验目录。

| 内容 | 来源与声明 |
| --- | --- |
| CNN ONNX | 从原项目正式热图模型导出；输入和校验值见 `models/manifest.json` |
| YOLO ONNX | 原项目基于 Ultralytics YOLO26n 的单类别微调模型，经 trtyolo-export 转换；Ultralytics 的默认许可为 AGPL-3.0，另有独立商业授权 |
| YOLO 原生运行库及补丁 | TensorRT-YOLO 6.4.0，GPL-3.0；完整许可见 `licenses/TensorRT-YOLO.txt`，本地改动见 `models/yolo/xyxy_float.patch` |
| 模板与 HUD 字形 | 来自原项目的 Aimlabs Gridshot 画面；用于目标和界面识别，游戏素材权利属于相应权利人 |
| Python 依赖、CUDA、TensorRT SDK | 保留各自许可；环境和 SDK 不打包到 Git 仓库 |

YOLO 模型的许可文本保留于 `licenses/Ultralytics.txt`。这些声明不改变第三方代码、模型或游戏素材的原有许可。

上游项目：
- [TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO)
- [trtyolo-export](https://github.com/laugh12321/trtyolo-export)
- [Ultralytics](https://github.com/ultralytics/ultralytics)

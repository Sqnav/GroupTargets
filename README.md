# GroupTargets

群目标视觉—雷达定位研究代码。项目源码位于 [`radar_vision_localization/`](radar_vision_localization/README.md)。

包含32 MHz、相机光心与观察机回波参考点重合条件下的七组实验，以及共享雷达证据 GNN 的三个对照分支。数据、模型权重、实验结果和缓存分别保存在外部目录，不随代码上传。

## 代码入口

- `radar_vision_localization/src/rvloc/models/`：RGB、单次查询雷视融合、融合内对象世界模型。
- `radar_vision_localization/tools/accelerated_training.py`：RGB及单帧融合训练。
- `radar_vision_localization/tools/accelerated_world_training.py`：世界模型融合训练。
- `radar_vision_localization/tools/parallel_training_controller.py`：七组实验流水线。
- `radar_vision_localization/gnn_research/`：shared / isolated / shuffled 图模型、训练、评价和机制审计。

## 安装与配置

需要Python 3.10或以上；在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e './radar_vision_localization[test]'
export RVLOC_PYTHON="$(command -v python)"
```

现有运行脚本和历史示例包含原服务器路径。换机运行时请设置 `RVLOC_PYTHON`，并根据实际数据修改外部实验配置中的 manifest、output_root、weights_root、cache_root 和检查点路径。默认模板中的空数据与权重路径需要由使用者绑定。

本次上传保留原源码；仅修正快照的 `.gitignore` 产物目录锚定，避免忽略 `src/rvloc/data/` 源码。模型结果以对应冻结数据和评价协议为准。

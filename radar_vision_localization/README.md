# 32 MHz 共点实验：视觉先验 → 单次雷达查询 → 最终深度

活动架构为 `single_query_depth_v3`。视觉分支输出深度均值 μ_v 和标准差 σ_v；融合分支仅生成一组候选、查询一次 RA/RE，然后融合候选、全局上下文和成员关系，一次输出最终连续深度。没有第一轮/第二轮深度更新，也没有 IQ 风险校正。

本目录只有代码、配置、测试和文档。默认模板的数据清单、标定和检查点保持为空；真实运行用代码目录以外的独立冻结配置，不改默认入口。旧权重保留；已通过 99 项合成测试的零查询 v2 源码在 `/data7/ysq/GroupTargets/code/legacy/single_query_before_20261126_105736`，更早两轮源码与可信 `_legacy` 数值实现也保留。

## 七组实验

| 编号 | 方法 | 当前流程 |
|---|---|---|
| 1 | FFT | 当前 IQ → 联合距离/角度功率体 → 峰检测 → 三维位置 |
| 2 | 视觉先验 | 冻结视觉 ROI＋框尺度＋相机/导航几何 → 成员交互 → μ_v、σ_v；位置由 μ_v 输出 |
| 3 | 单次查询融合 | 视觉 μ_v/σ_v → 一次 RA/RE 候选查询 → 候选/全局/成员融合 → 最终深度 |
| 4 | FFT＋Kalman | 第 1 组测量 → 纯雷达关联 → CV Kalman |
| 5 | 视觉＋Kalman | 第 2 组测量 → 纯视觉关联 → CV Kalman |
| 6 | 融合＋Kalman | 第 3 组测量 → 同一纯视觉关联 → CV Kalman |
| 7 | 世界状态参与融合 | 与第 3 组相同的一次查询；历史对象先验注入证据融合 token，再一次回归最终深度 |

第 5/6/7 组共用 TRAIN 冻结的纯视觉关联；不使用 GT 人数或身份。每条 episode 独立清空状态，只使用当前与历史帧，按 physical_time_s 处理 1 秒间隔。当前 KF 仍使用 TRAIN 标定的固定观测协方差，σ_v 不自动替代其 R。

## 视觉深度与不确定度

沿用现有冻结 DINOv3 导出的检测框和 1024 维 ROI；定位器不构造或训练视觉骨干，输入 ROI 在计算图中 detach。数据准备必须绑定实际前端权重 SHA；合成测试中的随机 ROI 不代表运行了真实 DINOv3。

ROI 与公开相机/导航几何编码、同帧成员交互后，一个视觉先验头输出 μ_v、σ_v。两者只依赖当前视觉与几何，不读取雷达，也不读取世界模型历史。μ_v 在公开深度范围内有界；σ_v 单位是米，模板上下界为 0.25–16 m，初始偏置对应 2 m。这些是待 TRAIN 验证的工程设置，并非已校准置信区间。

## 一次雷达证据查询

当前原生 IQ 经可信 FFT 得到 RA/RE；两个 CNN 各执行一次，其二维特征图同时用于全局上下文与局部采样，不重复编码。

候选由正常视觉先验生成：`rho_candidate = mu_v + sigma_v * z`。模板 z 为：

```text
[-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
```

全部成员的候选一起经可信 candidate_geometry 转为基站坐标，调用一次 sample_paired_spectra。RA 和 RE 的物理有效掩码分别保留。越界候选保持原地址并标记无效，不裁剪成重复候选、不插入 GT 地址。没有第二次围绕修正深度重新查询。

每个候选保留 RA/RE 特征、米制与标准化偏移、候选深度、基站距离、角度及有效性标志。经过一层候选 Transformer 交互与有效候选加权汇聚，再与全局雷达上下文、视觉 token 和 μ_v/σ_v 结合，通过成员 Transformer 融合群关系，最终深度头只输出一次连续值。

候选是读取证据的采样位置。最终深度不是候选分类、不是候选深度加权平均，也不限制在候选最小/最大深度之间；它仍受公开相机深度边界约束，最终位置为 `p = camera_origin + rho * ray_direction`。

当所有局部候选都不可用时，局部证据严格为零，保留全局与视觉路径；不会制造有效雷达证据。原始 RA/RE 数值内核未修改。

## 世界模型位置

对象 GRU 从历史后验产生当前先验，通过 memory_bridge 注入查询后的融合 token，再与当前证据共同决定最终深度。历史不会污染 μ_v/σ_v 的纯视觉含义，不改变候选位置，也不会引起额外查询。随后根据当前视觉与最终融合深度更新后验。None 身份不读写共同记忆，过期和跨 episode 记忆被清除。

## 监督与评价

基础训练损失包含最终深度 Huber、0.05×最终深度 MSE，以及 0.1×视觉先验高斯负对数似然：

```text
visual_prior_nll = 0.5*((rho_gt-mu_v)/sigma_v)^2 + log(sigma_v) + 0.5*log(2*pi)
```

该似然项让标准差受到误差监督，而非仅学习查询窗口宽度。G7 继续使用一步先验位置与隐状态动态辅助监督；当前训练器没有未来多步标签监督。没有第一轮/第二轮深度损失或旧初深辅助项。GT 只在当前前向之后单独加载，用于监督或离线评价，不能影响正常候选地址。

三维 RMSE 是主评价指标，同时报告 MAE、P95/P99、严重错误、定位覆盖和未匹配输出。`visual_prior_metrics.json` 额外记录视觉深度 RMSE、高斯 NLL、1σ/2σ经验覆盖；其 GT 匹配使用公开射线角度/支持，独立于最终三维匹配。经验覆盖不等于已经校准的置信保证。

正常预测 JSON 导出视觉均值/标准差和一次查询的候选深度、偏移、掩码与汇聚权重。权重用于证据汇聚，不是目标概率。默认不保存完整 CNN 特征图。原始预测始终先保存，再单独读取 GT 评价。

G2/G3/G7 共享公开 ray_depth_bounds、range_scale_m 和 ray_valid。RA/RE 的真实 RF 坐标轴、功率和测量掩码保持独立，不用 RF 距离球改写视觉输出范围。共享输入合同仍为 public_visual_depth_direct_v1/public_visual_rays_ra_re_direct_v1；这些名称描述数据，不表示使用旧 v2 模型。

## 入口、版本和存储

- `models/single_frame.py`：视觉先验、全局/局部融合及最终深度。
- `models/single_query.py`：一次物理候选查询与候选交互。
- `models/world_fusion.py`：融合内因果对象记忆。
- `models/losses.py`、`training.py`：监督与训练入口。
- `pipeline.py`、`calibration*.py`、`evaluation.py`：推理、TRAIN 标定与离线评价。

结果在 `/data7/ysq/GroupTargets/results/radar_vision_localization/`；权重在 `/data7/ysq/GroupTargets/weights/radar_vision_localization/`；缓存在 `/data7/ysq/GroupTargets/data/cache/radar_vision_localization/`。训练按同一 run ID 分开保存日志和权重。

```bash
cd /data7/ysq/GroupTargets/code/radar_vision_localization
./scripts/rvloc plan --config configs/seven_methods.json
./scripts/rvloc validate-data --manifest configs/manifest.pending.example.json --metadata-only
```

数据绑定后依次：训练 G2/G3 → 导出 G1/G2/G3 的 TRAIN 预测 → 按 episode fit/calibrate/audit 标定 → 冻结视觉关联后训练 G7 → 冻结协议 → 运行七组 DEV 推理与评价。G4/5/6 分别复用 G1/2/3 的观测源，不另训练定位器。G2/G3/G7 默认独立训练。

可用单一入口执行整条流水线：

```bash
./scripts/rvloc run-experiment --config /绝对路径/experiment.json \
  --run-dir /绝对路径/results/独立运行目录 --devices 4,5
tail -n 80 -F /绝对路径/results/独立运行目录/run.log
```

`experiment.py` 固定执行 43 个阶段：三组网络训练、TRAIN 预测与标定、七组 DEV 推理和评价、21 个配对统计与中文报告。训练使用预先声明的最终 epoch，不用 DEV 选检查点。`status.json` 每 5 秒记录阶段与子进程状态；训练 `progress.json` 记录 epoch、更新、loss 与耗时。失败阶段保留原输出；同命令加 `--resume` 只跳过 SHA 验证通过的完成阶段，未完成训练重新启动该阶段，不声称恢复优化器。

`data/register_camera_center.py` 将已有共点分支的冻结 TRAIN700／DEV150 导出为严格 manifest，逐帧核验 RGB、IQ 和缓存身份。prepared-only 模式要求完整 IQ SHA 认证收据，仍校验当前 IQ 文件身份与小文件内容 SHA；FFT 方法继续读取原始 IQ。可在 TRAIN 首轮缓存公开 ROI、RA/RE 功率和几何的 CPU 数组，后续 epoch 复用这些输入；CNN 的可训练输出不缓存。缓存条件下的模型耗时应与完整 RGB/IQ 前端耗时分开。

此数据分支让观察机雷达散射参考点与现有相机光心重合；机体导航中心与相机光心仍保留约 0.46 m 的安装偏移。按当前实验要求，视觉射线起点与网络中的 `observer_position` 都使用相机光心。机体导航只保留在原始正常观测记录中用于审计，不传入 G2/G3/G7 定位网络。

## 固定成员评价与单行实时日志

`tools/fixed_gt_correspondence.py` 保存每帧当前 ROI 对应的 GT 标签。G2/G3/G5/G6/G7 的输出保持检测行序，固定成员评价直接沿用这些标签，不根据预测三维位置重新寻找 GT。实例分割 GT 框用于独立核查，重复检测、歧义和无对应检测单独保留。GT 标签不进入正常关联或对象记忆；FFT G1/G4 继续单独报告群点集评价。

`tools/live_training_progress.py` 为正在运行的实验追加单行日志，不重启训练。每行同时显示 loss、全量 TRAIN 的三维 RMSE、检查点 epoch、本模型已用时间、预计剩余时间和总耗时。RMSE 由 `tools/checkpoint_train_rmse.py` 在独立进程中读取完整保存的权重并实际重放 TRAIN 得到，不从 loss 换算。`TRAIN_RMSE@E33` 表示 epoch 33 检查点的全量 TRAIN 结果，不是当前 batch、在线参数或 DEV 成绩；评估尚未完成时显示 pending。

当前实验的单行日志：

```bash
tail -n 20 -F /data7/ysq/GroupTargets/results/radar_vision_localization/live_progress_20261126_run1/progress.log
```

同样的行也会追加到原实验的 `run.log`。预计剩余时间针对当前模型，随实测速度更新；后续模型和标定阶段尚未测量时不虚构整条流水线的完成时间。原训练源码、配置、权重和默认入口保持原样；新增工具、固定标注与真实计算收据分别保存在代码、数据和结果目录中。

新检查点 schema 为 `rvloc_seven_groups_single_query_depth_v3`，v1 两轮权重、v2 零查询权重与旧标定不直接混用。历史 TEST 不读取，DEV 不用于反复调参。候选倍数、标准差范围和损失权重尚未在真实数据上认证。

合成测试检查一次查询计数、视觉先验独立性、真实采样梯度、候选/成员交互、最终输出连续性、无效证据和因果记忆等。减少查询次数不等于已经测得端到端提速；ROI/RARE缓存条件下的加载与定位计时也不能代表完整 RGB/IQ→输出的成本。真实精度、标准差校准和时延都需要后续实验验证。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests
```

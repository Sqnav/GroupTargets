# 修复后数据的接入合同

此合同不修改原数据，不假定用户的新数据已经共点。当前神经定位架构为 single_query_depth_v3，使用共享公开相机射线支持。`configs/manifest.pending.example.json` 是可解析的空清单模板，不能训练/推理。请将正式清单、传感器工件和标签放在代码目录外。

## 必须明确的物理定义

- RF：4.9 GHz、32 MHz、40 脉冲、128 频点、16×16 阵列、PRI 2.5 ms。一个 CPI 内脉冲相对时间为 `0, .0025, …, .0975 s`，跨帧不拼接相位。
- 频率数组为显式的 128 个中心化基带采样：`(arange(128)-64)*250000 Hz`。不能用含左右端点的 linspace 偷换频率步长。
- 时间：`physical_time_s`，episode 内连续帧间隔 1 s；不使用 `sim_timestamp_ns` 推算速度。导航样本必须有真实物理时间，只允许当前及历史时间，最后一个样本对应当前帧。
- 坐标：固定基站局部 `base_local`，轴为前/右/上；基站为零点，阵列基为单位阵。相机旋转将相机 FRD 向量映到该坐标，允许 FRD/FRU 的手性差异。其他源坐标必须在导出时显式转换。
- “中心重合”指**观察机自身回波散射参考点等于现有相机光心**。导航机体中心可以保留安装偏移，不能把机体导航值直接冒充光心。射线起点使用实际相机光心。
- 本合同要求每帧散射参考点与相机位置数值相同，以及带 SHA 的独立生成/安装证明。仅修改 manifest 的标记不等于回波已经按新位置生成；生成器一致性需要数据生产端证明。

## 外部目录示例

```text
co_center_32mhz/
├── manifest.json
├── center_evidence.json
└── episode_0001/
    ├── frame_0000.normal.json
    ├── frame_0000.labels.json
    ├── frame_0000.visual.npz
    ├── frame_0000.iq.npy
    └── frame_0000.radar.npz
```

清单只包含 train/dev 的显式 episode ID 和逐帧路径/SHA，不遍历其他数据、不包含 TEST。`normal_path` 和 `labels_path` 必须不同；正常输入严禁 GT 位置、真实速度、目标数、GT ID、散射源身份和未来导航。即使标签路径出现在清单里，正常加载器也不打开它。

## 清单与逐帧字段

完整字段结构见可运行的 `tests/test_data.py::synthetic_manifest`，此函数只生成测试工件。正式数据不应复制其合成数值。

| 对象 | 字段 | 维度/含义 |
|---|---|---|
| manifest | schema/status/data_root/coordinate_frame | `rvloc.manifest.v1`、ready 或 pending_data_fix、外部数据根、base_local |
| rf | carrier_hz/bandwidth_hz/pulse_interval_s/shape | `4.9e9/32e6/.0025/[40,128,16,16]` |
| rf | spacing_y_m/spacing_z_m/frequencies_hz/frequency_grid/frame_interval_s | 真实阵元间距、128 基带频率、centered_baseband_uniform、1 |
| visual_geometry | depth_min_m/depth_max_m/range_scale_m | 事先公开的米制深度支持及归一化尺度，RGB 路径不从 IQ 决定边界 |
| center_contract | reference/co_location/status/evidence | observer_self_scatter_reference / camera_optical_center / verified / `{path,sha256}` |
| splits | train/dev | 按 episode 隔离、互不重叠的 ID 列表 |
| episodes | episode_id/frames | 每帧 frame_idx、physical_time_s、normal_path/SHA、labels_path/SHA |
| normal | schema/episode_id/frame_idx/physical_time_s | rvloc.observation.v1 及与清单完全一致的身份 |
| camera | K/position_m/rotation_frd_to_frame/image_size_wh | `[3,3]/[3]/[3,3]/[2]`，真实分辨率，不硬编码缩放 |
| normal | observer_position_m/observer_scatter_reference_position_m | 导航机体点与回波参考点各 `[3]`，后者等于 camera.position_m |
| navigation | physical_time_s/position_m/velocity_mps | `[T]/[T,3]/[T,3]`，当前及历史公开导航 |
| calibration | station_position_m/array_basis_fru_to_frame | `[3]/[3,3]`，本版固定零点/单位阵 |
| visual / iq / radar | path/sha256 | 相对 data_root 的工件引用；IQ 为复杂数 npy；radar 可选缓存 |
| visual.npz | bbox_xyxy/detection_scores/detector_roi_features | `[N,4]/[N]/[N,1024]`，纯 RGB 检测与外观，固定 detector 身份 |
| labels | schema/episode_id/frame_idx/physical_time_s/positions_m/member_ids | `rvloc.labels.v1`；`[M,3]` 及仅供监督/评价的 M 个 ID；可选 velocities_mps `[M,3]` |

共点证明 JSON 的字段为：`schema=rvloc.center_evidence.v1`、`coordinate_frame=base_local`、`reference=observer_self_scatter_reference`、`co_location=camera_optical_center`、`camera_to_scatter_reference_offset_m=[0,0,0]`、非空 `source`、非空 `capture_revision`。它应引用实际数据生成变更与安装定义，不得用“已验证”替代真实核查。

## 特征准备

RGB 路径沿用固定检测器的框与 1024 维 ROI 特征，可导出已有缓存或通过 `prepare-visual` 提供明确 `module:function` 回调；回调只收到 RGB。代码不下载/训练检测器，不自动选择另一套视觉编码器。回调返回上述三项数组，检测器权重 SHA 显式传入。

`prepare-radar` 从一个原生 IQ CPI 使用可信 FFT 处理生成 RA/RE 缓存，并保存 IQ/RF 身份。融合加载验证该缓存与实际 IQ/RF 一致；缓存缺失时可从 IQ 准备。RGB-only 不打开 IQ、雷达缓存或标签；FFT-only 不打开 RGB/ROI。通用 numpy 工件禁止 pickle。

请先运行 metadata-only 检查，再做完整的传感器验证。完整验证会读取真实传感器工件，但仍不读 GT；本次实现阶段没有对用户数据执行该命令。

## 单次查询版本的共享几何

G2/G3/G7 的 ray_depth_bounds、range_scale_m 和 ray_valid 均来自相同公开视觉几何；融合不得再以 RF 距离球覆盖它们。range_scale_m 只负责归一化，不限制 GT 的基站距离或 x 方向。雷达谱仍使用真实 RF 距离/角度轴与独立测量掩码。融合 feature schema 为 public_visual_rays_ra_re_direct_v1，共享 depth_schema 为 public_visual_depth_direct_v1。离线监督匹配入口为 supervision.match_direct_ray_labels，正常推理不读取标签。

## 单次查询的视觉先验

视觉均值与米制标准差只由当前冻结 ROI 和公开几何预测；候选地址为 mu_v + sigma_v*z。世界历史只进入查询后的融合。正常推理输出保留先验与候选掩码供审计；任何 GT 的先验 NLL、覆盖率和候选覆盖统计仅在预测保存后离线计算。

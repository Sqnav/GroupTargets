# 代码与工件分离

按照“此文件夹下只放代码”的要求，原项目整目录以文件系统 rename 方式迁移，不复制大型结果，不删除原文件或权重：

```text
原路径：/data7/ysq/WM/radar_vision_localization
归档：  /data7/ysq/WM/radar_vision_localization_artifacts/legacy/20261126_before_seven_groups
新源码：/data7/ysq/WM/radar_vision_localization
新工件：/data7/ysq/WM/radar_vision_localization_artifacts/runs
新缓存：/data7/ysq/WM/radar_vision_localization_artifacts/cache
```

原项目的模型、结果、缓存、默认部署描述与旧代码在归档中保留。新目录不自动加载或发布历史模型；新的统一入口为 `scripts/rvloc`。历史文件中记录的绝对路径仍是原始身份信息，没有批量改写。重放旧实验时，需要把以原路径开头的工件路径映射到归档路径；不能把旧部署脚本的原绝对路径直接当作新工程路径。

`configs/legacy_sources.json` 记录复用源码的原相对路径和 SHA；`src/rvloc/_legacy/` 为这些可信源码的代码副本，不包含数据/模型/结果。适配器复用原 D，而不重新实现 RA/RE 精修物理。目录迁移收据与本次合成测试收据写在外部工件根目录。

未操作 `dataset_construction` 中的数据；未读取历史 TEST；未进行真实数据训练/评估。归档本身不是新实验结果。

# Reward 数据审核与裁剪

在仓库根目录运行。工具只处理离线文件，不启动 Ray、机器人或 reward model；依赖现有环境中的 Torch、NumPy、OpenCV、OmegaConf。审核窗口需要桌面显示和带 GUI 支持的 OpenCV；`--dry-run` 和裁剪可在无桌面节点运行。

## 1. 审核正负帧

推荐审核 `raw_reward_episodes/`，再重新划分训练/验证集：

```bash
python -m toolkits.dexhand.review_classifier_data \
  --input logs/dexhand_frames/raw_reward_episodes \
  --output-dir logs/reviewed_reward
```

也可用 `--input /path/train.pt /path/val.pt` 审核已有划分，或传入引号包裹的 glob。审核工具读取当前 RLinf 的 `images / labels / metadata` `.pt` 格式，支持双相机 VHWC 和旧单相机 HWC/CHW；不读取 franka infra 的 transition pickle。

| 按键 | 功能 |
| --- | --- |
| `n` / 右箭头、`p` / 左箭头 | 下一帧、上一帧 |
| `g`、`b` | 保留、丢弃当前样本，然后进入下一帧 |
| `1`、`2`、`0` | 仅看正帧、仅看负帧、查看全部 |
| `s` | 确认后保存 |
| `q` / Esc | 退出；有未保存修改时询问是否保存 |

各相机按 metadata 中的顺序并排显示，顶部显示标签、来源、审核数量和丢弃数量。**未审核样本默认保留**，标记不会修改原始标签。

每次保存生成 `logs/reviewed_reward/review_<时间>/`，保留原文件名，并同步筛选 `step_ids`、`episode_ids`。同时写入 `review.json` 和保留帧的 PNG；`images/<success或failure>/view_<序号>/` 中的序号对应 metadata 相机顺序。添加 `--no-export-images` 可关闭 PNG 导出。

添加 `--replace-inputs` 会在确认后备份全部输入到 `review_backup_<时间>/`，再覆盖输入文件；默认另存不覆盖。`--dry-run` 只检查数据并打印正负样本统计，不打开窗口、不写文件。

## 2. 裁剪 reward 或 demo 数据

修改 [config/classifier_crop.yaml](config/classifier_crop.yaml)，按相机名指定 `[top, left, bottom, right]`。默认示例是全图，不会缩小视野。所有数值都是 0～1 的比例；`coordinates: stored` 表示相对于**已保存的图像**：

```yaml
coordinates: stored
cameras:
  wrist_1:
    crop: [0.2, 0.1, 0.9, 0.8]
  global:
    crop: [0.0, 0.0, 1.0, 1.0]
```

裁剪审核后的 reward 数据：

```bash
python -m toolkits.dexhand.crop_classifier_data \
  --data-dir logs/reviewed_reward/review_<时间> \
  --output-dir logs/cropped_reward \
  --crop-config toolkits/dexhand/config/classifier_crop.yaml
```

裁剪 demo，必须指定主相机在前的相机名列表。其余相机按当前 `RealWorldEnv` 的字母顺序映射到 `extra_view_images`：

```bash
python -m toolkits.dexhand.crop_classifier_data \
  --data-dir logs/dexhand_demos/demos \
  --output-dir logs/cropped_demos \
  --crop-config toolkits/dexhand/config/classifier_crop.yaml \
  --camera-keys wrist_1 global
```

同一命令支持 `CollectEpisode` 输出的 `collected_data/` 目录。支持格式为：reward `.pt`、replay-buffer 的 `curr_obs/next_obs` `.pt`、`CollectEpisode` 的 `observations` `.pkl`；不处理 LeRobot、模型 checkpoint 或 infra 的 pickle 列表。

图像裁剪后用 OpenCV 双线性插值恢复原尺寸，保持 uint8、视角顺序、帧数和终止帧。标签、动作、状态等不变。目录结构和索引文件原样复制，裁剪记录写入 `crop_manifest.json`；reward metadata 也记录裁剪参数。原 PNG 预览会原样复制，**不会代表裁剪后的图像**，需要重新运行审核工具查看/导出新预览。

输出目录必须不存在或为空，且不能与输入目录互相包含。添加 `--dry-run` 可校验所有文件但不写入。输入文件不会被覆盖；中途失败不发布部分裁剪结果。

### 如果使用原始相机坐标

当前环境在保存图像前已经裁剪和缩放，不能直接把原始相机 ROI 套到保存图像上。使用 `coordinates: raw`，同时提供 `source_region`（保存图像在原始相机中的范围）和新的 `crop`：

```yaml
coordinates: raw
cameras:
  wrist_1:
    source_region: [0.0, 0.125, 1.0, 0.875]
    crop: [0.25, 0.3125, 0.75, 0.6875]
```

此示例的 `source_region` 对应 640×480 原始画面的默认居中正方形裁剪。请根据实际采集设置填写；如果之前又做过离线裁剪，也要更新来源范围。工具会换算到保存图像坐标，并拒绝超出来源范围的裁剪，因为已丢失的像素无法恢复。对已缩放图片再裁剪不能完全等同于原始高分辨率画面直接裁剪。

部署前同步在线 `env.eval.override_cfg.camera_crop_regions`（以相机序列号为键，坐标相对于原始画面），保持训练与推理视野一致；工具不会自动修改硬件配置。

## 3. 重新生成训练/验证集

对审核/裁剪后的原始 `episode_*.pt` 执行：

```bash
PYTHONPATH=. python examples/reward/preprocess_reward_dataset.py \
  --raw-format labeled_frames \
  --raw-data-path logs/cropped_reward \
  --output-dir logs/processed_reward
```

输出 `train.pt`、`val.pt`，继续使用现有训练命令。若直接处理的是已有 `train.pt` / `val.pt`，不必再次划分。审核过多样本可能导致无法形成均含正负样本的两个回合集合，此时需要补采数据。

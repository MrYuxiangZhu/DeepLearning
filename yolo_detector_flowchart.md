# YOLO 视频人类与动物检测模型完整流程图与参数详解

本文档整理 `video_human_animal_detector.py` 中基于 Ultralytics YOLO 的视频检测、目标跟踪、时序平滑、中文可视化和公开视频测试完整流程。

---

## 1. 总体架构概览

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                         YOLO 视频检测与跟踪系统                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  输入视频 / 公开视频下载                                                      │
│          │                                                                   │
│          ▼                                                                   │
│  cv2.VideoCapture 读取视频元信息                                               │
│          │                                                                   │
│          ▼                                                                   │
│  逐帧读取 BGR 图像                                                             │
│          │                                                                   │
│          ▼                                                                   │
│  YOLO.track() 检测 + ByteTrack/BOTSort 跟踪                                    │
│          │                                                                   │
│          ▼                                                                   │
│  类别过滤：person + COCO 常见动物                                              │
│          │                                                                   │
│          ▼                                                                   │
│  坐标裁剪 + track_id 提取 + 置信度提取                                         │
│          │                                                                   │
│          ▼                                                                   │
│  TrackHistorySmoother 多帧类别加权投票                                         │
│          │                                                                   │
│          ▼                                                                   │
│  PIL 中文绘制：检测框 + 类别 + ID + 分数                                       │
│          │                                                                   │
│          ▼                                                                   │
│  cv2.VideoWriter 写入输出视频                                                  │
│          │                                                                   │
│          ▼                                                                   │
│  输出统计：帧数、耗时、平均 FPS、类别计数                                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 完整主流程图

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                              程序入口 main()                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. 设置运行参数                                                               │
│     - use_public_sample_video                                                 │
│     - public_sample_name                                                      │
│     - custom_input_video_path                                                 │
│     - output_video_path                                                       │
│     - model_path                                                              │
│     - confidence_threshold                                                    │
│     - iou_threshold                                                           │
│     - image_size                                                              │
│     - device                                                                  │
│     - tracker_config                                                          │
│                                                                              │
│  2. 判断输入来源                                                               │
│        ┌──────────────────────────────┐                                      │
│        │ use_public_sample_video=True │                                      │
│        └──────────────┬───────────────┘                                      │
│                       │                                                       │
│           ┌───────────▼───────────┐                                           │
│           │ PublicVideoDataset    │                                           │
│           │ Loader.download()     │                                           │
│           └───────────┬───────────┘                                           │
│                       │                                                       │
│                       ▼                                                       │
│              获得 input_video_path                                            │
│                                                                              │
│        ┌───────────────────────────────┐                                     │
│        │ use_public_sample_video=False │                                     │
│        └──────────────┬────────────────┘                                     │
│                       │                                                       │
│                       ▼                                                       │
│              使用 custom_input_video_path                                     │
│                                                                              │
│  3. 创建 VideoHumanAnimalDetector                                             │
│        - 加载 YOLO 模型                                                       │
│        - 初始化类别映射                                                       │
│        - 初始化颜色映射                                                       │
│        - 初始化多帧平滑器                                                     │
│        - 加载中文字体                                                         │
│                                                                              │
│  4. 调用 detector.process_video()                                             │
│                                                                              │
│  5. 调用 print_stats() 输出处理统计                                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 视频处理主流程图

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                         process_video(input, output)                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  检查输入视频是否存在                                                          │
│          │                                                                   │
│          ├── 不存在 → FileNotFoundError                                      │
│          │                                                                   │
│          ▼                                                                   │
│  创建输出目录 os.makedirs()                                                    │
│          │                                                                   │
│          ▼                                                                   │
│  打开输入视频 cv2.VideoCapture                                                 │
│          │                                                                   │
│          ├── 打开失败 → RuntimeError                                          │
│          │                                                                   │
│          ▼                                                                   │
│  读取视频元信息                                                                │
│     - frame_count 总帧数                                                       │
│     - fps 帧率                                                                 │
│     - frame_width 宽度                                                         │
│     - frame_height 高度                                                        │
│          │                                                                   │
│          ▼                                                                   │
│  创建输出视频 cv2.VideoWriter                                                  │
│     - fourcc = mp4v                                                           │
│     - fps = 原视频 fps                                                        │
│     - size = 原视频宽高                                                       │
│          │                                                                   │
│          ├── 创建失败 → RuntimeError                                          │
│          │                                                                   │
│          ▼                                                                   │
│  初始化统计变量                                                                │
│     - start_time                                                              │
│     - processed_frames                                                        │
│     - class_counter                                                           │
│     - tqdm progress                                                           │
│          │                                                                   │
│          ▼                                                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                         逐帧处理循环                                   │  │
│  │                                                                        │  │
│  │  capture.read()                                                        │  │
│  │       │                                                                │  │
│  │       ├── success=False → break                                        │  │
│  │       │                                                                │  │
│  │       ▼                                                                │  │
│  │  detect_and_track_frame(frame_bgr)                                     │  │
│  │       │                                                                │  │
│  │       ▼                                                                │  │
│  │  统计检测类别 class_counter                                            │  │
│  │       │                                                                │  │
│  │       ▼                                                                │  │
│  │  draw_detections(frame_bgr, detections)                                │  │
│  │       │                                                                │  │
│  │       ▼                                                                │  │
│  │  writer.write(visualized_frame)                                        │  │
│  │       │                                                                │  │
│  │       ▼                                                                │  │
│  │  processed_frames += 1                                                 │  │
│  │  progress.update(1)                                                    │  │
│  │                                                                        │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│          │                                                                   │
│          ▼                                                                   │
│  finally 释放资源                                                              │
│     - progress.close()                                                        │
│     - capture.release()                                                       │
│     - writer.release()                                                        │
│          │                                                                   │
│          ▼                                                                   │
│  计算 elapsed_seconds 与 average_fps                                           │
│          │                                                                   │
│          ▼                                                                   │
│  返回 VideoProcessStats                                                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 单帧 YOLO 检测与跟踪流程图

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                    detect_and_track_frame(frame_bgr)                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  输入：frame_bgr                                                              │
│     - 类型：np.ndarray                                                        │
│     - 格式：OpenCV BGR                                                        │
│     - 形状：(H, W, 3)                                                         │
│                                                                              │
│          │                                                                   │
│          ▼                                                                   │
│  调用 YOLO.track()                                                            │
│     - source=frame_bgr                                                        │
│     - conf=confidence_threshold                                               │
│     - iou=iou_threshold                                                       │
│     - imgsz=image_size                                                        │
│     - device=device                                                           │
│     - tracker=tracker_config                                                  │
│     - persist=True                                                            │
│     - verbose=False                                                           │
│                                                                              │
│          │                                                                   │
│          ▼                                                                   │
│  获取 results[0]                                                              │
│          │                                                                   │
│          ├── 无结果 / 无 boxes → 返回空列表                                   │
│          │                                                                   │
│          ▼                                                                   │
│  遍历 result.boxes                                                            │
│          │                                                                   │
│          ▼                                                                   │
│  解析单个 box_item                                                            │
│     - class_id = int(box_item.cls.item())                                     │
│     - label_en = result.names[class_id]                                       │
│     - score = float(box_item.conf.item())                                     │
│     - xyxy = box_item.xyxy                                                    │
│     - track_id = box_item.id                                                  │
│          │                                                                   │
│          ▼                                                                   │
│  类别过滤 allowed_labels                                                       │
│          │                                                                   │
│          ├── 不在允许类别 → continue                                          │
│          │                                                                   │
│          ▼                                                                   │
│  坐标裁剪 _clip_box()                                                          │
│          │                                                                   │
│          ├── 非法框 x2<=x1 或 y2<=y1 → continue                              │
│          │                                                                   │
│          ▼                                                                   │
│  多帧平滑 smoother.update(track_id, label_en, score)                          │
│          │                                                                   │
│          ▼                                                                   │
│  英文标签映射中文 label_map_zh                                                 │
│          │                                                                   │
│          ▼                                                                   │
│  封装 DetectionResult                                                          │
│     - label_en                                                                │
│     - label_zh                                                                │
│     - score                                                                   │
│     - box                                                                     │
│     - track_id                                                                │
│     - raw_label_en                                                            │
│          │                                                                   │
│          ▼                                                                   │
│  返回 List[DetectionResult]                                                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 单帧处理横向流程图

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                               单帧处理流程                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  BGR 帧                                                                       │
│    │                                                                         │
│    ▼                                                                         │
│  YOLO.track() ──► 检测框 boxes                                                │
│    │                 │                                                       │
│    │                 ├── xyxy 坐标                                           │
│    │                 ├── cls 类别 ID                                         │
│    │                 ├── conf 置信度                                         │
│    │                 └── id 跟踪 ID                                          │
│    │                                                                         │
│    ▼                                                                         │
│  类别过滤 allowed_labels                                                      │
│    │                                                                         │
│    ▼                                                                         │
│  坐标裁剪 _clip_box                                                           │
│    │                                                                         │
│    ▼                                                                         │
│  track_id 历史缓存                                                            │
│    │                                                                         │
│    ▼                                                                         │
│  多帧加权投票                                                                 │
│    │                                                                         │
│    ▼                                                                         │
│  中文标签映射                                                                 │
│    │                                                                         │
│    ▼                                                                         │
│  PIL 绘制矩形框 + 标签 + ID + 分数                                            │
│    │                                                                         │
│    ▼                                                                         │
│  输出标注帧                                                                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 目标跟踪与时序平滑流程图

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                         TrackHistorySmoother                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  输入：track_id, label_en, score                                               │
│          │                                                                   │
│          ▼                                                                   │
│  判断 track_id 是否存在                                                        │
│          │                                                                   │
│          ├── track_id=None                                                    │
│          │       │                                                           │
│          │       ▼                                                           │
│          │   无法确认跨帧身份，直接返回当前 label_en, score                   │
│          │                                                                   │
│          ▼                                                                   │
│  根据 track_id 读取历史队列                                                     │
│     histories[track_id] = deque(maxlen=window_size)                           │
│          │                                                                   │
│          ▼                                                                   │
│  追加当前帧结果                                                                │
│     history.append((label_en, score))                                         │
│          │                                                                   │
│          ▼                                                                   │
│  统计窗口内每个类别的加权分数                                                   │
│     weighted_scores[label] += score                                           │
│     vote_counts[label] += 1                                                   │
│          │                                                                   │
│          ▼                                                                   │
│  选择加权总分最高类别                                                           │
│     best_label = argmax(weighted_scores)                                      │
│          │                                                                   │
│          ▼                                                                   │
│  判断该类别票数是否达到 min_votes                                               │
│          │                                                                   │
│          ├── 未达到 min_votes                                                  │
│          │       │                                                           │
│          │       ▼                                                           │
│          │   返回当前帧 label_en, score                                       │
│          │                                                                   │
│          └── 达到 min_votes                                                    │
│                  │                                                           │
│                  ▼                                                           │
│              返回 best_label 和平均平滑分数                                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 6.1 时序平滑公式

```text
对同一个 track_id，在最近 N 帧窗口中：

weighted_score(c) = Σ score_i, 其中 label_i = c
vote_count(c)     = 类别 c 出现次数

best_label = argmax_c weighted_score(c)

如果 vote_count(best_label) >= min_votes：
    smoothed_score = weighted_score(best_label) / vote_count(best_label)
否则：
    使用当前帧 label 和 score
```

---

## 7. 中文可视化绘制流程图

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                           draw_detections()                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  输入 frame_bgr 与 detections                                                  │
│          │                                                                   │
│          ▼                                                                   │
│  BGR → RGB                                                                    │
│     cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2RGB)                         │
│          │                                                                   │
│          ▼                                                                   │
│  np.ndarray → PIL.Image                                                       │
│          │                                                                   │
│          ▼                                                                   │
│  ImageDraw.Draw(pil_image)                                                    │
│          │                                                                   │
│          ▼                                                                   │
│  遍历每个 DetectionResult                                                      │
│          │                                                                   │
│          ▼                                                                   │
│  读取 box、label、score、track_id                                               │
│          │                                                                   │
│          ▼                                                                   │
│  根据类别读取 color_map                                                        │
│          │                                                                   │
│          ▼                                                                   │
│  绘制矩形框                                                                    │
│     draw.rectangle([(x1,y1),(x2,y2)], outline=color, width=3)                 │
│          │                                                                   │
│          ▼                                                                   │
│  计算文本尺寸 draw.textbbox()                                                  │
│          │                                                                   │
│          ▼                                                                   │
│  绘制文字背景条                                                                │
│          │                                                                   │
│          ▼                                                                   │
│  绘制中文标签                                                                  │
│     例如：人 ID:3 0.87                                                        │
│          │                                                                   │
│          ▼                                                                   │
│  PIL.Image → np.ndarray → BGR                                                  │
│          │                                                                   │
│          ▼                                                                   │
│  返回 output_frame                                                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. 参数详解

### 8.1 `main()` 运行参数

| 参数名 | 默认值 | 类型 | 作用 | 调优建议 |
|---|---:|---|---|---|
| `use_public_sample_video` | `True` | `bool` | 是否使用公开示例视频测试 | 初次验证建议为 `True`；处理自己的视频时改为 `False` |
| `public_sample_name` | `"vtest_people"` | `str` | 公开视频名称 | 可选 `vtest_people`、`people_street` |
| `custom_input_video_path` | `"你的视频.mp4"` | `str` | 自定义输入视频路径 | 使用自己视频时必须改成真实路径 |
| `output_video_path` | `"./outputs/yolo_human_animal_tracking_result.mp4"` | `str` | 输出视频保存路径 | 建议使用 `.mp4` 后缀 |
| `model_path` | `"yolo11x.pt"` | `str` | YOLO 模型权重 | 精度优先用 `yolo11x.pt`；速度优先用 `yolo11n.pt` 或 `yolo11s.pt` |
| `confidence_threshold` | `0.35` | `float` | 检测置信度阈值 | 误检多则提高；漏检多则降低 |
| `iou_threshold` | `0.5` | `float` | NMS IoU 阈值 | 重复框多则降低；相邻目标被压掉则提高 |
| `image_size` | `1280` | `int` | YOLO 推理输入尺寸 | 小目标多建议增大；速度慢建议降低 |
| `device` | `None` | `str | None` | 推理设备 | `None` 自动选择；也可设为 `"cuda"`、`"cpu"`、`"0"` |
| `tracker_config` | `"bytetrack.yaml"` | `str` | 跟踪器配置 | 常用 `bytetrack.yaml` 或 `botsort.yaml` |

---

### 8.2 `VideoHumanAnimalDetector.__init__()` 参数

| 参数名 | 默认值 | 说明 | 影响 |
|---|---:|---|---|
| `model_path` | `"yolo11x.pt"` | YOLO 权重文件路径或模型名 | 决定检测模型大小、速度、精度 |
| `confidence_threshold` | `0.35` | 最低检测置信度 | 越高误检越少，但漏检可能增加 |
| `iou_threshold` | `0.5` | NMS 重叠阈值 | 控制重叠框合并强度 |
| `image_size` | `1280` | 推理图片尺寸 | 越大越利于小目标，显存和耗时越高 |
| `device` | `None` | 推理设备 | GPU 显著快于 CPU |
| `tracker_config` | `"bytetrack.yaml"` | 跟踪器配置文件 | 影响 track_id 稳定性 |
| `smoothing_window` | `12` | 类别平滑窗口帧数 | 越大越稳定，但响应更慢 |
| `smoothing_min_votes` | `3` | 切换到平滑类别所需最小票数 | 越大越保守 |

---

### 8.3 YOLO `track()` 核心参数

| 参数名 | 示例值 | 说明 |
|---|---:|---|
| `source` | `frame_bgr` | 输入图像，可以是图片、视频、摄像头、NumPy 数组 |
| `conf` | `0.35` | 置信度阈值，低于该值的检测框会被过滤 |
| `iou` | `0.5` | NMS IoU 阈值，用于去除重复检测框 |
| `imgsz` | `1280` | 模型输入尺寸 |
| `device` | `None` | 推理设备，`None` 自动选择 |
| `tracker` | `"bytetrack.yaml"` | 跟踪器配置文件 |
| `persist` | `True` | 保持视频连续帧之间的跟踪状态 |
| `verbose` | `False` | 是否打印 YOLO 推理日志 |

---

### 8.4 模型规格选择建议

| 模型 | 特点 | 适用场景 |
|---|---|---|
| `yolo11n.pt` | 最小、最快、精度较低 | CPU、边缘设备、快速预览 |
| `yolo11s.pt` | 小模型、速度快 | 实时视频、普通 GPU |
| `yolo11m.pt` | 中等模型 | 精度和速度折中 |
| `yolo11l.pt` | 大模型 | 精度优先，GPU 较好 |
| `yolo11x.pt` | 最大、精度最高、速度最慢 | 离线处理、精度优先 |

---

## 9. 允许检测类别

当前脚本使用 COCO 预训练模型，因此动物类别受 COCO 数据集限制。

```text
allowed_labels = {
    person,
    bird,
    cat,
    dog,
    horse,
    sheep,
    cow,
    elephant,
    bear,
    zebra,
    giraffe
}
```

### 9.1 中文映射

| 英文类别 | 中文类别 |
|---|---|
| `person` | 人 |
| `bird` | 鸟 |
| `cat` | 猫 |
| `dog` | 狗 |
| `horse` | 马 |
| `sheep` | 羊 |
| `cow` | 牛 |
| `elephant` | 大象 |
| `bear` | 熊 |
| `zebra` | 斑马 |
| `giraffe` | 长颈鹿 |

注意：COCO 预训练模型默认不包含 `tiger`、`deer`、`monkey` 等类别。如果必须检测这些动物，需要训练或微调自定义 YOLO 模型。

---

## 10. 数据结构说明

### 10.1 `DetectionResult`

| 字段 | 类型 | 说明 |
|---|---|---|
| `label_en` | `str` | 平滑后的英文类别 |
| `label_zh` | `str` | 中文类别 |
| `score` | `float` | 平滑后或当前帧置信度 |
| `box` | `Tuple[int, int, int, int]` | 检测框，格式为 `(x1, y1, x2, y2)` |
| `track_id` | `Optional[int]` | 跟踪 ID，没有 ID 时为 `None` |
| `raw_label_en` | `Optional[str]` | YOLO 当前帧原始类别 |

### 10.2 `VideoProcessStats`

| 字段 | 类型 | 说明 |
|---|---|---|
| `input_video_path` | `str` | 输入视频路径 |
| `output_video_path` | `str` | 输出视频路径 |
| `frame_count` | `int` | 视频元信息中的总帧数 |
| `processed_frames` | `int` | 实际处理帧数 |
| `elapsed_seconds` | `float` | 总耗时，单位秒 |
| `average_fps` | `float` | 平均处理速度 |
| `class_counter` | `Counter` | 检测类别统计 |

---

## 11. 坐标、IoU 与 NMS 说明

### 11.1 检测框坐标格式

YOLO 输出的框使用 `xyxy` 格式：

```text
x1, y1 = 左上角坐标
x2, y2 = 右下角坐标
box = (x1, y1, x2, y2)
```

### 11.2 坐标裁剪

```text
x1 = max(0, min(frame_width  - 1, x1))
y1 = max(0, min(frame_height - 1, y1))
x2 = max(0, min(frame_width  - 1, x2))
y2 = max(0, min(frame_height - 1, y2))
```

作用：防止模型输出框越界，避免裁剪或绘制时报错。

### 11.3 IoU 公式

```text
IoU = Area(A ∩ B) / Area(A ∪ B)
```

含义：两个检测框的重叠程度。

在 YOLO 中，`iou_threshold` 主要影响 NMS：

```text
如果两个框重叠过高，并且属于同类，通常保留高置信度框，抑制低置信度框。
```

---

## 12. 公开示例视频下载流程

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                        PublicVideoDatasetLoader.download()                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  输入 sample_name                                                             │
│          │                                                                   │
│          ▼                                                                   │
│  检查 sample_name 是否存在于 SAMPLE_VIDEOS                                     │
│          │                                                                   │
│          ├── 不存在 → ValueError                                              │
│          │                                                                   │
│          ▼                                                                   │
│  根据 URL 推断文件后缀                                                         │
│          │                                                                   │
│          ▼                                                                   │
│  拼接输出路径 dataset_dir/sample_name.suffix                                  │
│          │                                                                   │
│          ▼                                                                   │
│  如果文件已存在且 overwrite=False                                              │
│          │                                                                   │
│          ├── 直接返回本地路径                                                  │
│          │                                                                   │
│          ▼                                                                   │
│  requests.get(url, stream=True, timeout=60)                                   │
│          │                                                                   │
│          ▼                                                                   │
│  分块写入本地文件                                                              │
│          │                                                                   │
│          ▼                                                                   │
│  tqdm 显示下载进度                                                             │
│          │                                                                   │
│          ▼                                                                   │
│  返回下载后视频路径                                                            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

可选公开视频：

| 名称 | URL |
|---|---|
| `people_street` | `https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/cv/tracking/768x576.avi` |
| `vtest_people` | `https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi` |

---

## 13. 准确率提升路线图

当前代码已经使用 YOLO + 跟踪 + 多帧平滑，比零样本单帧检测更适合工程落地。但如果目标是稳定达到 95%+，建议继续按照下面路线优化。

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                            95%+ 精度优化路线                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. 收集真实业务视频                                                           │
│          │                                                                   │
│          ▼                                                                   │
│  2. 抽帧并标注检测框                                                           │
│     - person                                                                 │
│     - dog/cat/horse/...                                                      │
│     - tiger/deer/monkey 等自定义类别                                          │
│          │                                                                   │
│          ▼                                                                   │
│  3. 训练自定义 YOLO 检测模型                                                    │
│          │                                                                   │
│          ▼                                                                   │
│  4. 用验证集评估                                                               │
│     - Precision                                                              │
│     - Recall                                                                 │
│     - mAP@0.5                                                                │
│     - mAP@0.5:0.95                                                           │
│          │                                                                   │
│          ▼                                                                   │
│  5. 增加跟踪器调参                                                             │
│     - ByteTrack                                                              │
│     - BoT-SORT                                                               │
│          │                                                                   │
│          ▼                                                                   │
│  6. 增加多帧投票和置信度平滑                                                    │
│          │                                                                   │
│          ▼                                                                   │
│  7. 对 person 增加专用属性模型                                                  │
│     - man                                                                    │
│     - woman                                                                  │
│     - unknown_person                                                         │
│          │                                                                   │
│          ▼                                                                   │
│  8. 对动物增加细分类模型                                                        │
│     - tiger                                                                  │
│     - deer                                                                   │
│     - monkey                                                                 │
│     - 其他业务类别                                                            │
│          │                                                                   │
│          ▼                                                                   │
│  9. 在独立测试集上确认是否达到 95%+                                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 14. 推荐调参策略

### 14.1 误检较多

```text
现象：背景、其他物体被误识别为人或动物
建议：
1. 提高 confidence_threshold，例如 0.35 → 0.45
2. 降低 image_size 可能减少小噪声误检，但会牺牲小目标
3. 使用更大的模型，例如 yolo11s → yolo11l / yolo11x
4. 使用业务数据微调模型
```

### 14.2 漏检较多

```text
现象：真实人或动物没有检测出来
建议：
1. 降低 confidence_threshold，例如 0.35 → 0.25
2. 增大 image_size，例如 640 → 960 → 1280
3. 使用更大模型
4. 补充相似场景训练数据
```

### 14.3 ID 经常切换

```text
现象：同一个目标 track_id 经常变化
建议：
1. 尝试 botsort.yaml
2. 提高视频帧率或减少跳帧
3. 避免过高 confidence_threshold 导致中间帧断检
4. 调整跟踪器配置中的匹配阈值
```

### 14.4 类别闪烁

```text
现象：同一个目标在 dog/cat 或 person/animal 间跳变
建议：
1. 增大 smoothing_window，例如 12 → 20
2. 增大 smoothing_min_votes，例如 3 → 5
3. 引入专用分类模型
4. 使用自定义训练数据减少类别混淆
```

---

## 15. 安装与运行

### 15.1 安装依赖

```bash
pip install ultralytics opencv-python pillow numpy tqdm requests
```

如果需要 GPU，请根据 CUDA 版本安装对应的 PyTorch。

### 15.2 运行公开示例视频

保持：

```text
use_public_sample_video = True
```

然后运行：

```bash
python3 video_human_animal_detector.py
```

### 15.3 运行自定义视频

修改：

```text
use_public_sample_video = False
custom_input_video_path = "你的真实视频路径.mp4"
```

然后运行：

```bash
python3 video_human_animal_detector.py
```

---

## 16. 最终输出

```text
outputs/yolo_human_animal_tracking_result.mp4
```

视频中每个检测目标会显示：

```text
中文类别 ID:跟踪编号 置信度
```

示例：

```text
人 ID:5 0.86
狗 ID:8 0.79
猫 ID:12 0.72
```

终端会输出：

```text
处理完成
输入视频: ...
输出视频: ...
原始帧数: ...
处理帧数: ...
耗时: ... 秒
平均处理速度: ... FPS
检测类别统计:
  - person: ...
  - dog: ...
```

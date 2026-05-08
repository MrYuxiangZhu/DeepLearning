"""
基于 YOLO 的视频人类与动物检测、跟踪、时序平滑与公开示例视频测试框架。

整改目标：
1. 将原 OWLv2 零样本检测架构替换为更适合工程落地的 YOLO 检测架构。
2. 使用 Ultralytics YOLO 预训练 COCO 模型检测 person 与常见动物。
3. 使用 YOLO 内置 ByteTrack/BOTSort 跟踪接口，为同一视频目标分配 track_id。
4. 对同一个 track_id 的历史类别进行多帧加权投票，降低视频帧间抖动。
5. 提供公开示例视频下载能力，便于一键测试整改后的架构。
6. 输出带中文标签、置信度和跟踪 ID 的标注视频。

安装依赖：
pip install ultralytics opencv-python pillow numpy tqdm requests

可选依赖：
如果你希望使用 GPU，请按你的 CUDA 版本安装合适的 torch/torchvision。

运行方式：
python3 video_human_animal_detector.py

重要说明：
- 本脚本使用公开预训练模型，不等价于已针对你的业务场景完成 95%+ 准确率优化。
- 若要稳定达到 95%+，仍建议使用你的真实场景数据对 YOLO 进行微调，并补充专用 man/woman 或动物细分类模型。
"""

from __future__ import annotations

import os
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from ultralytics import YOLO


@dataclass
class DetectionResult:
    """保存单个检测或跟踪目标的结构化结果。"""

    label_en: str
    label_zh: str
    score: float
    box: Tuple[int, int, int, int]
    track_id: Optional[int] = None
    raw_label_en: Optional[str] = None


@dataclass
class VideoProcessStats:
    """保存视频处理过程中的统计信息。"""

    input_video_path: str
    output_video_path: str
    frame_count: int
    processed_frames: int
    elapsed_seconds: float
    average_fps: float
    class_counter: Counter


class TrackHistorySmoother:
    """对同一个 track_id 的类别进行多帧加权投票，降低视频识别抖动。"""

    def __init__(self, window_size: int = 12, min_votes: int = 3) -> None:
        self.window_size = window_size
        self.min_votes = min_votes
        self.histories: Dict[int, Deque[Tuple[str, float]]] = defaultdict(lambda: deque(maxlen=self.window_size))

    def update(self, track_id: Optional[int], label_en: str, score: float) -> Tuple[str, float]:
        if track_id is None:
            return label_en, score

        history = self.histories[track_id]
        history.append((label_en, score))

        weighted_scores: Dict[str, float] = defaultdict(float)
        vote_counts: Counter = Counter()
        for historical_label, historical_score in history:
            weighted_scores[historical_label] += historical_score
            vote_counts[historical_label] += 1

        best_label = max(weighted_scores.items(), key=lambda item: item[1])[0]
        if vote_counts[best_label] < self.min_votes:
            return label_en, score

        smoothed_score = weighted_scores[best_label] / max(1, vote_counts[best_label])
        return best_label, float(smoothed_score)


class VideoHumanAnimalDetector:
    """基于 YOLO 的视频人类与动物检测器。"""

    def __init__(
        self,
        model_path: str = "yolo11x.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        image_size: int = 1280,
        device: Optional[str] = None,
        tracker_config: str = "bytetrack.yaml",
        smoothing_window: int = 12,
        smoothing_min_votes: int = 3,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device
        self.tracker_config = tracker_config
        self.model = YOLO(model_path)
        self.smoother = TrackHistorySmoother(window_size=smoothing_window, min_votes=smoothing_min_votes)

        self.allowed_labels = {
            "person",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
        }

        self.label_map_zh: Dict[str, str] = {
            "person": "人",
            "bird": "鸟",
            "cat": "猫",
            "dog": "狗",
            "horse": "马",
            "sheep": "羊",
            "cow": "牛",
            "elephant": "大象",
            "bear": "熊",
            "zebra": "斑马",
            "giraffe": "长颈鹿",
        }

        self.color_map: Dict[str, Tuple[int, int, int]] = {
            "person": (0, 255, 255),
            "bird": (255, 128, 0),
            "cat": (0, 200, 255),
            "dog": (0, 255, 0),
            "horse": (255, 0, 255),
            "sheep": (180, 255, 180),
            "cow": (128, 128, 255),
            "elephant": (180, 180, 180),
            "bear": (80, 120, 180),
            "zebra": (255, 255, 255),
            "giraffe": (0, 165, 255),
        }
        self.draw_font = self._load_font(font_size=22)

    def _load_font(self, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        candidate_fonts = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for font_path in candidate_fonts:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, font_size)
                except OSError:
                    continue
        return ImageFont.load_default()

    def detect_and_track_frame(self, frame_bgr: np.ndarray) -> List[DetectionResult]:
        results = self.model.track(
            source=frame_bgr,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            tracker=self.tracker_config,
            persist=True,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        detections: List[DetectionResult] = []
        names = result.names

        for box_item in result.boxes:
            class_id = int(box_item.cls.item())
            label_en = str(names[class_id])
            if label_en not in self.allowed_labels:
                continue

            score = float(box_item.conf.item())
            xyxy = box_item.xyxy[0].detach().cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = self._clip_box(tuple(xyxy), frame_bgr.shape[1], frame_bgr.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue

            track_id = None
            if box_item.id is not None:
                track_id = int(box_item.id.item())

            smoothed_label, smoothed_score = self.smoother.update(track_id, label_en, score)
            label_zh = self.label_map_zh.get(smoothed_label, smoothed_label)

            detections.append(
                DetectionResult(
                    label_en=smoothed_label,
                    label_zh=label_zh,
                    score=smoothed_score,
                    box=(x1, y1, x2, y2),
                    track_id=track_id,
                    raw_label_en=label_en,
                )
            )

        return detections

    def _clip_box(self, box: Tuple[int, int, int, int], frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        x1 = max(0, min(frame_width - 1, x1))
        y1 = max(0, min(frame_height - 1, y1))
        x2 = max(0, min(frame_width - 1, x2))
        y2 = max(0, min(frame_height - 1, y2))
        return x1, y1, x2, y2

    def draw_detections(self, frame_bgr: np.ndarray, detections: Iterable[DetectionResult]) -> np.ndarray:
        output_rgb = cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(output_rgb)
        draw = ImageDraw.Draw(pil_image)

        for detection in detections:
            x1, y1, x2, y2 = detection.box
            color_bgr = self.color_map.get(detection.label_en, (0, 255, 0))
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            track_text = f" ID:{detection.track_id}" if detection.track_id is not None else ""
            raw_text = "" if detection.raw_label_en in (None, detection.label_en) else f"/{detection.raw_label_en}"
            text = f"{detection.label_zh}{raw_text}{track_text} {detection.score:.2f}"

            draw.rectangle([(x1, y1), (x2, y2)], outline=color_rgb, width=3)
            text_bbox = draw.textbbox((x1, y1), text, font=self.draw_font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            text_left = x1
            text_top = max(0, y1 - text_h - 12)
            text_right = min(output_rgb.shape[1] - 1, text_left + text_w + 12)
            text_bottom = min(output_rgb.shape[0] - 1, text_top + text_h + 10)
            draw.rectangle([(text_left, text_top), (text_right, text_bottom)], fill=color_rgb)
            draw.text((text_left + 6, text_top + 4), text, font=self.draw_font, fill=(0, 0, 0))

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def process_video(self, input_video_path: str, output_video_path: str) -> VideoProcessStats:
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"未找到输入视频: {input_video_path}")

        output_dir = os.path.dirname(output_video_path) or "."
        os.makedirs(output_dir, exist_ok=True)

        capture = cv2.VideoCapture(input_video_path)
        if not capture.isOpened():
            raise RuntimeError(f"无法打开输入视频: {input_video_path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_width <= 0 or frame_height <= 0:
            capture.release()
            raise RuntimeError("无法读取视频宽高，请检查输入视频编码是否正常。")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"无法创建输出视频: {output_video_path}")

        start_time = time.time()
        processed_frames = 0
        class_counter: Counter = Counter()
        progress = tqdm(total=frame_count if frame_count > 0 else None, desc="YOLO 视频检测跟踪")

        try:
            while True:
                success, frame_bgr = capture.read()
                if not success:
                    break

                detections = self.detect_and_track_frame(frame_bgr)
                for detection in detections:
                    class_counter[detection.label_en] += 1

                visualized_frame = self.draw_detections(frame_bgr, detections)
                writer.write(visualized_frame)
                processed_frames += 1
                progress.update(1)
        finally:
            progress.close()
            capture.release()
            writer.release()

        elapsed_seconds = time.time() - start_time
        average_fps = processed_frames / elapsed_seconds if elapsed_seconds > 0 else 0.0
        return VideoProcessStats(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            frame_count=frame_count,
            processed_frames=processed_frames,
            elapsed_seconds=elapsed_seconds,
            average_fps=average_fps,
            class_counter=class_counter,
        )


class PublicVideoDatasetLoader:
    """下载公开可访问的示例视频，用于快速测试整改后的检测架构。"""

    SAMPLE_VIDEOS: Dict[str, str] = {
        "people_street": "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/cv/tracking/768x576.avi",
        "vtest_people": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi",
    }

    def __init__(self, dataset_dir: str = "./public_video_samples") -> None:
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def download(self, sample_name: str = "vtest_people", overwrite: bool = False) -> str:
        if sample_name not in self.SAMPLE_VIDEOS:
            available = ", ".join(sorted(self.SAMPLE_VIDEOS))
            raise ValueError(f"未知公开示例视频: {sample_name}。可选项: {available}")

        url = self.SAMPLE_VIDEOS[sample_name]
        suffix = Path(url).suffix or ".mp4"
        output_path = self.dataset_dir / f"{sample_name}{suffix}"

        if output_path.exists() and not overwrite and output_path.stat().st_size > 0:
            return str(output_path)

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with output_path.open("wb") as file_obj, tqdm(
            total=total_size if total_size > 0 else None,
            unit="B",
            unit_scale=True,
            desc=f"下载公开视频 {sample_name}",
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                file_obj.write(chunk)
                progress.update(len(chunk))

        return str(output_path)


def print_stats(stats: VideoProcessStats) -> None:
    print("\n处理完成")
    print(f"输入视频: {stats.input_video_path}")
    print(f"输出视频: {stats.output_video_path}")
    print(f"原始帧数: {stats.frame_count}")
    print(f"处理帧数: {stats.processed_frames}")
    print(f"耗时: {stats.elapsed_seconds:.2f} 秒")
    print(f"平均处理速度: {stats.average_fps:.2f} FPS")
    if stats.class_counter:
        print("检测类别统计:")
        for label, count in stats.class_counter.most_common():
            print(f"  - {label}: {count}")
    else:
        print("未检测到 person 或 COCO 常见动物类别。")


def main() -> None:
    use_public_sample_video = True
    public_sample_name = "vtest_people"
    custom_input_video_path = "你的视频.mp4"
    output_video_path = "./outputs/yolo_human_animal_tracking_result.mp4"

    model_path = "yolo11x.pt"
    confidence_threshold = 0.35
    iou_threshold = 0.5
    image_size = 1280
    device = None
    tracker_config = "bytetrack.yaml"

    if use_public_sample_video:
        loader = PublicVideoDatasetLoader(dataset_dir="./public_video_samples")
        input_video_path = loader.download(sample_name=public_sample_name, overwrite=False)
    else:
        input_video_path = custom_input_video_path

    detector = VideoHumanAnimalDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        image_size=image_size,
        device=device,
        tracker_config=tracker_config,
        smoothing_window=12,
        smoothing_min_votes=3,
    )

    stats = detector.process_video(input_video_path=input_video_path, output_video_path=output_video_path)
    print_stats(stats)


if __name__ == "__main__":
    main()

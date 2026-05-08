"""
完整的视频人类与动物检测框架。  # 用模块文档字符串说明本文件的用途，便于打开文件时快速理解整体目标。

功能说明：  # 下面列出脚本支持的主要功能。
1. 从输入视频中逐帧读取图像。  # 第一步是把视频拆成一帧一帧的图像进行处理。
2. 使用零样本目标检测模型检测人物与动物。  # 通过开放词汇检测模型识别 person、dog、cat、tiger 等类别。
3. 对检测到的人物区域再次做男人/女人二分类。  # 通过第二个视觉语言模型对人物框内区域做更细粒度识别。
4. 在视频帧上绘制检测框、类别名称与置信度。  # 方便直接观察检测结果。
5. 将处理后的视频重新保存到磁盘。  # 最终输出一个带标注的新视频文件。

安装依赖示例：  # 提供一个推荐的依赖安装命令，便于首次运行。
pip install torch torchvision transformers pillow opencv-python tqdm  # 这些库分别用于深度学习、视觉模型、图像处理、视频读写和进度条。

运行方式：  # 补充说明如何从终端启动脚本。
直接执行 python3 video_human_animal_detector.py。  # 不需要再传命令行参数。
输入路径、输出路径、阈值等请在 main() 函数开头自行修改。  # 把所有可调参数集中到 main() 更方便实验。
"""  # 模块说明结束。

import os  # 导入操作系统模块，用于创建输出目录和处理文件路径。
from dataclasses import dataclass  # 导入 dataclass 装饰器，便于定义结构清晰的检测结果对象。
from typing import Dict, List, Tuple  # 导入常用类型标注工具，提升代码可读性。

import cv2  # 导入 OpenCV，用于读取视频、写入视频和绘制检测框。
import numpy as np  # 导入 NumPy，用于在 OpenCV 和 PIL/张量之间做数组转换。
import torch  # 导入 PyTorch 主库，用于模型推理和张量运算。
from PIL import Image  # 导入 PIL 图像类，用于把视频帧转换成视觉模型可接受的图像对象。
from PIL import ImageDraw  # 导入 PIL 绘图模块，用于在图像上绘制中文文本。
from PIL import ImageFont  # 导入 PIL 字体模块，用于加载中文字体并避免 OpenCV 中文乱码。
from tqdm import tqdm  # 导入 tqdm 进度条，用于显示视频处理进度。
from transformers import CLIPModel  # 导入 CLIP 模型，用于对人物框内区域做男人/女人分类。
from transformers import CLIPProcessor  # 导入 CLIP 处理器，用于把裁剪图和文本标签编码成模型输入。
from transformers import Owlv2ForObjectDetection  # 导入 OWLv2 目标检测模型，用于零样本检测人和动物。
from transformers import Owlv2Processor  # 导入 OWLv2 处理器，用于把图像和文本提示词变成模型输入。


@dataclass  # 使用 dataclass 简化样板代码，让检测结果对象更容易创建和访问。
class DetectionResult:  # 定义一个检测结果数据结构，用于统一保存单个目标的信息。
    label_en: str  # 保存英文类别名称，便于和模型标签保持一致。
    label_zh: str  # 保存中文类别名称，便于绘制到视频帧上时更直观。
    score: float  # 保存该检测结果的置信度分数。
    box: Tuple[int, int, int, int]  # 保存检测框坐标，格式为 (x1, y1, x2, y2)。


class VideoHumanAnimalDetector:  # 定义完整的视频检测器类，负责模型加载、逐帧推理和结果可视化。
    def __init__(  # 定义初始化函数，用于配置模型、类别和推理设备。
        self,  # self 表示当前对象实例本身。
        detection_threshold: float = 0.15,  # 设置检测阈值，分数低于该值的框会被过滤。
        nms_iou_threshold: float = 0.4,  # 设置 NMS 去重阈值，用于去掉高度重叠的重复框。
        device: str | None = None,  # 允许用户手动指定设备；若为空则自动选择 CUDA 或 CPU。
    ) -> None:  # 初始化函数不返回值。
        self.detection_threshold = detection_threshold  # 保存检测阈值配置，供后续推理时使用。
        self.nms_iou_threshold = nms_iou_threshold  # 保存 NMS 的 IoU 阈值配置。
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择当前可用的推理设备。

        self.detector_model_name = "google/owlv2-base-patch16-ensemble"  # 指定零样本检测模型名称，首次运行时会自动下载。
        self.clip_model_name = "openai/clip-vit-base-patch32"  # 指定 CLIP 模型名称，用于人物性别分类。

        self.detector_processor = Owlv2Processor.from_pretrained(self.detector_model_name)  # 加载 OWLv2 处理器，负责图像和文本提示词编码。
        self.detector_model = Owlv2ForObjectDetection.from_pretrained(self.detector_model_name).to(self.device)  # 加载 OWLv2 模型并移动到目标设备。
        self.detector_model.eval()  # 切换到评估模式，表示这里只做推理不训练。

        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)  # 加载 CLIP 处理器，用于编码人物裁剪图和文本标签。
        self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)  # 加载 CLIP 模型并移动到目标设备。
        self.clip_model.eval()  # 切换到评估模式，减少不必要的训练态行为。

        self.detect_prompts = [  # 定义目标检测阶段使用的开放词汇提示词列表。
            "person",  # 检测人物大类，后续再对人物区域细分男人或女人。
            "dog",  # 检测狗。
            "cat",  # 检测猫。
            "tiger",  # 检测老虎。
            "horse",  # 检测马。
            "sheep",  # 检测羊。
            "cow",  # 检测牛。
            "bear",  # 检测熊。
            "elephant",  # 检测大象。
            "deer",  # 检测鹿。
            "monkey",  # 检测猴子。
            "bird",  # 检测鸟。
            "zebra",  # 检测斑马。
            "giraffe",  # 检测长颈鹿。
        ]  # 检测提示词列表定义结束。

        self.gender_prompts = ["man", "woman"]  # 定义人物二次分类使用的标签，只在人物框内区域上使用。

        self.label_map_zh: Dict[str, str] = {  # 定义英文类别到中文类别的映射，便于输出中文结果。
            "person": "人",  # person 对应中文“人”。
            "man": "男人",  # man 对应中文“男人”。
            "woman": "女人",  # woman 对应中文“女人”。
            "dog": "狗",  # dog 对应中文“狗”。
            "cat": "猫",  # cat 对应中文“猫”。
            "tiger": "老虎",  # tiger 对应中文“老虎”。
            "horse": "马",  # horse 对应中文“马”。
            "sheep": "羊",  # sheep 对应中文“羊”。
            "cow": "牛",  # cow 对应中文“牛”。
            "bear": "熊",  # bear 对应中文“熊”。
            "elephant": "大象",  # elephant 对应中文“大象”。
            "deer": "鹿",  # deer 对应中文“鹿”。
            "monkey": "猴子",  # monkey 对应中文“猴子”。
            "bird": "鸟",  # bird 对应中文“鸟”。
            "zebra": "斑马",  # zebra 对应中文“斑马”。
            "giraffe": "长颈鹿",  # giraffe 对应中文“长颈鹿”。
        }  # 中英文映射字典定义结束。

        self.color_map: Dict[str, Tuple[int, int, int]] = {  # 定义不同类别绘制框时使用的颜色，格式为 BGR。
            "人": (0, 255, 255),  # 人类相关目标使用黄色。
            "男人": (255, 0, 0),  # 男人使用蓝色。
            "女人": (255, 0, 255),  # 女人使用紫色。
            "狗": (0, 255, 0),  # 狗使用绿色。
            "猫": (0, 200, 255),  # 猫使用橙色。
            "老虎": (0, 128, 255),  # 老虎使用深橙色。
        }  # 颜色映射字典定义结束。
        self.draw_font = self._load_font(font_size=20)  # 加载绘制文字所需的字体，优先使用支持中文的系统字体。

    def _load_font(self, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:  # 定义字体加载函数，优先找支持中文的字体并回退到默认字体。
        candidate_fonts = [  # 定义一个候选字体路径列表，尽量覆盖常见 Linux 环境。
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Noto CJK 常见于许多 Linux 发行版。
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # 某些系统会把 Noto CJK 安装在 opentype 目录。
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # 文泉驿正黑是常见的中文字体。
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # 若没有中文字体，则回退到 DejaVu Sans。
        ]  # 候选字体列表定义结束。

        for font_path in candidate_fonts:  # 依次尝试加载每一个候选字体。
            if os.path.exists(font_path):  # 如果当前字体路径在系统中存在。
                try:  # 使用 try 防止字体文件损坏导致程序崩溃。
                    return ImageFont.truetype(font_path, font_size)  # 成功则直接返回该字体对象。
                except OSError:  # 如果当前字体无法被正常加载。
                    continue  # 则继续尝试下一个字体文件。

        return ImageFont.load_default()  # 如果所有字体都不可用，则回退到 PIL 默认字体。

    def classify_person_crop(self, crop_rgb: np.ndarray) -> Tuple[str, float]:  # 定义人物区域二次分类函数，返回男人或女人及其分数。
        if crop_rgb.size == 0:  # 如果裁剪出的图像区域为空。
            return "person", 0.0  # 则直接返回 person 和 0 分，避免后续模型推理报错。

        crop_image = Image.fromarray(crop_rgb)  # 把 NumPy 数组转为 PIL 图像，供 CLIP 处理器使用。
        inputs = self.clip_processor(  # 使用 CLIP 处理器把图像和文本标签编码成模型输入。
            text=self.gender_prompts,  # 文本候选类别为 man 和 woman。
            images=crop_image,  # 图像输入为人物框裁剪图。
            return_tensors="pt",  # 让处理器返回 PyTorch 张量。
            padding=True,  # 对文本进行对齐补齐，保证批次维度一致。
        )  # CLIP 输入编码结束。
        inputs = {key: value.to(self.device) for key, value in inputs.items()}  # 把所有输入张量移动到当前设备上。

        with torch.no_grad():  # 在无梯度环境下做推理，节省显存和时间。
            outputs = self.clip_model(**inputs)  # 执行 CLIP 前向传播，得到图像与文本的相似度得分。
            logits_per_image = outputs.logits_per_image  # 取出图像对各个文本标签的匹配分数。
            probs = logits_per_image.softmax(dim=-1).squeeze(0)  # 对分数做 softmax，转成 man/woman 概率。

        best_index = int(probs.argmax().item())  # 找到概率最大的标签下标。
        best_label = self.gender_prompts[best_index]  # 取出对应的英文标签名称。
        best_score = float(probs[best_index].item())  # 取出对应的概率分数，转成普通 Python 浮点数。
        return best_label, best_score  # 返回性别英文标签和置信度。

    def compute_iou(self, box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:  # 定义计算两个框 IoU 的函数，用于后续 NMS 去重。
        x1 = max(box_a[0], box_b[0])  # 计算两个框交集左上角的 x 坐标。
        y1 = max(box_a[1], box_b[1])  # 计算两个框交集左上角的 y 坐标。
        x2 = min(box_a[2], box_b[2])  # 计算两个框交集右下角的 x 坐标。
        y2 = min(box_a[3], box_b[3])  # 计算两个框交集右下角的 y 坐标。

        inter_w = max(0, x2 - x1)  # 计算交集区域宽度，若无重叠则为 0。
        inter_h = max(0, y2 - y1)  # 计算交集区域高度，若无重叠则为 0。
        inter_area = inter_w * inter_h  # 计算交集面积。

        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])  # 计算第一个框的面积。
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])  # 计算第二个框的面积。
        union_area = area_a + area_b - inter_area  # 计算并集面积。

        if union_area <= 0:  # 如果并集面积非正，说明框异常。
            return 0.0  # 直接返回 0，避免除零错误。

        return inter_area / union_area  # 返回标准 IoU 值。

    def nms(self, detections: List[DetectionResult]) -> List[DetectionResult]:  # 定义简易 NMS，用于移除高度重叠的重复检测框。
        sorted_detections = sorted(detections, key=lambda item: item.score, reverse=True)  # 按置信度从高到低排序，优先保留高分框。
        kept_detections: List[DetectionResult] = []  # 初始化保留结果列表。

        while sorted_detections:  # 只要还有候选框就持续循环。
            best_detection = sorted_detections.pop(0)  # 取出当前分数最高的框。
            kept_detections.append(best_detection)  # 把它加入最终保留列表。

            remaining: List[DetectionResult] = []  # 初始化剩余候选列表。
            for candidate in sorted_detections:  # 遍历其余候选框。
                same_label = candidate.label_en == best_detection.label_en  # 判断两个框是否属于同一英文类别。
                iou_value = self.compute_iou(candidate.box, best_detection.box)  # 计算当前候选框与最佳框的 IoU。
                if not (same_label and iou_value > self.nms_iou_threshold):  # 如果不是同类高度重叠框，则保留继续参与竞争。
                    remaining.append(candidate)  # 把未被抑制的框加入剩余列表。

            sorted_detections = remaining  # 更新候选列表，继续下一轮选择。

        return kept_detections  # 返回经过 NMS 处理后的检测结果。

    def detect_frame(self, frame_bgr: np.ndarray) -> List[DetectionResult]:  # 定义单帧检测函数，输入一帧 BGR 图像，输出检测结果列表。
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # 把 OpenCV 的 BGR 图像转换成 RGB，便于 PIL 和模型处理。
        pil_image = Image.fromarray(frame_rgb)  # 把 RGB 数组包装成 PIL 图像对象。

        inputs = self.detector_processor(  # 使用 OWLv2 处理器编码图像和文本提示词。
            text=[self.detect_prompts],  # 注意这里传入的是二维列表，表示一张图像对应一组查询类别。
            images=pil_image,  # 图像输入为当前视频帧。
            return_tensors="pt",  # 返回 PyTorch 张量。
        )  # 检测模型输入编码结束。
        inputs = {key: value.to(self.device) for key, value in inputs.items()}  # 把所有输入张量移动到推理设备。

        with torch.no_grad():  # 在无梯度环境下执行推理，减少资源开销。
            outputs = self.detector_model(**inputs)  # 执行 OWLv2 前向传播，得到检测原始输出。

        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)  # 构造目标图像尺寸张量，格式为 [高度, 宽度]。
        results = self.detector_processor.post_process_object_detection(  # 把模型原始输出转换成实际框坐标、标签和分数。
            outputs=outputs,  # 传入模型原始输出。
            threshold=self.detection_threshold,  # 传入置信度阈值，低于阈值的候选框会被过滤。
            target_sizes=target_sizes,  # 传入原图大小，用于把框映射回原始像素坐标系。
        )  # 后处理结束。

        frame_h, frame_w = frame_rgb.shape[:2]  # 取出当前帧的高度和宽度，用于后续裁剪和边界裁切。
        raw_detections: List[DetectionResult] = []  # 初始化原始检测结果列表。

        for box, score, label in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"]):  # 遍历当前帧的所有候选框。
            x1, y1, x2, y2 = box.tolist()  # 把张量形式的检测框坐标转成普通 Python 数值。
            x1 = max(0, min(frame_w - 1, int(x1)))  # 把左上角 x 坐标裁切到图像范围内，并转成整数。
            y1 = max(0, min(frame_h - 1, int(y1)))  # 把左上角 y 坐标裁切到图像范围内，并转成整数。
            x2 = max(0, min(frame_w - 1, int(x2)))  # 把右下角 x 坐标裁切到图像范围内，并转成整数。
            y2 = max(0, min(frame_h - 1, int(y2)))  # 把右下角 y 坐标裁切到图像范围内，并转成整数。

            if x2 <= x1 or y2 <= y1:  # 如果框宽或高不合法。
                continue  # 直接跳过该异常框。

            label_en = self.detect_prompts[int(label.item())]  # 根据标签下标取出当前检测结果对应的英文类别名。
            score_value = float(score.item())  # 取出当前检测框的置信度分数。

            if label_en == "person":  # 如果当前检测到的是人物大类。
                crop_rgb = frame_rgb[y1:y2, x1:x2]  # 从原始 RGB 帧中裁剪出人物框区域。
                gender_label, gender_score = self.classify_person_crop(crop_rgb)  # 对人物区域进一步分类为 man 或 woman。
                if gender_score >= 0.5:  # 如果性别分类置信度足够高。
                    label_en = gender_label  # 则把最终英文标签替换成更细粒度的 man 或 woman。
                    score_value = score_value * gender_score  # 把检测分和分类分相乘，得到一个更保守的综合置信度。

            label_zh = self.label_map_zh.get(label_en, label_en)  # 根据英文标签映射出中文名称，若无映射则保留英文。
            raw_detections.append(  # 把当前检测结果打包后加入原始结果列表。
                DetectionResult(  # 创建一个 DetectionResult 对象。
                    label_en=label_en,  # 保存英文类别名。
                    label_zh=label_zh,  # 保存中文类别名。
                    score=score_value,  # 保存置信度分数。
                    box=(x1, y1, x2, y2),  # 保存框坐标。
                )  # 当前检测结果对象创建结束。
            )  # 当前结果加入列表结束。

        filtered_detections = self.nms(raw_detections)  # 对所有检测结果执行 NMS，减少同类目标的重复框。
        return filtered_detections  # 返回当前帧的最终检测结果。

    def draw_detections(self, frame_bgr: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:  # 定义绘制函数，把检测框和类别信息画到图像上。
        output_frame = frame_bgr.copy()  # 先复制一份原始帧，避免直接修改输入帧。
        output_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)  # 把 BGR 图像转换成 RGB，供 PIL 正确绘制文本使用。
        pil_image = Image.fromarray(output_rgb)  # 把 RGB 数组转成 PIL 图像对象。
        draw = ImageDraw.Draw(pil_image)  # 创建 PIL 绘图对象，用于绘制中文文字和背景条。

        for detection in detections:  # 遍历当前帧中的每个检测目标。
            x1, y1, x2, y2 = detection.box  # 解包当前检测框的四个坐标。
            color = self.color_map.get(detection.label_zh, (0, 255, 0))  # 读取对应类别的颜色；若无特殊颜色则默认绿色。
            text = f"{detection.label_zh} {detection.score:.2f}"  # 构造显示文本，包含中文类别和置信度。

            draw.rectangle([(x1, y1), (x2, y2)], outline=(color[2], color[1], color[0]), width=2)  # 使用 PIL 绘制检测框，并把 OpenCV 的 BGR 颜色转换成 RGB。
            text_bbox = draw.textbbox((x1, max(0, y1 - 28)), text, font=self.draw_font)  # 先估计文本框大小，便于绘制合适的背景条。
            text_left = x1  # 设置文本背景条左边界与检测框左边界对齐。
            text_top = max(0, y1 - (text_bbox[3] - text_bbox[1]) - 10)  # 计算文本背景条顶部坐标，尽量放在检测框上方。
            text_right = min(output_frame.shape[1] - 1, text_left + (text_bbox[2] - text_bbox[0]) + 12)  # 根据文本宽度计算背景条右边界。
            text_bottom = min(output_frame.shape[0] - 1, text_top + (text_bbox[3] - text_bbox[1]) + 10)  # 根据文本高度计算背景条底部边界。
            draw.rectangle([(text_left, text_top), (text_right, text_bottom)], fill=(color[2], color[1], color[0]))  # 使用 PIL 在文本后方绘制实心背景条，注意这里要把 BGR 转成 RGB。
            draw.text((text_left + 6, text_top + 4), text, font=self.draw_font, fill=(0, 0, 0))  # 使用 PIL 绘制标签文本，优先支持中文显示。

        output_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)  # 把 PIL 图像重新转回 OpenCV 使用的 BGR 数组。
        return output_frame  # 返回绘制完成后的图像帧。

    def process_video(self, input_video_path: str, output_video_path: str) -> None:  # 定义视频处理主函数，负责逐帧检测并输出新视频。
        if not os.path.exists(input_video_path):  # 如果输入视频路径不存在。
            raise FileNotFoundError(f"未找到输入视频: {input_video_path}")  # 抛出异常并提示用户检查路径。

        os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)  # 确保输出视频所在目录存在，不存在则自动创建。

        capture = cv2.VideoCapture(input_video_path)  # 打开输入视频文件，准备逐帧读取。
        if not capture.isOpened():  # 如果视频打开失败。
            raise RuntimeError(f"无法打开输入视频: {input_video_path}")  # 抛出异常并终止执行。

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # 读取视频总帧数，供进度条显示使用。
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0  # 读取视频帧率；若读取失败则默认使用 25 帧每秒。
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # 读取视频帧宽度，用于创建输出视频。
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 读取视频帧高度，用于创建输出视频。

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 指定输出视频编码格式为 mp4v，兼容性较好。
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))  # 创建视频写入器对象。
        if not writer.isOpened():  # 如果输出视频写入器创建失败。
            capture.release()  # 先释放输入视频资源。
            raise RuntimeError(f"无法创建输出视频: {output_video_path}")  # 抛出异常提示用户检查输出路径和编码环境。

        progress = tqdm(total=frame_count if frame_count > 0 else None, desc="处理视频帧")  # 创建总进度条，用于显示视频处理整体进度。

        try:  # 使用 try/finally 确保即使中途报错也能释放视频资源。
            while True:  # 进入逐帧读取循环。
                success, frame_bgr = capture.read()  # 从输入视频中读取下一帧图像。
                if not success:  # 如果读取失败，通常表示视频已经到结尾。
                    break  # 跳出循环，结束处理。

                detections = self.detect_frame(frame_bgr)  # 对当前帧执行目标检测和人物细分类。
                visualized_frame = self.draw_detections(frame_bgr, detections)  # 在当前帧上绘制检测结果。
                writer.write(visualized_frame)  # 把标注后的帧写入输出视频文件。
                progress.update(1)  # 进度条向前推进 1 帧。

        finally:  # 无论是否报错，都会执行资源释放逻辑。
            progress.close()  # 关闭进度条，清理终端显示。
            capture.release()  # 释放输入视频读取器。
            writer.release()  # 释放输出视频写入器，确保文件正确保存。


def main() -> None:  # 定义脚本主函数，用于串联模型构建和视频处理流程；所有可调路径与超参在此处集中设置。
    input_video_path = "你的视频.mp4"  # 输入视频的本地路径字符串，请将占位符改成你机器上的真实视频文件路径。
    output_video_path = "./outputs/result.mp4"  # 输出视频的保存路径，目录不存在时 process_video 会自动创建上层目录。
    threshold = 0.2  # 目标检测置信度阈值；越高则候选框更少但更严格。
    nms_iou = 0.4  # NMS 去重 IoU 阈值；越大越容易保留邻近框，越小越会去重重叠框。
    device = None  # 推理设备；None 表示自动选择 CUDA 或 CPU，也可设为 "cuda" / "cpu" 强制指定。

    detector = VideoHumanAnimalDetector(  # 创建视频检测器对象。
        detection_threshold=threshold,  # 传入在 main() 中设置的检测阈值。
        nms_iou_threshold=nms_iou,  # 传入在 main() 中设置的 NMS 阈值。
        device=device,  # 传入在 main() 中设置的设备字符串或自动选择。
    )  # 检测器对象创建结束。

    detector.process_video(input_video_path, output_video_path)  # 调用主处理函数，对输入视频执行检测并保存结果视频。
    print(f"处理完成，结果已保存到: {output_video_path}")  # 在终端打印处理完成提示信息。


if __name__ == "__main__":  # 当当前文件作为脚本直接运行时，执行以下主流程。
    main()  # 调用主函数，启动完整的视频检测与分类过程。

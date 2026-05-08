"""
Complete Object Detection Framework
Backbone + Neck (FPN) + 2 Heads (Classification + Regression)

基于 RetinaNet 架构的完整目标检测框架
支持多尺度特征融合和多任务学习
"""

import argparse
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================================
# 第一部分: Backbone (ResNet50)
# ============================================================================

class Backbone(nn.Module):
    """
    ResNet50 Backbone for feature extraction
    输出多层特征图用于 FPN
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet50(weights=weights)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        self.layer1 = resnet.layer1  # 输出通道: 256
        self.layer2 = resnet.layer2  # 输出通道: 512
        self.layer3 = resnet.layer3  # 输出通道: 1024
        self.layer4 = resnet.layer4  # 输出通道: 2048

        self.out_channels = [256, 512, 1024, 2048]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        返回多层特征图 [C3, C4, C5, C6]
        C3: layer1 输出, stride=8
        C4: layer2 输出, stride=16
        C5: layer3 输出, stride=32
        C6: layer4 输出, stride=64
        """
        x = self.stem(x)

        c1 = self.layer1(x)   # stride=4,  64*4=256 channels
        c2 = self.layer2(c1)  # stride=8,  64*8=512 channels
        c3 = self.layer3(c2)  # stride=16, 64*16=1024 channels
        c4 = self.layer4(c3)  # stride=32, 64*32=2048 channels

        return [c1, c2, c3, c4]


# ============================================================================
# 第二部分: Neck (Feature Pyramid Network)
# ============================================================================

class ConvModule(nn.Module):
    """卷积模块: Conv + BatchNorm + ReLU"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN)
    自顶向下特征融合 + 横向连接
    """

    def __init__(self, in_channels: List[int], out_channels: int = 256) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in in_channels:
            lateral_conv = nn.Conv2d(in_ch, out_channels, kernel_size=1)
            fpn_conv = ConvModule(out_channels, out_channels, kernel_size=3)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        FPN 前向传播
        inputs: [C2, C3, C4, C5] from backbone
        outputs: [P2, P3, P4, P5] 融合后的特征
        """
        assert len(inputs) == len(self.in_channels)

        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )

        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]

        return outs


class PAFPN(nn.Module):
    """
    Path Aggregation Feature Pyramid Network (PAFPN)
    在 FPN 基础上增加自底向上路径聚合
    """

    def __init__(self, in_channels: List[int], out_channels: int = 256) -> None:
        super().__init__()
        self.fpn = FPN(in_channels, out_channels)

        self.downsample_convs = nn.ModuleList()
        self.upsample_convs = nn.ModuleList()

        for i in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvModule(out_channels, out_channels, kernel_size=3, stride=2)
            )
            self.upsample_convs.append(
                ConvModule(out_channels, out_channels, kernel_size=3)
            )

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        out = self.fpn(inputs)

        for i in range(len(out) - 1):
            down = self.downsample_convs[i](out[i])
            up = F.interpolate(out[i + 1], size=down.shape[-2:], mode="nearest")
            out[i + 1] = out[i + 1] + self.upsample_convs[i](down + up)

        return out


# ============================================================================
# 第三部分: Head (Classification + Regression)
# ============================================================================

class AnchorGenerator(nn.Module):
    """
    锚框生成器
    为每个特征图位置生成多个锚框
    """

    def __init__(
        self,
        scales: List[float] = [1.0, 1.2, 1.5],
        ratios: List[float] = [0.5, 1.0, 2.0],
        strides: List[int] = [8, 16, 32, 64],
    ) -> None:
        super().__init__()
        self.scales = scales
        self.ratios = ratios
        self.strides = strides

        self.num_anchors = len(scales) * len(ratios)

    def generate_anchors(self, feature_size: Tuple[int, int], stride: int, device: torch.device) -> torch.Tensor:
        """为单个特征图生成锚框"""
        h, w = feature_size
        shifts_x = torch.arange(0, w, dtype=torch.float32, device=device) * stride
        shifts_y = torch.arange(0, h, dtype=torch.float32, device=device) * stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        base_anchors = []
        for scale in self.scales:
            for ratio in self.ratios:
                anchor_w = scale * stride * math.sqrt(ratio)
                anchor_h = scale * stride / math.sqrt(ratio)
                base_anchors.append([-anchor_w / 2, -anchor_h / 2, anchor_w / 2, anchor_h / 2])

        base_anchors = torch.tensor(base_anchors, dtype=torch.float32, device=device)

        anchors = base_anchors.unsqueeze(0) + torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).unsqueeze(1)
        anchors = anchors.reshape(-1, 4)

        return anchors

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        """为所有特征图生成锚框"""
        device = feature_maps[0].device
        anchors = []
        for i, feat in enumerate(feature_maps):
            h, w = feat.shape[-2:]
            anchors.append(self.generate_anchors((h, w), self.strides[i], device))
        return anchors


class ClassificationHead(nn.Module):
    """
    分类 Head
    预测每个锚框的类别概率
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int,
        feature_channels: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Conv2d(feature_channels, num_anchors * num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.output.bias, bias_value)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """返回每个特征图的分类 logits"""
        outputs = []
        for feat in features:
            feat = self.conv(feat)
            logits = self.output(feat)
            B, _, H, W = logits.shape
            logits = logits.view(B, self.num_anchors, self.num_classes, H, W)
            logits = logits.permute(0, 3, 4, 1, 2)
            logits = logits.reshape(B, H * W * self.num_anchors, self.num_classes)
            outputs.append(logits)
        return outputs


class RegressionHead(nn.Module):
    """
    回归 Head
    预测每个锚框的边界框偏移量 (dx, dy, dw, dh)
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        feature_channels: int = 256,
    ) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.box_dim = 4

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Conv2d(feature_channels, num_anchors * self.box_dim, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.constant_(self.output.bias, 0.0)
        nn.init.normal_(self.output.weight, std=0.01)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """返回每个特征图的边界框预测"""
        outputs = []
        for feat in features:
            feat = self.conv(feat)
            bbox_pred = self.output(feat)
            B, _, H, W = bbox_pred.shape
            bbox_pred = bbox_pred.view(B, self.num_anchors, self.box_dim, H, W)
            bbox_pred = bbox_pred.permute(0, 3, 4, 1, 2)
            bbox_pred = bbox_pred.reshape(B, H * W * self.num_anchors, self.box_dim)
            outputs.append(bbox_pred)
        return outputs


# ============================================================================
# 第四部分: 完整检测器
# ============================================================================

class DetectionModel(nn.Module):
    """
    完整目标检测模型
    Backbone + Neck + 2 Heads
    """

    def __init__(
        self,
        num_classes: int = 80,
        backbone_pretrained: bool = True,
        neck_type: str = "FPN",
        fpn_channels: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.backbone = Backbone(pretrained=backbone_pretrained)

        if neck_type == "FPN":
            self.neck = FPN(
                in_channels=self.backbone.out_channels,
                out_channels=fpn_channels,
            )
        elif neck_type == "PAFPN":
            self.neck = PAFPN(
                in_channels=self.backbone.out_channels,
                out_channels=fpn_channels,
            )
        else:
            raise ValueError(f"Unknown neck type: {neck_type}")

        num_anchors = 9
        self.anchor_generator = AnchorGenerator()

        self.cls_head = ClassificationHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )

        self.reg_head = RegressionHead(
            in_channels=fpn_channels,
            num_anchors=num_anchors,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        features = self.backbone(x)
        fpn_features = self.neck(features)

        cls_outputs = self.cls_head(fpn_features)
        reg_outputs = self.reg_head(fpn_features)

        if return_features:
            return cls_outputs, reg_outputs, fpn_features

        return cls_outputs, reg_outputs


# ============================================================================
# 第五部分: 损失函数
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for dense object detection"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class DetectionLoss(nn.Module):
    """
    完整检测损失
    分类损失 (Focal Loss) + 回归损失 (Smooth L1 Loss)
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.25,
        gamma: float = 2.0,
        box_loss_weight: float = 1.0,
        cls_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.box_loss_weight = box_loss_weight
        self.cls_loss_weight = cls_loss_weight

        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(
        self,
        cls_outputs: List[torch.Tensor],
        reg_outputs: List[torch.Tensor],
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        anchors: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(gt_labels)
        total_cls_loss = 0.0
        total_reg_loss = 0.0
        total_positives = 0

        for b in range(batch_size):
            pos_losses = []

            for level_idx in range(len(cls_outputs)):
                cls_pred = cls_outputs[level_idx][b]
                reg_pred = reg_outputs[level_idx][b]
                anchor = anchors[level_idx]

                gt_label = gt_labels[b]
                gt_bbox = gt_bboxes[b]

                if len(gt_label) == 0:
                    continue

                matched_cls_loss = self.compute_cls_loss(cls_pred, reg_pred, gt_label, gt_bbox, anchor)
                total_cls_loss += matched_cls_loss["cls_loss"]

                if matched_cls_loss["num_positives"] > 0:
                    total_positives += matched_cls_loss["num_positives"]
                    pos_losses.append(matched_cls_loss["reg_loss"])

            if pos_losses:
                total_reg_loss += sum(pos_losses) / len(pos_losses)

        num_positives = max(total_positives, 1)

        loss_dict = {
            "cls_loss": total_cls_loss / num_positives * self.cls_loss_weight,
            "reg_loss": total_reg_loss / batch_size * self.box_loss_weight,
            "total_loss": (total_cls_loss / num_positives + total_reg_loss / batch_size),
        }

        return loss_dict

    def compute_cls_loss(
        self,
        cls_pred: torch.Tensor,
        reg_pred: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        anchors: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        num_anchors = anchors.shape[0]

        if len(gt_labels) == 0:
            valid_mask = torch.zeros(num_anchors, dtype=torch.bool, device=cls_pred.device)
            cls_loss = F.cross_entropy(
                cls_pred[valid_mask],
                torch.zeros_like(gt_labels, dtype=torch.long),
                reduction="mean",
            )
            return {"cls_loss": cls_loss, "reg_loss": torch.tensor(0.0), "num_positives": 0}

        ious = self.box_iou(anchors, gt_bboxes)
        max_iou, best_gt_idx = ious.max(dim=1)

        pos_threshold = 0.5
        neg_threshold = 0.4

        pos_mask = max_iou >= pos_threshold
        neg_mask = (max_iou >= neg_threshold) & (~pos_mask)

        target_cls = torch.zeros(num_anchors, dtype=torch.long, device=cls_pred.device)
        target_cls[pos_mask] = gt_labels[best_gt_idx[pos_mask]]
        target_cls[neg_mask] = 0

        cls_loss = F.cross_entropy(cls_pred, target_cls, reduction="mean", weight=None)

        pos_reg_loss = torch.tensor(0.0, device=cls_pred.device)
        if pos_mask.sum() > 0:
            pos_anchor = anchors[pos_mask]
            pos_gt_bbox = gt_bboxes[best_gt_idx[pos_mask]]

            pos_reg_pred = reg_pred[pos_mask]
            reg_target = self.encode_boxes(pos_anchor, pos_gt_bbox)

            pos_reg_loss = F.smooth_l1_loss(pos_reg_pred, reg_target, reduction="mean")

        return {
            "cls_loss": cls_loss,
            "reg_loss": pos_reg_loss,
            "num_positives": pos_mask.sum().item(),
        }

    @staticmethod
    def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2 - inter
        iou = inter / union.clamp(min=1e-6)

        return iou

    @staticmethod
    def encode_boxes(anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        cx = (anchors[:, 0] + anchors[:, 2]) / 2
        cy = (anchors[:, 1] + anchors[:, 3]) / 2
        w = anchors[:, 2] - anchors[:, 0]
        h = anchors[:, 3] - anchors[:, 1]

        gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]

        dx = (gt_cx - cx) / w.clamp(min=1e-6)
        dy = (gt_cy - cy) / h.clamp(min=1e-6)
        dw = torch.log(gt_w / w.clamp(min=1e-6))
        dh = torch.log(gt_h / h.clamp(min=1e-6))

        return torch.stack([dx, dy, dw, dh], dim=1)


# ============================================================================
# 第六部分: 数据集与数据加载
# ============================================================================

class SyntheticDetectionDataset(torch.utils.data.Dataset):
    """
    合成目标检测数据集 (用于演示)
    实际使用时替换为 COCO/VOC 数据集
    """

    def __init__(
        self,
        num_samples: int = 5000,
        image_size: int = 640,
        num_classes: int = 80,
        max_objects: int = 10,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.max_objects = max_objects

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        image = torch.rand(3, self.image_size, self.image_size) * 0.5 + 0.25

        num_objects = torch.randint(1, self.max_objects + 1, (1,)).item()
        labels = torch.randint(0, self.num_classes, (num_objects,))
        bboxes = []

        for _ in range(num_objects):
            x1 = torch.randint(0, self.image_size - 50, (1,)).item()
            y1 = torch.randint(0, self.image_size - 50, (1,)).item()
            w = torch.randint(20, min(100, self.image_size - x1), (1,)).item()
            h = torch.randint(20, min(100, self.image_size - y1), (1,)).item()
            bboxes.append([x1, y1, x1 + w, y1 + h])

        bboxes = torch.tensor(bboxes, dtype=torch.float32)

        target = {
            "labels": labels,
            "bboxes": bboxes,
        }

        return image, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def build_dataloader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    image_size: int = 640,
    num_classes: int = 80,
    train: bool = True,
) -> DataLoader:
    dataset = SyntheticDetectionDataset(
        num_samples=10000 if train else 1000,
        image_size=image_size,
        num_classes=num_classes,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=train,
    )

    return dataloader


# ============================================================================
# 第七部分: 训练与验证
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0.0,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    num_batches = 0

    anchor_generator = model.module.anchor_generator if hasattr(model, "module") else model.anchor_generator

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)

        gt_labels_list = [t["labels"].to(device) for t in targets]
        gt_bboxes_list = [t["bboxes"].to(device) for t in targets]

        optimizer.zero_grad()

        cls_outputs, reg_outputs = model(images)

        anchors = anchor_generator([feat.detach() for feat in cls_outputs])

        criterion = DetectionLoss(num_classes=model.num_classes)
        losses = criterion(cls_outputs, reg_outputs, gt_labels_list, gt_bboxes_list, anchors)

        loss = losses["total_loss"]
        loss.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        total_loss += losses["total_loss"].item()
        total_cls_loss += losses["cls_loss"].item()
        total_reg_loss += losses["reg_loss"].item()
        num_batches += 1

        if batch_idx % 50 == 0:
            print(
                f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                f"Loss: {losses['total_loss'].item():.4f} "
                f"Cls: {losses['cls_loss'].item():.4f} "
                f"Reg: {losses['reg_loss'].item():.4f}"
            )

    return {
        "total_loss": total_loss / num_batches,
        "cls_loss": total_cls_loss / num_batches,
        "reg_loss": total_reg_loss / num_batches,
    }


@torch.no_grad()
def validate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    num_batches = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)

        gt_labels_list = [t["labels"].to(device) for t in targets]
        gt_bboxes_list = [t["bboxes"].to(device) for t in targets]

        cls_outputs, reg_outputs = model(images)

        anchor_generator = model.anchor_generator
        anchors = anchor_generator([feat for feat in cls_outputs])

        criterion = DetectionLoss(num_classes=model.num_classes)
        losses = criterion(cls_outputs, reg_outputs, gt_labels_list, gt_bboxes_list, anchors)

        total_loss += losses["total_loss"].item()
        num_batches += 1

    return {"val_loss": total_loss / num_batches}


# ============================================================================
# 第八部分: 主训练脚本
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete Detection Model Training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--image-size", type=int, default=640, help="Input image size")
    parser.add_argument("--num-classes", type=int, default=80, help="Number of object classes")
    parser.add_argument("--backbone-pretrained", action="store_true", help="Use pretrained backbone")
    parser.add_argument("--no-pretrain", dest="backbone_pretrained", action="store_false")
    parser.add_argument("--neck", type=str, default="FPN", choices=["FPN", "PAFPN"], help="Neck type")
    parser.add_argument("--fpn-channels", type=int, default=256, help="FPN output channels")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = DetectionModel(
        num_classes=args.num_classes,
        backbone_pretrained=args.backbone_pretrained,
        neck_type=args.neck,
        fpn_channels=args.fpn_channels,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model: Total parameters = {total_params:.2f}M, Trainable = {trainable_params:.2f}M")

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader = build_dataloader(
        data_dir=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        num_classes=args.num_classes,
        train=True,
    )

    val_loader = build_dataloader(
        data_dir=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        num_classes=args.num_classes,
        train=False,
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch + 1,
            max_norm=10.0,
        )

        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch [{epoch + 1}/{args.epochs}] "
            f"Time: {epoch_time:.1f}s, LR: {current_lr:.6f}\n"
            f"Train - Loss: {train_metrics['total_loss']:.4f}, "
            f"Cls: {train_metrics['cls_loss']:.4f}, "
            f"Reg: {train_metrics['reg_loss']:.4f}\n"
            f"Val   - Loss: {val_metrics['val_loss']:.4f}\n"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }

        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "last.pth"))

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, "best.pth"))
            print(f"Saved best model with val_loss = {best_val_loss:.4f}")

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

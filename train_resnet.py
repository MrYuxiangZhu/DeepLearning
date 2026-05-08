"""
ResNet-50 Training Code
完整的 ResNet-50 训练脚本，支持 CIFAR-10/CIFAR-100 数据集
"""

import argparse
import os
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: type, layers: List[int], num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: type, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        layers.extend([block(self.in_channels, out_channels) for _ in range(1, blocks)])

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet50(num_classes: int = 10) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet101(num_classes: int = 10) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def resnet152(num_classes: int = 10) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


MODEL_CONFIGS: Dict[str, nn.Module] = {
    "ResNet50": resnet50,
    "ResNet101": resnet101,
    "ResNet152": resnet152,
}


def build_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    dataset: str = "CIFAR10",
) -> Tuple[DataLoader, DataLoader]:
    if dataset == "CIFAR10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        num_classes = 10
    elif dataset == "CIFAR100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    else:
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer = None,
    scaler: torch.cuda.amp.GradScaler = None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            if is_train and scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        current_batch_size = labels.size(0)
        total_loss += loss.item() * current_batch_size
        total_acc += accuracy(outputs, labels) * current_batch_size
        total_samples += current_batch_size

    return total_loss / total_samples, total_acc / total_samples


def save_checkpoint(state: dict, checkpoint_dir: str, filename: str) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, filename))


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer = None, scheduler: optim.lr_scheduler._LRScheduler = None) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def prompt_if_missing(value, prompt_text: str, cast_func, default, choices=None):
    if value not in (None, ""):
        return value

    while True:
        raw_value = input(f"{prompt_text} [default: {default}]: ").strip()
        if raw_value == "":
            return default

        try:
            parsed_value = cast_func(raw_value)
        except ValueError:
            print("输入格式不正确，请重新输入。")
            continue

        if choices is not None and parsed_value not in choices:
            print(f"输入必须是以下之一: {', '.join(map(str, choices))}")
            continue

        return parsed_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet model on CIFAR-10/100.")
    parser.add_argument("--model", type=str, default=None, choices=MODEL_CONFIGS.keys(), help="ResNet variant.")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory.")
    parser.add_argument("--dataset", type=str, default=None, choices=["CIFAR10", "CIFAR100"], help="Dataset to use.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Warmup epochs.")
    parser.add_argument("--label-smoothing", type=float, default=None, help="Label smoothing factor.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_name = prompt_if_missing(args.model, "请输入模型名称", str, "ResNet50", choices=MODEL_CONFIGS.keys())
    data_dir = prompt_if_missing(args.data_dir, "请输入数据集目录", str, "./data")
    checkpoint_dir = prompt_if_missing(args.checkpoint_dir, "请输入模型保存目录", str, "./checkpoints")
    dataset_name = prompt_if_missing(args.dataset, "请输入数据集名称", str, "CIFAR10", choices=["CIFAR10", "CIFAR100"])
    epochs = prompt_if_missing(args.epochs, "请输入训练轮数", int, 200)
    batch_size = prompt_if_missing(args.batch_size, "请输入批大小", int, 128)
    lr = prompt_if_missing(args.lr, "请输入学习率", float, 0.1)
    weight_decay = prompt_if_missing(args.weight_decay, "请输入权重衰减", float, 1e-4)
    num_workers = prompt_if_missing(args.num_workers, "请输入 DataLoader workers 数量", int, 4)
    seed = prompt_if_missing(args.seed, "请输入随机种子", int, 42)
    warmup_epochs = prompt_if_missing(args.warmup_epochs, "请输入预热轮数", int, 5)
    label_smoothing = prompt_if_missing(args.label_smoothing, "请输入标签平滑系数", float, 0.1)

    num_classes = 100 if dataset_name == "CIFAR100" else 10

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader = build_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        dataset=dataset_name,
    )
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    model = MODEL_CONFIGS[model_name](num_classes=num_classes).to(device)
    print(f"Model: {model_name}, Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True,
    )

    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_items=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], [warmup_epochs])

    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    start_epoch = 1
    best_val_acc = 0.0

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint.get("epoch", 1) + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")

    history = []

    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )

        val_loss, val_acc = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
            "time": elapsed,
        })

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] "
            f"lr={current_lr:.6f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"time={elapsed:.1f}s"
        )

        checkpoint = {
            "epoch": epoch,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "args": {
                "model": model_name,
                "data_dir": data_dir,
                "checkpoint_dir": checkpoint_dir,
                "dataset": dataset_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "num_workers": num_workers,
                "seed": seed,
                "warmup_epochs": warmup_epochs,
                "label_smoothing": label_smoothing,
            },
            "history": history,
        }

        save_checkpoint(checkpoint, checkpoint_dir, "last.pth")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint["best_val_acc"] = best_val_acc
            save_checkpoint(checkpoint, checkpoint_dir, "best.pth")
            print(f"Saved best model with val_acc={best_val_acc:.4f}")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()

import argparse  # 导入命令行参数解析模块，用于读取脚本运行时的外部参数。
import os  # 导入操作系统模块，用于创建目录和处理文件路径。
import time  # 导入时间模块，用于记录每轮训练耗时。
from typing import Tuple  # 导入类型标注工具，用于标记函数返回值类型。

import torch  # 导入 PyTorch 主库，用于张量运算和神经网络训练。
import torch.nn as nn  # 导入神经网络子模块，并命名为 nn 方便调用。
import torch.optim as optim  # 导入优化器子模块，并命名为 optim 方便调用。
from torch.utils.data import DataLoader  # 导入数据加载器，用于批量读取数据。
from torchvision import datasets, transforms  # 导入 torchvision 数据集和图像变换工具。


class SimpleCNN(nn.Module):  # 定义一个简单 CNN 分类网络。
    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:  # 初始化网络结构与分类参数。
        super().__init__()  # 调用父类初始化方法，完成 nn.Module 基础设置。
        self.features = nn.Sequential(  # 定义卷积特征提取部分。
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),  # 第一层卷积，将 3 通道输入映射为 64 通道特征图。
            nn.BatchNorm2d(64),  # 对 64 通道特征做批归一化。
            nn.ReLU(inplace=True),  # 使用 ReLU 激活函数增加非线性。
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),  # 第二层卷积，继续提取 64 通道特征。
            nn.BatchNorm2d(64),  # 对第二层卷积输出做批归一化。
            nn.ReLU(inplace=True),  # 再次使用 ReLU 激活函数。
            nn.MaxPool2d(kernel_size=2, stride=2),  # 通过最大池化将特征图尺寸减半。
            nn.Dropout(0.25),  # 添加轻度 Dropout，减轻过拟合。
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),  # 第三层卷积，将通道数扩展到 128。
            nn.BatchNorm2d(128),  # 对 128 通道特征做批归一化。
            nn.ReLU(inplace=True),  # 使用 ReLU 激活函数。
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  # 第四层卷积，继续提取 128 通道特征。
            nn.BatchNorm2d(128),  # 对第四层卷积输出做批归一化。
            nn.ReLU(inplace=True),  # 使用 ReLU 激活函数。
            nn.MaxPool2d(kernel_size=2, stride=2),  # 再次下采样，缩小空间尺寸。
            nn.Dropout(0.25),  # 再次添加轻度 Dropout。
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),  # 第五层卷积，将通道数扩展到 256。
            nn.BatchNorm2d(256),  # 对 256 通道特征做批归一化。
            nn.ReLU(inplace=True),  # 使用 ReLU 激活函数。
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),  # 第六层卷积，继续提取 256 通道特征。
            nn.BatchNorm2d(256),  # 对第六层卷积输出做批归一化。
            nn.ReLU(inplace=True),  # 使用 ReLU 激活函数。
            nn.AdaptiveAvgPool2d((1, 1)),  # 使用自适应平均池化把空间维度收缩为 1x1。
        )  # 特征提取部分定义结束。
        self.classifier = nn.Sequential(  # 定义分类器部分。
            nn.Flatten(),  # 将卷积输出展平成一维向量。
            nn.Linear(256, 256),  # 第一层全连接，把 256 维输入映射到 256 维隐藏层。
            nn.ReLU(inplace=True),  # 使用 ReLU 激活函数。
            nn.Dropout(dropout),  # 使用可配置的 Dropout 做正则化。
            nn.Linear(256, num_classes),  # 输出层映射到具体类别数。
        )  # 分类器定义结束。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义模型前向传播过程。
        x = self.features(x)  # 先提取图像的卷积特征。
        x = self.classifier(x)  # 再通过分类器得到最终预测结果。
        return x  # 返回模型输出 logits。


def build_dataloaders(  # 定义数据加载器构建函数。
    data_dir: str,  # 数据集所在目录。
    batch_size: int,  # 每个批次的样本数量。
    num_workers: int,  # 后台读取数据使用的进程数。
) -> Tuple[DataLoader, DataLoader]:  # 返回训练集和验证集的数据加载器。
    train_transform = transforms.Compose(  # 定义训练集的数据增强与预处理管道。
        [  # 下面依次执行每个训练预处理步骤。
            transforms.RandomCrop(32, padding=4),  # 先对图像随机裁剪并填充边界。
            transforms.RandomHorizontalFlip(),  # 再随机水平翻转图像增强样本多样性。
            transforms.ToTensor(),  # 将图像转换为张量格式。
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 使用 CIFAR-10 统计值做标准化。
        ]  # 训练变换列表结束。
    )  # 训练预处理组合结束。
    test_transform = transforms.Compose(  # 定义验证集的数据预处理管道。
        [  # 验证阶段只保留基础张量化与标准化。
            transforms.ToTensor(),  # 将图像转换为张量。
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 使用相同统计值标准化。
        ]  # 验证变换列表结束。
    )  # 验证预处理组合结束。

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)  # 创建 CIFAR-10 训练集对象。
    val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)  # 创建 CIFAR-10 测试集对象作为验证集。

    train_loader = DataLoader(  # 创建训练集数据加载器。
        train_dataset,  # 指定训练数据集。
        batch_size=batch_size,  # 指定每个训练批次的大小。
        shuffle=True,  # 训练阶段打乱顺序以提升泛化能力。
        num_workers=num_workers,  # 指定并行加载数据的进程数。
        pin_memory=torch.cuda.is_available(),  # 如果使用 GPU，则启用锁页内存以提高传输效率。
    )  # 训练加载器定义结束。
    val_loader = DataLoader(  # 创建验证集数据加载器。
        val_dataset,  # 指定验证数据集。
        batch_size=batch_size,  # 指定每个验证批次的大小。
        shuffle=False,  # 验证阶段不打乱数据顺序。
        num_workers=num_workers,  # 指定并行加载数据的进程数。
        pin_memory=torch.cuda.is_available(),  # 如果使用 GPU，则启用锁页内存。
    )  # 验证加载器定义结束。
    return train_loader, val_loader  # 返回训练加载器和验证加载器。


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:  # 定义批次准确率计算函数。
    preds = outputs.argmax(dim=1)  # 取每个样本预测得分最高的类别作为预测标签。
    correct = (preds == targets).sum().item()  # 统计预测正确的样本个数。
    return correct / targets.size(0)  # 用正确数除以样本数得到准确率。


def run_one_epoch(  # 定义单个 epoch 的训练或验证过程。
    model: nn.Module,  # 当前要使用的模型。
    loader: DataLoader,  # 当前阶段对应的数据加载器。
    criterion: nn.Module,  # 当前使用的损失函数。
    device: torch.device,  # 当前使用的计算设备。
    optimizer: optim.Optimizer = None,  # 若提供优化器则执行训练，否则执行验证。
) -> Tuple[float, float]:  # 返回该轮的平均损失和平均准确率。
    is_train = optimizer is not None  # 根据优化器是否为空判断是否为训练阶段。
    model.train() if is_train else model.eval()  # 切换模型到训练模式或评估模式。

    total_loss = 0.0  # 初始化累计损失。
    total_acc = 0.0  # 初始化累计准确率。
    total_samples = 0  # 初始化累计样本数量。

    for images, labels in loader:  # 遍历数据加载器中的每个批次。
        images = images.to(device, non_blocking=True)  # 将图像张量移动到目标设备。
        labels = labels.to(device, non_blocking=True)  # 将标签张量移动到目标设备。

        with torch.set_grad_enabled(is_train):  # 训练阶段开启梯度，验证阶段关闭梯度。
            outputs = model(images)  # 前向传播得到模型输出。
            loss = criterion(outputs, labels)  # 计算当前批次损失。

            if is_train:  # 如果当前是训练阶段。
                optimizer.zero_grad()  # 先清空历史梯度。
                loss.backward()  # 反向传播计算梯度。
                optimizer.step()  # 使用优化器更新参数。

        current_batch_size = labels.size(0)  # 获取当前批次样本数量。
        total_loss += loss.item() * current_batch_size  # 按样本数累计损失和。
        total_acc += accuracy(outputs, labels) * current_batch_size  # 按样本数累计准确率和。
        total_samples += current_batch_size  # 累加已处理样本总数。

    return total_loss / total_samples, total_acc / total_samples  # 返回整轮平均损失与平均准确率。


def save_checkpoint(state: dict, checkpoint_dir: str, filename: str) -> None:  # 定义保存训练状态的函数。
    os.makedirs(checkpoint_dir, exist_ok=True)  # 确保检查点目录存在，不存在时自动创建。
    torch.save(state, os.path.join(checkpoint_dir, filename))  # 将状态字典保存到指定路径文件。


def prompt_if_missing(  # 定义参数补全函数。
    value,  # 当前参数值。
    prompt_text: str,  # 终端提示文本。
    cast_func,  # 输入值转换函数。
    default,  # 默认值。
    choices=None,  # 可选合法值集合。
):  # 参数定义结束。
    if value not in (None, ""):  # 如果参数已经存在有效值。
        return value  # 直接返回已有值，不再提示用户输入。

    while True:  # 持续循环直到获得合法输入。
        raw_value = input(f"{prompt_text} [default: {default}]: ").strip()  # 读取用户输入并去除首尾空白。
        if raw_value == "":  # 如果用户直接回车。
            return default  # 返回默认值。

        try:  # 尝试进行类型转换。
            parsed_value = cast_func(raw_value)  # 按指定函数转换输入内容。
        except ValueError:  # 如果转换失败。
            print("输入格式不正确，请重新输入。")  # 提示用户格式错误。
            continue  # 继续循环重新输入。

        if choices is not None and parsed_value not in choices:  # 如果设置了合法值范围且输入不在其中。
            print(f"输入必须是以下之一: {', '.join(map(str, choices))}")  # 提示用户合法输入范围。
            continue  # 继续循环重新输入。

        return parsed_value  # 返回校验通过后的参数值。


def parse_args() -> argparse.Namespace:  # 定义命令行参数解析函数。
    parser = argparse.ArgumentParser(description="Train a CNN model on CIFAR-10.")  # 创建命令行参数解析器。
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory.")  # 添加数据目录参数。
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory.")  # 添加检查点目录参数。
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")  # 添加训练轮数参数。
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")  # 添加批大小参数。
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate.")  # 添加学习率参数。
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay.")  # 添加权重衰减参数。
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")  # 添加数据加载进程数参数。
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate.")  # 添加 Dropout 参数。
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")  # 添加随机种子参数。
    return parser.parse_args()  # 解析命令行参数并返回。


def main() -> None:  # 定义脚本主函数。
    args = parse_args()  # 先获取命令行输入参数。

    data_dir = prompt_if_missing(args.data_dir, "请输入数据集目录", str, "./data")  # 获取数据集目录，缺失时交互输入。
    checkpoint_dir = prompt_if_missing(args.checkpoint_dir, "请输入模型保存目录", str, "./checkpoints_cnn")  # 获取模型保存目录，缺失时交互输入。
    epochs = prompt_if_missing(args.epochs, "请输入训练轮数", int, 30)  # 获取训练轮数，缺失时交互输入。
    batch_size = prompt_if_missing(args.batch_size, "请输入批大小", int, 128)  # 获取批大小，缺失时交互输入。
    lr = prompt_if_missing(args.lr, "请输入学习率", float, 0.001)  # 获取学习率，缺失时交互输入。
    weight_decay = prompt_if_missing(args.weight_decay, "请输入权重衰减", float, 5e-4)  # 获取权重衰减系数，缺失时交互输入。
    num_workers = prompt_if_missing(args.num_workers, "请输入 DataLoader workers 数量", int, 4)  # 获取数据加载进程数，缺失时交互输入。
    dropout = prompt_if_missing(args.dropout, "请输入 dropout 比例", float, 0.5)  # 获取 Dropout 比例，缺失时交互输入。
    seed = prompt_if_missing(args.seed, "请输入随机种子", int, 42)  # 获取随机种子，缺失时交互输入。

    torch.manual_seed(seed)  # 设置 CPU 随机种子，便于实验复现。
    if torch.cuda.is_available():  # 如果当前设备支持 CUDA。
        torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置随机种子。

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择使用 GPU 或 CPU。
    print(f"Using device: {device}")  # 打印当前训练设备信息。

    train_loader, val_loader = build_dataloaders(  # 构建训练集和验证集加载器。
        data_dir=data_dir,  # 传入数据集目录。
        batch_size=batch_size,  # 传入批大小。
        num_workers=num_workers,  # 传入数据加载进程数。
    )  # 数据加载器构建结束。

    model = SimpleCNN(num_classes=10, dropout=dropout).to(device)  # 创建 CNN 模型并移动到目标设备。
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数。
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # 使用 Adam 优化器更新参数。
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # 定义余弦退火学习率调度器。

    best_val_acc = 0.0  # 初始化最佳验证准确率。

    for epoch in range(1, epochs + 1):  # 按设定轮数进行循环训练。
        start_time = time.time()  # 记录本轮开始时间。

        train_loss, train_acc = run_one_epoch(  # 执行一轮训练。
            model=model,  # 传入模型。
            loader=train_loader,  # 传入训练集加载器。
            criterion=criterion,  # 传入损失函数。
            device=device,  # 传入计算设备。
            optimizer=optimizer,  # 传入优化器以启用参数更新。
        )  # 训练过程结束。
        val_loss, val_acc = run_one_epoch(  # 执行一轮验证。
            model=model,  # 传入模型。
            loader=val_loader,  # 传入验证集加载器。
            criterion=criterion,  # 传入损失函数。
            device=device,  # 传入计算设备。
        )  # 验证过程结束。
        scheduler.step()  # 当前轮结束后调整学习率。

        elapsed = time.time() - start_time  # 计算当前轮总耗时。
        current_lr = optimizer.param_groups[0]["lr"]  # 获取当前生效学习率。
        print(  # 打印当前轮训练日志。
            f"Epoch [{epoch:02d}/{epochs:02d}] "  # 输出当前轮编号及总轮数。
            f"lr={current_lr:.6f} "  # 输出当前学习率。
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "  # 输出训练损失和训练准确率。
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "  # 输出验证损失和验证准确率。
            f"time={elapsed:.1f}s"  # 输出本轮耗时。
        )  # 日志打印结束。

        checkpoint = {  # 组装本轮需要保存的训练状态。
            "epoch": epoch,  # 保存当前轮次编号。
            "model_name": "SimpleCNN",  # 保存当前模型名称。
            "model_state_dict": model.state_dict(),  # 保存模型权重参数。
            "optimizer_state_dict": optimizer.state_dict(),  # 保存优化器内部状态。
            "scheduler_state_dict": scheduler.state_dict(),  # 保存学习率调度器状态。
            "best_val_acc": best_val_acc,  # 保存当前历史最佳验证准确率。
            "args": {  # 保存训练超参数信息。
                "data_dir": data_dir,  # 记录数据目录。
                "checkpoint_dir": checkpoint_dir,  # 记录检查点目录。
                "epochs": epochs,  # 记录训练轮数。
                "batch_size": batch_size,  # 记录批大小。
                "lr": lr,  # 记录学习率。
                "weight_decay": weight_decay,  # 记录权重衰减。
                "num_workers": num_workers,  # 记录数据加载进程数。
                "dropout": dropout,  # 记录 Dropout 比例。
                "seed": seed,  # 记录随机种子。
            },  # 超参数字典结束。
        }  # 检查点字典构建结束。
        save_checkpoint(checkpoint, checkpoint_dir, "last.pth")  # 将当前最新状态保存为 last.pth。

        if val_acc > best_val_acc:  # 如果当前验证准确率优于历史最佳。
            best_val_acc = val_acc  # 更新最佳验证准确率。
            checkpoint["best_val_acc"] = best_val_acc  # 同步更新检查点中的最佳准确率字段。
            save_checkpoint(checkpoint, checkpoint_dir, "best.pth")  # 将最佳模型保存为 best.pth。
            print(f"Saved best model with val_acc={best_val_acc:.4f}")  # 打印最佳模型保存提示。

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")  # 训练结束后打印最佳验证准确率。


if __name__ == "__main__":  # 仅当脚本被直接执行时才运行下面代码。
    main()  # 调用主函数启动训练流程。

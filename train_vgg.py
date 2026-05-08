import argparse  # 导入命令行参数解析模块，用于读取用户启动脚本时传入的参数。
import os  # 导入操作系统模块，用于创建目录和拼接文件路径。
import time  # 导入时间模块，用于统计每个训练轮次的耗时。
from typing import Dict, List, Tuple  # 导入类型标注工具，提升代码可读性与提示效果。

import torch  # 导入 PyTorch 主库，用于张量计算与深度学习训练。
import torch.nn as nn  # 导入神经网络模块，并简写为 nn 方便调用层与损失函数。
import torch.optim as optim  # 导入优化器模块，并简写为 optim 方便创建优化器和学习率调度器。
from torch.utils.data import DataLoader  # 导入数据加载器，用于按批次迭代数据集。
from torchvision import datasets, transforms  # 导入 torchvision 中的数据集和图像预处理工具。


VGG_CONFIGS: Dict[str, List] = {  # 定义不同 VGG 变体的网络结构配置表。
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],  # VGG11 的卷积与池化布局。
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],  # VGG13 的卷积与池化布局。
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],  # VGG16 的卷积与池化布局。
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],  # VGG19 的卷积与池化布局。
}  # 结构配置表定义结束。


class VGG(nn.Module):  # 定义 VGG 网络类，继承自 PyTorch 的神经网络基类。
    def __init__(self, vgg_name: str, num_classes: int = 10, dropout: float = 0.5) -> None:  # 初始化网络结构与超参数。
        super().__init__()  # 调用父类构造函数，完成 nn.Module 必需初始化。
        self.features = self._make_layers(VGG_CONFIGS[vgg_name])  # 根据配置表构建卷积特征提取部分。
        self.classifier = nn.Sequential(  # 定义全连接分类器部分。
            nn.Linear(512, 512),  # 第一层全连接，将卷积输出映射到 512 维特征。
            nn.ReLU(inplace=True),  # 使用 ReLU 激活函数增强非线性表达能力。
            nn.Dropout(dropout),  # 使用 Dropout 降低过拟合风险。
            nn.Linear(512, 512),  # 第二层全连接，继续提炼高层特征。
            nn.ReLU(inplace=True),  # 再次使用 ReLU 激活函数。
            nn.Dropout(dropout),  # 再次使用 Dropout 做正则化。
            nn.Linear(512, num_classes),  # 最后一层映射到类别数输出 logits。
        )  # 分类器定义结束。

    def _make_layers(self, config: List) -> nn.Sequential:  # 根据配置列表动态生成卷积层序列。
        layers: List[nn.Module] = []  # 初始化一个空列表，用来按顺序保存网络层。
        in_channels = 3  # 设置输入通道数为 3，对应 RGB 彩色图像。

        for item in config:  # 依次遍历配置中的每个元素。
            if item == "M":  # 如果当前元素是 M，表示这里需要插入池化层。
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 添加 2x2 最大池化层进行下采样。
            else:  # 如果当前元素不是 M，说明它是卷积层的输出通道数。
                layers.extend(  # 一次性追加一个卷积块中的多层结构。
                    [  # 下面这个列表表示一个标准卷积块。
                        nn.Conv2d(in_channels, item, kernel_size=3, padding=1, bias=False),  # 添加 3x3 卷积层并保持特征图尺寸不变。
                        nn.BatchNorm2d(item),  # 添加批归一化层，加速收敛并稳定训练。
                        nn.ReLU(inplace=True),  # 添加 ReLU 激活函数引入非线性。
                    ]  # 卷积块内部层定义结束。
                )  # 将卷积块添加进总层列表。
                in_channels = item  # 更新下一层卷积的输入通道数。

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))  # 最后添加自适应平均池化，把空间维度压缩到 1x1。
        return nn.Sequential(*layers)  # 将层列表打包为顺序容器并返回。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播逻辑。
        x = self.features(x)  # 先让输入通过卷积特征提取部分。
        x = torch.flatten(x, 1)  # 将卷积输出展平成二维张量，便于输入全连接层。
        x = self.classifier(x)  # 将展平后的特征送入分类器得到输出。
        return x  # 返回模型的最终预测结果。


def build_dataloaders(  # 定义数据加载器构建函数。
    data_dir: str,  # 数据集根目录参数。
    batch_size: int,  # 每个批次包含的样本数参数。
    num_workers: int,  # 后台加载数据的进程数量参数。
) -> Tuple[DataLoader, DataLoader]:  # 返回训练集和验证集两个数据加载器。
    train_transform = transforms.Compose(  # 定义训练集的数据增强与预处理流程。
        [  # 下面按顺序执行训练集变换。
            transforms.RandomCrop(32, padding=4),  # 对图像做随机裁剪并补边，提升泛化能力。
            transforms.RandomHorizontalFlip(),  # 以一定概率随机水平翻转图像。
            transforms.ToTensor(),  # 将 PIL 图像或数组转换为张量。
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 按 CIFAR-10 的均值和方差做标准化。
        ]  # 训练集变换列表结束。
    )  # 训练集预处理组合定义结束。
    test_transform = transforms.Compose(  # 定义验证集的数据预处理流程。
        [  # 验证集只做基础预处理，不做随机增强。
            transforms.ToTensor(),  # 将图像转换为张量格式。
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 使用同样的均值方差进行标准化。
        ]  # 验证集变换列表结束。
    )  # 验证集预处理组合定义结束。

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)  # 构建 CIFAR-10 训练数据集对象。
    val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)  # 构建 CIFAR-10 测试集作为验证集对象。

    train_loader = DataLoader(  # 创建训练集数据加载器。
        train_dataset,  # 指定要迭代的训练数据集。
        batch_size=batch_size,  # 设置训练时的批大小。
        shuffle=True,  # 训练阶段打乱数据顺序，有助于提升训练效果。
        num_workers=num_workers,  # 设置并行加载数据的进程数。
        pin_memory=torch.cuda.is_available(),  # 如果使用 GPU，则锁页内存以提升拷贝效率。
    )  # 训练集数据加载器定义结束。
    val_loader = DataLoader(  # 创建验证集数据加载器。
        val_dataset,  # 指定要迭代的验证数据集。
        batch_size=batch_size,  # 设置验证时的批大小。
        shuffle=False,  # 验证阶段不打乱数据顺序，保证评估稳定。
        num_workers=num_workers,  # 设置并行加载数据的进程数。
        pin_memory=torch.cuda.is_available(),  # 如果使用 GPU，则锁页内存以提升拷贝效率。
    )  # 验证集数据加载器定义结束。
    return train_loader, val_loader  # 返回训练与验证两个加载器。


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:  # 定义计算单个批次准确率的函数。
    preds = outputs.argmax(dim=1)  # 取每个样本预测分数最大的类别作为预测标签。
    correct = (preds == targets).sum().item()  # 统计当前批次中预测正确的样本数量。
    return correct / targets.size(0)  # 返回当前批次的平均准确率。


def run_one_epoch(  # 定义单轮训练或验证函数。
    model: nn.Module,  # 传入需要训练或评估的模型。
    loader: DataLoader,  # 传入当前阶段使用的数据加载器。
    criterion: nn.Module,  # 传入损失函数对象。
    device: torch.device,  # 传入计算设备信息。
    optimizer: optim.Optimizer = None,  # 如果传入优化器则执行训练，否则执行验证。
) -> Tuple[float, float]:  # 返回本轮平均损失与平均准确率。
    is_train = optimizer is not None  # 通过优化器是否为空来判断当前是否为训练阶段。
    model.train() if is_train else model.eval()  # 根据阶段切换模型到训练模式或评估模式。

    total_loss = 0.0  # 初始化累计损失。
    total_acc = 0.0  # 初始化累计准确率。
    total_samples = 0  # 初始化累计样本数。

    for images, labels in loader:  # 按批次遍历当前数据加载器中的图像和标签。
        images = images.to(device, non_blocking=True)  # 将图像张量移动到目标设备上。
        labels = labels.to(device, non_blocking=True)  # 将标签张量移动到目标设备上。

        with torch.set_grad_enabled(is_train):  # 仅在训练阶段开启梯度计算，验证阶段关闭以节省显存。
            outputs = model(images)  # 执行前向传播得到模型输出。
            loss = criterion(outputs, labels)  # 根据输出和标签计算当前批次损失。

            if is_train:  # 如果当前处于训练阶段，则执行反向传播和参数更新。
                optimizer.zero_grad()  # 先清空上一批次累积的梯度。
                loss.backward()  # 对当前损失执行反向传播，计算梯度。
                optimizer.step()  # 使用优化器根据梯度更新模型参数。

        current_batch_size = labels.size(0)  # 获取当前批次的实际样本数量。
        total_loss += loss.item() * current_batch_size  # 按样本数累计损失总和。
        total_acc += accuracy(outputs, labels) * current_batch_size  # 按样本数累计准确率总和。
        total_samples += current_batch_size  # 累计已处理的样本总数。

    return total_loss / total_samples, total_acc / total_samples  # 计算并返回整个轮次的平均损失和准确率。


def save_checkpoint(state: dict, checkpoint_dir: str, filename: str) -> None:  # 定义模型检查点保存函数。
    os.makedirs(checkpoint_dir, exist_ok=True)  # 如果目录不存在则创建目录，已存在则忽略报错。
    torch.save(state, os.path.join(checkpoint_dir, filename))  # 将训练状态字典保存到指定文件。


def prompt_if_missing(  # 定义参数补全函数，用于在命令行参数缺失时进行交互输入。
    value,  # 当前参数值，可能来自命令行。
    prompt_text: str,  # 终端提示文本内容。
    cast_func,  # 用于把输入字符串转换为目标类型的函数。
    default,  # 用户直接回车时采用的默认值。
    choices=None,  # 可选的合法输入范围集合。
):  # 参数列表结束。
    if value not in (None, ""):  # 如果参数已经有值，则直接返回该值。
        return value  # 返回已有参数，避免再次询问用户。

    while True:  # 循环提示，直到用户输入合法值为止。
        raw_value = input(f"{prompt_text} [default: {default}]: ").strip()  # 从终端读取输入并去掉首尾空白。
        if raw_value == "":  # 如果用户直接回车，表示使用默认值。
            return default  # 返回预设默认值。

        try:  # 尝试将用户输入转换为目标类型。
            parsed_value = cast_func(raw_value)  # 使用给定转换函数解析输入内容。
        except ValueError:  # 如果类型转换失败，则捕获异常。
            print("输入格式不正确，请重新输入。")  # 提示用户输入格式错误。
            continue  # 继续下一轮循环重新获取输入。

        if choices is not None and parsed_value not in choices:  # 如果设置了可选范围且输入不在范围内。
            print(f"输入必须是以下之一: {', '.join(map(str, choices))}")  # 提示用户合法的可选项。
            continue  # 继续循环重新输入。

        return parsed_value  # 输入合法时返回解析后的值。


def parse_args() -> argparse.Namespace:  # 定义命令行参数解析函数。
    parser = argparse.ArgumentParser(description="Train a VGG model on CIFAR-10.")  # 创建命令行解析器并设置脚本说明。
    parser.add_argument("--model", type=str, default=None, choices=VGG_CONFIGS.keys(), help="VGG variant.")  # 添加模型名称参数。
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory.")  # 添加数据集目录参数。
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoint directory.")  # 添加模型保存目录参数。
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")  # 添加训练轮数参数。
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")  # 添加批大小参数。
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate.")  # 添加初始学习率参数。
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay.")  # 添加权重衰减参数。
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")  # 添加数据加载进程数参数。
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate.")  # 添加 Dropout 比例参数。
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")  # 添加随机种子参数。
    return parser.parse_args()  # 解析命令行参数并返回结果对象。


def main() -> None:  # 定义主函数，组织完整训练流程。
    args = parse_args()  # 先读取命令行参数。

    model_name = prompt_if_missing(args.model, "请输入模型名称", str, "VGG16", choices=VGG_CONFIGS.keys())  # 获取模型名称，若缺失则终端交互输入。
    data_dir = prompt_if_missing(args.data_dir, "请输入数据集目录", str, "./data")  # 获取数据集目录，若缺失则终端交互输入。
    checkpoint_dir = prompt_if_missing(args.checkpoint_dir, "请输入模型保存目录", str, "./checkpoints")  # 获取检查点保存目录，若缺失则终端交互输入。
    epochs = prompt_if_missing(args.epochs, "请输入训练轮数", int, 30)  # 获取训练总轮数，若缺失则终端交互输入。
    batch_size = prompt_if_missing(args.batch_size, "请输入批大小", int, 128)  # 获取批大小，若缺失则终端交互输入。
    lr = prompt_if_missing(args.lr, "请输入学习率", float, 0.1)  # 获取学习率，若缺失则终端交互输入。
    weight_decay = prompt_if_missing(args.weight_decay, "请输入权重衰减", float, 5e-4)  # 获取权重衰减系数，若缺失则终端交互输入。
    num_workers = prompt_if_missing(args.num_workers, "请输入 DataLoader workers 数量", int, 4)  # 获取数据加载进程数，若缺失则终端交互输入。
    dropout = prompt_if_missing(args.dropout, "请输入 dropout 比例", float, 0.5)  # 获取 Dropout 比例，若缺失则终端交互输入。
    seed = prompt_if_missing(args.seed, "请输入随机种子", int, 42)  # 获取随机种子，若缺失则终端交互输入。

    torch.manual_seed(seed)  # 设置 CPU 端随机种子，便于实验复现。
    if torch.cuda.is_available():  # 如果当前环境支持 CUDA。
        torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设备设置随机种子。

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择 GPU 或 CPU 作为运行设备。
    print(f"Using device: {device}")  # 打印当前实际使用的设备信息。

    train_loader, val_loader = build_dataloaders(  # 构建训练与验证数据加载器。
        data_dir=data_dir,  # 传入数据集根目录。
        batch_size=batch_size,  # 传入批大小参数。
        num_workers=num_workers,  # 传入并行加载进程数。
    )  # 数据加载器构建结束。

    model = VGG(vgg_name=model_name, num_classes=10, dropout=dropout).to(device)  # 创建 VGG 模型并移动到目标设备。
    criterion = nn.CrossEntropyLoss()  # 定义分类任务常用的交叉熵损失函数。
    optimizer = optim.SGD(  # 使用带动量的 SGD 优化器训练模型。
        model.parameters(),  # 指定需要更新的模型参数。
        lr=lr,  # 指定优化器初始学习率。
        momentum=0.9,  # 设置动量系数以加速收敛。
        weight_decay=weight_decay,  # 设置 L2 正则化强度。
        nesterov=True,  # 开启 Nesterov 动量优化。
    )  # 优化器定义结束。
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # 使用余弦退火策略调度学习率。

    best_val_acc = 0.0  # 初始化历史最佳验证准确率。

    for epoch in range(1, epochs + 1):  # 按照设定轮数开始循环训练。
        start_time = time.time()  # 记录当前轮次开始时间。

        train_loss, train_acc = run_one_epoch(  # 执行一轮训练并返回训练损失与准确率。
            model=model,  # 传入当前模型。
            loader=train_loader,  # 传入训练集加载器。
            criterion=criterion,  # 传入损失函数。
            device=device,  # 传入运行设备。
            optimizer=optimizer,  # 传入优化器以启用参数更新。
        )  # 训练轮次执行结束。
        val_loss, val_acc = run_one_epoch(  # 执行一轮验证并返回验证损失与准确率。
            model=model,  # 传入当前模型。
            loader=val_loader,  # 传入验证集加载器。
            criterion=criterion,  # 传入损失函数。
            device=device,  # 传入运行设备。
        )  # 验证轮次执行结束。
        scheduler.step()  # 每轮结束后更新学习率。

        elapsed = time.time() - start_time  # 计算当前轮次耗时。
        current_lr = optimizer.param_groups[0]["lr"]  # 读取当前优化器实际学习率。
        print(  # 打印当前轮次的训练日志。
            f"Epoch [{epoch:02d}/{epochs:02d}] "  # 输出当前轮次编号和总轮次。
            f"lr={current_lr:.6f} "  # 输出当前学习率。
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "  # 输出训练损失和训练准确率。
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "  # 输出验证损失和验证准确率。
            f"time={elapsed:.1f}s"  # 输出本轮耗时。
        )  # 日志打印结束。

        checkpoint = {  # 组装当前轮次要保存的检查点信息。
            "epoch": epoch,  # 保存当前轮次编号。
            "model_name": model_name,  # 保存当前模型名称。
            "model_state_dict": model.state_dict(),  # 保存模型参数字典。
            "optimizer_state_dict": optimizer.state_dict(),  # 保存优化器状态字典。
            "scheduler_state_dict": scheduler.state_dict(),  # 保存学习率调度器状态字典。
            "best_val_acc": best_val_acc,  # 保存截至当前的最佳验证准确率。
            "args": {  # 保存关键训练参数，便于后续复现。
                "model": model_name,  # 记录模型名称参数。
                "data_dir": data_dir,  # 记录数据目录参数。
                "checkpoint_dir": checkpoint_dir,  # 记录检查点目录参数。
                "epochs": epochs,  # 记录训练轮数参数。
                "batch_size": batch_size,  # 记录批大小参数。
                "lr": lr,  # 记录学习率参数。
                "weight_decay": weight_decay,  # 记录权重衰减参数。
                "num_workers": num_workers,  # 记录数据加载进程数参数。
                "dropout": dropout,  # 记录 Dropout 比例参数。
                "seed": seed,  # 记录随机种子参数。
            },  # 参数记录字典结束。
        }  # 检查点字典构建结束。
        save_checkpoint(checkpoint, checkpoint_dir, "last.pth")  # 每轮都保存最新模型为 last.pth。

        if val_acc > best_val_acc:  # 如果当前验证准确率刷新历史最佳。
            best_val_acc = val_acc  # 更新历史最佳验证准确率。
            checkpoint["best_val_acc"] = best_val_acc  # 同步更新检查点中的最佳准确率字段。
            save_checkpoint(checkpoint, checkpoint_dir, "best.pth")  # 额外保存当前最佳模型为 best.pth。
            print(f"Saved best model with val_acc={best_val_acc:.4f}")  # 打印最佳模型保存提示。

    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")  # 训练完成后输出最终最佳验证准确率。


if __name__ == "__main__":  # 只有当脚本被直接执行时才进入主函数。
    main()  # 调用主函数启动整个训练流程。

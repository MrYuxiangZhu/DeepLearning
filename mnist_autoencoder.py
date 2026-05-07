import os  # 导入操作系统模块，用于创建目录和拼接文件路径。
from typing import Tuple  # 导入类型标注工具，用于标记函数返回值类型。

import torch  # 导入 PyTorch 主库，用于张量运算和模型训练。
import torch.nn as nn  # 导入神经网络模块，并简写为 nn 便于调用。
import torch.nn.functional as F  # 导入常用函数式接口，并简写为 F 便于使用激活函数。
from torch.optim import Adam  # 从优化器模块中导入 Adam 优化器。
from torch.optim.lr_scheduler import StepLR  # 导入按步长衰减学习率的调度器。
from torch.utils.data import DataLoader  # 导入数据加载器，用于批量读取数据。
from torchvision import datasets, transforms  # 导入 torchvision 的数据集和图像预处理工具。
from torchvision.utils import save_image  # 导入图像保存工具，用于保存重建结果对比图。


class DownConvLayer(nn.Module):  # 定义下采样卷积层类，用于提取特征并缩小空间尺寸。
    def __init__(self, dim: int) -> None:  # 定义初始化函数，dim 表示输入与输出通道数。
        super().__init__()  # 调用父类初始化方法，完成 nn.Module 的基础设置。
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)  # 定义卷积层，保持通道数不变且卷积后特征图尺寸不变。
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 定义最大池化层，将特征图的高和宽缩小为原来的一半。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播函数，输入为四维图像张量。
        x = F.relu(self.conv(x))  # 先进行卷积，再经过 ReLU 激活函数增强非线性表达能力。
        x = self.pool(x)  # 对激活后的特征图做最大池化，实现下采样。
        return x  # 返回下采样后的特征图。


class UpConvLayer(nn.Module):  # 定义上采样卷积层类，用于恢复空间尺寸并细化特征。
    def __init__(self, dim: int) -> None:  # 定义初始化函数，dim 表示输入与输出通道数。
        super().__init__()  # 调用父类初始化方法，完成 nn.Module 的基础设置。
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)  # 定义卷积层，保持通道数不变并提取局部特征。
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")  # 定义最近邻上采样层，将高和宽都放大 2 倍。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播函数，输入为待恢复分辨率的特征图。
        x = F.relu(self.conv(x))  # 先通过卷积和 ReLU 激活提取特征。
        x = self.upsample(x)  # 再对特征图进行上采样，恢复更大的空间尺寸。
        return x  # 返回上采样后的特征图。


class Encoder(nn.Module):  # 定义编码器类，用于逐步压缩输入图像并得到潜在表示。
    def __init__(self, dim: int, layer_num: int = 2) -> None:  # 定义初始化函数，layer_num 表示下采样层数。
        super().__init__()  # 调用父类初始化方法，完成编码器模块的注册。
        self.convs = nn.ModuleList([DownConvLayer(dim) for _ in range(layer_num)])  # 按照层数创建多个下采样卷积层并存入模块列表。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义编码器前向传播函数。
        for conv in self.convs:  # 依次遍历每一个下采样卷积层。
            x = conv(x)  # 将当前特征图送入当前下采样层并更新输出。
        return x  # 返回编码后的潜在特征图。


class Decoder(nn.Module):  # 定义解码器类，用于把潜在特征图恢复成图像。
    def __init__(self, dim: int, layer_num: int = 2) -> None:  # 定义初始化函数，layer_num 表示上采样层数。
        super().__init__()  # 调用父类初始化方法，完成解码器模块的注册。
        self.convs = nn.ModuleList([UpConvLayer(dim) for _ in range(layer_num)])  # 按照层数创建多个上采样卷积层并存入模块列表。
        self.final_conv = nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1)  # 定义最终卷积层，把特征图映射回单通道灰度图。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义解码器前向传播函数。
        for conv in self.convs:  # 依次遍历每一个上采样卷积层。
            x = conv(x)  # 将当前特征图送入当前上采样层并更新输出。
        reconstruct = torch.sigmoid(self.final_conv(x))  # 经过最终卷积并使用 Sigmoid 将像素值压到 0 到 1 范围。
        return reconstruct  # 返回最终重建出的图像。


class AutoEncoderModel(nn.Module):  # 定义完整的自编码器模型类。
    def __init__(self, layer_num: int = 2) -> None:  # 定义初始化函数，layer_num 表示编码器和解码器的层数。
        super().__init__()  # 调用父类初始化方法，完成模型模块注册。
        self.encoder = Encoder(dim=1, layer_num=layer_num)  # 创建编码器模块，这里输入和中间通道都设置为 1。
        self.decoder = Decoder(dim=1, layer_num=layer_num)  # 创建解码器模块，与编码器层数保持一致。

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # 定义自编码器的前向传播函数。
        latent = self.encoder(inputs)  # 先通过编码器把原图压缩为潜在特征表示。
        reconstruct_img = self.decoder(latent)  # 再通过解码器把潜在特征还原为图像。
        return reconstruct_img  # 返回重建图像。


def build_dataloader(batch_size: int) -> DataLoader:  # 定义数据加载器构建函数。
    transform = transforms.Compose(  # 定义图像预处理流程。
        [  # 下面按顺序执行每一个预处理步骤。
            transforms.Resize((32, 32)),  # 先把 MNIST 图像从 28x28 统一缩放到 32x32。
            transforms.ToTensor(),  # 再把 PIL 图像转换成张量，并将像素值归一化到 0 到 1。
        ]  # 预处理步骤列表结束。
    )  # 组合变换定义结束。
    train_dataset = datasets.MNIST(  # 创建 MNIST 训练集对象。
        root="./data",  # 指定数据下载和存放目录。
        train=True,  # 表示加载训练集而不是测试集。
        transform=transform,  # 指定前面定义好的图像预处理流程。
        download=True,  # 如果本地没有数据集就自动下载。
    )  # 训练集对象创建结束。
    train_loader = DataLoader(  # 创建训练集数据加载器。
        train_dataset,  # 指定要读取的数据集对象。
        batch_size=batch_size,  # 指定每个批次的样本数量。
        shuffle=True,  # 训练时随机打乱样本顺序以提升泛化能力。
        num_workers=2,  # 指定后台加载数据使用的子进程数量。
        pin_memory=torch.cuda.is_available(),  # 如果使用 GPU，则启用锁页内存以提升传输速度。
    )  # 数据加载器定义结束。
    return train_loader  # 返回构建好的训练数据加载器。


def save_reconstruction_samples(  # 定义保存原图与重建图对比结果的函数。
    model: nn.Module,  # 当前已经训练好的模型。
    loader: DataLoader,  # 用于取出一批样本的数据加载器。
    device: torch.device,  # 当前使用的计算设备。
    save_dir: str,  # 图像保存目录。
    max_images: int = 8,  # 最多保存多少组对比图片。
) -> None:  # 函数无返回值，只负责保存图片文件。
    model.eval()  # 将模型切换到评估模式，关闭 Dropout 等训练态行为。
    os.makedirs(save_dir, exist_ok=True)  # 如果保存目录不存在，则自动创建目录。
    images, _ = next(iter(loader))  # 从数据加载器中取出一个批次的图像和标签。
    images = images.to(device)  # 将图像移动到目标设备上以便前向计算。
    with torch.no_grad():  # 在不计算梯度的上下文中执行推理，节省显存和计算。
        reconstructed_images = model(images)  # 使用模型对输入图像进行重建。
    images = images[:max_images].cpu()  # 截取前若干张原图并移动回 CPU 以便保存。
    reconstructed_images = reconstructed_images[:max_images].cpu()  # 截取前若干张重建图并移动回 CPU。
    comparison = torch.cat([images, reconstructed_images], dim=0)  # 把原图和重建图沿批次维拼接，方便一次性保存。
    save_image(comparison, os.path.join(save_dir, "reconstruction.png"), nrow=max_images)  # 将拼接后的图像网格保存为 PNG 文件。


def train_one_epoch(  # 定义单轮训练函数。
    model: nn.Module,  # 当前要训练的模型。
    loader: DataLoader,  # 训练数据加载器。
    optimizer: Adam,  # 用于更新参数的优化器。
    criterion: nn.Module,  # 用于计算重建误差的损失函数。
    device: torch.device,  # 当前使用的计算设备。
) -> float:  # 返回这一轮训练的平均损失值。
    model.train()  # 将模型切换到训练模式。
    running_loss = 0.0  # 初始化累计损失变量。
    total_samples = 0  # 初始化累计样本数变量。
    for images, _ in loader:  # 遍历数据加载器中的每一个批次，这里标签不会参与训练。
        images = images.to(device, non_blocking=True)  # 把输入图像移动到目标设备上。
        optimizer.zero_grad()  # 在当前批次反向传播前清空旧梯度。
        reconstructed_images = model(images)  # 执行前向传播，得到重建图像。
        loss = criterion(reconstructed_images, images)  # 计算重建图像与原图之间的均方误差。
        loss.backward()  # 反向传播，计算模型参数的梯度。
        optimizer.step()  # 根据梯度更新模型参数。
        batch_size = images.size(0)  # 读取当前批次的样本数量。
        running_loss += loss.item() * batch_size  # 按样本数累加当前批次的损失总和。
        total_samples += batch_size  # 累加当前已经处理过的样本数量。
    return running_loss / total_samples  # 返回这一轮的平均损失。


def setup_seed(seed: int) -> None:  # 定义随机种子设置函数，便于实验复现。
    torch.manual_seed(seed)  # 设置 CPU 上的随机种子。
    if torch.cuda.is_available():  # 如果当前环境中存在可用的 CUDA 设备。
        torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设备设置相同的随机种子。


def prepare_output_dirs() -> Tuple[str, str]:  # 定义输出目录准备函数。
    weight_dir = "./outputs/mnist_autoencoder"  # 定义模型权重保存目录。
    image_dir = "./outputs/mnist_autoencoder/images"  # 定义重建图像保存目录。
    os.makedirs(weight_dir, exist_ok=True)  # 如果权重目录不存在则自动创建。
    os.makedirs(image_dir, exist_ok=True)  # 如果图像目录不存在则自动创建。
    return weight_dir, image_dir  # 返回两个输出目录路径。


def main() -> None:  # 定义脚本主函数，负责串联完整训练流程。
    batch_size = 128  # 设置每个批次的样本数量。
    learning_rate = 1e-3  # 设置 Adam 优化器的初始学习率。
    num_epochs = 10  # 设置总训练轮数。
    seed = 42  # 设置随机种子，保证实验结果尽可能可复现。
    layer_num = 2  # 设置编码器和解码器中的卷积层数量。
    step_size = 5  # 设置学习率调度器每隔多少轮衰减一次学习率。
    gamma = 0.5  # 设置每次衰减时学习率乘上的系数。
    setup_seed(seed)  # 调用随机种子设置函数。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择使用 GPU 或 CPU 训练。
    print(f"当前使用设备: {device}")  # 打印当前训练所使用的设备信息。
    train_loader = build_dataloader(batch_size=batch_size)  # 构建 MNIST 训练数据加载器。
    model = AutoEncoderModel(layer_num=layer_num).to(device)  # 创建自编码器模型并移动到目标设备上。
    optimizer = Adam(model.parameters(), lr=learning_rate)  # 创建 Adam 优化器用于更新模型参数。
    criterion = nn.MSELoss()  # 创建均方误差损失函数，衡量重建图像与原图的差异。
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  # 创建学习率调度器，按固定轮数衰减学习率。
    weight_dir, image_dir = prepare_output_dirs()  # 创建并获取模型权重和图像的保存目录。
    best_loss = float("inf")  # 使用正无穷初始化最佳损失，便于后续比较。

    for epoch in range(1, num_epochs + 1):  # 从第 1 轮开始循环训练直到最后一轮。
        epoch_loss = train_one_epoch(  # 调用单轮训练函数，执行一轮参数更新。
            model=model,  # 传入当前模型。
            loader=train_loader,  # 传入训练数据加载器。
            optimizer=optimizer,  # 传入优化器。
            criterion=criterion,  # 传入损失函数。
            device=device,  # 传入计算设备。
        )  # 单轮训练调用结束。
        scheduler.step()  # 在每轮训练结束后更新学习率。
        current_lr = optimizer.param_groups[0]["lr"]  # 读取当前优化器正在使用的学习率。
        print(f"Epoch [{epoch}/{num_epochs}] - loss: {epoch_loss:.6f} - lr: {current_lr:.6f}")  # 打印当前轮的损失和学习率。
        if epoch_loss < best_loss:  # 如果当前轮损失优于历史最佳损失。
            best_loss = epoch_loss  # 更新最佳损失值。
            torch.save(model.state_dict(), os.path.join(weight_dir, "best_autoencoder.pth"))  # 保存当前最佳模型参数。

    torch.save(model.state_dict(), os.path.join(weight_dir, "last_autoencoder.pth"))  # 在训练结束后额外保存最后一轮模型参数。
    save_reconstruction_samples(model=model, loader=train_loader, device=device, save_dir=image_dir)  # 保存一张原图和重建图的对比图片。
    print("训练完成，模型权重和重建图片已保存到 outputs/mnist_autoencoder 目录。")  # 打印训练完成提示信息。


if __name__ == "__main__":  # 当脚本被直接运行时执行下面的主流程。
    main()  # 调用主函数启动整个 MNIST 自编码器训练过程。

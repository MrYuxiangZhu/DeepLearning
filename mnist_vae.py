import os  # 导入操作系统模块，用于创建目录和拼接文件路径。
from typing import Tuple  # 导入类型标注工具，用于标记函数返回值类型。

import torch  # 导入 PyTorch 主库，用于张量运算和自动求导。
import torch.nn as nn  # 导入神经网络模块，并简写为 nn 便于调用。
import torch.nn.functional as F  # 导入函数式接口，并简写为 F 便于使用激活函数与损失函数。
from torch.optim import Adam  # 导入 Adam 优化器，用于更新模型参数。
from torch.optim.lr_scheduler import StepLR  # 导入学习率调度器，用于按设定步长衰减学习率。
from torch.utils.data import DataLoader  # 导入数据加载器，用于按批次读取 MNIST 数据。
from torchvision import datasets, transforms  # 导入 torchvision 中的数据集与预处理工具。
from torchvision.utils import save_image  # 导入图像保存函数，用于保存重建图像和随机生成图像。


class VAEModel(nn.Module):  # 定义变分自编码器模型类。
    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 20) -> None:  # 定义初始化函数。
        super().__init__()  # 调用父类初始化方法，完成 nn.Module 的基础设置。
        self.input_dim = input_dim  # 记录输入向量维度，MNIST 展平后为 28x28=784。
        self.hidden_dim = hidden_dim  # 记录隐藏层维度。
        self.latent_dim = latent_dim  # 记录潜变量维度。
        self.encoder = nn.Linear(input_dim, hidden_dim)  # 定义编码器第一层全连接层，将输入映射到隐藏特征。
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # 定义均值分支，用于预测潜变量分布的均值。
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)  # 定义对数方差分支，用于预测潜变量分布的对数方差。
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)  # 定义解码器第一层全连接层，将潜变量映射回隐藏特征。
        self.decoder = nn.Linear(hidden_dim, input_dim)  # 定义解码器输出层，将隐藏特征映射回原始图像维度。

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # 定义编码函数，返回均值和对数方差。
        x = F.relu(self.encoder(x))  # 输入先经过编码器线性层，再通过 ReLU 激活提取非线性特征。
        mu = self.fc_mu(x)  # 使用均值分支预测潜变量分布的均值向量。
        log_var = self.fc_log_var(x)  # 使用对数方差分支预测潜变量分布的对数方差向量。
        return mu, log_var  # 返回均值和对数方差。

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:  # 定义重参数化函数。
        std = torch.exp(0.5 * log_var)  # 先根据对数方差计算标准差，公式为 std=exp(0.5*log_var)。
        eps = torch.randn_like(std)  # 从标准正态分布中采样噪声张量，形状与标准差一致。
        z = mu + eps * std  # 使用重参数化公式 z = mu + eps * std 得到可导的潜变量采样结果。
        return z  # 返回采样得到的潜变量向量。

    def decode(self, z: torch.Tensor) -> torch.Tensor:  # 定义解码函数，将潜变量还原为图像向量。
        x = F.relu(self.fc_decode(z))  # 潜变量先经过线性层和 ReLU 激活恢复为隐藏特征。
        x = torch.sigmoid(self.decoder(x))  # 再通过输出层并使用 Sigmoid，将像素值限制在 0 到 1。
        return x  # 返回重建后的图像向量。

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 定义完整前向传播函数。
        mu, log_var = self.encode(x)  # 先对输入进行编码，得到潜变量分布的参数。
        z = self.reparameterize(mu, log_var)  # 再通过重参数化技巧从潜变量分布中采样。
        reconstructed = self.decode(z)  # 最后将采样到的潜变量送入解码器进行重建。
        return reconstructed, mu, log_var  # 返回重建结果、均值和对数方差。


def vae_loss(  # 定义 VAE 总损失函数。
    reconstructed: torch.Tensor,  # 模型重建得到的图像向量。
    original: torch.Tensor,  # 原始输入图像向量。
    mu: torch.Tensor,  # 编码器预测的均值向量。
    log_var: torch.Tensor,  # 编码器预测的对数方差向量。
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 返回总损失、重建损失和 KL 散度。
    recon_loss = F.binary_cross_entropy(reconstructed, original, reduction="sum")  # 计算二值交叉熵重建损失，并对批次内所有像素求和。
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # 计算潜变量分布与标准正态分布之间的 KL 散度。
    total_loss = recon_loss + kl_divergence  # 将重建损失与 KL 散度相加，得到 VAE 的总损失。
    return total_loss, recon_loss, kl_divergence  # 返回三种损失，便于训练时分别统计。


def build_dataloader(batch_size: int) -> DataLoader:  # 定义数据加载器构建函数。
    transform = transforms.Compose(  # 定义数据预处理流程。
        [  # 下面依次执行每一个预处理步骤。
            transforms.ToTensor(),  # 将 PIL 图像转换为张量，并把像素值归一化到 0 到 1。
        ]  # 预处理步骤列表结束。
    )  # 数据预处理组合结束。
    train_dataset = datasets.MNIST(  # 创建 MNIST 训练集对象。
        root="./data",  # 指定数据集下载与保存目录。
        train=True,  # 指定使用训练集。
        transform=transform,  # 指定对图像应用的预处理流程。
        download=True,  # 如果本地不存在数据集则自动下载。
    )  # 训练集对象创建结束。
    train_loader = DataLoader(  # 创建训练数据加载器。
        train_dataset,  # 指定要读取的数据集对象。
        batch_size=batch_size,  # 指定每个批次的样本数量。
        shuffle=True,  # 训练时打乱样本顺序，提高泛化能力。
        num_workers=2,  # 指定后台读取数据时使用的子进程数量。
        pin_memory=torch.cuda.is_available(),  # 如果使用 GPU，则启用锁页内存提高传输效率。
    )  # 数据加载器定义结束。
    return train_loader  # 返回构建好的训练数据加载器。


def setup_seed(seed: int) -> None:  # 定义随机种子设置函数，尽量保证实验可复现。
    torch.manual_seed(seed)  # 设置 CPU 上的随机种子。
    if torch.cuda.is_available():  # 如果当前环境中存在可用的 CUDA 设备。
        torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设备设置相同的随机种子。


def prepare_output_dirs() -> Tuple[str, str]:  # 定义输出目录准备函数。
    base_dir = "./outputs/mnist_vae"  # 定义 VAE 结果的根目录。
    image_dir = os.path.join(base_dir, "images")  # 定义生成图像保存目录。
    os.makedirs(base_dir, exist_ok=True)  # 如果根目录不存在，则自动创建。
    os.makedirs(image_dir, exist_ok=True)  # 如果图像目录不存在，则自动创建。
    return base_dir, image_dir  # 返回根目录和图像目录路径。


def train_one_epoch(  # 定义单轮训练函数。
    model: nn.Module,  # 当前待训练的 VAE 模型。
    loader: DataLoader,  # 训练数据加载器。
    optimizer: Adam,  # 用于更新参数的优化器。
    device: torch.device,  # 当前使用的计算设备。
) -> Tuple[float, float, float]:  # 返回平均总损失、平均重建损失和平均 KL 散度。
    model.train()  # 将模型切换到训练模式。
    total_loss_sum = 0.0  # 初始化总损失累计变量。
    recon_loss_sum = 0.0  # 初始化重建损失累计变量。
    kl_loss_sum = 0.0  # 初始化 KL 散度累计变量。
    total_samples = 0  # 初始化累计样本数量。

    for images, _ in loader:  # 遍历训练集中的每一个批次，这里标签不会参与训练。
        images = images.to(device, non_blocking=True)  # 将图像张量移动到目标设备上。
        images = torch.flatten(images, start_dim=1)  # 将每张 28x28 图像展平成长度为 784 的向量。
        optimizer.zero_grad()  # 在当前批次反向传播前清空历史梯度。
        reconstructed, mu, log_var = model(images)  # 执行前向传播，得到重建结果以及潜变量分布参数。
        total_loss, recon_loss, kl_divergence = vae_loss(reconstructed, images, mu, log_var)  # 计算总损失、重建损失和 KL 散度。
        total_loss.backward()  # 反向传播计算梯度。
        optimizer.step()  # 根据梯度更新模型参数。

        batch_size = images.size(0)  # 获取当前批次样本数量。
        total_loss_sum += total_loss.item()  # 累加当前批次的总损失。
        recon_loss_sum += recon_loss.item()  # 累加当前批次的重建损失。
        kl_loss_sum += kl_divergence.item()  # 累加当前批次的 KL 散度。
        total_samples += batch_size  # 统计已经处理过的样本总数。

    average_total_loss = total_loss_sum / total_samples  # 计算每个样本的平均总损失。
    average_recon_loss = recon_loss_sum / total_samples  # 计算每个样本的平均重建损失。
    average_kl_loss = kl_loss_sum / total_samples  # 计算每个样本的平均 KL 散度。
    return average_total_loss, average_recon_loss, average_kl_loss  # 返回这一轮训练的三个平均指标。


def save_reconstructions(  # 定义保存原图和重建图对比结果的函数。
    model: nn.Module,  # 当前训练好的 VAE 模型。
    loader: DataLoader,  # 用于提取一批训练样本的数据加载器。
    device: torch.device,  # 当前使用的计算设备。
    save_path: str,  # 保存图像的目标路径。
    max_images: int = 8,  # 最多保存多少张原图及其重建图。
) -> None:  # 函数不返回结果，只负责保存图片。
    model.eval()  # 将模型切换到评估模式。
    images, _ = next(iter(loader))  # 从数据加载器中取出一个批次的样本图像和标签。
    images = images.to(device)  # 将图像移动到目标设备。
    flat_images = torch.flatten(images, start_dim=1)  # 将原始图像展平，以适配 VAE 输入格式。

    with torch.no_grad():  # 在不计算梯度的上下文中执行推理。
        reconstructed, _, _ = model(flat_images)  # 使用模型对输入图像进行重建。

    original_images = images[:max_images].cpu()  # 截取前若干张原图并移动回 CPU。
    reconstructed_images = reconstructed[:max_images].view(-1, 1, 28, 28).cpu()  # 将前若干张重建结果还原为图像形状并移动回 CPU。
    comparison = torch.cat([original_images, reconstructed_images], dim=0)  # 将原图和重建图沿批次维拼接起来。
    save_image(comparison, save_path, nrow=max_images)  # 将拼接后的图片网格保存到指定路径。


def save_random_samples(  # 定义从标准正态分布随机采样并生成新图像的函数。
    model: VAEModel,  # 当前训练好的 VAE 模型。
    device: torch.device,  # 当前使用的计算设备。
    save_path: str,  # 保存图像的目标路径。
    sample_count: int = 16,  # 需要随机生成多少张新图像。
) -> None:  # 函数不返回结果，只负责保存图片。
    model.eval()  # 将模型切换到评估模式。
    with torch.no_grad():  # 在不计算梯度的上下文中执行随机采样生成。
        z = torch.randn(sample_count, model.latent_dim, device=device)  # 从标准正态分布中随机采样潜变量向量。
        generated = model.decode(z)  # 将随机潜变量送入解码器，生成新的手写数字图像。
    generated = generated.view(-1, 1, 28, 28).cpu()  # 将生成结果还原为图像形状并移动回 CPU。
    save_image(generated, save_path, nrow=4)  # 将生成的样本图像保存为 4x4 网格。


def main() -> None:  # 定义脚本主函数，负责串联完整的 VAE 训练与生成流程。
    batch_size = 128  # 设置每个批次的样本数量。
    learning_rate = 1e-3  # 设置优化器的初始学习率。
    num_epochs = 10  # 设置总训练轮数。
    hidden_dim = 400  # 设置编码器和解码器中间隐藏层维度。
    latent_dim = 20  # 设置潜变量空间维度。
    seed = 42  # 设置随机种子，便于实验复现。
    step_size = 5  # 设置学习率调度器每隔多少轮衰减一次。
    gamma = 0.5  # 设置每次衰减时学习率乘上的系数。

    setup_seed(seed)  # 调用随机种子设置函数。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择当前训练设备为 GPU 或 CPU。
    print(f"当前使用设备: {device}")  # 打印当前训练设备信息。

    train_loader = build_dataloader(batch_size=batch_size)  # 构建 MNIST 训练数据加载器。
    model = VAEModel(input_dim=784, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)  # 创建 VAE 模型并移动到目标设备。
    optimizer = Adam(model.parameters(), lr=learning_rate)  # 创建 Adam 优化器。
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)  # 创建按步长衰减学习率的调度器。
    base_dir, image_dir = prepare_output_dirs()  # 创建结果保存目录。
    best_loss = float("inf")  # 用正无穷初始化最佳损失，便于后续比较。

    for epoch in range(1, num_epochs + 1):  # 从第 1 轮循环训练到最后一轮。
        average_total_loss, average_recon_loss, average_kl_loss = train_one_epoch(  # 执行一整轮训练并返回平均损失指标。
            model=model,  # 传入当前待训练的模型。
            loader=train_loader,  # 传入训练数据加载器。
            optimizer=optimizer,  # 传入优化器。
            device=device,  # 传入当前设备。
        )  # 单轮训练调用结束。
        scheduler.step()  # 在当前轮训练结束后更新学习率。
        current_lr = optimizer.param_groups[0]["lr"]  # 读取当前优化器实际使用的学习率。
        print(  # 打印当前轮训练的日志信息。
            f"Epoch [{epoch}/{num_epochs}] "  # 输出当前轮数与总轮数。
            f"total_loss={average_total_loss:.4f} "  # 输出平均总损失。
            f"recon_loss={average_recon_loss:.4f} "  # 输出平均重建损失。
            f"kl_loss={average_kl_loss:.4f} "  # 输出平均 KL 散度。
            f"lr={current_lr:.6f}"  # 输出当前学习率。
        )  # 日志打印结束。

        if average_total_loss < best_loss:  # 如果当前轮总损失优于历史最佳结果。
            best_loss = average_total_loss  # 更新最佳损失值。
            torch.save(model.state_dict(), os.path.join(base_dir, "best_vae.pth"))  # 保存当前最佳模型权重。

    torch.save(model.state_dict(), os.path.join(base_dir, "last_vae.pth"))  # 训练结束后保存最后一轮模型权重。
    save_reconstructions(  # 保存原图与重建图的对比结果。
        model=model,  # 传入训练好的模型。
        loader=train_loader,  # 传入训练数据加载器。
        device=device,  # 传入当前设备。
        save_path=os.path.join(image_dir, "reconstruction.png"),  # 指定重建结果图的保存路径。
        max_images=8,  # 指定保存 8 组原图与重建图。
    )  # 原图与重建图保存结束。
    save_random_samples(  # 保存从标准正态分布中随机生成的新样本图像。
        model=model,  # 传入训练好的模型。
        device=device,  # 传入当前设备。
        save_path=os.path.join(image_dir, "generated.png"),  # 指定随机生成样本图的保存路径。
        sample_count=16,  # 指定生成 16 张新图像。
    )  # 随机采样图像保存结束。
    print("训练完成，模型权重、重建图片和随机生成图片已保存到 outputs/mnist_vae 目录。")  # 打印训练结束提示信息。


if __name__ == "__main__":  # 当脚本被直接运行时执行下面的主流程。
    main()  # 调用主函数启动整个 VAE 训练与生成过程。

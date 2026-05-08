import os  # 导入操作系统模块，用于创建输出目录和拼接文件路径。
from typing import Tuple  # 导入类型标注工具，用于标记函数返回值类型。

import torch  # 导入 PyTorch 主库，用于张量运算、自动求导和神经网络训练。
import torch.nn as nn  # 导入神经网络模块，并简写为 nn，便于定义模型结构。
from torch import optim  # 导入优化器模块，后续用来创建 Adam 优化器。
from torch.utils.data import DataLoader  # 导入数据加载器，用于按批次读取 MNIST 数据集。
from torchvision import datasets, transforms  # 导入 torchvision 中的数据集与图像预处理工具。
from torchvision.utils import save_image  # 导入图像保存函数，用于保存生成结果图。
from tqdm import tqdm  # 导入 tqdm 进度条工具，用于实时显示训练进度。


# GAN 的核心思想：
# 1. 生成器 Generator 学习把随机噪声映射成“看起来像真的”图像。
# 2. 判别器 Discriminator 学习区分输入图像是真实样本还是生成样本。
# 3. 二者通过对抗训练不断博弈：
#    - 判别器希望把真样本判成 1，把假样本判成 0。
#    - 生成器希望让判别器把假样本也判成 1。
# 4. 当训练较充分时，生成器会逐步学会生成越来越逼真的手写数字。


class Generator(nn.Module):  # 定义生成器类，负责把随机噪声变成手写数字图像。
    def __init__(self, noise_dim: int = 128, image_dim: int = 784) -> None:  # 定义初始化函数，noise_dim 是噪声维度，image_dim 是输出图像展平后的维度。
        super().__init__()  # 调用父类初始化函数，完成 nn.Module 的基础设置。
        self.noise_dim = noise_dim  # 保存噪声向量维度，便于后续使用或查看模型配置。
        self.image_dim = image_dim  # 保存图像维度，MNIST 展平后为 28x28=784。
        self.main = nn.Sequential(  # 使用顺序容器搭建多层感知机生成器。
            nn.Linear(noise_dim, 256),  # 第一层全连接：把低维随机噪声映射到 256 维隐藏特征。
            nn.ReLU(inplace=True),  # 使用 ReLU 激活函数引入非线性，使模型具备更强表达能力。
            nn.Linear(256, 512),  # 第二层全连接：继续扩大特征表示能力。
            nn.ReLU(inplace=True),  # 再次使用 ReLU，对隐藏特征做非线性变换。
            nn.Linear(512, 1024),  # 第三层全连接：进一步提升特征维度，为生成图像做准备。
            nn.ReLU(inplace=True),  # 使用 ReLU，让网络能够学习更复杂的映射关系。
            nn.Linear(1024, image_dim),  # 输出层：把高维隐藏特征映射到 784 维像素空间。
            nn.Tanh(),  # 使用 Tanh 把输出压到 [-1, 1]，与归一化后的 MNIST 数据范围对应。
        )  # 生成器网络结构定义结束。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播函数，输入是随机噪声张量。
        return self.main(x)  # 将噪声送入生成器网络，输出一张展平后的假图像。


class Discriminator(nn.Module):  # 定义判别器类，负责判断输入图像是真图还是假图。
    def __init__(self, image_dim: int = 784) -> None:  # 定义初始化函数，image_dim 是输入图像展平后的维度。
        super().__init__()  # 调用父类初始化函数，完成 nn.Module 的基础设置。
        self.image_dim = image_dim  # 保存输入图像维度配置。
        self.main = nn.Sequential(  # 使用顺序容器搭建多层感知机判别器。
            nn.Linear(image_dim, 1024),  # 第一层全连接：把输入图像映射到 1024 维隐藏空间。
            nn.LeakyReLU(0.2, inplace=True),  # 使用 LeakyReLU，避免普通 ReLU 在负半轴梯度完全为 0。
            nn.Linear(1024, 512),  # 第二层全连接：继续压缩并提取判别相关特征。
            nn.LeakyReLU(0.2, inplace=True),  # 再次使用 LeakyReLU，提高训练稳定性。
            nn.Linear(512, 256),  # 第三层全连接：进一步提取高层语义信息。
            nn.LeakyReLU(0.2, inplace=True),  # 使用 LeakyReLU，保持负值区域也有小梯度。
            nn.Linear(256, 1),  # 输出层：输出一个标量，表示“是真图”的概率分数。
            nn.Sigmoid(),  # 使用 Sigmoid 把输出压到 (0,1)，便于与 BCELoss 搭配。
        )  # 判别器网络结构定义结束。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播函数，输入是一批图像向量。
        return self.main(x)  # 返回判别器对每张图像的真伪概率预测。


def setup_seed(seed: int) -> None:  # 定义随机种子设置函数，尽量提升实验可复现性。
    torch.manual_seed(seed)  # 设置 CPU 端随机种子，影响张量随机初始化与随机采样。
    if torch.cuda.is_available():  # 如果当前环境中有可用 GPU。
        torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置相同随机种子，减少多次运行差异。


def build_dataloader(batch_size: int) -> DataLoader:  # 定义数据加载器构建函数，用于读取 MNIST 训练集。
    transform = transforms.Compose(  # 使用 Compose 串联多个图像预处理步骤。
        [  # 下面依次定义每一步预处理操作。
            transforms.ToTensor(),  # 把 PIL 图像转成张量，并把像素值从 [0,255] 缩放到 [0,1]。
            transforms.Normalize((0.5,), (0.5,)),  # 继续把像素从 [0,1] 线性映射到 [-1,1]，以匹配生成器的 Tanh 输出范围。
        ]  # 预处理步骤列表定义结束。
    )  # 预处理组合定义结束。
    train_dataset = datasets.MNIST(  # 创建 MNIST 训练集对象。
        root="./data",  # 指定数据集下载和缓存的本地目录。
        train=True,  # 指定读取训练集，而不是测试集。
        transform=transform,  # 指定对每张图像应用上面定义的预处理流程。
        download=True,  # 如果本地没有数据集，则自动下载。
    )  # 数据集对象创建结束。
    train_loader = DataLoader(  # 创建数据加载器，用于分批读取样本。
        train_dataset,  # 指定要读取的数据集对象。
        batch_size=batch_size,  # 指定每个 batch 的样本数量。
        shuffle=True,  # 训练时随机打乱样本顺序，有助于提升泛化和训练稳定性。
        num_workers=2,  # 使用 2 个子进程并行加载数据，提高吞吐效率。
        pin_memory=torch.cuda.is_available(),  # 若使用 GPU，则开启锁页内存，加快主机到显卡的数据拷贝。
    )  # 数据加载器创建结束。
    return train_loader  # 返回构建好的训练数据加载器。


def prepare_output_dirs() -> Tuple[str, str]:  # 定义输出目录准备函数，返回根目录和图像目录。
    base_dir = "./outputs/mnist_gan"  # 设置 GAN 训练结果的根目录。
    image_dir = os.path.join(base_dir, "images")  # 在根目录下创建一个 images 子目录用于保存生成图片。
    os.makedirs(base_dir, exist_ok=True)  # 若根目录不存在则创建；若已存在则不报错。
    os.makedirs(image_dir, exist_ok=True)  # 若图片目录不存在则创建；若已存在则不报错。
    return base_dir, image_dir  # 返回两个目录路径，便于主函数统一使用。


def save_generated_samples(  # 定义保存生成样本的函数。
    generator: Generator,  # 当前训练好的生成器模型。
    fixed_noise: torch.Tensor,  # 固定随机噪声，用于在不同 epoch 之间做可比的可视化。
    save_path: str,  # 生成图片的保存路径。
) -> None:  # 该函数只负责保存图片，不返回任何值。
    generator.eval()  # 将生成器切换到评估模式，虽然这里没有 BN/Dropout，但这是良好习惯。
    with torch.no_grad():  # 在不跟踪梯度的上下文中推理，可减少显存开销并加快速度。
        fake_images = generator(fixed_noise).view(-1, 1, 28, 28).cpu()  # 用固定噪声生成图像，并还原成 [batch,1,28,28] 形状，再移动到 CPU。
    save_image(fake_images, save_path, nrow=8, normalize=True, value_range=(-1, 1))  # 将生成结果保存为网格图，并把 [-1,1] 自动映射到可显示范围。


def train_one_epoch(  # 定义单轮训练函数，负责遍历一个 epoch 内的所有 batch。
    generator: Generator,  # 当前待训练的生成器。
    discriminator: Discriminator,  # 当前待训练的判别器。
    loader: DataLoader,  # 训练数据加载器。
    criterion: nn.Module,  # 损失函数，这里会传入 BCELoss。
    g_optimizer: optim.Optimizer,  # 生成器优化器。
    d_optimizer: optim.Optimizer,  # 判别器优化器。
    device: torch.device,  # 当前训练使用的设备，如 CPU 或 CUDA。
    noise_dim: int,  # 噪声向量维度。
    epoch: int,  # 当前是第几轮训练。
    num_epochs: int,  # 总训练轮数，用于进度条显示。
) -> Tuple[float, float, float, float]:  # 返回判别器损失、生成器损失、真样本得分和假样本得分的平均值。
    generator.train()  # 把生成器切换到训练模式。
    discriminator.train()  # 把判别器切换到训练模式。

    d_loss_sum = 0.0  # 初始化判别器损失累计值，用于统计整轮平均损失。
    g_loss_sum = 0.0  # 初始化生成器损失累计值，用于统计整轮平均损失。
    real_score_sum = 0.0  # 初始化真样本判别得分累计值，用于观察 D(x)。
    fake_score_sum = 0.0  # 初始化假样本判别得分累计值，用于观察 D(G(z))。
    total_batches = 0  # 初始化 batch 计数器，用于后续求平均。

    progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)  # 创建当前 epoch 的进度条。
    for images, _ in progress_bar:  # 逐 batch 读取图像和标签；GAN 训练里标签不会直接参与，所以用下划线忽略。
        images = images.to(device, non_blocking=True)  # 把当前批次图像移动到目标设备上。
        real_images = images.view(images.size(0), -1)  # 把每张 28x28 的图像展平成 784 维向量，供全连接网络输入。
        batch_size = real_images.size(0)  # 记录当前 batch 的样本数量。

        real_labels = torch.ones(batch_size, 1, device=device)  # 为真图构造标签 1，表示“真实样本”。
        fake_labels = torch.zeros(batch_size, 1, device=device)  # 为假图构造标签 0，表示“生成样本”。

        # 第一步：训练判别器。
        # 判别器的目标是：
        # - 看到真图时输出接近 1；
        # - 看到假图时输出接近 0。
        # 因此判别器损失由“真图损失 + 假图损失”两部分组成。
        noise = torch.randn(batch_size, noise_dim, device=device)  # 从标准正态分布采样一批随机噪声，作为生成器输入。
        fake_images = generator(noise)  # 用当前生成器根据噪声生成一批假图像。

        d_optimizer.zero_grad()  # 在当前 batch 更新前清空判别器上一个 batch 的历史梯度。
        real_outputs = discriminator(real_images)  # 把真图送入判别器，得到其“为真”的概率输出。
        d_loss_real = criterion(real_outputs, real_labels)  # 计算真图损失，希望判别器把真图判断为 1。

        fake_outputs = discriminator(fake_images.detach())  # 把假图送入判别器；detach 用来切断生成器梯度，避免训练 D 时反向更新 G。
        d_loss_fake = criterion(fake_outputs, fake_labels)  # 计算假图损失，希望判别器把假图判断为 0。

        d_loss = d_loss_real + d_loss_fake  # 将真图损失和假图损失相加，得到判别器总损失。
        d_loss.backward()  # 对判别器损失做反向传播，计算判别器参数梯度。
        d_optimizer.step()  # 使用 Adam 根据梯度更新判别器参数。

        # 第二步：训练生成器。
        # 生成器并不直接看真实图像，而是通过“骗过判别器”来学习。
        # 因此生成器的目标是：让判别器把假图也预测成 1。
        noise = torch.randn(batch_size, noise_dim, device=device)  # 再采样一批新的随机噪声，供生成器训练使用。
        generated_images = generator(noise)  # 根据新的噪声生成一批假图像。

        g_optimizer.zero_grad()  # 在当前 batch 更新前清空生成器历史梯度。
        generator_outputs = discriminator(generated_images)  # 把生成器输出送入判别器，观察判别器给出的真假判断。
        g_loss = criterion(generator_outputs, real_labels)  # 用真标签 1 计算生成器损失，目标是让判别器误以为这些假图都是真的。
        g_loss.backward()  # 对生成器损失反向传播，计算生成器参数梯度。
        g_optimizer.step()  # 使用 Adam 更新生成器参数，使其更会“伪造”图像。

        d_loss_sum += d_loss.item()  # 累加当前 batch 的判别器损失。
        g_loss_sum += g_loss.item()  # 累加当前 batch 的生成器损失。
        real_score_sum += real_outputs.mean().item()  # 累加真图平均得分，便于后续计算整轮平均 D(x)。
        fake_score_sum += fake_outputs.mean().item()  # 累加假图平均得分，便于后续计算整轮平均 D(G(z))。
        total_batches += 1  # 记录已经处理的 batch 数量。

        progress_bar.set_postfix(  # 在进度条右侧实时显示关键训练指标。
            d_loss=f"{d_loss.item():.4f}",  # 显示当前 batch 的判别器损失。
            g_loss=f"{g_loss.item():.4f}",  # 显示当前 batch 的生成器损失。
            dx=f"{real_outputs.mean().item():.2f}",  # 显示当前 batch 中判别器对真图的平均打分 D(x)。
            dgz=f"{fake_outputs.mean().item():.2f}",  # 显示当前 batch 中判别器对假图的平均打分 D(G(z))。
        )  # 本次进度条指标更新结束。

    average_d_loss = d_loss_sum / total_batches  # 计算整轮训练的平均判别器损失。
    average_g_loss = g_loss_sum / total_batches  # 计算整轮训练的平均生成器损失。
    average_real_score = real_score_sum / total_batches  # 计算整轮训练中真图平均得分 D(x)。
    average_fake_score = fake_score_sum / total_batches  # 计算整轮训练中假图平均得分 D(G(z))。
    return average_d_loss, average_g_loss, average_real_score, average_fake_score  # 返回整轮训练统计指标。


def main() -> None:  # 定义主函数，用于串联完整的 GAN 训练流程。
    batch_size = 64  # 设置每个 batch 的样本数；较小 batch 更灵活，也常用于 GAN 训练。
    learning_rate = 2e-4  # 设置 Adam 的学习率，这是 DCGAN/GAN 中很常见的起始值。
    num_epochs = 50  # 设置训练轮数，轮数越多通常生成效果越好，但训练时间也更长。
    noise_dim = 128  # 设置输入给生成器的随机噪声维度，表示潜在空间大小。
    seed = 42  # 设置随机种子，尽量保证实验结果可复现。

    setup_seed(seed)  # 调用随机种子设置函数。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动检测当前使用 GPU 还是 CPU。
    print(f"当前使用设备: {device}")  # 打印设备信息，便于确认训练运行环境。

    train_loader = build_dataloader(batch_size=batch_size)  # 构造 MNIST 训练集的数据加载器。
    generator = Generator(noise_dim=noise_dim).to(device)  # 实例化生成器，并移动到训练设备。
    discriminator = Discriminator().to(device)  # 实例化判别器，并移动到训练设备。

    criterion = nn.BCELoss()  # 定义二分类交叉熵损失函数，用于真假二分类任务。
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))  # 为生成器创建 Adam 优化器；betas 是 GAN 中常见设置。
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))  # 为判别器创建 Adam 优化器；使用相同学习率与动量参数。

    base_dir, image_dir = prepare_output_dirs()  # 创建保存模型和图像的输出目录。
    fixed_noise = torch.randn(64, noise_dim, device=device)  # 固定一批随机噪声，用于每轮结束后观察生成器是否逐渐学到更清晰的数字。

    for epoch in range(1, num_epochs + 1):  # 从第 1 轮循环训练到第 num_epochs 轮。
        average_d_loss, average_g_loss, average_real_score, average_fake_score = train_one_epoch(  # 执行一整轮对抗训练，并接收统计结果。
            generator=generator,  # 传入当前生成器。
            discriminator=discriminator,  # 传入当前判别器。
            loader=train_loader,  # 传入训练数据加载器。
            criterion=criterion,  # 传入 BCE 损失函数。
            g_optimizer=g_optimizer,  # 传入生成器优化器。
            d_optimizer=d_optimizer,  # 传入判别器优化器。
            device=device,  # 传入当前计算设备。
            noise_dim=noise_dim,  # 传入噪声维度配置。
            epoch=epoch,  # 传入当前 epoch 编号。
            num_epochs=num_epochs,  # 传入总 epoch 数，用于进度条展示。
        )  # 当前轮训练调用结束。

        save_generated_samples(  # 每轮训练结束后保存一次固定噪声下的生成结果。
            generator=generator,  # 传入当前生成器。
            fixed_noise=fixed_noise,  # 使用固定噪声，便于横向比较每一轮生成效果。
            save_path=os.path.join(image_dir, f"epoch_{epoch:03d}.png"),  # 以 epoch 编号命名图片文件，方便按时间顺序查看。
        )  # 当前轮图片保存结束。

        print(  # 打印当前 epoch 的训练日志。
            f"Epoch [{epoch}/{num_epochs}] "  # 输出当前轮数和总轮数。
            f"d_loss={average_d_loss:.4f} "  # 输出判别器平均损失。
            f"g_loss={average_g_loss:.4f} "  # 输出生成器平均损失。
            f"D(x)={average_real_score:.2f} "  # 输出判别器对真图的平均置信度。
            f"D(G(z))={average_fake_score:.2f}"  # 输出判别器对假图的平均置信度。
        )  # 当前轮日志打印结束。

    torch.save(generator.state_dict(), os.path.join(base_dir, "generator.pth"))  # 保存训练完成后的生成器权重。
    torch.save(discriminator.state_dict(), os.path.join(base_dir, "discriminator.pth"))  # 保存训练完成后的判别器权重。

    save_generated_samples(  # 训练结束后再额外保存一张最终生成结果图。
        generator=generator,  # 传入训练完成的生成器。
        fixed_noise=fixed_noise,  # 仍使用固定噪声，便于与中间轮次结果比较。
        save_path=os.path.join(image_dir, "generated_final.png"),  # 指定最终图片保存路径。
    )  # 最终生成图片保存结束。
    print("训练完成，模型权重和生成图片已保存到 outputs/mnist_gan 目录。")  # 打印训练结束提示信息。


if __name__ == "__main__":  # 当该脚本被直接执行时，才会进入下面的主流程。
    main()  # 调用主函数，启动完整的 GAN 训练与结果保存过程。

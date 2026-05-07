# MNIST VAE 流程图逐框详解

本文档对应 `mnist_vae_flowchart.md` 与 `mnist_vae.py`。下面按章节、按流程图中的每一个“框”展开文字说明，便于对照图与代码阅读。

---

## 第 1 节：`mnist_vae.py` 主流程图

### 框 1：`mnist_vae.py` 主流程

这是脚本的顶层概念框，表示整个文件承担的唯一职责：在 MNIST 上训练一个全连接变分自编码器（VAE），保存权重，并输出重建图与随机生成图。当你在终端执行 `python3 mnist_vae.py` 时，Python 会加载该模块，并在 `if __name__ == "__main__":` 处调用 `main()`，从而进入这里所描述的整条流水线。

### 框 2：`main()` 主入口函数

`main()` 是训练与收尾逻辑的总调度函数。它不负责定义网络细节，而是串联：随机种子、设备选择、数据加载器、`VAEModel`、`Adam` 优化器、`StepLR` 调度器、输出目录、多轮 `train_one_epoch`、最佳模型判断、最终保存与可视化。把所有可变超参（如 `batch_size`、`num_epochs`）集中写在 `main()` 开头，便于你以后改实验设定而不用翻遍整个文件。

### 框 3：场景 1——初始化与训练准备

这一整块对应 `main()` 前半段，尚未进入 epoch 循环。

- **`batch_size = 128`**：每个训练步从 DataLoader 取出的样本数。较大 batch 梯度更稳但显存占用更高；128 是常见折中。
- **`learning_rate = 1e-3`**：`Adam` 的初始步长。VAE 同时优化重建项和 KL 项，学习率过大可能震荡，过小则收敛慢。
- **`num_epochs = 10`**：完整遍历训练集的次数。轮数少时重建可能偏模糊；可增加轮数或配合调度器观察损失曲线。
- **`hidden_dim = 400`**：编码器第一层与解码器中间层的宽度。决定从 784 维像素到潜空间之间的“压缩表达能力”。
- **`latent_dim = 20`**：潜变量 `z` 的维度。维度过小会限制生成多样性；过大则可能加重 KL 与重建的权衡难度。
- **`seed = 42`**：固定随机种子，使 `torch`、`cuda` 上的随机操作更可复现；不同 seed 会给出略不同的初始化与打乱顺序。
- **`step_size = 5`**：**`StepLR`** 每隔多少个 epoch 把学习率乘以 `gamma`。与 `num_epochs=10` 搭配时，大约在第 6 轮起学习率会衰减。
- **`gamma = 0.5`**：学习率衰减因子；例如从 `1e-3` 变为 `5e-4`。
- **`setup_seed(seed)`**：内部调用 `torch.manual_seed` 与（若有 GPU）`torch.cuda.manual_seed_all`，减少“同代码不同机器结果差很多”的情况。
- **`device = cuda / cpu`**：根据 `torch.cuda.is_available()` 选择计算设备；后续所有张量与模型都应 `.to(device)`，避免 CPU/GPU 混用报错。
- **`train_loader = build_dataloader(batch_size)`**：构造 MNIST 训练集与 `DataLoader`，细节见下文 `build_dataloader` 相关说明。

### 框 4：场景 2——构建 VAE 模型与训练组件

- **`model = VAEModel(784, 400, 20)`**：实例化网络，输入维固定为 MNIST 展平后的 784。`to(device)` 在下一行代码中完成（流程图里合并表述）。
- **`optimizer = Adam(..., lr=1e-3)`**：自适应学习率优化器，更新所有 `nn.Parameter`。VAE 的损失非凸，Adam 是常用默认选择。
- **`scheduler = StepLR(..., step_size=5, gamma=0.5)`**：每个 epoch 结束后 `scheduler.step()`，按阶梯降低学习率，有助于训练后期细化。
- **`prepare_output_dirs()`**：创建 `./outputs/mnist_vae` 与 `./outputs/mnist_vae/images`，避免保存权重或图片时因目录不存在而失败。
- **`best_loss = inf`**：用“史上最优 epoch 平均总损失”做比较；首轮几乎一定会刷新并保存 `best_vae.pth`。

### 框 5：场景 3——进入 epoch 训练循环

- **`for epoch in range(1, num_epochs + 1)`**：外层epoch；内向 `DataLoader` 迭代 batch。
- **`train_one_epoch(...)`**：返回本轮在全数据集上的平均 `total_loss`、`recon_loss`、`kl_loss`（实现上按样本数累加再平均，见第 5 节详述）。
- **`scheduler.step()`**：在**本轮所有 batch 跑完后**更新学习率；注意顺序不要与“每个 batch 调一次”混淆。
- **`print(...)`**：打印监控量，便于人工判断是否过拟合、KL 是否过大等。
- **`if average_total_loss < best_loss: 保存 best_vae.pth`**：保留验证集意义上“历史最好”的一版参数（此处仅用训练损失近似；若需严谨可加验证集）。

### 框 6：场景 4——训练结束后的结果保存

- **`last_vae.pth`**：最后一轮参数，便于复现“训练结束瞬间”的模型。
- **`save_reconstructions`**：取一批真实图，前向得到重建，拼成网格保存为 `reconstruction.png`，直观对比重建质量。
- **`save_random_samples`**：从 **N(0,1)** 采样 `z`，仅走解码器得到新图 `generated.png`，体现 VAE 生成能力（无需输入图像）。
- **输出训练完成提示**：脚本级用户体验，确认输出路径。

### 框 7：`结束`

程序正常退出；若在集群或脚本链中运行，可用退出码 0 表示成功。

---

## 第 2 节：VAE 模型构建流程图

### 框 1：`VAEModel` 模型构建流程（总标题）

表示 `__init__` 中如何**拼装**子模块，而非一次前向计算。

### 框 2：`VAEModel(input_dim=784, hidden_dim=400, latent_dim=20)`

构造函数入口三件套：`input_dim` 与 MNIST 展平维度一致；`hidden_dim` 控制瓶颈前宽度；`latent_dim` 控制潜空间维数。三者一起决定了参数量与拟合能力。

### 框 3：编码器主干——`encoder: Linear(784,400) + ReLU`

将每个样本的 784 维向量线性映射到 400 维，再经 ReLU 注入非线性。没有这一层，后面的 `mu`/`log_var` 只是输入的线性函数，表达能力不足。

### 框 4：分布参数分支——`fc_mu` 与 `fc_log_var`

二者共享“编码器主干”输出的 400 维特征，但参数不共享：

- **`fc_mu`**：`Linear(400, 20)`，输出 **q(z|x) 的均值 μ**。
- **`fc_log_var`**：`Linear(400, 20)`，输出 **log σ²**（对数方差），而非直接输出方差，避免 σ² 必须为正且优化数值更稳。

**在图中写作“方差 σ²”时，代码里对应的是 `log_var`，两者通过指数与开方互相换算。**

### 框 5：解码器结构——`fc_decode` 与 `decoder`

- **`fc_decode`**：`Linear(20,400) + ReLU`，把潜向量扩回 400 维隐藏表示。
- **`decoder`**：`Linear(400,784) + Sigmoid`，输出与像素同长度的向量，且像素被压到 (0,1)，与 `ToTensor()` 后的 MNIST 标签尺度一致，便于 **BCE** 重建损失。

### 框 6：编码结果输出两组参数——`mu` 与 `log_var`

强调 VAE 与 AE 的本质差别：编码器输出的是**分布参数**，而不是一个确定的码向量。后续 `reparameterize` 用这两组数定义 **N(μ, σ²)** 并完成可导采样。

---

## 第 3 节：VAE 前向传播流程图

### 框 1：`VAE 前向传播 forward()`（总标题）

对应 `VAEModel.forward`：串联 encode → reparameterize → decode。

### 框 2：输入图像 `images`，shape `[batch, 1, 28, 28]`

来自 MNIST 与 `ToTensor()`：单通道、空间 28×28。此时仍是“图像排版”的张量，尚未展平。

### 框 3：展平 `flatten`，shape `[batch, 784]`

全连接 VAE 的标准做法：把空间维展成一条长向量。`start_dim=1` 表示保留 batch 维，从通道与宽高维展平。

### 框 4：`encode(x)`——`Linear(784,400) -> ReLU`

见第 2 节编码器主干；返回进入双头前的**共享隐表示**。

### 框 5：`fc_mu(x)` → 得到 `mu`

每个样本 20 维均值向量，batch 整体为 `[batch, 20]`。

### 框 6：`fc_log_var(x)` → 得到 `log_var`

每个样本 20 维对数方差；与 `mu` 同形，逐维定义对角高斯。

### 框 7：`reparameterize(mu, log_var)`

- **`std = exp(0.5 * log_var)`**：因 `log_var = log(σ²)`，故 `σ = exp(0.5 log_var)`。
- **`eps = randn_like(std)`**：从标准正态采样，与 `std` 同设备同形状。
- **`z = mu + eps * std`**：**重参数化技巧**：随机性来自 `eps`，对 `mu` 与 `log_var` 仍可通过链式法则反传。

若省略此步、直接 `z ~ N(mu, std)` 用非重参数采样，则无法对采样算子求导，难以用常规反向传播端到端训练。

### 框 8：`decode(z)`

`z` 经 `fc_decode`（ReLU）再经 `decoder`（Sigmoid），得到与输入同长的重建向量 `[batch, 784]`。

### 框 9：`reconstructed`，shape `[batch, 784]`

前向最终图像版输出，与展平后的 `original` 对齐，供 `vae_loss` 中 BCE 使用。若要可视化，需 `view(-1,1,28,28)` 还原空间维。

---

## 第 4 节：`vae_loss()` 损失函数流程图

### 框 1：`vae_loss()` 损失计算（总标题）

实现变分下界中的典型两项：**重建项 + KL(q||p)**，本实现用 BCE 作重建、用闭式 KL（先验为标准正态）作正则。

### 框 2：`reconstructed`——模型重建结果

解码器输出，已与 `original` 同为 `[batch, 784]`，且在 Sigmoid 后处于 (0,1)。

### 框 3：`original`——原始输入图像

与训练循环里 `flatten` 后的 `images` 一致；**标签未使用**，VAE 是无监督重建+隐空间正则。

### 框 4：`mu` / `log_var`——编码器输出的分布参数

仅参与 **KL 项**：把 **q(z|x)=N(μ, diag(σ²))** 拉向 **p(z)=N(0,I)**，使潜空间规整，便于从 N(0,1) 采样生成。

### 框 5：重建损失 `recon_loss`

- **`binary_cross_entropy(reconstructed, original, reduction="sum")`**：对 batch 内所有元素求和（而非 mean）。训练循环里再按样本数平均，需注意**“sum 与后续平均方式”**与 `train_one_epoch` 的统计一致；当前实现是**每 batch 的 loss 已是 sum over 全像素**，累加 `total_loss.item()` 后除以 `total_samples`（样本数），等价于“先对像素求和再对样本平均”，与常见实现略有不同但自洽——你只要明白监控到的标量是“按样本平均的总目标”的近似统计即可。

直观上：重建项越小，**单张图**越接近原图。

### 框 6：KL 散度 `kl_divergence`

- **公式形态**：`-0.5 * sum(1 + log_var - mu^2 - exp(log_var))` 是多维对角高斯相对标准正态的 KL 闭式解按维求和再取负号整理后的常见写法。
- 作用：惩罚过大的 \|μ\| 与过大的方差偏离，使 **大部分 mass** 落在先验附近，从而让 **从 N(0,1) 采样** 时解码器仍能得到合理图像。

### 框 7：`total_loss = recon_loss + kl_loss`

联合目标。KL 权重大时会偏“规整但糊”的潜空间；重建权重大（相对意义上的“KL 弱”）可能更锐利但采样质量差。**本脚本未引入 β-VAE 系数**，即 β=1。

### 补充：`vae_loss` 为什么要这样写（设计思路与每部分特点）

下面与上图各框对应，从**变分目标**出发说明：损失在优化什么、每一项的目的与特点，以及为何把三个标量都返回。

#### 1. 整体：在优化什么（ELBO）

VAE 的训练目标来自变分推断中的 **ELBO（证据下界）**。最大化 ELBO，相当于同时要求：

- **解码器**在从潜变量 \(z\) 还原数据时，让数据似然（或其代理）尽量大——在实现里通常体现为**重建误差要尽量小**；
- **编码器**给出的近似后验 \(q_\phi(z \mid x)\) 不要离先验 \(p(z)\) 太远——在实现里体现为 **\(\mathrm{KL}\big(q_\phi(z \mid x)\,\|\,p(z)\big)\)** 要尽量小。

实务上若写成**最小化**的损失，常见形式为：

\[
\text{loss} \approx -\text{ELBO} = \underbrace{\text{重建项（负对数似然的近似）}}_{\text{本代码中的 BCE}} + \underbrace{\mathrm{KL}\big(q_\phi(z \mid x)\,\|\,p(z)\big)}_{\text{本代码中的闭式 KL}}
\]

因此 `total_loss = recon_loss + kl_divergence` 与「最小化负 ELBO」的符号约定一致：两项**相加**后反传，就是在联合优化「拟合数据」与「正则化潜空间」。

#### 2. 重建损失：`F.binary_cross_entropy(..., reduction="sum")`

**目的：** 度量「给定重参数化采样得到的 \(z\)，解码器输出 \(\hat{x}\) 与真实 \(x\) 有多接近」。

**为何用 BCE：** 本项目中解码器末层为 **Sigmoid**，输出在 \((0,1)\)，与 `ToTensor()` 后 MNIST 像素落在 \([0,1]\) 一致。此时把每个像素看作 Bernoulli 率（或用 BCE 作为交叉熵重建），**BCE 与这一生成假设匹配**。若把像素当作高斯噪声下的连续观测，则更常见 **MSE**；二者对应不同的概率模型，需与**输出激活与像素尺度**一致。

**`reduction="sum"` 的特点：** 对当前 batch 内所有元素上的 BCE **求和**，得到的是与「batch 内总像素数」成比例的一个标量。它与 `reduction="mean"` 仅差常数倍，但会改变**重建项与 KL 项的相对量级**（以及和学习率、潜在 \(\beta\) 权重的搭配）。本仓库中 `train_one_epoch` 如何再对 batch 累加、除以 `total_samples`，需与 `vae_loss` 的 `reduction` 一并理解，避免把「按像素 mean」与「按样本平均」混读。

#### 3. KL 项：`-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())`

**目的：** 将编码器给出的 **\(q(z \mid x) = \mathcal{N}(\mu, \mathrm{diag}(\sigma^2))\)** 拉向先验 **\(p(z) = \mathcal{N}(0, I)\)**，使潜空间**更规整、更连续**，便于从 **\(\mathcal{N}(0,I)\)** 采样再解码（对应 `save_random_samples` 的生成方式）。

**公式来源（特点）：** 在对角协方差、先验为标准正态时，KL 有**闭式解**，计算快、梯度稳定。代码用 **`log_var`（即 \(\log\sigma^2\)）** 参数化方差，是为了保证 \(\sigma^2 > 0\) 且在 \(\sigma\) 很小时梯度行为更可控。对单维，KL 可取 \(\frac{1}{2}\big(\sigma^2 + \mu^2 - 1 - \log\sigma^2\big)\) 的形式；对所有潜维与 batch 维求和并整理符号后，即得到常见的 `-0.5 * sum(1 + log_var - mu^2 - exp(log_var))` 写法。

**直观作用简要归纳：**

- **\(\mu^2\) 相关项**：把各维均值往 0 拉，避免后验整体漂离先验中心；
- **\(\exp(\log\_var)\) 与 \(-\log\_var\)**：共同约束方差，避免无意义的路径（例如方差塌缩或失控），使表示留在「先验附近」的有效区域；
- **整体**：鼓励 \(q(z|x)\) 接近标准正态，从而**随机采样**时解码器仍能得到合理样本；但若 KL 相对过大，也可能出现重建偏糊或与 **posterior collapse** 相关的问题——需靠架构、容量或 \(\beta\)-VAE 等权衡。

#### 4. 为何要返回 `(total_loss, recon_loss, kl_divergence)` 三个量

- **`total_loss`**：唯一需要 **`.backward()`** 的量，反向传播用这个标量即可。
- **`recon_loss` / `kl_divergence`**：便于在 `train_one_epoch` 与 `main()` 里**分别记录**：重建是否在下降、KL 是否偏大或失效、两项是否失衡，而不用在日志里只能从总Loss反推。

#### 5. 小结对照表

| 部分 | 在优化什么 | 目的 | 特点与注意 |
|------|------------|------|------------|
| 重建（BCE） | 解码器拟合观测 \(x\) | 单张重建尽量像原图 | 与 Sigmoid + \([0,1]\) 像素假设一致；`sum`/`mean` 影响与 KL 的相对权重感 |
| KL | \(q(z|x)\) 接近 \(\mathcal{N}(0,I)\) | 潜空间规整、支持从标准正态生成 | 闭式快；`log_var` 数值更稳；与重建存在经典权衡 |
| 相加 | ELBO 的两项联合优化 | 无监督同时学表示与生成 | 本脚本未乘 β（等价 β=1）；工程中常用 β-VAE 等调节 |

---

## 第 5 节：`train_one_epoch()` 单轮训练流程图

### 框 1：`train_one_epoch()`（总标题）

负责**一个 epoch** 内遍历 `train_loader` 的全部 batch。

### 框 2：`model.train()`

将 `Dropout`、`BatchNorm` 等模块置于训练态；本网络虽为基本全连接+ReLU，仍保持习惯写法。

### 框 3：初始化累计变量

- **`total_loss_sum / recon_loss_sum / kl_loss_sum`**：累加每个 batch 返回的标量 `.item()`。
- **`total_samples`**：累加 `batch_size`，用于最后除以样本数得到**按样本平均**的三种损失。

### 框 4：`for images, _ in loader`

**忽略标签 `_`**，仅图像参与前向与损失；符合 VAE 无监督设定。

### 框 5：数据预处理

- **`images.to(device, non_blocking=True)`**：把数据搬到 GPU/CPU，non_blocking 在 pin_memory 时可减少同步等待。
- **`flatten`**：与第 3 节一致，供全连接层使用。

### 框 6：模型前向传播

- **`reconstructed, mu, log_var = model(images)`**：一次前向拿到重建与分布参数。

### 框 7：损失与参数更新

- **`vae_loss(...)`**：得到 `total_loss` 与分拆项。
- **`total_loss.backward()`**：对 `model` 中所有 `requires_grad=True` 的参数计算梯度。
- **`optimizer.step()`**：Adam 更新参数。
- **`optimizer.zero_grad()`**：本实现在每个 batch **开头**清空梯度（见 `mnist_vae.py`），避免梯度累积。

### 框 8：累计统计当前 batch 损失与样本数

将当前 batch 的 sum 型损失累加到 epoch 总量上，并增加 `total_samples`。注意若 `vae_loss` 内部是**全 batch 像素 sum**，则这里累加的是 batch 总和，与“按样本平均”的公式要一致；阅读代码时建议打开 `vae_loss` 确认 `reduction` 与除数含义。

### 框 9：所有 batch 结束后求平均

- **`average_total_loss = total_loss_sum / total_samples`**（以及 recon、kl 同理）作为 `main()` 打印与 `best_loss` 比较的依据。

### 框 10：返回三个平均损失指标

供 `main()` 打印与记录 `best_vae.pth`，不参与反向（返回的是 Python float）。

---

## 第 6 节：`main()` 中的训练保存分支图

### 框 1：训练保存分支图（总标题）

描述 epoch 循环内部的**并行逻辑块**（概念上分支，代码上顺序执行）。

### 框 2：进入 epoch 循环

外层驱动整个训练时间表。

### 框 3：分支 1——训练 `train_one_epoch`

算损失、反传、更新权重的主体。

### 框 4：分支 2——调度器 `scheduler.step()`

在 epoch 末尾调整学习率；与 `StepLR` 的 `step_size`、`gamma` 对应。

### 框 5：分支 3——日志与最佳模型

打印三项损失与学习率；若本轮 `average_total_loss` 创新低则覆盖写 `best_vae.pth`。

### 框 6：判断是否还有下一轮 epoch

由 `for` 循环控制；未到 `num_epochs` 则回到下一 epoch。

### 框 7：是——继续训练

循环继续累加优化。

### 框 8：否——训练结束后处理

跳出循环后执行保存与可视化。

### 框 9：保存 `last_vae.pth`、重建图、生成图、提示信息

与第 1 节场景 4 一致；这是**必执行**的收尾，与是否出现过 `best_loss` 无关。

---

## 第 7 节：张量尺寸变化图

### 框 1：张量尺寸变化流程（总标题）

与第 3 节前向一致，强调**张量 rank 与形状**。

### 框 2：输入图像 `[batch, 1, 28, 28]`

MNIST 原生空间尺寸。

### 框 3：flatten 后 `[batch, 784]`

### 框 4：encoder 输出 `[batch, 400]`

**注意**：图中的“encoder 输出”指 ReLU 后的隐藏向量，不是 `mu`/`log_var` 本身。

### 框 5：`mu` 向量 `[batch, 20]`

### 框 6：`log_var` 向量 `[batch, 20]`

### 框 7：重参数化得到 `z`——`[batch, 20]`

与 `latent_dim` 一致。

### 框 8：`fc_decode` 输出 `[batch, 400]`

### 框 9：decoder 输出 `[batch, 784]`

### 框 10：reshape 成图像 `[batch, 1, 28, 28]`

用于 `save_image` 或可视化；数学上仍是同一张图的另一种排布。

---

## 第 8 节：VAE 核心原理图补充（教材风格）

### 框 1：原始数据

一张手写数字图像或其展平向量；数据分布记为对 x 的经验分布。**训练目标**之一是：给定 x，解码器从采样的 z 能较好重建 x。

### 框 2：编码器

把 x 变成足以预测 **q(z|x)** 的特征；实现里即 `encode`：先升维到 400 再分叉出 μ 与 log σ²。

### 框 3：均值 μ

对角高斯各维的中心；若训练得当，同类数字的 μ 在潜空间中会形成聚类趋势（不保证线性可分）。

### 框 4：方差 σ²（代码中为 `log_var`）

各维“不确定度”；方差大说明该维在重构 x 时容忍噪声更大，采样 z 时该维波动更大。

### 框 5：定义正态分布 N(μ, σ²)

**近似后验 q(z|x)**。训练和生成时都依赖这一族分布：训练时用重参数化采样；生成时若训练充分，可从 **N(0,1)** 近似代替。

### 框 6：标准正态分布 N(0, 1)

**先验 p(z)**。KL 项迫使 q(z|x) 接近该先验，避免后验坍塌到远离原点的狭窄区域，否则随机采样难以泛化。

### 框 7：分布一致 Loss（教材用语）

在代码里即 **KL 散度**；不直接对“整张图像”算，而对**潜变量分布**算。

### 框 8：重建 Loss

连接 **原始数据** 与 **重建数据**，代码为 **BCE**（像素视为 Bernoulli 率或用于 0–1 像素的交叉熵）。也可用 MSE，但需与输出激活、像素范围一致。

### 框 9：从 N(μ, σ²) 采样得到 z

随机节点；通过 **z=μ+εσ** 与 ε∼N(0,1) 实现可导。

### 框 10：潜变量 z

低维、用于承载语义与风格的压缩码；维数 `latent_dim` 控制容量。

### 框 11：解码器

把 z 映射回高维像素；实现里两层全连接 + Sigmoid。

### 框 12：重建数据

模型对 x 的近似 x̂；训练时与 x 比对；生成时若 z 来自先验，则 x̂ 是“新样本”。

---

## 第 9 节：与辅助函数、输出文件的全局对照

### `build_dataloader`

- **MNIST `root=./data`**：数据落盘位置。
- **`ToTensor()`**：uint8 像素 → float [0,1]。
- **`DataLoader`**：`shuffle=True` 打乱；`num_workers=2`；`pin_memory` 随 CUDA 开启。

### `setup_seed` / `prepare_output_dirs`

已在前文各框覆盖；略。

### `save_reconstructions`

- **`model.eval()`**：关闭训练期行为（如有 BN/Dropout）。
- **`torch.no_grad()`**：推理不建计算图，省显存。
- **拼接 `original` 与 `reconstruction`**：便于一眼对比。

### `save_random_samples`

- **`z ~ N(0,1)`**：**不经过编码器**，检验“潜空间+解码器”是否学到**可生成**的映射。
- **`nrow=4`**：保存为 4 列网格图。

### 输出文件小结

| 路径 | 含义 |
|------|------|
| `outputs/mnist_vae/best_vae.pth` | 训练过程中平均总损失最优的权重 |
| `outputs/mnist_vae/last_vae.pth` | 最后一轮权重 |
| `outputs/mnist_vae/images/reconstruction.png` | 原图与重建对比 |
| `outputs/mnist_vae/images/generated.png` | 纯随机 z 生成样本 |

---

## 结语

读完本文档后，你可以从任意一张 `mnist_vae_flowchart.md` 中的方框出发，在本文件中找到对应的**目的、输入输出、与 `mnist_vae.py` 的代码位置、以及与其他框的依赖关系**。若你希望下一步把 **CVAE** 或 **β-VAE** 也写成同样风格的流程与逐框说明，可以在同目录下再开新文件延续这一结构。

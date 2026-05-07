# Diffusion 与 MaskGIT 训练/推理概览

本项目包含两条 MNIST 生成路线：

- `diffusion.py` + `train_mnist.py`：先训练连续潜变量 VAE，再在 VAE latent 上训练 DiT/DDPM。
- `maskgit.py` + `train_maskgit_mnist.py`：先训练离散潜变量 VQ-VAE，再在 VQ token 上训练 MaskGIT。

## Diffusion 路线

### 训练流程

1. `ConvVAE` 将 `1x28x28` MNIST 图像编码为 `latent_channels x 7 x 7` 的连续 Gaussian latent。
2. VAE 用 BCE 重建损失和 KL 损失训练，checkpoint 保存为 `vae.pt`。
3. 训练 DiT 时，冻结 VAE，用 `ConvVAE.encode` 得到 `mu/logvar`，再重参数化采样 latent。
4. `Diffusion.q_sample` 随机选择 timestep `t`，把 latent 加噪为 `x_t`，并返回真实噪声 `epsilon`。
5. `DiT` 输入 `x_t`、`t` 和数字类别标签，预测噪声。
6. DiT 用预测噪声与真实噪声之间的 MSE 训练，checkpoint 保存为 `dit.pt`。

### 推理流程

1. 从标准高斯噪声 latent 开始。
2. `Diffusion.sample` 从最大 timestep 反向迭代到 0。
3. 每一步调用 `DiT` 预测噪声，并用 DDPM 公式更新 latent。
4. 条件采样时可使用 classifier-free guidance：同时预测有标签和无标签噪声，再按 `cfg_scale` 外推。
5. 最终 latent 经过 `ConvVAE.decode` 还原为图像 logits，再用 sigmoid 得到像素。

## MaskGIT 路线

### 训练流程

1. `VQVAE` 将 MNIST 图像编码为 `7x7` 的离散 codebook token。
2. VQ-VAE 用 BCE 重建损失、codebook 损失和 commitment 损失训练，checkpoint 保存为 `vqvae.pt`。
3. 训练 MaskGIT 时，冻结 VQ-VAE，把 `7x7` token grid 展平为长度 49 的序列。
4. `MaskGIT.random_mask` 随机遮盖一部分 token，并返回 mask ratio。
5. `MaskGIT` 输入被遮盖序列、mask ratio 和可选类别标签，预测所有位置的 code logits。
6. 只在被 mask 的位置计算 cross entropy。
7. 项目先训练无条件模型 `maskgit_uncond.pt`，再可继续训练条件模型 `maskgit_cond.pt`。

### 推理流程

1. 从全 `[MASK]` 的长度 49 token 序列开始。
2. 每一步 MaskGIT 并行预测所有 masked token 的分布。
3. 从分布中采样 token，并根据采样概率作为 confidence。
4. 按 cosine schedule 重新 mask 一部分低 confidence token。
5. 迭代结束后，把长度 49 的序列还原为 `7x7` token grid。
6. `VQVAE.decode_indices` 把 token grid 映射回 codebook latent 并解码为图像。
7. 条件采样同样支持 classifier-free guidance。

### mask ratio embedding 的作用

`mask_ratio` 表示当前序列里有多少比例的 token 仍然是 `[MASK]`。它类似 diffusion 里的 timestep：告诉模型“现在处在生成过程的哪个阶段”。

训练时，`MaskGIT.random_mask` 会为每张图随机采样不同的遮盖比例。有些样本只缺少少量 token，模型主要做局部补全；有些样本几乎全被遮住，模型必须根据很少的上下文甚至只靠类别先验来生成。把 `mask_ratio` 通过 `SinusoidalEmbedding` 变成条件向量并加到每个 token embedding 上，可以让同一个 Transformer 根据遮盖程度调整预测策略。

推理时，MaskGIT 从全 mask 开始，然后逐步填 token、重新 mask 低 confidence 的位置。早期 `mask_ratio` 高，模型知道上下文很少，预测应该更依赖全局结构和类别条件；后期 `mask_ratio` 低，模型知道大部分内容已经确定，预测可以更偏向局部一致性和细节修正。

如果没有这个 embedding，模型仍然能从 `[MASK]` token 的数量间接推断阶段，但这需要 Transformer 自己统计全局 mask 数量。显式加入 `mask_ratio` 会让阶段信息更直接、更稳定，也让训练时的随机遮盖比例和推理时的逐步解码过程更好对齐。

## 类职责

### `diffusion.py`

- `ConvVAE`：连续 latent 自编码器，负责把图像压缩到 `B C 7 7` latent 并从 latent 重建图像。
- `SinusoidalTimeEmbedding`：把 timestep 标量映射成 DiT 可用的条件向量。
- `DiTBlock`：带 adaptive LayerNorm 的 Transformer block，用 timestep/class 条件调制 attention 和 MLP。
- `DiT`：latent diffusion 主模型，把 latent map 展平成 token 序列，预测 DDPM 噪声。
- `Diffusion`：维护 beta/alpha schedule，提供前向加噪 `q_sample` 和反向采样 `sample`。

### `maskgit.py`

- `VectorQuantizer`：维护 codebook，把连续 encoder latent 最近邻量化为离散 token。
- `VQVAE`：离散 latent 自编码器，负责图像到 VQ token grid 的编码和 token grid 到图像的解码。
- `SinusoidalEmbedding`：把 mask ratio 等标量映射成 Transformer 条件向量。
- `MaskGIT`：双向 Transformer，基于上下文并行预测被 mask 的 VQ token。

### 训练与可视化脚本

- `train_mnist.py`：负责 VAE/DiT 的训练、加载、采样和 TensorBoard 记录。
- `train_maskgit_mnist.py`：负责 VQ-VAE/MaskGIT 的训练、加载、采样和 TensorBoard 记录。
- `visualize_dit_sampling.py`：加载训练好的 VAE/DiT，保存反向扩散过程的帧、GIF 和 contact sheet。

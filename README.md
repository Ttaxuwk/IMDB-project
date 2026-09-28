# IMDB 情感分析：从零 LSTM、从零 Transformer 与 BERT 迁移学习对比

## 一、项目简介

本项目在 IMDB 电影评论数据集上完成情感分析（二分类：正面 / 负面），对比三种模型：

1. **LSTM**：从零实现双向 LSTM，理解序列建模基础
2. **Transformer**：从零实现 Multi-Head Self-Attention，理解注意力机制
3. **BERT**：基于预训练 BERT 的迁移学习，冻结底层 + 微调高层

目标是通过对比实验，理解序列建模的演进路径、迁移学习在 NLP 中的价值，以及不同架构的适用场景。

## 二、项目结构

```
IMDB-project/
├── data/                      # IMDB 数据集（自动下载）
├── models/                    # 保存的模型权重
│   ├── best_lstm.pth
│   ├── best_transformer.pth
│   └── best_bert.pth
├── results/                   # 训练曲线和错误分析图
│   ├── lstm_training_curve.png
│   ├── transformer_training_curve.png
│   └── bert_training_curve.png
├── data_utils.py              # 数据加载、词表构建（LSTM / Transformer 共用）
├── train_lstm.py              # 双向 LSTM
├── train_transformer.py       # 从零 Transformer
├── train_bert.py              # BERT 迁移学习
└── README.md
```

## 三、数据集

**IMDB 电影评论**：50000 条真实电影评论，二分类（正面 / 负面）。

- 训练集：22500 条
- 验证集：2500 条
- 测试集：25000 条

**两种分词方式**：

| 模型 | 分词方式 | 词表大小 | 序列长度 |
|---|---|---|---|
| LSTM / Transformer | 词级（小写 + 去标点） | 10000 | 256 |
| BERT | 子词级（WordPiece） | 30522 | 256 |

## 四、实验一：双向 LSTM

### 4.1 模型结构

```
输入 [batch, 256]
  ↓ Embedding (10000 → 128)
  ↓ 双向 LSTM (128 → 128×2)
  ↓ Masked Mean Pooling
  ↓ Dropout(0.3) + Linear(256 → 2)
```

- 参数量：1,544,706
- 双向 LSTM 同时利用左右上下文

### 4.2 训练配置

| 配置 | 值 |
|---|---|
| 优化器 | Adam, lr=1e-3, weight_decay=1e-4 |
| 损失函数 | CrossEntropyLoss |
| Batch size | 64 |
| 梯度裁剪 | max_norm=1.0 |
| 早停 | patience=5 |
| Epochs | 最多 20（实际约 7 触发早停） |

### 4.3 结果

| 指标 | 值 |
|---|---|
| 最佳验证准确率 | 89.52% |
| 测试准确率 | **86.46%** |
| 测试损失 | 0.3415 |

### 4.4 关键发现

- 第 1-4 个 epoch 验证准确率快速上升
- 第 5 个 epoch 后训练准确率继续上升，验证准确率开始下降 → 过拟合
- 早停在 epoch 7 触发，避免无效训练

### 4.5 调参历程

| 实验 | 改动 | 验证准确率 | 测试准确率 | 结论 |
|---|---|---|---|---|
| 基线 | - | 89.08% | 86.46% | - |
| +早停 | patience=3 | 89.52% | 85.90% | 避免无效训练 |
| +减小容量 | 128→64 | 84.76% | 85.75% | ❌ 反而变差 |
| +weight decay + 梯度裁剪 | wd=1e-4, clip=1.0 | 90.04% | 86.22% | 略提升 |
| +强正则 | 2层 LSTM + dropout 0.5 | 93.92% | 85.88% | 验证虚高 |
| +lr调度 | ReduceLROnPlateau | 93.92% | 85.88% | 无提升 |

**结论**：LSTM 测试准确率上限约 86%，继续调参收益递减。

## 五、实验二：从零 Transformer

### 5.1 模型结构

```
输入 [batch, 256]
  ↓ Embedding (10000 → 128) + 位置编码
  ↓ Transformer Encoder Block × 2
      ├── Multi-Head Self-Attention (4 头)
      ├── 残差 + LayerNorm (Pre-Norm)
      ├── Feed-Forward (128 → 256 → 128)
      └── 残差 + LayerNorm
  ↓ LayerNorm
  ↓ Masked Mean Pooling
  ↓ Dropout + Linear(128 → 2)
```

- 参数量：1,578,242
- 2 层 Encoder Block，每层 4 头注意力

### 5.2 核心组件

**Multi-Head Self-Attention**：

```
Q = X @ W_Q,  K = X @ W_K,  V = X @ W_V
scores = Q @ K^T / sqrt(d_k)
attention = softmax(scores)
output = attention @ V
```

**Pre-Norm 残差结构**：

```
x = LayerNorm(x + Dropout(Sublayer(x)))
```

### 5.3 训练配置

| 配置 | 值 |
|---|---|
| 优化器 | Adam, lr=5e-4, weight_decay=1e-4 |
| 学习率调度 | Warmup(10%) + 线性衰减 |
| 损失函数 | CrossEntropyLoss |
| Batch size | 64 |
| 梯度裁剪 | max_norm=1.0 |
| 早停 | patience=5 |

### 5.4 结果

| 指标 | 值 |
|---|---|
| 最佳验证准确率 | 86.28% |
| 测试准确率 | **83.34%** |
| 测试损失 | 0.3729 |

### 5.5 调参历程（重要的负面实验）

| 配置 | 最佳验证 | 测试准确率 | 现象 |
|---|---|---|---|
| lr=1e-3 | 83.80% | 78.08% | **Epoch 2 崩溃**（83% → 57%） |
| lr=5e-4 | 84.12% | 76.15% | Epoch 2 仍崩溃 |
| +warmup | 86.28% | 83.34% | 缓解但未解决 |
| +dropout=0.3 | 88.20% | 81.80% | 验证虚高 |
| +lr=3e-4 | 89.00% | 79.36% | 学不动，测试下降 |

**结论**：从零训练的 Transformer 在 IMDB 上测试准确率约 83%，低于 LSTM 的 86.46%。原因：

1. IMDB 只有 22500 条训练数据，对 Transformer 偏少
2. Transformer 缺少 LSTM 的归纳偏置
3. Transformer 对学习率极其敏感，需要 warmup

### 5.6 关于 Transformer 训练的重要发现

**现象**：无论 lr 是 1e-3 还是 5e-4，第 2 个 epoch 验证准确率都会从 84% 暴跌到 57%。

**原因**：

- 训练初期参数随机初始化，梯度方向混乱
- 大学习率让参数跳到糟糕区域
- 需要 warmup 让学习率先小后大

**解决方案**：前 10% 步数做 warmup，学习率从 0 线性增加到峰值。

**这是 Transformer 训练中最经典的坑**，原始论文专门用了 4000 步 warmup。

## 六、实验三：BERT 迁移学习

### 6.1 模型结构

```
输入 [batch, 256]
  ↓ BERT Tokenizer（子词级）
  ↓ BERT Encoder × 12（冻结前 8 层）
  ↓ 取 [CLS] token 输出
  ↓ Dropout(0.3) + Linear(768 → 2)
```

- 总参数量：109,483,778
- 可训练参数量：28,943,618（26.4%）

### 6.2 关键：冻结底层

```python
# 冻结 embeddings 和前 8 层
for param in self.bert.embeddings.parameters():
    param.requires_grad = False
for i in range(freeze_layers):
    for param in self.bert.encoder.layer[i].parameters():
        param.requires_grad = False
```

**为什么冻结**：

- BERT 前 8 层学的是通用语法特征，不需要改
- 后 4 层学的是高层语义，需要适应新任务
- 冻结后：参数量减少 74%，内存减半，速度翻倍，过拟合减少

### 6.3 训练配置

| 配置 | 值 |
|---|---|
| 优化器 | AdamW, lr=2e-5, weight_decay=0.01 |
| 学习率调度 | Warmup(10%) + 线性衰减 |
| Batch size | 8 |
| Max length | 256 |
| 梯度裁剪 | max_norm=1.0 |
| 早停 | patience=2 |
| Epochs | 最多 5 |

**为什么学习率 2e-5**：比从零训练小 100 倍。预训练权重已经很好了，只需要小幅调整。

### 6.4 结果

| 指标 | 值 |
|---|---|
| 最佳验证准确率 | 92.64% |
| 测试准确率 | **91.34%** |
| 测试损失 | 0.2750 |

### 6.5 关键发现

- 第 1 个 epoch 验证准确率就到 92.52%
- 第 2 个 epoch 达到最佳 92.64%
- 第 3 个 epoch 开始过拟合（train 94%，val 87.68%）
- 早停在 epoch 4 后触发

### 6.6 BERT 的三种失败与解决

**失败 1**：直接跑 32 batch + 256 max_len，16GB Mac 内存爆了。

**解决**：batch_size 降到 8。

**失败 2**：MPS 不支持 `scaled_dot_product_attention` 的 dropout。

**报错**：`NotImplementedError: scaled_dot_product_attention for MPS does not support dropout.`

**解决**：`BertModel.from_pretrained(..., attn_implementation="eager")`。

**失败 3**：test_set 里有个别样本的文本是 None，导致 tokenizer 报错。

**解决**：加载后过滤掉 None 和空文本。

## 七、三个模型对比

### 7.1 最终结果

| 模型 | 训练方式 | 参数量 | 测试准确率 |
|---|---|---|---|
| LSTM | 从零训练 | 154 万 | 86.46% |
| Transformer | 从零训练 | 158 万 | 83.34% |
| **BERT** | **预训练 + 冻结微调** | **1.09 亿** | **91.34%** |

### 7.2 训练效率对比

| 模型 | 每个 epoch 时间 | 总训练时间 |
|---|---|---|
| LSTM | 约 2 分钟 | 约 15 分钟（7 epoch） |
| Transformer | 约 1 分钟 | 约 10 分钟（10 epoch） |
| BERT（冻结） | 约 10 分钟 | 约 40 分钟（4 epoch） |

### 7.3 核心结论

1. **迁移学习显著优于从零训练**：BERT（91.34%）比 LSTM（86.46%）高 4.9 个百分点，比从零 Transformer（83.34%）高 8 个百分点。

2. **从零 LSTM 打败从零Transformer**：在中小规模数据上，LSTM 的归纳偏置（序列建模天性）比 Transformer 更有效。

3. **Transformer 对学习率极其敏感**：需要 warmup，否则第 2 个 epoch 会崩溃。

4. **冻结底层是 BERT 微调的关键**：可训练参数从 1.09 亿降到 2890 万，速度翻倍，效果更好。

5. **预训练数据量决定上限**：BERT 在 33 亿词上预训练，这是它碾压从零模型的根本原因。

## 八、个人思考与踩坑记录

### 8.1 关于序列建模

- **RNN/LSTM**：串行处理，天然适合序列，但有梯度消失问题
- **Transformer**：并行处理，Self-Attention 直接建模任意位置关系，但需要更多数据
- **BERT**：预训练 + 微调，把“学语言”和“做任务”分离

### 8.2 关于 Attention 的直观理解

> 每个词拿着自己的问题（Q），去问所有词的名牌（K），根据匹配度（Softmax），把所有词的信息（V）加权融合。

- **Self-Attention**：自己问自己
- **Multi-Head**：同时问多个问题
- **Cross-Attention**：Decoder 问 Encoder

### 8.3 关于过拟合

- LSTM 从 epoch 5 开始过拟合
- Transformer 从 epoch 3 开始过拟合
- BERT 从 epoch 3 开始过拟合
- **早停是最简单有效的正则化手段**

### 8.4 关于工程细节

- Mac 上 `num_workers=0`，否则多进程报错
- HuggingFace 用 `HF_ENDPOINT=https://hf-mirror.com` 镜像
- BERT 的 tokenization 要预 tokenize，否则训练时每次都重新 tokenize
- MPS 上 BERT 需要 `attn_implementation="eager"`

### 8.5 关于调参的认知

**调参不能解决所有问题。** Transformer 从 78% 调到 83% 花了 6 组实验，但始终没超过 LSTM。这时候应该换架构，而不是继续调参。

**架构选择比参数微调重要 10 倍。**

## 九、如何运行

```bash
# 安装依赖
pip install torch transformers datasets tqdm matplotlib numpy

# 训练 LSTM
python train_lstm.py

# 训练从零 Transformer
python train_transformer.py

# 训练 BERT
python train_bert.py
```

设备支持：CUDA / MPS（Mac M 芯片）/ CPU

## 十、参考

- [IMDB 数据集](https://huggingface.co/datasets/stanfordnlp/imdb)
- [BERT 论文](https://arxiv.org/abs/1810.04805)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Stanford CS224n
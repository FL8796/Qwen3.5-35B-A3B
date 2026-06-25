# Qwen3.5-35B-A3B 模型结构技术文档

> **面向受众**: 技术专家 / 架构师 / 资深算法工程师
> **文档性质**: 技术宣讲材料


---

## 目录

1. [宏观定位与设计哲学](#1-宏观定位与设计哲学)
2. [整体架构概览](#2-整体架构概览)
3. [核心配置参数速查](#3-核心配置参数速查)
4. [Hybrid Attention: Gated DeltaNet × Gated Attention](#4-hybrid-attention-gated-deltanet--gated-attention)
5. [Sparse Mixture-of-Experts: 全层 MoE 设计](#5-sparse-mixture-of-experts-全层-moe-设计)
6. [3D Multimodal RoPE: 统一多模态位置编码](#6-3d-multimodal-rope-统一多模态位置编码)
7. [Vision Encoder: 视觉信号的前端处理](#7-vision-encoder-视觉信号的前端处理)
8. [Tokenizer 与词表设计](#8-tokenizer-与词表设计)
9. [Dense vs MoE: 与 Qwen3.5-27B 的架构对比](#9-dense-vs-moe-与-qwen35-27b-的架构对比)
10. [推理优化策略](#10-推理优化策略)
11. [类继承关系全景图](#11-类继承关系全景图)
12. [关键源码文件索引](#12-关键源码文件索引)

---

## 1. 宏观定位与设计哲学

### 1.1 一句话定位

Qwen3.5-35B-A3B 是阿里通义团队在 2026 年发布的**原生多模态混合专家（MoE）大语言模型**，总参数量 350 亿，每次推理仅激活约 30 亿参数，以 "大模型的知识容量 + 小模型的推理成本" 为设计目标。

### 1.2 三大设计支柱

```
┌─────────────────────────────────────────────────────────┐
│                   Qwen3.5-35B-A3B                        │
├─────────────────┬─────────────────┬─────────────────────┤
│   Hybrid        │   Sparse MoE    │   Native            │
│   Attention     │   FFN           │   Multimodal        │
│                 │                 │                     │
│  3:1 GDN:Attn   │  256 experts    │  Text + Image +     │
│  O(N) + O(N²)   │  Top-8 routed   │  Video early fusion │
│  per-block      │  + shared       │  MRoPE 3D position  │
└─────────────────┴─────────────────┴─────────────────────┘
```

### 1.3 关键数字

| 指标 | 数值 |
|------|------|
| 总参数 | 35B |
| 激活参数 | ~3B |
| 隐藏维度 | 2,048 |
| 层数 | 40 |
| 全注意力层数 | 10（每 4 层中 1 层） |
| 线性注意力层数 | 30（每 4 层中 3 层） |
| 专家总数 | 256 |
| 每 token 激活专家 | 8（路由）+ 1（共享） |
| 原生上下文 | 262K tokens |
| 最大上下文 | 1,010K tokens |
| 词表大小 | 248,320 |
| 支持语言 | 201 种 |

---

## 2. 整体架构概览

### 2.1 宏观数据流

```
                    ┌──────────────────────┐
   Image/Video ────▶│  Vision Encoder      │
                    │  (ViT, 27 layers)    │──── Visual Embeddings ──┐
                    └──────────────────────┘                        │
                                                                     ▼
   Text ───────────▶│  Token Embedding     │──────────────────────────┤
                    │  (248,320 × 2,048)   │                          │
                    └──────────────────────┘                          │
                                                                     ▼
                    ┌──────────────────────────────────────────────────┐
                    │              Qwen3_5MoeTextModel                 │
                    │                                                  │
                    │  ┌──────────────────────────────────────────┐   │
                    │  │  Layer 0:  GatedDeltaNet  →  MoE FFN     │   │
                    │  │  Layer 1:  GatedDeltaNet  →  MoE FFN     │   │
                    │  │  Layer 2:  GatedDeltaNet  →  MoE FFN     │   │  ×10
                    │  │  Layer 3:  GatedAttention →  MoE FFN     │   │
                    │  └──────────────────────────────────────────┘   │
                    │                                                  │
                    │  RMSNorm → lm_head → logits                     │
                    └──────────────────────────────────────────────────┘
```

### 2.2 单层 Decoder 结构

```
Input (hidden_states)
    │
    ├── input_layernorm (RMSNorm)
    │       │
    │       ▼
    │   Token Mixer ──── 分支选择 ────┐
    │       │                         │
    │       │    layer_type ==        │    layer_type ==
    │       │    "linear_attention"   │    "full_attention"
    │       │          │              │          │
    │       │   GatedDeltaNet         │    GatedAttention
    │       │   (O(N) 线性注意力)      │    (O(N²) 全注意力)
    │       │          │              │          │
    │       └──────────┴──────────────┘          │
    │                    │                       │
    ├── Residual Add ◄───┘                       │
    │                    │                       │
    ├── post_attention_layernorm (RMSNorm)       │
    │       │                                     │
    │   SparseMoeBlock (所有层)                    │
    │   ┌─────────────────────────┐              │
    │   │ Router → Top-8 experts  │              │
    │   │ Shared Expert           │              │
    │   │ Σ(weight × expert) +    │              │
    │   │   gate × shared         │              │
    │   └─────────────────────────┘              │
    │       │                                     │
    └── Residual Add ◄────────────────────────────┘
    │
    ▼
Output
```

---

## 3. 核心配置参数速查

### 3.1 Text Config (`Qwen3_5MoeTextConfig`)

| 参数 | 值 | 说明 |
|------|-----|------|
| `model_type` | `qwen3_5_moe_text` | 模型类型标识 |
| `vocab_size` | 248,320 | 词表大小（已填充至 256 的倍数） |
| `hidden_size` | 2,048 | 隐藏层维度 |
| `num_hidden_layers` | 40 | Transformer 层数 |
| `num_attention_heads` | 16 | Full Attention 的 Query 头数 |
| `num_key_value_heads` | 2 | Full Attention 的 KV 头数（GQA 8:1） |
| `head_dim` | 256 | Full Attention 每头维度 |
| `hidden_act` | `silu` | 激活函数 |
| `max_position_embeddings` | 32,768 | 默认最大位置编码 |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `initializer_range` | 0.02 | 权重初始化标准差 |
| `attention_bias` | `False` | 注意力投影不使用偏置 |
| `attention_dropout` | 0.0 | 注意力 dropout 率 |
| `tie_word_embeddings` | `False` | 词嵌入与 lm_head 不共享 |
| `partial_rotary_factor` | 0.25 | RoPE 部分应用比例 |

### 3.2 Linear Attention Config

| 参数 | 值 | 说明 |
|------|-----|------|
| `linear_conv_kernel_dim` | 4 | 因果卷积核大小 |
| `linear_key_head_dim` | 128 | GDN 中 Key 每头维度 |
| `linear_value_head_dim` | 128 | GDN 中 Value 每头维度 |
| `linear_num_key_heads` | 16 | GDN 中 QK 头数 |
| `linear_num_value_heads` | 32 | GDN 中 V 头数（V:K = 2:1） |

### 3.3 MoE Config

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_experts` | 256 | 专家总数 |
| `num_experts_per_tok` | 8 | 每 token 激活的路由专家数 |
| `moe_intermediate_size` | 512 | 单个专家 FFN 中间维度 |
| `shared_expert_intermediate_size` | 512 | 共享专家 FFN 中间维度 |
| `router_aux_loss_coef` | 0.001 | 路由辅助损失系数 |
| `output_router_logits` | `False` | 默认不输出路由 logits |

### 3.4 Vision Config (`Qwen3_5MoeVisionConfig`)

| 参数 | 值 | 说明 |
|------|-----|------|
| `depth` | 27 | ViT 层数 |
| `hidden_size` | 1,152 | 视觉隐藏维度 |
| `intermediate_size` | 4,304 | 视觉 FFN 中间维度 |
| `num_heads` | 16 | 视觉注意力头数 |
| `in_channels` | 3 | 输入通道（RGB） |
| `patch_size` | 16 | Patch 大小 |
| `spatial_merge_size` | 2 | 空间合并因子 |
| `temporal_patch_size` | 2 | 时序 patch 大小（视频） |
| `out_hidden_size` | 3,584 | 视觉输出投影维度 |
| `num_position_embeddings` | 2,304 | 最大位置编码数 |
| `hidden_act` | `gelu_pytorch_tanh` | 视觉激活函数 |

---

## 4. Hybrid Attention: Gated DeltaNet × Gated Attention

### 4.1 设计动机

传统 Transformer 的 Self-Attention 复杂度为 O(N²)，在长上下文（262K+ tokens）场景下计算开销巨大。Qwen3.5 采用 **3:1 混合注意力堆叠**：每 4 层中，3 层使用 O(N) 复杂度的 Gated DeltaNet（线性注意力），1 层使用 O(N²) 的 Gated Attention（全注意力）。

```
Layer 0:  GatedDeltaNet  ████████████  O(N)    ┐
Layer 1:  GatedDeltaNet  ████████████  O(N)    │  Group 0
Layer 2:  GatedDeltaNet  ████████████  O(N)    │
Layer 3:  GatedAttention ████████████  O(N²)   ┘
Layer 4:  GatedDeltaNet  ████████████  O(N)    ┐
   ...                                          │ ×10 groups
Layer 39: GatedAttention ████████████  O(N²)   ┘
```

**设计权衡**: 30 层线性注意力提供高效的长程建模，10 层全注意力提供精确的局部语义捕获。

### 4.2 Gated Attention（全注意力层）

#### 4.2.1 数据流

```
Input (B, S, 2048)
    │
    ├── q_proj: (2048, 16×256×2) ──────▶ Q, gate = chunk(..., 2, dim=-1)
    │       Q: (B, S, 16, 256)          gate: (B, S, 16)
    │       │
    │       ▼
    │   q_norm (RMSNorm per-head)
    │       │
    ├── k_proj: (2048, 2×256) ────────▶ K: (B, S, 2, 256)
    │       │
    │       ▼
    │   k_norm (RMSNorm per-head)
    │       │
    ├── v_proj: (2048, 2×256) ────────▶ V: (B, S, 2, 256)
    │       │
    ├── RoPE applied to Q, K
    │       │
    ├── KV Cache update (if inference)
    │       │
    ├── Scaled Dot-Product Attention
    │   (FlashAttention-2 / SDPA / Eager)
    │   GQA: 16 Q heads → 2 KV heads (repeat 8×)
    │       │
    └── attn_output * sigmoid(gate) ────── Gated 机制
            │
            ▼
    o_proj: (4096, 2048) ─────────────▶ Output
```

#### 4.2.2 关键设计点

**GQA (Grouped Query Attention)**:
- Q 头数 16，KV 头数 2，分组比 8:1
- 相比标准 Multi-Head Attention，KV Cache 压缩 8×，显著降低推理显存

**Q/K 归一化 (QK-Norm)**:
- 在注意力计算前对 Q 和 K 分别做 per-head RMSNorm
- 稳定训练过程，防止注意力 logits 的数值溢出
- 这是从 Qwen3 系列引入的改进

**Gated 输出**:
```python
query_states, gate = torch.chunk(q_proj(hidden_states), 2, dim=-1)
# ... attention ...
attn_output = attn_output * torch.sigmoid(gate)
```
- 从 Qwen3Next 继承的 gating 机制
- q_proj 输出维度为 `num_heads × head_dim × 2`，后半部分作为 gate
- gate 通过 sigmoid 函数控制每个注意力头的输出幅度

### 4.3 Gated DeltaNet（线性注意力层）

#### 4.3.1 核心思想

Gated DeltaNet 是一种基于 **Delta Rule（Delta 规则）** 的线性注意力机制，其核心思想源于经典的线性联想记忆理论。它将注意力建模为一种在线学习过程：给定键值对 (K, V)，模型通过 Delta 规则增量更新一个隐式状态矩阵 S（recurrent state），然后用 Query 从状态矩阵中读取结果。

```
Recurrent State Update (Delta Rule):
    S_t = S_{t-1} * g_t + K_t^T * (V_t - K_t * S_{t-1}) * β_t

Output Read:
    O_t = Q_t * S_t
```

其中：
- `g_t = -exp(A_log) * softplus(a_t + dt_bias)` 是时间衰减因子
- `β_t = sigmoid(b_t)` 是门控更新强度
- `S_t ∈ R^{k_dim × v_dim}` 是循环状态矩阵

#### 4.3.2 数据流

```
Input (B, S, 2048)
    │
    ├── in_proj_qkv: (2048, 2×2048 + 4096) ──▶ Q, K, V 联合投影
    │       │
    │       ▼
    │   CausalConv1d (kernel=4, groups=6144, SiLU)
    │   因果卷积提供局部上下文感知
    │       │
    │       ▼
    │   Q, K, V 拆分: Q(2048), K(2048), V(4096)
    │   重塑: Q→(B,S,16,128), K→(B,S,16,128), V→(B,S,32,128)
    │
    ├── in_proj_z: (2048, 4096) ──▶ Z: (B, S, 32, 128)
    │       │
    ├── in_proj_b: (2048, 32) ──▶ b: (B, S, 32)
    │       │  β = sigmoid(b)
    │
    ├── in_proj_a: (2048, 32) ──▶ a: (B, S, 32)
    │       │  g = -exp(A_log) * softplus(a + dt_bias)
    │
    │   Q/K L2 Normalize
    │       │
    │   ┌─────────────────────────────────────┐
    │   │  Gated Delta Rule 计算               │
    │   │                                     │
    │   │  Prefill: chunk_gated_delta_rule    │
    │   │  (分块并行, chunk_size=64)           │
    │   │                                     │
    │   │  Decode:  fused_recurrent_gated_    │
    │   │  delta_rule (单步循环)               │
    │   └─────────────────────────────────────┘
    │       │
    │   RMSNormGated(core_attn_out, Z)
    │   门控归一化: norm(out) * silu(Z)
    │       │
    └── out_proj: (4096, 2048) ──────────────▶ Output
```

#### 4.3.3 关键组件详解

**因果卷积 (CausalConv1d)**:
```
Conv1d(
    in_channels=6144,   # 2*2048(Q,K) + 4096(V)
    out_channels=6144,
    kernel_size=4,
    groups=6144,        # Depthwise 卷积
    padding=3,
    bias=False
)
```
- 在 Q/K/V 联合投影后应用，提供局部上下文窗口
- 使用 `causal-conv1d` (Dao-AILab) 加速库，否则回退到 PyTorch 实现
- 激活函数为 SiLU

**时间离散化 (Time Discretization)**:
```python
# 可学习参数
dt_bias: nn.Parameter(ones(32))        # 每 V 头的时间步偏置
A_log:   nn.Parameter(log(U(0,16)))    # 每 V 头的衰减因子（对数空间）

# 前向计算
g = -exp(A_log) * softplus(a + dt_bias)
```
- 使用 softplus 保证正的衰减率
- `dt_bias` 初始化为 1，`A_log` 初始化为 (0, 16) 均匀分布的对数

**Q/K L2 归一化**:
```python
query = l2norm(query, dim=-1, eps=1e-6)
key = l2norm(key, dim=-1, eps=1e-6)
```
- 在 Delta Rule 计算前对 Q 和 K 进行 L2 归一化
- 稳定训练过程，防止数值不稳定

**RMSNormGated**:
```python
# 不使用 FLA 加速时
hidden_states = RMSNorm(hidden_states) * silu(gate)
# 使用 FLA 加速时（融合算子）
hidden_states = FusedRMSNormGated(hidden_states, gate)
```
- 将 RMSNorm 和门控机制融合
- 门控信号 Z 通过 SiLU 激活后与归一化输出相乘

**Prefill vs Decode 分派**:
```
seq_len == 1 且有缓存状态?
    YES → fused_recurrent_gated_delta_rule (单步循环)
    NO  → chunk_gated_delta_rule (分块并行, chunk_size=64)
```
- Prefill 阶段使用分块并行算法，充分利用 GPU 并行性
- Decode 阶段使用单步循环算法，避免重复计算

**缓存状态**:
- `conv_state`: 卷积缓存，形状 `(B, 6144, kernel_size)`
- `recurrent_state`: 循环状态矩阵，形状 `(B, 32, 128, 128)`

---

## 5. Sparse Mixture-of-Experts: 全层 MoE 设计

### 5.1 设计决策

Qwen3.5-35B-A3B 的关键架构决策：**所有 40 层均使用 MoE FFN**，没有 Dense MLP 层。这与 Qwen3Next 不同（Qwen3Next 通过 `mlp_only_layers` 控制哪些层使用 Dense MLP）。

```python
# Qwen3_5MoeTextConfig.__post_init__
def __post_init__(self, **kwargs):
    super().__post_init__(**kwargs)
    del self.mlp_only_layers  # 删除 dense MLP 层列表 → 全 MoE
```

### 5.2 SparseMoeBlock 数据流

```
Input (B, S, 2048)
    │
    ├── Router (TopKRouter)
    │   ┌──────────────────────────────┐
    │   │ router_logits = x @ W.T       │  W: (256, 2048)
    │   │ router_probs = softmax(logits)│
    │   │ topk(probs, k=8)              │
    │   │ weights = topk / sum(topk)    │  归一化
    │   └──────────────────────────────┘
    │       │
    │       8 个专家索引 + 权重
    │       │
    ├── Routed Experts (256 个中选 8 个)
    │   ┌──────────────────────────────┐
    │   │ Expert_i:                     │
    │   │   gate_up: (2048, 2×512)     │  gate + up 合并
    │   │   down:    (512, 2048)       │
    │   │   silu(gate) * up → down    │  SwiGLU
    │   └──────────────────────────────┘
    │       │  Σ(weight_i × Expert_i(x))  ← 8 个专家加权求和
    │       │
    ├── Shared Expert
    │   ┌──────────────────────────────┐
    │   │ SwiGLU FFN:                   │
    │   │   gate_proj: (2048, 512)     │
    │   │   up_proj:   (2048, 512)     │
    │   │   down_proj: (512, 2048)     │
    │   └──────────────────────────────┘
    │       │
    │   shared_gate: (2048, 1) → sigmoid → scale
    │       │  scale × shared_expert_output
    │       │
    └── Final Output:
        routed_output + sigmoid(gate) × shared_output
```

### 5.3 专家网络 (Expert Network)

每个专家是一个 **SwiGLU FFN**：

```python
class Qwen3_5MoeExperts:
    # 所有 256 个专家的参数存储为 3D 张量
    gate_up_proj: Parameter(256, 2×512, 2048)  # 合并的 gate + up
    down_proj:    Parameter(256, 2048, 512)     # 输出投影
```

具体计算：
```
h = x @ gate_up_proj.T          # (B, 256, 2×512)
gate, up = chunk(h, 2, dim=-1)  # gate:(B, 256, 512), up:(B, 256, 512)
output = down_proj(silu(gate) * up)  # (B, 256, 2048)
```

### 5.4 Router 设计

**关键**: Qwen3.5-35B-A3B 的 Router 继承自 `Qwen3VLMoeTextTopKRouter`，始终强制 `norm_topk_prob=True`（与 `Qwen2MoeTopKRouter` 不同，后者默认 `norm_topk_prob=False`）。

```python
class Qwen3_5MoeTopKRouter:
    weight: Parameter(256, 2048)

    def forward(hidden_states):
        # 1. 计算路由 logits
        logits = hidden_states @ weight.T       # (seq_len, 256)

        # 2. Softmax 归一化
        probs = softmax(logits)

        # 3. Top-8 选择
        values, indices = topk(probs, k=8)

        # 4. 强制归一化（始终执行）
        values = values / values.sum(dim=-1, keepdim=True)

        return logits, values, indices
```

### 5.5 共享专家 (Shared Expert)

- 始终激活，为所有 token 提供基础语言能力
- 通过可学习的门控因子 `sigmoid(shared_expert_gate(x))` 控制贡献强度
- 与路由专家输出相加：`final = routed + sigmoid(gate) * shared`

### 5.6 辅助损失 (Auxiliary Loss)

目标：鼓励 Router 均匀分配 token 到各专家，避免专家负载不均。

```python
# 配置
router_aux_loss_coef = 0.001  # 辅助损失系数
```

---

## 6. 3D Multimodal RoPE: 统一多模态位置编码

### 6.1 设计动机

Qwen3.5 是原生多模态模型，需要同时处理文本（1D 序列）、图像（2D 空间）、视频（3D 时空）的 token。传统 RoPE 只能编码 1D 位置，Qwen3.5 引入 **3D MRoPE (Multimodal Rotary Position Embedding)** 来统一编码不同模态的位置信息。

### 6.2 位置编码分片

```python
mrope_section = [11, 11, 10]  # [Temporal, Height, Width]
```

- **T (Temporal)**: 11 个维度，编码时间位置（文本: 序列位置；视频: 帧位置）
- **H (Height)**: 11 个维度，编码空间高度
- **W (Width)**: 10 个维度，编码空间宽度
- RoPE 总维度: `(11+11+10) × 2 = 64`（`head_dim × partial_rotary_factor = 256 × 0.25 = 64`）

### 6.3 Position IDs 结构

```python
position_ids.shape = (4, batch_size, seq_len)
# dim 0: text position_ids  (1D, 文本序列位置)
# dim 1: temporal position_ids  (视频帧位置)
# dim 2: height position_ids  (空间高度)
# dim 3: width position_ids   (空间宽度)
```

### 6.4 Interleaved MRoPE 变换

标准 MRoPE 将频率按 `[TTT...HHH...WWW]` 分块排列。Qwen3.5 使用 **Interleaved MRoPE**，将其重组为交错排列 `[THWTHWTHW...TT]`：

```python
def apply_interleaved_mrope(freqs, mrope_section):
    freqs_t = freqs[0]  # 以 T 维度为基础
    for dim, offset in enumerate((1, 2), start=1):  # H, W
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)  # 每 3 个位置取一个
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t
```

**效果**: 保持频率连续性，避免分块造成的频率不连续问题。

---

## 7. Vision Encoder: 视觉信号的前端处理

### 7.1 架构

```
Image/Video Input
    │
    ▼
Patch Embedding (16×16 patch → 1152 dim)
    │
    ▼
2D RoPE + Bilinear Interpolation Position Encoding
    │
    ▼
┌─────────────────────────────────────┐
│  ViT Blocks × 27                    │
│  ┌───────────────────────────────┐  │
│  │ VisionAttention + FFN         │  │
│  │ (hidden=1152, intermediate=   │  │
│  │  4304, heads=16,              │  │
│  │  activation=gelu_pytorch_tanh)│  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
    │
    ▼
Spatial Merge (2×2 → 1)
    │
    ▼
Output Projection: 1152 → 3584 (vision-language projector)
    │
    ▼
Visual Embeddings (送入 Text Decoder)
```

### 7.2 关键参数

- **ViT 深度**: 27 层
- **Patch 大小**: 16×16（空间），2（时序）
- **空间合并**: 2×2 合并为 1，减少 token 数量
- **输出投影**: 1152 → 3584，与文本空间对齐

---

## 8. Tokenizer 与词表设计

### 8.1 词表规模

Qwen3.5 的词表大小为 **248,320**（已填充至 256 的倍数），远超 Qwen2 系列的 151,936。更大的词表提供了：
- 更高效的 token 编码（更少的 token 表示相同内容）
- 更好的多语言支持（201 种语言和方言）

### 8.2 分词器架构

```
BPE Tokenizer
├── Normalizer: NFC (Unicode 规范化)
├── PreTokenizer: Sequence
│   ├── Split (Regex) ── 处理缩写、字母、数字、标点
│   └── ByteLevel ── 字节级编码
└── Decoder: ByteLevel
```

### 8.3 特殊 Token

| Token | ID | 说明 |
|-------|-----|------|
| `<\|endoftext\|>` | 0 | pad / eos / unk 三合一 |
| `vision_start_token_id` | 248,053 | 视觉序列开始 |
| `vision_end_token_id` | 248,054 | 视觉序列结束 |
| `image_token_id` | 248,056 | 图像占位符 |
| `video_token_id` | 248,057 | 视频占位符 |

---

## 9. Dense vs MoE: 与 Qwen3.5-27B 的架构对比

| 特性 | Qwen3.5-27B (Dense) | Qwen3.5-35B-A3B (MoE) |
|------|---------------------|----------------------|
| **model_type** | `qwen3_5` | `qwen3_5_moe` |
| **总参数量** | 27B | 35B |
| **激活参数量** | 27B（全激活） | ~3B（稀疏激活） |
| **hidden_size** | 4,096 | 2,048 |
| **层数** | 64 | 40 |
| **Hybrid 模式** | 3:1 GDN:Attn | 3:1 GDN:Attn |
| **FFN 类型** | Dense SwiGLU MLP | SparseMoE（256 experts） |
| **FFN intermediate** | 17,408（单层） | 512（每专家） |
| **MoE 层比例** | 0%（全 Dense） | 100%（全 MoE） |
| **KV 头数** | 4 | 2 |
| **GQA 分组比** | 4:1 | 8:1 |
| **继承关系** | Qwen3Next + Qwen3VL | Qwen3Next + Qwen3VL_MoE |

**核心设计权衡**:
- 27B: 更深的网络（64 层）、更宽的隐藏维度（4096），但每层都全量计算
- 35B-A3B: 更浅的网络（40 层）、更窄的隐藏维度（2048），但 MoE 提供 256 个专家的知识容量，实际激活仅 3B

---

## 10. 推理优化策略

### 10.1 加速库

```bash
# 必备加速库
pip install causal-conv1d         # GatedDeltaNet 因果卷积加速
pip install flash-linear-attention # GatedDeltaNet 的 Delta Rule 加速

# 推荐
pip install flash-attn            # 全注意力层加速
```

### 10.2 并行策略

| 并行方式 | 适用场景 | 配置位置 |
|---------|---------|---------|
| Tensor Parallelism | 单层权重切分 | `base_model_tp_plan` |
| Expert Parallelism | MoE 专家分布 | `base_model_ep_plan` |
| Pipeline Parallelism | 层间流水线 | `base_model_pp_plan` |

### 10.3 量化

官方提供 GPTQ 4-bit 量化版本 (`Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`)，可在消费级 GPU 上运行。

### 10.4 缓存策略

| 缓存类型 | 适用层 | 形状 |
|---------|-------|------|
| KV Cache | GatedAttention (10 层) | `(B, 2, S, 256)` |
| Conv State | GatedDeltaNet (30 层) | `(B, 6144, 4)` |
| Recurrent State | GatedDeltaNet (30 层) | `(B, 32, 128, 128)` |

GatedDeltaNet 的循环状态矩阵 `(B, 32, 128, 128)` 在长上下文下保持恒定大小，不随序列长度增长。

---

## 11. 类继承关系全景图

### 11.1 配置类

```
PreTrainedConfig
├── Qwen3NextConfig
│   └── Qwen3_5MoeTextConfig  (model_type="qwen3_5_moe_text")
│       ├── 覆盖: vocab_size=248320, hidden_size=2048
│       ├── 覆盖: num_hidden_layers=40, num_experts=256
│       ├── 删除: mlp_only_layers, intermediate_size
│       └── 删除: decoder_sparse_step, norm_topk_prob
├── Qwen3_5VisionConfig
│   └── Qwen3_5MoeVisionConfig  (model_type="qwen3_5_moe_vision")
└── Qwen3VLConfig
    └── Qwen3_5MoeConfig  (model_type="qwen3_5_moe")
```

### 11.2 模型类

```
PreTrainedModel
└── Qwen3NextPreTrainedModel
    └── Qwen3_5MoePreTrainedModel
        ├── Qwen3_5MoeVisionModel  → Qwen3_5VisionModel → Qwen3VLVisionModel
        ├── Qwen3_5MoeTextModel    → Qwen3_5TextModel → Qwen3NextModel
        ├── Qwen3_5MoeModel        → Qwen3_5Model → Qwen3VLModel
        ├── Qwen3_5MoeForCausalLM  → Qwen3NextForCausalLM
        └── Qwen3_5MoeForConditionalGeneration → Qwen3VLMoeForConditionalGeneration
```

### 11.3 关键组件

```
Qwen3_5MoeGatedDeltaNet
  └── Qwen3_5GatedDeltaNet
      └── Qwen3NextGatedDeltaNet

Qwen3_5MoeAttention
  └── Qwen3NextAttention
      └── Qwen3MoeAttention

Qwen3_5MoeSparseMoeBlock
  └── Qwen3NextSparseMoeBlock
      └── Qwen2MoeSparseMoeBlock
          ├── Router: Qwen3_5MoeTopKRouter → Qwen3VLMoeTextTopKRouter
          ├── Experts: Qwen3_5MoeExperts → Qwen3NextExperts → Qwen2MoeExperts
          └── Shared Expert: Qwen2MoeMLP (SwiGLU)

Qwen3_5MoeRMSNorm
  └── Qwen3NextRMSNorm → Gemma3RMSNorm
```

---

## 12. 关键源码文件索引

| 文件路径 | 说明 |
|---------|------|
| [configuration_qwen3_5_moe.py](file:///d:/transformers/src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py) | MoE 变体配置类（含所有默认参数） |
| [modular_qwen3_5_moe.py](file:///d:/transformers/src/transformers/models/qwen3_5_moe/modular_qwen3_5_moe.py) | MoE 变体模块化源文件 |
| [modeling_qwen3_5_moe.py](file:///d:/transformers/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py) | MoE 变体自动生成模型实现 |
| [configuration_qwen3_5.py](file:///d:/transformers/src/transformers/models/qwen3_5/configuration_qwen3_5.py) | Dense 变体配置类 |
| [modular_qwen3_5.py](file:///d:/transformers/src/transformers/models/qwen3_5/modular_qwen3_5.py) | Dense 变体模块化源文件（含 GatedDeltaNet 覆写） |
| [modeling_qwen3_5.py](file:///d:/transformers/src/transformers/models/qwen3_5/modeling_qwen3_5.py) | Dense 变体自动生成模型实现 |
| [modular_qwen3_next.py](file:///d:/transformers/src/transformers/models/qwen3_next/modular_qwen3_next.py) | Qwen3Next 基础架构 |
| [modeling_qwen2_moe.py](file:///d:/transformers/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py) | MoE 基础组件（SparseMoeBlock、Experts、Router） |
| [modular_qwen3_vl_moe.py](file:///d:/transformers/src/transformers/models/qwen3_vl_moe/modular_qwen3_vl_moe.py) | Qwen3VL-MoE（Router 继承源） |
| [tokenization_qwen3_5.py](file:///d:/transformers/src/transformers/models/qwen3_5/tokenization_qwen3_5.py) | 分词器实现 |
| [model_doc/qwen3_5.md](file:///d:/transformers/docs/source/en/model_doc/qwen3_5.md) | 官方文档 |

---

*文档基于 HuggingFace Transformers 源码分析生成，代码版本与项目 `d:\transformers` 一致。*

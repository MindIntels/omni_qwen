# Qwen VL 模型架构演进：从 Qwen2-VL 到 Qwen3-NeXT

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 模型架构迭代总览](#2-模型架构迭代总览)
- [3. 核心组件详解](#3-核心组件详解)
  - [3.1 RoPE 位置编码演进](#31-rope-位置编码演进)
  - [3.2 Vision Transformer (ViT)](#32-vision-transformer-vit)
  - [3.3 Window Attention](#33-window-attention)
  - [3.4 视觉-语言投影器](#34-视觉-语言投影器)
  - [3.5 Decoder Backbone](#35-decoder-backbone)
  - [3.6 SwiGLU + RMSNorm](#36-swiglu--rmsnorm)
  - [3.7 Mixture-of-Experts (MoE)](#37-mixture-of-experts-moe)
  - [3.8 Gated DeltaNet (线性注意力)](#38-gated-deltanet-线性注意力)
  - [3.9 Gated Attention (门控注意力)](#39-gated-attention-门控注意力)
- [4. 模型组装](#4-模型组装)
  - [4.1 Qwen2-VL](#41-qwen2-vl)
  - [4.2 Qwen2.5-VL](#42-qwen25-vl)
  - [4.3 Qwen3-VL](#43-qwen3-vl)
  - [4.4 Qwen3-NeXT](#44-qwen3-next)
- [5. 架构对比分析](#5-架构对比分析)
- [6. 关键改进优化点](#6-关键改进优化点)
- [7. 目录结构](#7-目录结构)
- [8. 快速开始](#8-快速开始)
- [9. 测试说明](#9-测试说明)

---

## 1. 项目概述

本项目实现了 Qwen VL（Vision-Language）系列模型的**核心架构组件**，覆盖从
Qwen2-VL 到 Qwen3-NeXT 的完整演进链。每个模块均为独立可测试的 PyTorch 实现，
侧重于**架构设计的清晰展示**和**关键创新点的可理解性**。

```
Qwen2-VL ──▶ Qwen2.5-VL ──▶ Qwen3-VL ──▶ Qwen3-NeXT
 (2D-RoPE)    (3D-RoPE)     (MoE FFN)    (DeltaNet+Attn)
```

---

## 2. 模型架构迭代总览

| 特性 | Qwen2-VL | Qwen2.5-VL | Qwen3-VL | Qwen3-NeXT |
|------|----------|------------|----------|------------|
| **ViT RoPE** | 2-D (H, W) | 3-D (T, H, W) | 3-D | 3-D |
| **ViT Attention** | Window + Global | Window + Global | Window + Global | Window + Global |
| **Visual Projector** | 2×2 Merge + MLP | 2×2 Merge + MLP | Merge / Perceiver | Merge + MLP |
| **Decoder Attention** | GQA + M-RoPE | GQA + M-RoPE | GQA + M-RoPE | **Gated DeltaNet** + Gated Attn |
| **Decoder FFN** | Dense SwiGLU | Dense SwiGLU | **Sparse MoE** | Dense SwiGLU |
| **位置编码** | M-RoPE (3-way) | M-RoPE (3-way) | M-RoPE | M-RoPE |
| **推理复杂度** | O(S²d) | O(S²d) | O(S²d) × top_k/E | **O(Sd²)** (DeltaNet层) |
| **代表规模** | 2B / 7B / 72B | 3B / 7B / 72B | 235B-A22B | — |

---

## 3. 核心组件详解

### 3.1 RoPE 位置编码演进

Rotary Position Embedding (RoPE) 是 Qwen 系列的核心位置编码方案。

#### 1-D RoPE（文本 Decoder）

标准旋转嵌入，将 head_dim 按对配对，每对旋转角度 `position × θ_i`：

```
θ_i = base^{-2i / d}

[x_{2i}, x_{2i+1}] → [x_{2i}·cos(mθ) - x_{2i+1}·sin(mθ),
                       x_{2i}·sin(mθ) + x_{2i+1}·cos(mθ)]
```

**优势**: 天然编码相对位置，外推性好，无可训练参数。

#### 2-D RoPE（Qwen2-VL ViT）

将 head_dim 等分为两半：

```
head_dim = [───── height_dim ─────|───── width_dim ─────]
```

- 前半用 **行坐标 h** 旋转
- 后半用 **列坐标 w** 旋转

**优势**: 编码 2-D 空间关系，支持动态分辨率（不同图像尺寸自适应）。

#### 3-D RoPE（Qwen2.5-VL ViT）

将 head_dim 等分为三段：

```
head_dim = [── time_dim ──|── height_dim ──|── width_dim ──]
```

- 第一段用 **帧索引 t** 旋转
- 第二段用 **行坐标 h** 旋转
- 第三段用 **列坐标 w** 旋转

**优势**: 原生支持视频理解，时间维度与空间维度统一编码。

#### M-RoPE（多模态 RoPE，Decoder 使用）

为解码器设计的多模态位置编码。head_dim 分三段，但使用**三路独立位置 ID**：

```
(pos_t, pos_h, pos_w)
```

| 模态 | pos_t | pos_h | pos_w |
|------|-------|-------|-------|
| 文本 | token_pos | token_pos | token_pos |
| 图像 | image_placeholder_pos | patch_row | patch_col |
| 视频 | frame_index | patch_row | patch_col |

**关键洞察**: 对纯文本，三路 ID 退化为相同值 → 等价于标准 1-D RoPE。

### 3.2 Vision Transformer (ViT)

```
Input Image [B, 3, img_H, img_W]
        │
        ▼
┌──────────────────┐
│   PatchEmbed2D   │  Conv2d(3, C, patch_size, stride=patch_size)
│   → [B, N, C]    │  N = (H/p) × (W/p)
└──────────────────┘
        │
        ▼
┌──────────────────┐  ×(num_layers - num_layers/global_every) 次
│  Window ViT Block │  RMSNorm → Window Attention(2D/3D RoPE) → Residual
│  (local attention)│  RMSNorm → SwiGLU MLP → Residual
└──────────────────┘
        │
        ▼  (每 global_every 层)
┌──────────────────┐  ×(num_layers/global_every) 次
│  Global ViT Block │  RMSNorm → Full Attention(2D/3D RoPE) → Residual
│  (全局 attention) │  RMSNorm → SwiGLU MLP → Residual
└──────────────────┘
        │
        ▼
┌──────────────────┐
│     RMSNorm      │
│  → [B, N, C]     │
└──────────────────┘
```

**设计要点**:
- Window Attention 将复杂度从 O(N²) 降到 O(N × w²)，适合高分辨率
- 周期性全局层（如每 4 层一次）保持远距离信息流通
- 无可训练位置嵌入，纯靠 RoPE → 对未见分辨率零样本泛化

### 3.3 Window Attention

```
输入: [B, H×W, C]  (展平的 patch tokens)
                │
    ┌───────────┴───────────┐
    │  Window Partition      │  按 (win_h, win_w) 切分
    │  → [B×nH×nW, w², C]   │  w² = win_h × win_w
    └───────────┬───────────┘
                │
    ┌───────────┴───────────┐
    │  Multi-Head Attention  │  Q, K, V projection + SDPA
    │  (窗口内独立计算)       │  可选 2D/3D RoPE
    └───────────┬───────────┘
                │
    ┌───────────┴───────────┐
    │  Window Un-partition   │  恢复原始空间排列
    │  → [B, H×W, C]        │
    └───────────────────────┘
```

**内存优势**: 对 224×224 图像（14×14 patch = 196 tokens），attn 矩阵从 196² = 38416 降到 4×(49²) = 9604（window=7×7），约 **4× 节省**。对更高分辨率，节省更加显著。

### 3.4 视觉-语言投影器

#### Qwen2-VL / 2.5-VL: 空间合并 + MLP

```
[B, H×W, vit_dim]
      │
      │  reshape → [B, H/2, 2, W/2, 2, C]
      │  merge   → [B, H/2 × W/2, 4C]
      ▼
┌────────────┐
│  Linear    │  4C → llm_dim
│  GELU      │
│  Linear    │  llm_dim → llm_dim
└────────────┘
      │
      ▼
[B, N/4, llm_dim]   ← 视觉 token 数量压缩 4 倍
```

#### Perceiver Projector（高级版本）

使用固定数量的可学习 query tokens 对 ViT 特征做 cross-attention，
实现**分辨率无关**的 token 压缩。

### 3.5 Decoder Backbone

标准 Transformer 解码器，核心改进：

```
x → RMSNorm → GQA Self-Attention (M-RoPE) → + Residual
  → RMSNorm → SwiGLU FFN                  → + Residual → output
```

**GQA (Grouped Query Attention)**: Q heads 分组共享 K/V heads

```
例: num_q_heads=32, num_kv_heads=8
→ 每 4 个 Q heads 共享 1 组 K/V
→ KV cache 大小降为 MHA 的 1/4
```

**KV-Cache 支持**: 自回归生成时，已计算的 K/V 缓存并逐步拼接，避免重复计算。

### 3.6 SwiGLU + RMSNorm

#### SwiGLU FFN

```
FFN(x) = W_down · (SiLU(W_gate · x) ⊙ W_up · x)
```

- **SiLU (Swish)**: `x · σ(x)`，平滑非线性激活
- **门控机制**: `W_gate` 路径控制信息流，`W_up` 路径提供值
- 中间维度通常为 `hidden_size × 8/3`（对齐到 128）

**vs 标准 FFN**: SwiGLU 在相当参数量下精度更优（PaLM, LLaMA 验证）。

#### RMSNorm

```
RMSNorm(x) = x / RMS(x) · γ
RMS(x) = √(mean(x²) + ε)
```

**vs LayerNorm**: 省略均值中心化步骤，计算量略少，在 LLM 中效果等价。

### 3.7 Mixture-of-Experts (MoE)

Qwen3 的 MoE 架构（类 DeepSeek-MoE 设计）：

```
                 x
                 │
        ┌────────┴────────┐
        │                 │
   ┌────▼────┐    ┌───────▼────────┐
   │ Shared  │    │   Top-K Router │
   │ Expert  │    │  → select K of │
   │ (always │    │    N experts   │
   │  active)│    └───────┬────────┘
   └────┬────┘            │
        │         ┌───────▼────────┐
        │         │ Σ w_k·Expert_k │
        │         │  (weighted sum │
        │         │   of top-K)    │
        │         └───────┬────────┘
        └────────┬────────┘
                 │ (add)
                 ▼
              output
```

**关键设计**:

| 特性 | 说明 |
|------|------|
| Shared Expert | 始终激活，提供稠密基线信号，防止 token 被完全丢弃 |
| Top-K Routing | 每个 token 只激活 K 个专家（如 K=8 / N=128） |
| Load Balancing Loss | `L_aux = N · Σ f_e · P_e`，鼓励均匀利用 |
| Expert 结构 | 每个 expert 是独立的 SwiGLU FFN |

**参数效率**: Qwen3-235B 总参 235B，但每次推理只激活约 22B 参数。

### 3.8 Gated DeltaNet (线性注意力)

Qwen3-NeXT 的核心创新 — 用**线性注意力**替代大部分标准注意力层。

#### Delta Rule 原理

标准线性注意力的状态更新：`S_t = S_{t-1} + v_t k_t^T`（只写不删）

Delta Rule 改进：先**擦除**旧关联，再写入新值：

```
S_t = S_{t-1} + β_t · (v_t − S_{t-1}^T k_t) ⊗ k_t

其中:
  S ∈ R^{d_v × d_k}  — 状态矩阵（联想记忆）
  β_t ∈ (0, 1)        — 学习的写入强度门控
  (v_t − S_{t-1}^T k_t) — delta（期望值与当前存储值之差）
```

**直觉**: 类似 Hopfield 网络的误差校正学习规则，能有效覆写旧记忆。

#### Gated DeltaNet 完整架构

```
x ─┬─→ Q_proj → ShortConv → SiLU ─→ Q ─┐
   ├─→ K_proj → ShortConv → SiLU ─→ K  ─┤
   ├─→ V_proj → ShortConv → SiLU ─→ V  ─┤
   ├─→ β_proj → Sigmoid ──────────→ β  ─┤  Delta Rule
   │                                      │  Recurrence
   │                           ┌──────────▼──────────┐
   │                           │  S_t = S_{t-1} +    │
   │                           │  β·(v - S^T·k)⊗k    │
   │                           │  o_t = S_t · q_t    │
   │                           └──────────┬──────────┘
   │                                      │
   └─→ α_proj → Sigmoid ─→ α ──→ α ⊙ o  ─→ GroupNorm → O_proj → output
       (output gate)
```

**复杂度对比**:

| 模式 | 时间复杂度 | 空间复杂度 | 适用场景 |
|------|-----------|-----------|---------|
| 标准 Attention | O(S²d) | O(S²) | 短序列 |
| DeltaNet 递归 | O(Sd²) | O(d²) | 推理 / 长序列 |
| DeltaNet 分块 | O(S·C·d) | O(C² + d²) | 训练（C ≪ S） |

### 3.9 Gated Attention (门控注意力)

```
x → Q/K/V Proj → [optional RoPE] → GQA Scaled Dot-Product Attention
                                              │
                                         GroupNorm
                                              │
x → Gate Proj → Sigmoid → α ───────→ α ⊙ attn_output
                                              │
                                         O_proj → output
```

**为什么需要**: 纯 DeltaNet（线性注意力）在需要**精确检索**的任务上
（如 in-context learning、长距离依赖）会有质量下降。少量全注意力层
（如每 4 层一个）能补充这一能力。

**门控的作用**:
- 每个特征维度可以独立**抑制或放大**注意力输出
- 训练更稳定（类 GRU/LSTM 门控效果）
- 参数开销极小（仅增加一个线性投影）

---

## 4. 模型组装

### 4.1 Qwen2-VL

```
┌─────────────────────────────────────────────────────────────────┐
│                        Qwen2-VL                                 │
│                                                                 │
│  Image ──→ ViT (2-D RoPE, Window Attn) ──→ 2×2 Merger ──┐     │
│                                                           │     │
│  Text ──→ Token Embedding ────────────────────────────────┤     │
│                                                           │     │
│           [ visual_tokens ‖ text_tokens ]                 │     │
│                      │                                    │     │
│                      ▼                                    │     │
│         LLM Decoder (M-RoPE + GQA + SwiGLU) × N          │     │
│                      │                                    │     │
│                      ▼                                    │     │
│                   LM Head → logits                        │     │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Qwen2.5-VL

与 Qwen2-VL 结构相同，关键差异：
- ViT 使用 **3-D RoPE**（支持视频帧）
- 支持 Dynamic FPS 视频采样
- 更大规模预训练数据（OCR、空间定位增强）

### 4.3 Qwen3-VL

```
┌─────────────────────────────────────────────────────────────────┐
│                        Qwen3-VL                                 │
│                                                                 │
│  Image ──→ ViT (3-D RoPE) ──→ Merger ──┐                      │
│  Text  ──→ Embedding ──────────────────┤                       │
│                                         ▼                       │
│         MoE Decoder Block × N:                                  │
│         ┌─────────────────────────────────────┐                │
│         │ RMSNorm → GQA Attention → Residual  │                │
│         │ RMSNorm → **MoE FFN** → Residual    │                │
│         │  (shared expert + top-K routed)      │                │
│         └─────────────────────────────────────┘                │
│                         │                                       │
│                      LM Head                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Qwen3-NeXT

```
┌─────────────────────────────────────────────────────────────────┐
│                       Qwen3-NeXT                                │
│                                                                 │
│  Image ──→ ViT (3-D RoPE) ──→ Merger ──┐                      │
│  Text  ──→ Embedding ──────────────────┤                       │
│                                         ▼                       │
│         Hybrid Decoder × N:                                     │
│                                                                 │
│   Layer 1: [DN] RMSNorm → GatedDeltaNet → Res → SwiGLU → Res  │
│   Layer 2: [DN] RMSNorm → GatedDeltaNet → Res → SwiGLU → Res  │
│   Layer 3: [DN] RMSNorm → GatedDeltaNet → Res → SwiGLU → Res  │
│   Layer 4: [GA] RMSNorm → GatedAttention → Res → SwiGLU → Res │
│   Layer 5: [DN] ...                                             │
│   Layer 6: [DN] ...                                             │
│   Layer 7: [DN] ...                                             │
│   Layer 8: [GA] ...                                             │
│   ...                                                           │
│         DN = Gated DeltaNet (线性, O(Sd²))                      │
│         GA = Gated Attention (二次, O(S²d))                     │
│                         │                                       │
│                      LM Head                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 架构对比分析

### 推理效率

| 模型 | Prefill 复杂度 | Decode 复杂度 (per token) | KV Cache / State |
|------|---------------|--------------------------|------------------|
| Qwen2-VL | O(S²d) | O(Sd) | O(S·d) 线性增长 |
| Qwen2.5-VL | O(S²d) | O(Sd) | O(S·d) 线性增长 |
| Qwen3-VL | O(S²d) | O(Sd) | O(S·d) + MoE routing |
| Qwen3-NeXT | **O(S·C·d + S²d/R)** | **O(d²)** (DeltaNet) + O(Sd/R) | **O(d²)** 常量 (DeltaNet) |

> R = attn_every（全注意力层间隔），C = chunk_size

**Qwen3-NeXT 的关键优势**: DeltaNet 层的状态为固定大小 d² 矩阵，不随序列增长。
这使得超长上下文（128K+）推理成为可能，而无需巨大 KV cache。

### 模型容量 vs 计算成本

```
                    模型容量
                      ▲
  Qwen3-VL (MoE)    │  ★      ← 高容量，但只激活子集
                      │
  Qwen3-NeXT         │    ★   ← 混合架构平衡效率与容量
                      │
  Qwen2.5-VL         │  ★     ← 稠密模型，容量=计算
                      │
  Qwen2-VL            │★
                      └──────────────────▶ 推理成本
```

### 长序列性能

```
latency (log)
    │
    │\
    │ \  Standard Attention O(S²)
    │  \
    │   \
    │    \            Qwen3-NeXT hybrid
    │     ╲ ·  ·  ·  ·  ·  ·  ·
    │      ╲              ↗ O(S·C + S²/R)
    │       ·
    │        ·      DeltaNet O(Sd²)
    │         ·  ·  ·  ·  ·  ·  ·
    └──────────────────────────────▶ S (seq length)
         1K     4K    16K   64K  128K
```

---

## 6. 关键改进优化点

### 6.1 从 Qwen2-VL → Qwen2.5-VL

| 改进 | 技术细节 | 收益 |
|------|---------|------|
| 2D→3D RoPE | ViT head_dim 三分段 (t, h, w) | 原生视频理解，无需额外时序模块 |
| Dynamic FPS | 时间位置 ID 按实际时间戳分配 | 适应不同帧率的视频 |
| 更多预训练数据 | OCR、Grounding、Agent 数据增强 | 文档/图表理解大幅提升 |

### 6.2 从 Qwen2.5-VL → Qwen3-VL

| 改进 | 技术细节 | 收益 |
|------|---------|------|
| Dense→MoE FFN | Shared Expert + Top-K Routing | 参数量 10×↑ 但计算量仅 ~2×↑ |
| Load Balancing | L_aux = N·Σ(fₑ·Pₑ) | 防止专家坍缩，提高利用率 |
| Thinking Mode | `<think>` token 触发 CoT | 复杂推理能力增强 |

### 6.3 从 Qwen3-VL → Qwen3-NeXT

| 改进 | 技术细节 | 收益 |
|------|---------|------|
| Delta Rule 替代线性注意力 | S += β·(v − S^T·k)⊗k | 联想记忆有效覆写，避免容量饱和 |
| 混合架构 | 大部分层 DeltaNet + 少量全注意力 | 线性效率 + 保留精确检索能力 |
| Output Gating | α = σ(Wα·x) 应用于输出 | 训练稳定性 + 逐特征选择性抑制 |
| Short Convolution | Depthwise Conv1D (kernel=4) | 本地特征混合，类 Mamba/RWKV |
| GroupNorm 替代 Softmax | 对 DeltaNet 输出做 per-head norm | 稳定线性注意力的数值范围 |
| Chunkwise 并行 | 序列分块训练，块内并行 | 训练效率接近标准 Attention |

### 6.4 整体技术趋势

```
Qwen2-VL    →    Qwen2.5-VL    →    Qwen3-VL    →    Qwen3-NeXT
   │                  │                  │                  │
   │  空间编码增强    │   多模态增强     │  容量扩展       │  效率革新
   │  2D→位置泛化    │   3D→时序建模    │  MoE→稀疏化     │  线性注意力
   │                  │                  │                  │
   └──── 感知能力 ─── ┴── 理解深度 ──── ┴── 知识容量 ──── ┘
                                                推理效率 ──┘
```

---

## 7. 目录结构

```
omni_qwen/
├── README.md                    # 本文档
├── run_tests.sh                 # 一键运行全部测试
├── __init__.py                  # 包入口
│
│  ── 基础组件 ──
├── rope.py                      # 1-D / 2-D / 3-D / M-RoPE 位置编码
├── rms_norm.py                  # RMSNorm
├── swiglu_mlp.py                # SwiGLU FFN
│
│  ── 视觉模块 ──
├── window_attention.py          # 窗口注意力 (ViT 用)
├── vit.py                       # Vision Transformer (PatchEmbed + ViT blocks)
├── visual_projector.py          # 视觉→语言投影器 (Merger / Perceiver)
│
│  ── 语言模块 ──
├── decoder.py                   # LLM Decoder (GQA + KV-cache)
├── moe.py                       # MoE 层 (Shared + Routed experts)
├── gated_deltanet.py            # Gated DeltaNet 线性注意力
├── gated_attention.py           # Gated Self-Attention 门控注意力
│
│  ── 模型组装 ──
├── qwen2_vl.py                  # Qwen2-VL (2D-RoPE ViT + Dense Decoder)
├── qwen25_vl.py                 # Qwen2.5-VL (3D-RoPE ViT + Dense Decoder)
├── qwen3_vl.py                  # Qwen3-VL (3D-RoPE ViT + MoE Decoder)
├── qwen3_next.py                # Qwen3-NeXT (3D-RoPE ViT + Hybrid Decoder)
│
│  ── 测试 ──
└── tests/
    ├── __init__.py
    ├── test_rope.py              # RoPE 所有变体测试 (15 tests)
    ├── test_window_attention.py  # 窗口注意力测试 (7 tests)
    ├── test_vit.py               # ViT 测试 (8 tests)
    ├── test_decoder.py           # Decoder 测试 (10 tests)
    ├── test_moe.py               # MoE 测试 (10 tests)
    ├── test_gated_deltanet.py    # Gated DeltaNet 测试 (11 tests)
    ├── test_gated_attention.py   # Gated Attention 测试 (7 tests)
    └── test_models.py            # 模型集成测试 (16 tests)
```

---

## 8. 快速开始

### 环境要求

- Python >= 3.10
- PyTorch >= 2.0

### 安装依赖

```bash
cd llm
pip install -r requirements.txt
```

### 运行测试

```bash
# 运行全部 omni_qwen 测试
bash omni_qwen/run_tests.sh

# 运行单个模块测试
python -m pytest omni_qwen/tests/test_rope.py -v
python -m pytest omni_qwen/tests/test_gated_deltanet.py -v
python -m pytest omni_qwen/tests/test_models.py -v
```

### 基本使用

```python
import torch
from omni_qwen.qwen3_next import Qwen3NeXT

# 创建小型模型
model = Qwen3NeXT(
    vit_hidden=64, llm_hidden=96, llm_layers=8,
    num_q_heads=4, num_kv_heads=2,
    intermediate_size=192, vocab_size=256,
    attn_every=4,
)

# 纯文本
ids = torch.randint(0, 256, (1, 32))
logits = model(input_ids=ids)
print(logits.shape)  # [1, 32, 256]

# 多模态 (图像 + 文本)
img = torch.randn(1, 3, 28, 28)
logits = model(input_ids=ids, pixel_values=img)
```

---

## 9. 测试说明

| 测试文件 | 覆盖内容 | 测试数 |
|---------|---------|--------|
| test_rope.py | 1D/2D/3D RoPE 形状、边界、正交性；M-RoPE 退化等价 | 15 |
| test_window_attention.py | 分窗/恢复往返、padding、多窗口大小、梯度 | 7 |
| test_vit.py | PatchEmbed 形状、ViTBlock、完整 ViT 梯度、3D-RoPE | 8 |
| test_decoder.py | GQA KV-cache、因果掩码、RoPE 集成、Backbone 梯度 | 10 |
| test_moe.py | Router top-K、权重归一、负载均衡 loss、shared expert | 10 |
| test_gated_deltanet.py | 递归/分块一致性、状态更新、增量推理、变长序列 | 11 |
| test_gated_attention.py | 输出形状、KV-cache、门控效应、因果掩码 | 7 |
| test_models.py | 四模型全流程：纯文本/纯图/多模态、梯度、层调度 | 16 |
| **总计** | | **~84** |

每个测试均可独立运行，也支持 `pytest-xdist` 并行化。

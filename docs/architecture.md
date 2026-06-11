# Architecture

ST-LaneNet is a dual-branch lane detection network operating at 368×640 resolution. One branch proposes lane edges in bird's-eye view via Inverse Perspective Mapping; the other extracts global context via a Swin Transformer. Their feature maps are concatenated and decoded to a binary lane mask.

---

## System Overview

```
          Image 3×368×640
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
  Branch 1 (BEV)   Branch 2 (front-view)
        │               │
   IPM warp (H)    Swin-Tiny (4 stages)
   3×368×640       768×12×20
        │               │
  SpatialPriorHead  loc_reduce 768→128
   1×368×640        128×12×20
        │               │
   EdgeEncoder      bilinear upsample
  [1ch Seq|3ch Par] 128×368×640
   256×46×80             │
        │               │
   CARAFE ×3            │
   64×368×640            │
        │               │
   warp (H⁻¹)           │
   64×368×640            │
        │               │
        └───────┬───────┘
                │
         concat 192×368×640
                │
        Conv 3×3 + Conv 1×1
                │
         Lane mask 1×368×640
```

The two variants (`STLaneNet_Seq` and `STLaneNet_Par`) differ only in where the binary prior mask is inserted into the edge branch — see [Par vs Seq](#par-vs-seq-variants) below.

---

## Branch 1 — Edge Proposal

### IPM Transform

A fixed homography $H \in \mathbb{R}^{3\times3}$ projects the front-view image to bird's-eye view:

$$\tilde{\mathbf{x}}' \sim H\,\tilde{\mathbf{x}}$$

where $\tilde{\mathbf{x}} = (u,v,1)^\top$ is a homogeneous front-view pixel and $\tilde{\mathbf{x}}'$ is its top-down position. $H$ is derived from four manually calibrated control points for the TuSimple camera geometry:

| Point | Role | Front-view $(u,v)$ | Top-down $(u',v')$ |
|-------|------|--------------------|---------------------|
| 1 | Left horizon corner | (280, 200) | (0, 0) |
| 2 | Right horizon corner | (360, 200) | (640, 0) |
| 3 | Bottom-right | (640, 368) | (640, 368) |
| 4 | Bottom-left | (0, 368) | (0, 368) |

$H^{-1}$ reprojects edge features back to front-view before fusion.

> **`.detach()` is mandatory** on the IPM output: `x_top_down = warp(x, H).detach()`. Kornia attempts to differentiate through $H$ during backward, but $H$ is a fixed buffer with no defined Jacobian in this context — this causes `cudaErrorInvalidArgument`. Detaching cuts the graph at the warp; no gradient quality is lost since $H$ has no learnable parameters. See [Training](training.md#detach-constraint) for details.

---

### SpatialPriorHead

Takes the top-down image (3×368×640) and produces a binary prior mask (1×368×640).

| Layer | Operation | Input shape | Output shape |
|-------|-----------|-------------|--------------|
| `init_conv[0]` | Conv 7×7, s=2, BN, ReLU | 3×368×640 | 64×184×320 |
| `init_conv[3]` | MaxPool 3×3, s=2 | 64×184×320 | 64×92×160 |
| `init_conv[4]` | Conv 3×3, BN, ReLU | 64×92×160 | 256×92×160 |
| `DenseBlock` | Conv 3×3 + concat | 256×92×160 | 320×92×160 |
| `ImprovementBlock` | Conv 3×3 + residual | 320×92×160 | 64×92×160 |
| `final_conv` | Conv 3×3 → Conv 1×1 → bilinear upsample | 64×92×160 | 1×368×640 |

The Dense Block concatenates new feature maps with the input (à la DenseNet), increasing representational capacity. The Improvement Block uses a residual connection to refine quality without parameter overhead.

---

### EdgeEncoder

Replaces standard convolution with **Depthwise Separable Convolution (DSConv)** to reduce compute cost by ~9×:

**DSConv cost ratio:**

$$\rho = \frac{D_K^2 \cdot M + M \cdot N}{D_K^2 \cdot M \cdot N} = \frac{1}{N} + \frac{1}{D_K^2}$$

For $D_K=3$, $N=64$: $\rho \approx 1/9$.

**Dilated convolution** expands the receptive field without extra parameters:

$$k_{\text{eff}} = d\,(k-1) + 1$$

The encoder uses $d \in \{1,2,4\}$ for receptive fields of 3, 5, 9 px — covering multi-scale context.

| Layer | Operation | Dilation | Stride | Input | Output |
|-------|-----------|----------|--------|-------|--------|
| layer1 | DSConv | 1 | 2 | 1/3×368×640 | 64×184×320 |
| layer2 | DSConv | 2 | 2 | 64×184×320 | 128×92×160 |
| layer3 | DSConv | 4 | 2 | 128×92×160 | 256×46×80 |

*Input channels: 1 for Seq variant, 3 for Par variant.*

---

### EdgeDecoder (CARAFE)

CARAFE (Content-Aware Reassembly of Features) replaces transposed convolution for upsampling. It dynamically predicts a content-specific reassembly kernel at each spatial position:

$$\hat{X}_s = \sum_{(\delta_x,\delta_y)\,\in\,\mathcal{R}} W_s(\delta_x,\delta_y) \cdot X_{\lfloor s/\sigma \rfloor + (\delta_x,\delta_y)}$$

where $W_s$ is the predicted kernel at position $s$, $\sigma$ is the upsampling factor, and $\mathcal{R}$ is the local window. A residual connection prevents gradient vanishing.

| Layer | Operation | Input | Output |
|-------|-----------|-------|--------|
| carafe1 + skip $x_2$ | CARAFE ×2 | 256×46×80 | 128×92×160 |
| carafe2 + skip $x_1$ | CARAFE ×2 | 128×92×160 | 64×184×320 |
| carafe3 | CARAFE ×2 | 64×184×320 | 64×368×640 |
| IPM⁻¹ warp | `warp_perspective(H⁻¹)` | 64×368×640 | 64×368×640 |

---

## Branch 2 — Swin Transformer Localization

ST-LaneNet uses `swin_tiny_patch4_window7_224` with `features_only=True`, adapted for input size 368×640. The backbone is trained **from scratch** (no ImageNet pretrain).

### Patch Partition and Linear Embedding

The input image is tiled into non-overlapping 4×4 patches and projected to dimension $C=96$. This produces $92\times160$ tokens at stage 1. Subsequent stages use **Patch Merging** (2×2 neighbourhood merge, spatial resolution ÷2, channels ×2).

### W-MSA and SW-MSA

Standard ViT self-attention has quadratic complexity in the number of tokens:

$$\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C$$

Swin replaces this with window-local attention (W-MSA), reducing the quadratic term:

$$\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2 hw C$$

For window size $M=7$ and resolution $92\times160$, the reduction factor is $(92\times160)/49 \approx 300$.

Alternating W-MSA and Shifted-Window MSA (SW-MSA) blocks:
- **W-MSA**: attention within fixed local windows → linear complexity.
- **SW-MSA**: windows shifted by $\lfloor M/2 \rfloor$ → cross-window information flow.

Attention formula with relative position bias $B$:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + B\right)V$$

where $B$ is drawn from a learnable table of size $(2M-1)\times(2M-1)$.

### Stage Hierarchy

```
  Image 3×368×640
        │
        ▼
  ┌──────────────────────────────┐
  │  Patch Partition             │
  │  4×4 tiles, embed C=96       │  →  96×92×160
  └──────────────┬───────────────┘
                 │
                 ▼
  ┌──────────────────────────────┐
  │  Stage 1                     │
  │  2×(W-MSA → SW-MSA)          │  →  96×92×160
  └──────────────┬───────────────┘
                 │  Patch Merging  (÷2 spatial, ×2 channels)
                 ▼
  ┌──────────────────────────────┐
  │  Stage 2                     │
  │  2×(W-MSA → SW-MSA)          │  →  192×46×80
  └──────────────┬───────────────┘
                 │  Patch Merging
                 ▼
  ┌──────────────────────────────┐
  │  Stage 3                     │
  │  6×(W-MSA → SW-MSA)          │  →  384×23×40
  └──────────────┬───────────────┘
                 │  Patch Merging
                 ▼
  ┌──────────────────────────────┐
  │  Stage 4                     │
  │  2×(W-MSA → SW-MSA)          │  →  768×12×20
  └──────────────┬───────────────┘
                 │
                 ▼
  ┌──────────────────────────────┐
  │  loc_reduce                  │
  │  Conv 1×1, BN, ReLU          │  →  128×12×20
  │  768 → 128 channels          │
  └──────────────┬───────────────┘
                 │  bilinear upsample
                 ▼
          128×368×640  →  to fusion head
```

> **Why 768→128?** Upsampling 768×12×20 directly to 768×368×640 at batch=32 produces ~7.2B elements, overflowing CUDA 32-bit kernel indexing. At 128 channels: ~1.5B elements (safe).

---

## Fusion Head

```
concat([edge_features 64ch, loc_features 128ch])  →  192×368×640
Conv 3×3 + BN + ReLU                              →  256×368×640
Conv 1×1                                           →  1×368×640  (lane logit)
```

Returns `(out, prior_logits)` where `prior_logits` is the SpatialPriorHead output (used for auxiliary loss during training).

---

## Par vs Seq Variants

The two variants share all components. They differ only in the position of the binary prior mask in the edge branch.

### STLaneNet_Seq (Sequential)

The binary mask is the **input** to the EdgeEncoder. Data flow is strictly linear.

### STLaneNet_Par (Parallel)

The EdgeEncoder runs in parallel with the SpatialPriorHead. The binary mask is applied **after encoding** as a spatial attention gate (element-wise multiply ⊗).

### Side-by-Side Data Flow

```
       STLaneNet_Seq              |        STLaneNet_Par
       ─────────────              |        ─────────────
                                  |
   IPM top-down (3ch)             |    IPM top-down (3ch)
           │                      |         ┌─────┴─────┐
           ▼                      |         │           │
   SpatialPriorHead               |   EdgeEncoder  SpatialPriorHead
           │                      |   (3ch input)       │
           ▼                      |         │      binary mask (1ch)
   binary mask (1ch)              |         └──────  ⊗  ──────┘
           │                      |                  │
           ▼                      |                  ▼
   EdgeEncoder (1ch input)        |               CARAFE
           │                      |                  │
           ▼                      |                  ▼
        CARAFE                    |            warp (H⁻¹)
           │                      |                  │
           ▼                      |                  ▼
      warp (H⁻¹)                  |        edge features
           │                      |        (64×368×640)
           ▼                      |
     edge features                |
     (64×368×640)                 |
```

The only structural difference is the position of the binary mask: Seq feeds it as input to the encoder; Par feeds it as a post-encoding attention mask.

### Comparison Table

| Component | STLaneNet_Seq | STLaneNet_Par |
|-----------|--------------|--------------|
| EdgeEncoder input channels | 1 (binary mask) | 3 (IPM RGB) |
| Role of binary mask | Encoder input | Post-encoding spatial attention (⊗) |
| Feature richness | Low (binary) | High (texture + gradients) |
| First-layer compute | Low | Higher |
| FPS (measured) | **162.9** | 130.2 |
| F1 (validation) | **83.82%** | 83.63% |

Parameter difference: **146 parameters** (only the first DSConv layer differs). The 25% FPS gap is purely computational (3ch vs 1ch convolution), not parametric.

---

## Full Layer-by-Layer Table

All dimensions for batch=1 at 368×640.

| Module | Operation | Input | Output |
|--------|-----------|-------|--------|
| **Edge Branch — top-down** | | | |
| IPM warp | Homography $H$ (fixed) | 3×368×640 | 3×368×640 |
| `init_conv[0]` | Conv 7×7, s=2, BN, ReLU | 3×368×640 | 64×184×320 |
| `init_conv[3]` | MaxPool 3×3, s=2 | 64×184×320 | 64×92×160 |
| `init_conv[4]` | Conv 3×3, BN, ReLU | 64×92×160 | 256×92×160 |
| DenseBlock | Conv 3×3 + concat | 256×92×160 | 320×92×160 |
| ImprovementBlock | Conv 3×3 + residual | 320×92×160 | 64×92×160 |
| `final_conv` | Conv 3×3 → Conv 1×1 → upsample | 64×92×160 | 1×368×640 |
| layer1 | DSConv, d=1, s=2 | 1/3×368×640 | 64×184×320 |
| layer2 | DSConv, d=2, s=2 | 64×184×320 | 128×92×160 |
| layer3 | DSConv, d=4, s=2 | 128×92×160 | 256×46×80 |
| carafe1 + $x_2$ | CARAFE ×2 + skip | 256×46×80 | 128×92×160 |
| carafe2 + $x_1$ | CARAFE ×2 + skip | 128×92×160 | 64×184×320 |
| carafe3 | CARAFE ×2 | 64×184×320 | 64×368×640 |
| IPM⁻¹ warp | Homography $H^{-1}$ (fixed) | 64×368×640 | 64×368×640 |
| **Localization Branch — front-view** | | | |
| Patch Partition | 4×4 tiles, C=96 | 3×368×640 | 96×92×160 |
| Stage 1 | 2×(W-MSA→SW-MSA) | 96×92×160 | 96×92×160 |
| Merging + Stage 2 | 2×(W-MSA→SW-MSA) | 96×92×160 | 192×46×80 |
| Merging + Stage 3 | 6×(W-MSA→SW-MSA) | 192×46×80 | 384×23×40 |
| Merging + Stage 4 | 2×(W-MSA→SW-MSA) | 384×23×40 | 768×12×20 |
| `loc_reduce` | Conv 1×1, BN, ReLU | 768×12×20 | 128×12×20 |
| Bilinear upsample | — | 128×12×20 | 128×368×640 |
| **Fusion Head** | | | |
| Concat | channel | 64+128 ch | 192×368×640 |
| `final_conv[0]` | Conv 3×3, BN, ReLU | 192×368×640 | 256×368×640 |
| `final_conv[3]` | Conv 1×1 | 256×368×640 | 1×368×640 |

---

## Parameter Count

| Module | Parameters | % total |
|--------|-----------|---------|
| Swin-Tiny backbone (4 stages) | ~28,300K | 95.4% |
| SpatialPriorHead | ~527K | 1.8% |
| Fusion head (`final_conv`) | ~443K | 1.5% |
| EdgeDecoder (3× CARAFE) | ~243K | 0.8% |
| `loc_reduce` (768→128) | ~99K | 0.3% |
| EdgeEncoder | ~44K | 0.1% |
| **Total (Seq / Par)** | **~29,656K** | 100% |

The Swin-Tiny backbone accounts for ~95% of all parameters. The near-identical parameter counts of Seq and Par confirm that the 25% FPS gap is purely computational.

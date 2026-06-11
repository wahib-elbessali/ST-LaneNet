# Training Strategy

This document covers the loss function, hyperparameters, dataset preparation, and implementation-specific decisions that deviate from the original paper.

---

## Loss Function: Focal Loss

### Class Imbalance Problem

In lane segmentation, lane pixels represent fewer than 2% of the image (≈3,000 pixels out of 235,520). Standard cross-entropy loss is dominated by the overwhelming majority of background pixels, preventing the network from learning minority lane features.

### From Cross-Entropy to Focal Loss

Binary cross-entropy:

$$\text{CE}(p, y) = \begin{cases} -\log(p) & y = 1 \\ -\log(1-p) & y = 0 \end{cases}$$

With $p_t$ denoting the predicted probability for the correct class:

$$\text{CE}(p_t) = -\log(p_t)$$

**Step 1 — Class weighting ($\alpha_t$):** Upweight positive (lane) pixels:

$$\text{CE}(p_t) = -\alpha_t \log(p_t), \quad \alpha_t = \begin{cases} \alpha & y=1 \\ 1-\alpha & y=0 \end{cases}$$

**Step 2 — Hard-example focusing ($(1-p_t)^\gamma$):** Reduces loss contribution from easy-to-classify pixels, forcing the model to focus on ambiguous or difficult ones:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

When $\gamma \to 0$, FL reduces to weighted CE. When $p_t \approx 1$ (confident correct prediction), the factor $(1-p_t)^\gamma \to 0$, suppressing that pixel's gradient.

### Total Loss

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{FL}} + 0.4 \cdot \mathcal{L}_{\text{prior}}$$

$\mathcal{L}_{\text{prior}}$ is the Focal Loss on the SpatialPriorHead output (top-down binary mask). Weight $\lambda = 0.4$ chosen to guide the prior branch without dominating the main loss:
- $\lambda > 1$: model optimises bird's-eye view at the expense of the final mask.
- $\lambda < 0.1$: prior branch becomes redundant and slows edge-branch convergence.

### Hyperparameters

```
alpha = 0.75   (positive class weight)
gamma = 2.0    (focus factor)
prior_weight = 0.4
```

---

## Dataset: TuSimple

72,000 front-view highway images, JSON-Lines annotations with per-lane $x$-coordinates at fixed $y$ positions (`h_samples`, rows 240–710 px, step 10). Images resized to 368×640.

**Train/val split:** 80% train (~57,600), 20% validation (~14,400). Our subset uses ~3,200 samples per epoch at batch=32 → ~81 batches/epoch.

### Augmentation (training only)

| Transform | Parameter |
|-----------|-----------|
| Rotation | ±10° |
| Translate | ±5% (horizontal + vertical) |
| Horizontal flip | p=0.5 |
| Brightness/contrast | p=0.2 |
| Normalize | ImageNet mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) |

### Ground-Truth Mask Generation

The model requires two GT masks per training sample — one per branch and one per loss term.

**Front-view mask $M_{GT}$:** TuSimple polylines drawn at thickness=5px on a 368×640 black canvas. Points with $x = -2$ (absent) are skipped.

**Top-down mask $M^{\text{IPM}}_{GT}$:** The front-view mask is warped by $H$ then thresholded:

$$M^{\text{IPM}}_{GT} = \mathbf{1}\!\left[H \cdot M_{GT} > 0.5\right]$$

The 0.5 threshold is mandatory. Bilinear interpolation during warping produces fractional values (e.g., 0.7); direct integer truncation (`.long()`) would zero out most lane pixels in the GT mask, silently destroying the prior supervision signal.

---

## Optimizer and Scheduler

The paper uses SGD with momentum=0.9 and a polynomial decay learning rate schedule.

**Our implementation:** AdamW — chosen for faster, more stable convergence when training from scratch (no pretrained Swin weights). SGD requires careful warm-up tuning; AdamW converges reliably without it.

| Hyperparameter | Paper | Our Implementation |
|----------------|-------|--------------------|
| Optimizer | SGD, momentum=0.9 | AdamW |
| Learning rate | 0.1 + polynomial decay | 5×10⁻⁴ + PolynomialLR(power=0.9) |
| Weight decay | 0.005 | 0.005 |
| Batch size | 32 | 32 |
| Epochs | 80,000 steps (~988 epochs) | 50 epochs (~4,050 steps) |
| Swin pretrain | Not specified | From scratch |

PolynomialLR decays the learning rate smoothly to zero over `total_epochs`. **Do not increase `total_epochs` without updating `total_iters` in the scheduler call** — the LR collapses to zero at epoch 50 if `total_iters` is not updated.

---

## Implementation Constraints

### `.detach()` on IPM Output

```python
x_top_down = warp_perspective(x_front, H).detach()
```

Kornia's `warp_perspective` attempts to compute $\partial H / \partial H$ during backward. Since $H$ is a fixed registered buffer (not a learned parameter), its Jacobian is undefined, causing `cudaErrorInvalidArgument` on the GPU.

`.detach()` cuts the computational graph at the warp operation. Zero impact on training quality — $H$ has no learnable parameters and requires no gradient update.

---

### bfloat16, Not float16

Training uses `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` for the forward pass and loss computation, with float32 parameters updated during the backward pass. **No GradScaler is used.**

| Property | float16 | bfloat16 |
|----------|---------|---------|
| Mantissa bits | 10 | 7 |
| Exponent bits | 5 | 8 |
| Minimum value | ~6×10⁻⁵ | ~1.2×10⁻³⁸ |
| Dynamic range | ±65504 | Same as float32 |

In Focal Loss, the modulation term $(1-p_t)^\gamma$ reaches ~10⁻⁴ for well-classified pixels ($p_t \approx 0.99$, $\gamma=2$). This value is **below the float16 minimum** (~6×10⁻⁵), causing underflow to zero — silently zeroing the gradient for all confident predictions, catastrophically impairing learning.

bfloat16 has float32 dynamic range (~10⁻³⁸), eliminating this underflow. The tradeoff is 3 fewer mantissa bits (lower precision), but precision is not the bottleneck here.

---

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
```

Applied at every optimizer step:

$$\hat{g} = g \cdot \min\!\left(1,\; \frac{5.0}{\|g\|_2}\right)$$

Prevents gradient explosions in the early epochs when the Swin Transformer starts from random initialisation. Without clipping, the large untrained Swin weights produce extreme gradient magnitudes that destabilise training within the first 3–5 epochs.

---

## Infrastructure and Training Time

| Platform | GPU | VRAM | Time per epoch |
|----------|-----|------|----------------|
| Google Colab (free) | NVIDIA T4 | 12 GB | >45 min |
| Kaggle (free) | NVIDIA T4 | 12 GB | >45 min |
| Marimo (free) | RTX 6000 Blackwell | 96 GB | ~4 min |

**At 45 min/epoch on T4:** 80,000 steps (~988 epochs) would require >45 days of continuous compute. Colab and Kaggle are not viable for this scale.

**Marimo provides the RTX 6000 Blackwell for free**, reducing each epoch to ~4 min. However, Marimo's session timeout interrupts training after 3–4 epochs. The workaround: checkpoints are saved after each epoch (`STLaneNet_{Par,Seq}_epoch_{N}.pth`) and the best checkpoint is saved as `STLaneNet_{Par,Seq}_best.pth`. Training resumes on a new session via the `_resume()` helper in Cell 8.

**Total training cost:** 2 variants × 50 epochs × ~4 min = ~400 min (~7h) of effective GPU compute, spread across multiple Marimo sessions.

Both variants are backed up to Kaggle after each session via `auto_backup_to_kaggle()`.

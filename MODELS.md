# Models

Architecture and training configuration for the six classifiers compared in the
[README](README.md). All models classify the same 14 classes of the Skin-DS
dataset (see [DATASET.md](DATASET.md)). Two are custom CNNs. Four are KAN-ViT
hybrids developed in sequence. Accuracy figures below are best validation
accuracy from the training logs. Held-out test accuracies (with test-time
augmentation for the CNN) are noted where they differ.

## Shared KAN building blocks

The KAN-ViT hybrids reuse the same components, defined inline in their notebooks
(efficient_kan style, so no external KAN package is needed).

- `KANLinear` is a linear layer whose weights are learned B-splines over a fixed
  grid, plus a residual base transform. It replaces a standard `nn.Linear` with a
  learnable activation on each edge.
- `EfficientKAN` stacks `KANLinear` layers into a small feed-forward network,
  used in place of a transformer block's MLP.
- `KANViTBlock` keeps timm multi-head self-attention, but its feedforward is
  either `EfficientKAN` (spline) or a standard MLP. The two full-KAN models use
  `EfficientKAN` in every block. The two alternating models toggle per block with
  `use_kan = (i % 2 == 0)`, so blocks 0 and 2 are KAN and blocks 1 and 3 are MLP.
- `EfficientKANViT` wraps a pretrained EfficientNet backbone in `features_only`
  mode, projects its final feature map to `embed_dim` with a 1x1 convolution,
  flattens the spatial grid into tokens, adds a learned positional embedding,
  runs `depth` `KANViTBlock`s, pools, and applies a linear head to `num_classes`.
  The B3 model detects the backbone channel count and token grid with a dummy
  forward pass (384 channels, 7x7 = 49 tokens) rather than hardcoding them.

`create_kan_vit` builds an `EfficientKANViT`. Its `num_classes` argument defaults
to 7 but is always called with 14. Use 14.

## Training configuration

Common to every run: `AdamW`, cross-entropy loss (except Custom CNN V2, which
uses focal loss), ImageNet normalization, `RandomResizedCrop` with horizontal and
vertical flips, and a `WeightedRandomSampler` to counter class imbalance. Batch
size is 32. `mlp_ratio` is 4.0 for the full-KAN hybrids and 3.0 for the
alternating KAN-MLP hybrids (hidden dim 1152 instead of 3072).

| Model | Input | Backbone | Feedforward | embed_dim / depth / heads | Loss | Epochs | Val acc / test acc |
|-------|-------|----------|-------------|---------------------------|------|--------|--------------------|
| Custom CNN V1 | 224 | from scratch | n/a | n/a | CrossEntropy (+ TTA at eval) | 100 | 87.19% / 87.7% |
| Custom CNN V2 | 224 | from scratch | n/a | n/a | Focal | 100 | 86.72% / — |
| KAN-ViT | 160 | timm ViT, pretrained | full KAN | mlp_ratio 4.0 | CrossEntropy | 30 | 80.03% / 79.62% |
| EfficientNetB0-KANViT | 160 | EfficientNet-B0, frozen | full KAN | 768 / 4 / 8 | CrossEntropy | 35 | 81.99% / 83.01% |
| EfficientNetB0-KANViT-MLP | 160 | EfficientNet-B0, last blocks fine-tuned | alternating KAN-MLP | 384 / 4 / 8 | CrossEntropy | 25 | 81.39% / 82.46% |
| EfficientNetB3-KANViT-MLP | 224 | EfficientNet-B3, fine-tuned | alternating KAN-MLP | 384 / 4 / 8 | CrossEntropy | 50 | 84.40% / 85.44% |

Optimizer detail: Custom CNN V1/V2 use AdamW lr 1e-3, wd 1e-4, ReduceLROnPlateau
(factor 0.5, patience 5). KAN-ViT uses AdamW lr 1e-4 then a short lr 1e-5
fine-tune. The EfficientNet hybrids use AdamW lr 1e-4. The alternating models use
`CosineAnnealingWarmRestarts`.

## Custom CNN V1 (SkinMaster)

`SkinLesionCNN`, trained from random initialization at 224px, built from five
custom layers: `SqueezeExcitation` (channel attention, embedded in every conv
block), `SkinLesionConvBlock` (depthwise separable conv + BatchNorm + GELU + SE +
spatial dropout), `ResidualSkinBlock` (stacked conv blocks with a skip
connection), a multi-scale feature block, and a KAN-inspired
`MultiHeadSelfAttention` layer (8 heads). A two-layer head (Linear to 256, then
Linear to 14) produces the logits. Trained with cross-entropy and
`ReduceLROnPlateau`, and evaluated with test-time augmentation. Best validation
accuracy 87.19% at epoch 92, and 87.7% test accuracy with TTA, the strongest
model overall.

## Custom CNN V2

Identical architecture and hyperparameters to V1, with focal loss instead of
cross-entropy to weight hard and rare examples more heavily. It did not beat V1:
best validation accuracy 86.72% at epoch 81.

## KAN-ViT

The first hybrid: a plain timm vision transformer (pretrained) at 160px with its
block MLPs fully replaced by KAN spline layers (`mlp_ratio` 4.0). Trained with
cross-entropy, AdamW lr 1e-4, then a short lr 1e-5 fine-tune, 30 epochs. Best
validation accuracy 80.03% at epoch 28 (79.62% test). The near-zero
generalization gap indicates underfitting.

## EfficientNetB0-KANViT

The plain ViT backbone swapped for a frozen `efficientnet_b0` in `features_only`
mode, still full-KAN feedforward, at 160px. Tokens are projected to `embed_dim`
768 through 4 `KANViTBlock`s (8 heads). Best validation accuracy 81.99% at epoch
32 (83.01% test). The stronger convolutional features lift accuracy over the
plain-ViT hybrid, confirming the gain came from the backbone, not the
transformer. This is the largest checkpoint (772 MB), driven by the 768-dim
tokens.

## EfficientNetB0-KANViT-MLP

The alternating KAN-MLP novelty on the B0 backbone: blocks 0 and 2 use KAN
feedforward, blocks 1 and 3 use standard MLP (`mlp_ratio` 3.0, `embed_dim` 384),
with the last three B0 blocks unfrozen and fine-tuned, at 160px. Trained with
cross-entropy and `CosineAnnealingWarmRestarts`, 25 epochs. Best validation
accuracy 81.39% at epoch 19 (82.46% test). Accuracy dropped slightly versus the
full-KAN B0 model, likely from optimization and overfitting introduced by the
added complexity and partial fine-tuning. The checkpoint filename says
`50_epochs` while the log is 25 epochs. The log is authoritative.

## EfficientNetB3-KANViT-MLP

The alternating KAN-MLP design scaled up to an `efficientnet_b3` backbone at
224px (`embed_dim` 384, depth 4, 8 heads, `mlp_ratio` 3.0), with the backbone
channel count and 7x7 token grid detected dynamically (49 tokens). Trained in two
stages, backbone frozen then fine-tuned, with `CosineAnnealingWarmRestarts`, 50
epochs. Best validation accuracy 84.40% at epoch 44 (85.44% test), the strongest
hybrid and second overall. It also carries the largest generalization gap
(6.69%). The accuracy gain did not justify the added compute, which motivated the
shift to the custom CNN.

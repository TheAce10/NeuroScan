# NeuroScan — Technical Documentation

Brain tumour MRI classification using transfer learning. An ensemble of EfficientNetB3, VGG16, and DenseNet121 classifies MRI scans into four classes — glioma, meningioma, pituitary tumour, and no tumour — with Grad-CAM heatmap visualisation.

---

## Table of Contents

1. [Dataset](#1-dataset)
2. [Preprocessing Pipeline](#2-preprocessing-pipeline)
3. [Model Architectures](#3-model-architectures)
4. [Algorithm and Training Strategy](#4-algorithm-and-training-strategy)
5. [Ensemble Inference](#5-ensemble-inference)
6. [Results Summary](#6-results-summary)

---

## 1. Dataset

**Brain Tumor MRI Dataset** — Masoud Nickparvar  
Source: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

| Split    | Images per class | Total |
|----------|-----------------|-------|
| Training | 1,400           | 5,600 |
| Testing  | 400             | 1,600 |

Classes: `glioma` · `meningioma` · `pituitary` · `notumor`

### 1.1 Why This Dataset

**Quality and community validation.** At the time of selection the dataset held a Kaggle usability score of 10.0 and had been used in over 3,000 public notebooks, making it one of the most widely benchmarked brain MRI datasets available. Established accuracy baselines exist for direct comparison, and community-reported issues (mislabelled images, class distribution) are well documented.

**Class structure aligned with clinical categories.** The four classes directly reflect the primary tumour types a radiologist would need to distinguish — glioma, meningioma, pituitary tumour, and no tumour. This makes the classification task clinically meaningful rather than an artificial proxy problem.

**Pre-split training and test sets.** The dataset ships with a fixed train/test split, ensuring reported metrics are on a held-out set that was never used during training or hyperparameter tuning. This prevents inadvertent data leakage and makes results comparable across models.

**Known hard boundary cases.** The glioma/meningioma distinction is the hardest in the dataset — both are intracranial tumours that can appear visually similar on T1/T2-weighted MRI. This provided a concrete, clinically relevant challenge for the focal loss and class-weighting strategy applied to DenseNet121.

**Licence and accessibility.** The dataset is publicly available under a CC0 licence and downloadable via the Kaggle API, making it reproducible for any researcher without access to restricted clinical data.

### 1.2 Test Set

Contains 1,600 human brain MRI images reserved for model evaluation. Images are organised into the same four classes as the training set. Each class contains 400 images, ensuring fair and unbiased evaluation. No test images were used during training, validation, or hyperparameter search.

### 1.3 Local Structure

```
data/
  Training/
    glioma/         (1,400 images)
    meningioma/     (1,400 images)
    notumor/        (1,400 images)
    pituitary/      (1,400 images)
  Testing/
    glioma/         (400 images)
    meningioma/     (400 images)
    notumor/        (400 images)
    pituitary/      (400 images)
```

---

## 2. Preprocessing Pipeline

Implemented in `src/training/train.py` — `build_loaders()`. The pipeline is identical across all three models. Model-specific differences are in the loss function, not the input transforms.

### 2.1 RGB Conversion

```python
Image.open(path).convert("RGB")
```

Some images in the dataset are single-channel (grayscale) JPEG exports from DICOM scans. All three pretrained backbones expect 3-channel RGB input. Converting to RGB copies the single channel into all three, making the pipeline consistent regardless of source image format.

### 2.2 Resize to 224×224

```python
transforms.Resize((224, 224))
```

Applied to training and test images. Raw MRI images vary in resolution (typically 256×256 to 512×512). All three backbones were pretrained on ImageNet at 224×224, and their classifier heads require a fixed spatial input size. A uniform 224×224 also keeps GPU memory consumption predictable across batches.

> EfficientNetB3's native ImageNet resolution is 300×300, but 224×224 was used to maintain a single shared data loader across all three models and reduce memory pressure on the SLURM A10G nodes.

### 2.3 Data Augmentation (training set only)

The test loader uses only resize + normalise. Augmentation is applied exclusively during training to avoid inflating evaluation metrics.

| Transform | Parameters | Rationale |
|---|---|---|
| `RandomHorizontalFlip` | p=0.5 | Brain anatomy is approximately bilaterally symmetric; flipping doubles effective dataset size without introducing unrealistic samples |
| `RandomVerticalFlip` | p=0.1 | Low probability for mild regularisation without generating anatomically implausible images |
| `RandomRotation` | ±20° | MRI scanner head positioning varies slightly between subjects; rotation invariance reduces sensitivity to acquisition angle |
| `ColorJitter` | brightness=0.3, contrast=0.3, saturation=0.1 | Scanner contrast and brightness differ across MRI machines and acquisition protocols; jitter prevents overfitting to a specific scanner's intensity range |
| `RandomAffine` | translate=(0.1, 0.1) | Small spatial shifts (±10% of image size) simulate off-centre head positioning in the scanner bore |

Augmentations were chosen to be MRI-plausible — no elastic distortions, no drastic colour inversions, and no cutout/mixup, which could obscure diagnostically relevant lesion boundaries.

### 2.4 ToTensor

```python
transforms.ToTensor()
```

Converts a PIL image (H×W×C, uint8 0–255) to a PyTorch float32 tensor (C×H×W, 0.0–1.0). The channel-first layout is required by `nn.Conv2d`, and the 0–1 float range is required before normalisation.

### 2.5 Normalise — ImageNet Statistics

```python
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Applied to training and test images. These are the per-channel mean and standard deviation of the ImageNet training set. All three backbones were pretrained with this normalisation — their convolutional filters are calibrated to inputs in this distribution. Applying the same statistics at fine-tuning time ensures the pretrained feature detectors receive inputs in the expected range, keeping the Phase 1 gradient signal well-scaled from the first epoch.

The transform per channel: `(pixel − mean) / std`, shifting each channel to approximately N(0, 1).

### 2.6 Pipeline Summary

| Step | Training | Test | Purpose |
|---|---|---|---|
| RGB conversion | ✓ | ✓ | Normalise channel count across image formats |
| Resize 224×224 | ✓ | ✓ | Match pretrained backbone input size |
| Random flip / rotate / jitter / affine | ✓ | ✗ | Regularisation, scanner variability simulation |
| ToTensor | ✓ | ✓ | PIL → float32 tensor, HWC → CHW layout |
| Normalize (ImageNet μ/σ) | ✓ | ✓ | Align input distribution with pretrained weights |

---

## 3. Model Architectures

Three CNN architectures were chosen with different design philosophies. Using multiple architectures means each model makes different types of errors — their disagreements on hard cases are informative, and the ensemble is only confident when all three agree.

### 3.1 EfficientNetB3

EfficientNet scales a baseline network simultaneously along depth, width, and resolution using a compound coefficient. B3 sits in the middle of the family — more expressive than B0/B1, less memory-intensive than B5/B7. This made it well-suited to the SLURM node's 24 GB VRAM budget while still outperforming architectures with far more parameters.

The primary motivation for choosing EfficientNet is its high accuracy-per-parameter ratio. Brain MRI classification is a fine-grained task where subtle textural differences distinguish tumour types. EfficientNet's compound scaling preserves spatial detail across all three dimensions simultaneously, which benefits feature extraction from small lesion regions.

**Custom classifier head:**
```
Dropout(p=0.3) → Linear(in_features → 4)
```

Dropout at 0.3 is conservative — EfficientNet's backbone is already regularised via stochastic depth, so a lighter dropout suffices.

**Layers unfrozen in Phase 2:** `features.6`, `features.7`, `classifier`

---

### 3.2 VGG16

VGG16 is a deep sequential architecture with 13 convolutional layers followed by three fully connected layers, using only 3×3 convolutions throughout. It was included as a well-understood, thoroughly benchmarked baseline. Its linear, non-branching structure makes it easier to reason about where features are being learned and why certain predictions fail.

VGG16 is computationally heavier than EfficientNetB3 for similar accuracy, but its extensive use in the medical imaging literature means its behaviour on MRI data is well-characterised. Including it provides a direct point of comparison against published results on similar datasets.

**Custom classifier head:**
```
Linear(25088 → 4096) → ReLU → Dropout(0.4)
→ Linear(4096 → 4096) → ReLU → Dropout(0.4)
→ Linear(4096 → 4)
```

Dropout at 0.4 is higher than EfficientNetB3 because VGG16's large FC layers (4096 → 4096) are more prone to co-adaptation between units.

**Layers unfrozen in Phase 2:** `features[24:]`, `classifier`

---

### 3.3 DenseNet121

DenseNet connects every layer to every subsequent layer within a dense block — each layer receives the concatenated feature maps of all preceding layers as input. This encourages feature reuse and produces compact representations without requiring very deep networks.

DenseNet121 was chosen specifically because of the glioma/meningioma confusion problem. Its dense connectivity means earlier, lower-level features (edges, texture gradients) remain directly available at deeper layers — helping distinguish tumour boundary characteristics that VGG16 and EfficientNet may have progressively compressed away. This was the primary motivation for assigning the more aggressive training strategy (focal loss + class weighting) to DenseNet121: it was the architecture most likely to benefit from being forced to attend to hard boundary cases.

**Custom classifier head:**
```
Dropout(p=0.4) → Linear(1024 → 4)
```

**Layers unfrozen in Phase 2:** `denseblock4`, `norm5`, `classifier`

DenseNet's dense connections mean earlier blocks still influence later layers via concatenation, so the entire final dense block must be unfrozen — not just the last few layers — to allow meaningful domain adaptation.

---

## 4. Algorithm and Training Strategy

### 4.1 Transfer Learning

All three models were initialised with weights pretrained on ImageNet (1.28 million images, 1,000 classes). Training from scratch on 5,600 MRI images would cause severe overfitting — the models contain millions of parameters and have insufficient data to learn good generalisable representations from a random initialisation.

Transfer learning exploits the fact that low-level visual features (edges, textures, gradients) are domain-general. ImageNet-pretrained backbones already encode these features in their early layers. The task-specific learning required is primarily in the later, more abstract layers and in the classification head.

### 4.2 Two-Phase Training

Training is structured into two sequential phases per model. This is a standard approach for fine-tuning large pretrained networks and is motivated by the risk of destroying pretrained features too early.

#### Phase 1 — Frozen Backbone

All backbone parameters are frozen (`requires_grad = False`). Only the new classification head is trained.

**Rationale:** The classification head is initialised randomly. If the entire network were unfrozen immediately, large random gradients from the untrained head would backpropagate through the backbone and corrupt the pretrained features before the head has had any opportunity to converge. Freezing the backbone in Phase 1 lets the head stabilise, using the backbone purely as a fixed feature extractor.

| Parameter | Value |
|---|---|
| Epochs | 10 (all models) |
| Learning rate | 1e-3 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=10) |
| Trainable | Classifier head only |

#### Phase 2 — Partial Unfreeze / Fine-Tuning

The last convolutional blocks of each backbone are unfrozen and the full trainable portion is trained end-to-end at a much lower learning rate.

**Rationale:** Once the classifier head is stable, the backbone can be nudged to adapt its higher-level features to the MRI domain. Only the last blocks are unfrozen because early layers encode general low-level features equally valid for MRI as for natural images — updating them would be wasteful and risky. The final blocks encode the most abstract, task-specific features and are therefore the most valuable to adapt. The learning rate is reduced 20× (to 5e-5) to make small, careful updates that refine rather than overwrite the pretrained representations.

DenseNet121 runs 15 Phase 2 epochs versus 10 for the other two. Focal loss produces a flatter, slower loss curve — it requires more epochs to reach the same effective convergence point.

| Parameter | Value |
|---|---|
| Epochs | 10 (EfficientNetB3, VGG16) · 15 (DenseNet121) |
| Learning rate | 5e-5 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=epochs) |
| Trainable | Last backbone blocks + classifier head |

### 4.3 Optimizer — AdamW

AdamW computes adaptive per-parameter learning rates using first and second moment estimates of the gradients, making it far less sensitive to the global learning rate than SGD. This is particularly important for fine-tuning — different layers have very different gradient magnitudes: the randomly-initialised head needs large updates initially while the pretrained backbone needs small, careful ones.

The "W" in AdamW refers to decoupled weight decay. Standard Adam's weight decay is coupled with the adaptive learning rate, which diminishes its regularisation effect. AdamW decouples the two, making weight decay behave as L2 regularisation should in theory. Weight decay was set to 1e-4 in both phases.

### 4.4 Learning Rate Schedule — CosineAnnealingLR

The learning rate follows a cosine curve from its initial value down to approximately zero over each phase:

- **Early epochs:** higher LR for faster convergence when the model is far from the optimum
- **Late epochs:** low LR for fine-grained refinement near convergence, avoiding overshooting

Compared to step decay (dropping LR by a fixed factor every N epochs), cosine annealing produces smoother convergence, consistently better final accuracy on image classification tasks, and requires no manual tuning of decay steps.

### 4.5 Loss Function

#### CrossEntropyLoss — EfficientNetB3 and VGG16

Standard cross-entropy with no class weighting. The four training classes are balanced (1,400 images each), so there is no strong motivation for weighting.

#### FocalLoss — DenseNet121

Focal loss modifies cross-entropy to down-weight easy, well-classified examples and focus training signal on hard, misclassified ones:

```
FL(p_t) = −(1 − p_t)^γ · log(p_t)
```

When γ=0 this reduces to cross-entropy. At γ=2, examples the model already classifies with high confidence contribute very little to the total loss, while borderline examples (low p_t) retain a large contribution.

**Why DenseNet121 only, and why γ=2:** After Phase 1 baseline evaluation, glioma recall was consistently the lowest across all three models. The glioma/meningioma boundary is visually ambiguous — both present as hypointense regions with variable enhancement. Standard cross-entropy treats every example equally, so the model can achieve a decent aggregate loss by getting easy `notumor` cases right while still failing on glioma. Focal loss forces it to spend learning capacity on exactly the cases it is getting wrong. γ=2 is the value from the original focal loss paper (Lin et al., 2017) and is the established default for moderate class difficulty.

#### Class Weights — DenseNet121

In addition to focal loss, explicit class weights amplify the gradient penalty for the harder classes:

| Class | Weight | Reason |
|---|---|---|
| glioma | 2.0 | Worst recall across baseline models |
| meningioma | 1.2 | Frequently confused with glioma |
| notumor | 0.8 | Consistently high recall — reduce gradient dominance |
| pituitary | 0.8 | Consistently high recall — reduce gradient dominance |

**Note on loss comparability:** Because focal loss applies the `(1 − p_t)^γ` modulating factor, absolute loss values are numerically smaller than cross-entropy on the same data. DenseNet121's loss curves sit lower on the y-axis — this is a property of the loss function, not a sign of better or faster convergence. Loss curves across models are not directly comparable.

### 4.6 Checkpoint Strategy

During each phase the model state achieving the best validation accuracy is saved as `phase{N}_best.pt`. A checkpoint is also saved every 5 epochs (`phase{N}_epoch{N:02d}.pt`) to allow recovery from a mid-training failure.

At the end of Phase 2 the best-performing state is restored before final evaluation and before writing the output weight file. The final `.pt` file always reflects peak validation performance, not the state at the last epoch (which may have slightly degraded due to late-phase LR oscillation).

---

## 5. Ensemble Inference

At inference time the three models' softmax probability vectors are averaged (soft voting):

```
P_ensemble(c) = (1/3) · [P_effnet(c) + P_vgg16(c) + P_densenet121(c)]
```

The final predicted class is `argmax(P_ensemble)`.

**Why soft voting over hard voting:** Hard voting takes the majority class label. Soft voting averages the full probability distributions, so a model that is highly confident in a class contributes more than one that is barely over the decision boundary. A single model's 99% confidence correctly outweighs two models' 51% votes.

**Why an ensemble at all:** The three architectures make different types of errors. VGG16 and EfficientNetB3 tend to confuse glioma with meningioma; DenseNet121 (with focal loss) reduces glioma misclassification at the cost of slightly higher pituitary errors. Averaging the three distributions means the ensemble is only confident when all three agree — providing a natural measure of model uncertainty. Cases where the models strongly disagree are precisely the cases that warrant closer clinical review.

---

## 6. Results Summary

| Model | Accuracy | Glioma Recall | Meningioma Recall |
|---|---|---|---|
| EfficientNetB3 | 92.0% | — | — |
| VGG16 | 93.0% | — | — |
| DenseNet121 (focal loss) | 91.0% | 80.8% | — |

See `results/` for full per-class classification reports and confusion matrices.

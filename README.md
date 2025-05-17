# CLEVR Visual Question Answering with Transformers

This repository implements a Visual Question Answering (VQA) system using a multi-modal transformer architecture. Given an image and a corresponding question, the model predicts the correct answer. It was developed as part of **COL774: Machine Learning (Semester II, 2024-25)** at **IIT Delhi**.

---

## 🧠 Overview

The model uses:
- **ResNet-101** as a frozen/fine-tuned image encoder.
- **Transformer Encoder** (inspired by BERT) for text encoding.
- **Cross-Attention** to fuse visual and textual representations.
- **MLP Classifier** for answer prediction.

It is trained and evaluated on the [CLEVR dataset](https://cs.stanford.edu/people/jcjohns/clevr/), which requires reasoning over object attributes, spatial relationships, and logical constructs.

---

## 📁 Dataset

We use the **CLEVR** dataset containing:
- Synthetic 3D-rendered images.
- Associated questions and ground truth answers.

### Structure:
- `trainA`, `valA`, `testA` — used for training and evaluation.
- `testB` — used for zero-shot generalization testing.

> Dataset is not included in this repo. Download from [CLEVR Dataset](https://cs.stanford.edu/people/jcjohns/clevr/).

---

## Architecture Details

### 1. Image Encoder
- Pretrained **ResNet-101** from `torchvision.models`.
- Outputs a 2048-channel feature map.
- Projected to 768-dim via `nn.Linear`.

### 2. Text Encoder
- Tokenized questions via custom/BERT tokenizer.
- Positional embeddings are learnable.
- Transformer encoder: 6 layers, 8 heads, dim=768.

### 3. Cross-Attention Fusion
- Text queries attend to visual keys/values.
- `nn.MultiheadAttention` for feature fusion.

### 4. Classifier
- MLP: `Linear(768 → 500) → ReLU → Linear(500 → num_classes)`.

---

##  Features

-  Full training pipeline with checkpointing
-  ResNet fine-tuning support
-  Visualization of predictions and errors
-  Focal Loss integration (optional)
-  BERT Embedding initialization (optional)
-  Zero-shot evaluation on CLEVR-B dataset

---

## Usage

### Train
```bash
python part1.py --mode train --dataset <path_to_dataset> --save_path <model_save_path>
```

### Inference
```bash
python part1.py --mode inference --dataset <path_to_dataset> --model_path <saved_model>
```
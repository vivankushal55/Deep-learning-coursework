# CSC8637 – Deep Learning Coursework

This repository contains the coursework submission for the **Deep Learning** module (CSC8637), covering two tasks: fine-grained bird species image classification and a word-level language model for text generation.

---

## Repository Structure

```
├── DeepLearning_Task1.ipynb   # Task 1: Fine-Grained Image Classification
├── DeepLearning_Task2.ipynb   # Task 2: Language Model (Text Generation)
├── DeepLearning_Task1.pdf     # Task 1 Report
├── DeepLearning_Task2.pdf     # Task 2 Report
└── README.md
```

> **Note:** Trained model checkpoints (`.pth` / `.h5` files) should be downloaded separately and placed in the appropriate directories before running inference. See instructions below.

---

## Task 1 – Fine-Grained Bird Species Classification

### Overview

A fine-grained image classification model trained on the [CUB-200-2011](https://data.caltech.edu/records/65de6-vp158) dataset to classify 200 species of birds.

Two models were developed:
- **Model 1 (Transfer Learning):** EfficientNet-B3 pre-trained on ImageNet, fine-tuned on the CUB dataset.
- **Model 2 (Custom CNN):** A manually designed CNN architecture trained from scratch.

### Model 1 – Hyperparameters

| Parameter | Value |
|---|---|
| Model | EfficientNet-B3 |
| Optimizer | AdamW |
| Loss Function | Cross-Entropy Loss (label smoothing = 0.05) |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Batch Size | 16 |
| Epochs | 70 |
| Scheduler | CosineAnnealingLR (T_max = 50) |
| Image Size | 380×380 |
| Mixed Precision | AMP GradScaler (enabled) |

### Results (Test Set – Model 1)

| Metric | Value |
|---|---|
| Accuracy | 85.21% |
| Precision (macro) | 0.8561 |
| Recall (macro) | 0.8533 |
| F1 Score (macro) | 0.8508 |

### Data Augmentation

- Random resized crop
- Horizontal flip
- Random rotation
- Colour jitter
- Normalisation (ImageNet mean/std)

### How to Run

1. Download the CUB-200-2011 dataset from Canvas and place it in the expected directory (see notebook for the exact path).
2. Open `DeepLearning_Task1.ipynb`.
3. To **train** the model, run all cells from the top.
4. To **evaluate** using pre-trained weights, place the model checkpoint in the path specified in the notebook and run the evaluation cells only.

```bash
# Install dependencies
pip install torch torchvision timm scikit-learn matplotlib
```

> Random seeds are fixed throughout for reproducibility.

---

## Task 2 – Language Model (Agatha Christie Text Generation)

### Overview

A word-level language model trained on *Poirot Investigates* by Agatha Christie to generate text in a similar style. The model takes a few words as input and generates a sequence of text.

### Architecture

- Embedding layer
- Stacked LSTM layers
- Dense output layer with softmax
- Dropout for regularisation

### Training Details

| Parameter | Value |
|---|---|
| Model | LSTM-based RNN |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Cross-Entropy |
| Sequence Length | 40 words |
| Early Stopping | Enabled (best validation loss) |

### Results

- Best Validation Loss: **6.5595**
- Validation Perplexity: **705.93**

### How to Run

1. Download the dataset from Canvas and place it in the path specified in the notebook.
2. Open `DeepLearning_Task2.ipynb`.
3. To **train** the model, run all cells from the top.
4. To **generate text** using pre-trained weights:
   - Place the model checkpoint in the path specified in the notebook.
   - Run the inference cell, which will prompt you to enter a few starting words.

```bash
# Install dependencies
pip install tensorflow numpy
```

**Example usage at inference:**
```
Enter a few starting words (prompt): Poirot said
Temperature (default 0.8): 0.7
Top-k (default 30): 25
How many words to generate (default 80): 80
```

> The model generates text that reflects the vocabulary and basic sentence structure of Christie's writing, though long-range coherence is limited by the fixed context window and relatively small training corpus.

---

## Requirements

- Python 3.8+
- PyTorch (Task 1)
- TensorFlow / Keras (Task 2)
- See individual notebooks for full dependency lists.

---

## Reproducibility

All random seeds are fixed. Model checkpoints are saved at the best validation performance. To exactly reproduce the reported results, use the provided checkpoint files and run the evaluation cells only — do not retrain from scratch.

---

## Dataset Sources

- **Task 1:** [CUB-200-2011 Bird Dataset](https://data.caltech.edu/records/65de6-vp158)
- **Task 2:** [Poirot Investigates – Project Gutenberg](https://www.gutenberg.org/ebooks/61262)

Both datasets must be downloaded from Canvas prior to running the notebooks.

# Image Classification with Convolutional Neural Networks

This project focuses on image classification using **Convolutional Neural Networks (CNNs)** on different datasets, primarily **CIFAR-10** and an **Emotions** dataset sourced from Kaggle.
The objective is to understand why deep learning models — and CNNs in particular — are well-suited for visual data, and to build a complete training and evaluation pipeline using PyTorch.

---

## Project Goals

- Build a full deep learning pipeline for image classification
- Compare a naïve baseline with a convolutional architecture
- Understand the impact of convolution, pooling, and regularization
- Analyze model performance and failure cases

---

## Datasets

### CIFAR-10 Overview

- **Source:** CIFAR-10 (via `torchvision.datasets`)
- **Training samples:** 50,000
- **Test samples:** 10,000
- **Image size:** 3 × 32 × 32 (RGB)

### Emotions

- **Source:** Emotions dataset (from [kaggle.com](https://www.kaggle.com/datasets))
- **Classes:** 5 (angry, fear, happy, sad, surprise)
- **Training samples:** ~47,000
- **Test samples:** ~10,000
- **Image size:** resized to 64 × 64

---

## Cross-dataset evaluation (CIFAR-10 → Emotions)

- The model was initially trained on **CIFAR-10** (standard benchmark, 32×32). To evaluate generalization to real-world/finer-grained images, we tested (and in some experiments fine-tuned) this model on the **Emotions** dataset downloaded from Kaggle.
- This cross-dataset evaluation measures how a model trained on a standardized benchmark performs on a dataset with a different distribution and image sizes. To make inputs compatible, images were preprocessed (resizing, normalization and simple augmentations) so the model could be applied consistently across datasets.
- Hyperparameters were selected using a grid search over key parameters (examples: batch size, learning rate, dropout). The grid-search results and best configurations are logged with the experiment tracking.
- All training runs, hyperparameters and metrics were logged using **MLflow**; artifacts and run metadata are stored locally (see `mlartifacts/`) and can be inspected through the MLflow tracking UI. See the notebooks and scripts in `notebooks/` and `src/computer_vision/training/` for implementation details, preprocessing pipelines and per-experiment metrics.

---

## Baseline Model (MLP)

As a reference, a fully connected neural network is trained on flattened images.

**Purpose:**
- Highlight the limitations of dense architectures on image data
- Serve as a performance baseline

**Observation:**
- Limited performance due to loss of spatial information
- Strong overfitting despite regularization

---

## Convolutional Neural Network

### Architecture (simplified)

Input 
→ [Conv2D + ReLU + MaxPool]
→ Flatten
→ Softmax Output


Key components:
- Convolutional layers to capture spatial patterns
- Max pooling for spatial downsampling
- Dropout for regularization

---

## Training Setup

- **Framework:** PyTorch
- **Hyperparameter tuning:** grid search over selected parameters (batch size, learning rate, dropout, etc.) to find robust configurations.
- **Experiment tracking:** MLflow is used to log runs, hyperparameters, metrics and artifacts; see `mlartifacts/` and the helper in `src/utils/mlflow.py` for logging utilities.

Optional improvements include:
- data augmentation

---

## Evaluation

Model performance is evaluated using:
- classification accuracy on the test set
- confusion matrix
- per-class accuracy analysis

---

## Results Summary for Emotions Dataset

| Model | Test Accuracy |
|------|------------|
| MLP baseline | 50%        |
| CNN (no augmentation) | 70%        |
| CNN + data augmentation | 58%        |

---

## Key Learnings

- Convolutional layers preserve and exploit spatial structure
- CNNs dramatically reduce the number of parameters compared to dense networks
- Data augmentation don't always improve performance; it depends on the dataset and model capacity
- Error analysis is essential to understand model limitations

---

## Limitations and Future Work

Possible extensions:
- deeper architectures
- transfer learning with pretrained models (e.g. ResNet)
- batch normalization
- learning rate scheduling

---

## Technologies Used

- Python
- PyTorch
- torchvision
- NumPy
- matplotlib
- Jupyter Notebook
- MLflow (for experiment tracking)

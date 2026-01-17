# Machine Learning Portfolio

This repository contains a curated collection of **applied machine learning and deep learning projects** implemented from scratch and with standard frameworks.  
The goal of this portfolio is to explore core ML/DL domains through practical, well-structured, and reproducible projects.

The projects emphasize:
- solid problem formulation
- clean data pipelines
- model design and training
- rigorous evaluation
- clear documentation and analysis

---

## Project Structure

Each subdirectory focuses on a specific domain of machine learning:
```
├── classical_ml
├── computer_vision
├── nlp
├── README.md
└── recommender_systems
```

Each project follows a consistent structure:
- data loading and preprocessing
- model implementation
- training and evaluation
- discussion of results and limitations

---

## Getting Started

To set up the environment and install dependencies, run:
```commandline
uv sync
uv pip install -e .
```

---

## Domains Covered

### Classical Machine Learning
Supervised learning on tabular data with an emphasis on:
- feature engineering
- model comparison
- cross-validation
- interpretability

**Typical models:** Linear/Logistic Regression, Random Forests, Gradient Boosting.

---

### Computer Vision (Deep Learning)
Image-based tasks using convolutional neural networks.

**Topics include:**
- CNN architectures
- overfitting and regularization
- data augmentation
- performance analysis

**Framework:** PyTorch.

---

### Natural Language Processing
Text classification and representation learning using classical NLP techniques and neural models.

**Topics include:**
- Bag-of-Words and TF-IDF
- text embeddings
- sequence models

---

### Recommender Systems
Personalization and ranking problems.

**Approaches include:**
- content-based filtering
- collaborative filtering
- similarity-based ranking

---

### Anomaly Detection
Unsupervised and semi-supervised detection of abnormal patterns.

**Techniques include:**
- Isolation Forest
- One-Class SVM
- Autoencoders

---

## Technology Stack

- Python
- NumPy, pandas, scikit-learn
- PyTorch
- matplotlib / seaborn
- Jupyter Notebook

---

## Purpose of This Repository

This repository is intended to:
- build strong practical intuition in ML/DL
- compare different domains to identify specialization interests
- serve as a technical portfolio for internships or junior roles

Each project is self-contained and documented with clear assumptions, design choices, and results.

---

## Notes

This is an evolving repository. New projects and improvements are added progressively as skills deepen and more advanced topics are explored.

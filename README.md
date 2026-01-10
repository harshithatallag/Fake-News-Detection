# Detecting Deception: Fake News Classification Using NLP

**Machine Learning | Deep Learning | Transformers**

---

## Overview

End-to-end Natural Language Processing project focused on detecting fake news using supervised learning techniques, with strong emphasis on data integrity, realistic evaluation, and model interpretability.

The project builds and compares classical machine learning, deep learning, and transformer-based models while addressing critical real-world issues such as data leakage, inflated performance metrics, and dataset bias.

---

## Problem Statement

The rapid spread of misinformation across digital platforms poses serious risks to public trust, public health, and democratic processes. Manual fact-checking does not scale with the volume of online content.

This project addresses the core question:

**Can machine learning models reliably detect fake news using only article text while maintaining trustworthy and realistic evaluation?**

---

## Dataset

* ~39,000 labeled news articles
* Binary classification: Real vs Fake
* Moderately imbalanced dataset (~55% fake articles)
* Purely text-based, reflecting real-world unstructured data used in content moderation and media monitoring systems

---

## Key Technical Challenge: Data Leakage

Initial experiments produced near-perfect accuracy (>99%), indicating severe data leakage.

To ensure realistic evaluation, the following steps were taken:

* Removal of exact duplicate articles
* Detection and elimination of near-duplicate articles using text similarity analysis
* Text cleaning and normalization performed before dataset splitting
* Strict separation of training, validation, and test sets using stratified splits

This step was critical in preventing memorization-based performance and establishing reliable baselines.

---

## Project Workflow

* Data cleaning and leakage mitigation
* Text normalization and preprocessing
* Feature extraction and tokenization
* Model training across multiple architectures
* Robust evaluation and comparison
* Explainability and error analysis

The workflow mirrors production-oriented ML pipelines rather than isolated experiments.

---

## Modeling Approaches

### Classical Machine Learning (Baseline)

* TF-IDF + Logistic Regression
* Strong and interpretable lexical baseline
* Hyperparameter tuning using cross-validation

### Deep Learning

* Bidirectional LSTM
* Captures sequential and contextual dependencies
* Multiple configurations tested with early stopping

### Transformer-Based Model

* DistilBERT fine-tuned for binary text classification
* Leverages pretrained contextual embeddings
* Achieved the strongest overall performance

---

## Model Evaluation

Models were evaluated using multiple metrics to ensure balanced assessment:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC

Evaluation prioritized generalization and robustness, not headline accuracy.

---

## Explainability & Model Behavior

To improve trust and transparency:

* SHAP used for global feature importance
* LIME used for local, article-level explanations

Key findings:

* Models rely heavily on stylistic and linguistic cues
* Sensational or emotionally charged language strongly influences predictions
* Highlights limitations of purely text-based misinformation detection

---

## Key Insights

* Proper data cleaning has a greater impact than model complexity
* TF-IDF baselines remain highly competitive when evaluated correctly
* Deep learning improves contextual understanding but increases computational cost
* Transformer models offer marginal performance gains with higher resource requirements
* Dataset bias limits real-world generalization

---

## Limitations

* Evaluation limited to a single dataset
* No cross-source generalization testing
* Difficulty handling satire and subtle misinformation

These limitations highlight the need for diverse datasets and cautious real-world deployment.

---

## Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* TensorFlow / Keras
* Hugging Face Transformers
* NLTK
* SHAP, LIME

---

## Repository Structure

```
├── FakeNewsDetection.ipynb   # Full preprocessing, modeling, and evaluation
├── Final_Report.pdf         # Detailed methodology and findings
├── Presentation.pptx        # Executive-level summary
```

---

## Why This Project Matters

This project demonstrates:

* Strong understanding of NLP pipelines
* Awareness of common ML evaluation pitfalls
* Ability to balance model complexity with reliability
* Responsible ML practices including explainability and bias analysis



This README is **portfolio grade**.
Do not dilute it.

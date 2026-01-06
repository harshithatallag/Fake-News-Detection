# Fake News Detection Using Machine Learning and Deep Learning

## Overview

This project builds an end to end **fake news classification system** using Natural Language Processing techniques. The goal is to automatically classify news articles as **real or fake** based purely on textual content.

The project compares **traditional machine learning**, **deep learning**, and **transformer based models** to understand performance tradeoffs, model complexity, and real world applicability.

It reflects how text classification problems are handled in real production oriented analytics and ML workflows.

---

## Problem Statement

The rapid spread of misinformation across digital platforms creates serious social and economic risks. Manual verification does not scale.

This project addresses a core question:
Can machine learning models reliably detect fake news using only the article text?

---

## Dataset

The dataset contains labeled news articles with two classes:

Real news & Fake news

Each record includes the full article text and a binary label. The data represents real world unstructured text typically encountered in media monitoring and content moderation systems.

---

## Project Workflow

1. Text preprocessing and cleaning
2. Feature extraction and tokenization
3. Model training across multiple approaches
4. Model evaluation and comparison
5. Result interpretation and reporting

The workflow follows standard ML pipeline practices rather than isolated experiments.

---

## Text Preprocessing

- Lowercasing and text normalization
- Removal of punctuation and noise
- Stopword elimination
- Tokenization for classical and deep learning models

Proper preprocessing significantly impacted downstream model performance.

---

## Modeling Approaches

### Traditional Machine Learning

- Logistic Regression trained on TF IDF features
- Used as a strong and interpretable baseline

### Deep Learning

- LSTM based neural network
- Captures sequential and contextual dependencies in text

### Transformer Model

- DistilBERT fine tuned for binary text classification
- Leverages pre trained language representations for semantic understanding

---

## Model Evaluation

Models were evaluated using:

Accuracy


Precision


Recall


F1 Score


Confusion Matrix


ROC AUC

This ensured evaluation beyond surface level accuracy and exposed misclassification behavior.

---

## Key Results

Transformer based models achieved the highest classification performance


Logistic Regression provided a strong baseline with lower computational cost


LSTM models improved contextual understanding but required careful tuning


Model selection depends on accuracy requirements versus resource constraints

---

## Technologies Used

- Python
- Pandas and NumPy
- Scikit learn
- TensorFlow Keras
- Hugging Face Transformers
- NLTK

---

## Use Cases

Automated fake news detection


Content moderation pipelines


Media verification tools


Trust and safety analytics systems

---

## Why This Project Matters

This project demonstrates:
1. Ability to handle unstructured text data
2. Strong understanding of NLP pipelines
3. Comparison of classical ML and modern deep learning models
4. End to end problem solving rather than isolated modeling

---

## Repository Structure

FinalCode.ipynb
Complete implementation including preprocessing, modeling, and evaluation

Final Report.pdf
Detailed explanation of methodology and findings

Final Presentation.pptx
High level summary suitable for non technical stakeholders




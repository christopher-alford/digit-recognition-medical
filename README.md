# Handwritten Digit Recognition & Error Detection for Medical Record Digitization

A machine learning system for automated handwritten digit recognition with an integrated error detection layer — designed to support digitization of medical records including patient IDs, dosages, measurements, and test results. Built with TensorFlow/Keras and scikit-learn.

**Collaborators:** Christopher Alford & Morgan Childers

---

## The Problem

Handwritten entries in medical records are a persistent source of digitization errors. Manual transcription of patient IDs, dosages, and numeric test results is slow, expensive, and error-prone — with an average human error rate of ~2.5%. This project explores whether a CNN-based recognition system with confidence-based error flagging can match or improve on that benchmark.

---

## Results

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression (baseline) | ~93% | Flattened 784-feature input |
| **CNN (final model)** | **~99%** | Spatial feature learning via Conv2D layers |

The CNN achieves a **~6% accuracy improvement** over the logistic regression baseline by preserving spatial structure in the 28×28 pixel images rather than flattening them.

---

## Error Detection System

Beyond raw accuracy, the system includes a confidence-based flagging layer that mirrors real-world medical validation workflows:

- Predictions below a **0.90 confidence threshold** are flagged as `"Review Needed"`
- Predictions above the threshold are marked `"Accepted"`
- At the 0.90 threshold, **~2.56% of predictions are flagged** — closely matching the human error rate baseline
- Threshold testing across 0.80, 0.90, and 0.95 demonstrates the reliability vs. workload trade-off

The most common misclassifications involve visually similar digit pairs: **3/5**, **4/9**, and poorly penned digits with ambiguous structure.

---

## Model Architecture

```
Input (28x28x1)
│
├── Conv2D(32, 3x3, relu)
├── MaxPooling2D(2x2)
├── Conv2D(64, 3x3, relu)
├── MaxPooling2D(2x2)
├── Flatten
├── Dense(128, relu)
├── Dropout(0.3)
└── Dense(10, softmax)
```

Trained for 5 epochs with batch size 128, Adam optimizer, categorical cross-entropy loss, and 10% validation split.

---

## Tech Stack

- **Python** — NumPy, matplotlib
- **TensorFlow / Keras** — CNN architecture, training, evaluation
- **scikit-learn** — Logistic Regression baseline, classification report, confusion matrix
- **Dataset** — MNIST (60,000 training / 10,000 test images)

---

## Project Structure

```
digit-recognition-medical/
│
├── Copy_of_3010Project.ipynb     # Full notebook: baseline, CNN, error detection
└── README.md
```

---

## How to Run

1. Clone the repo
2. Open `Copy_of_3010Project.ipynb` in Google Colab or Jupyter
3. Run all cells in order — MNIST dataset loads automatically via Keras

---

## Practical Implications

- Enables scaled automation of handwritten data ingestion
- Built-in error flagging requests human review only when necessary
- Reduces digitization errors while minimizing reviewer workload
- Applicable beyond medical records to any domain with handwritten numeric data

---

## Authors

**Christopher Alford** — [linkedin.com/in/alford-christopher](https://linkedin.com/in/alford-christopher) | [github.com/christopher-alford](https://github.com/christopher-alford)

**Morgan Childers**

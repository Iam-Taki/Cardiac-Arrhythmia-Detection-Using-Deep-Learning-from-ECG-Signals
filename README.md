# Cardiac Arrhythmia Detection Using Deep Learning from ECG Signals

**Author:** Abdullah Al Noman Taki (VR528988)
**Course:** Machine Learning & Deep Learning (Deep Learning) — University of Verona
**Program:** MSc in Artificial Intelligence, A.Y. 2024-25

---

## 📋 Overview

This project implements and compares **five deep learning architectures** for multi-class ECG heartbeat classification using the **MIT-BIH Arrhythmia Dataset**. The goal is to automatically classify heartbeats into five categories, supporting automated arrhythmia detection in clinical settings.

## 🫀 Dataset

- **Source:** [MIT-BIH Arrhythmia Dataset (Kaggle)](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
- **Samples:** 7,023 ECG heartbeat recordings (187 time steps each)
- **Classes:**
  | Class | Heartbeat Type |
  |---|---|
  | 0 | Normal (N) |
  | 1 | Supraventricular (S) |
  | 2 | Ventricular (V) |
  | 3 | Fusion (F) |
  | 4 | Unknown (Q) |

- **Class imbalance handled:** Original dataset heavily skewed toward Normal beats (82.8%). Balanced via resampling to 3,000 samples/class.
- **Final split:** 12,750 training / 2,250 validation / 1,500 test samples.

## 🏗️ Models Implemented

| Model | Type | Test Accuracy | ROC-AUC |
|---|---|---|---|
| **1D CNN** | Baseline convolutional | **92.20%** | **0.9917** |
| Residual CNN | Convolutional + skip connections | 90.40% | 0.9855 |
| CNN + LSTM | Hybrid spatial-sequential | 88.80% | 0.9843 |
| Transformer | Self-attention based | 88.20% | 0.9816 |
| GRU | Recurrent (gated) | 86.40% | 0.9754 |

🏆 **Best Model: 1D CNN** — local morphological feature extraction (P wave, QRS complex, T wave) proved more effective than sequence modeling for this short-signal classification task.

## ⚙️ Preprocessing Pipeline

1. Load raw ECG signals (187 features + label) from MIT-BIH CSV files
2. Balance classes via resampling (3,000 samples/class)
3. Train/validation split (85/15, stratified)
4. Min-Max normalization to [0, 1]
5. Reshape to `(samples, 187, 1)` for sequential model input

## 🔧 Hyperparameters

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 64 |
| Max Epochs | 20 (EarlyStopping, patience=7) |
| LR Reduction | ReduceLROnPlateau (factor=0.5, patience=5) |
| Loss | Categorical Crossentropy |

## 📊 Evaluation Metrics

Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC (one-vs-rest, macro-averaged).

## 📁 Repository Structure

```
├── ECG_01_Preprocessing.ipynb     # Data loading, balancing, normalization
├── ECG_02_Models.ipynb            # All 5 models, training, evaluation
├── report/
│   └── Cardiac_Arrhythmia_Detection_Report.pdf
├── figures/
│   ├── class_distribution.png
│   ├── ecg_samples.png
│   ├── training_curves.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── model_comparison.png
└── README.md
```

## 🚀 How to Run

1. Open `ECG_01_Preprocessing.ipynb` in Google Colab
2. Upload your Kaggle API key (`kaggle.json`) when prompted
3. Run all cells — preprocessed data will be saved to Google Drive
4. Open `ECG_02_Models.ipynb`
5. Mount Google Drive and run all cells to train and evaluate all 5 models

## 📈 Key Findings

- **Local feature extraction outperforms sequence modeling** for short ECG segments (187 time steps)
- The **'Unknown' class** was the easiest to classify across all models
- **'Normal' vs 'Supraventricular'** beats were the most frequently confused pair, due to morphological similarity
- **MobileNet-style efficiency**: lightweight convolutional models matched or outperformed more complex sequential/attention-based architectures

## 📚 References

1. Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE Eng. Med. Biol. Mag.*, 20(3), 45-50.
2. He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*.
3. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
4. Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP*.
5. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.

---

*This project was developed as part of the Machine Learning & Deep Learning (Deep Learning) course at the University of Verona.*

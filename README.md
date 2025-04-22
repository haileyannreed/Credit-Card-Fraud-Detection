# Balanced Random Forest for Credit Card Fraud Detection

This repository implements the **Balanced Random Forest (BRF)** algorithm on the highly imbalanced Credit Card Fraud Detection dataset. The project follows the methodology outlined in the paper:  
**"Using Random Forest to Learn Imbalanced Data" by Chen, Liaw, and Breiman (2004)**.

---

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Fraudulent Transactions**: 492 (only 0.17%)
- **Features**: 
  - `Time`, `Amount`, and `Class` (target)
  - `V1` through `V28`: PCA-transformed features to preserve anonymity

---

## ⚙️ Preprocessing & Algorithm

- Features (`X`): All columns except `Class`
- Target (`y`): `Class` (0 = legitimate, 1 = fraud)

### Balanced Random Forest Steps:

1. Perform **10-fold cross-validation**.
2. For each fold and each tree:
   - Draw a **balanced bootstrap**: downsample majority class to match minority class.
   - Grow a **full-size, unpruned CART tree**.
   - At each node, randomly choose `mtry = sqrt(p)` features for splitting (standard RF practice).
3. Aggregate all trees for final prediction.

### Parameters Used:

| Parameter      | Value               |
|----------------|---------------------|
| `n_estimators` | 100 (default)       |
| `max_features` | `sqrt` (random subset at each node) |
| `max_depth`    | Not set (grow fully)|
| `criterion`    | `gini`              |
| `bootstrap`    | `True`              |

No hyperparameter tuning was used to align with the paper’s original methodology.

---

## 📑 The Paper

The BRF algorithm was introduced in *"Using Random Forest to Learn Imbalanced Data"* by Chen, Liaw, and Breiman. It compares BRF with:

- **One-sided Sampling**
- **SHRINK**
- **SMOTE + Downsampling** (500% SMOTE with 50% and 100% downsampling)
- **WRF** (Weighted Random Forest)

📌 *BRF* stood out in F-measure, while *WRF* had the best overall performance in terms of G-Mean and weighted accuracy.

---

## 📈 Performance Metrics

The following metrics were used due to the class imbalance:

| Metric             | Value     |
|--------------------|-----------|
| Accuracy⁺ (Recall) | 89.02%    |
| Accuracy⁻          | 99.07%    |
| Precision          | 14.25%    |
| F1-Score           | 24.54%    |
| G-Mean             | 93.89%    |
| Weighted Accuracy  | 94.04%    |

### Metric Definitions

- **Precision**: TP / (TP + FP)
- **Recall (Acc⁺)**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of Precision and Recall
- **G-Mean**: Geometric mean of class-wise accuracy
- **Weighted Accuracy**: Mean of Acc⁺ and Acc⁻

---

## 🔍 Analysis

BRF is effective at detecting minority class instances due to:
- Balanced bootstrapping
- Unpruned, deep CART trees
- Random feature selection (`mtry`) at each node

However, **precision is low**, indicating a trade-off between capturing more frauds and reducing false positives. This matches findings from the original paper.

---

## 🤔 Reflection & Future Work

**a. Strengths & Weaknesses:**
- BRF effectively balances recall and specificity but struggles with precision.
- Results are consistent with the paper, which strengthens its validity.
- Possible critiques: No hyperparameter tuning, low interpretability, and relatively low precision.

**b. Broader Impact & Future Work:**
- Can be adapted for healthcare, finance, cybersecurity — anywhere imbalanced classification matters.
- Future work could include:
  - Threshold optimization (e.g., cutoff = 0.6)
  - Hybrid methods (e.g., BRF + SMOTE)
  - Feature importance visualization
  - Integration with explainability tools like SHAP or LIME

---

## 🧪 How to Run

Clone the repo and open the notebook in Google Colab or Jupyter Notebook:

```bash
git clone https://github.com/yourusername/brf-creditcard-fraud.git

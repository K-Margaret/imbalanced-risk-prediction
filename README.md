# 🚗 Car Auction Risk Prediction (BadBuy Classification)

Predicting "BadBuy" vehicles at car auctions to mitigate financial risk. 

## 🎓 Project Context
This project was developed for the Data Science branch at **School 21 (ex-École 42)**. The official project checklist set a passing threshold of a **0.15 Gini score** - a baseline that many standard implementations barely cross. 

Instead of relying on complex, black-box ensemble models just to scrape by, this implementation focuses on rigorous data engineering. By applying advanced preprocessing to handle high-cardinality categoricals and severe class imbalance, even the simplest baseline models easily exceed a **0.30 Gini score**, and the fully optimized pipeline achieves an exceptional **~0.48-0.50 Gini score**, vastly outperforming the project's baseline expectations.

---

## 📊 The Business Problem
In the real world, most cars sold at auction are good deals. "BadBuys" (lemons) represent only ~11% of the data. Because of this severe class imbalance, standard **Accuracy** is a misleading metric. A dummy model predicting "GoodBuy" every time would achieve ~89% accuracy but be completely useless to the business. 

Instead, we optimize for the **Gini Coefficient** (derived from ROC-AUC), which measures how well the model ranks high-risk vehicles above low-risk ones.

---

## 🏆 Key Results

| Model | Preprocessing Strategy | Validation Gini | Test Gini |
| :--- | :--- | :---: | :---: |
| **Baseline** (Gaussian NB) | Standard One-Hot Encoding | 0.375 | - |
| **Final** (Logistic Regression) | **Frequency Encoding + Non-Linear + L1** | **0.474** | **0.506** |

*Note: The Test Gini is actually higher than the Validation Gini, proving the robustness of the preprocessing pipeline and lack of overfitting.*

---

## 💡 The Main Contribution: Custom Frequency Encoding
The massive jump in performance (from a ~0.37 baseline to **~0.50 Gini**) was driven primarily by a custom approach to high-cardinality categorical features. 

Features like `Make`, `Model`, and `SubModel` have hundreds of unique values. Standard One-Hot Encoding causes the "curse of dimensionality," creating thousands of sparse columns that degrade model performance. Instead, I designed and implemented a custom `FrequencyEncoder` that maps categories to their relative frequencies in the training set. This preserved the statistical weight of the categories while keeping dimensionality strictly low, allowing simple, interpretable linear models to perform exceptionally well and vastly outperform the project's baseline expectations.

## 🛡️ Robust Pipeline & Best Practices
To support this custom preprocessing, I ensured the rest of the pipeline adhered to strict data science best practices (and project requirements) to prevent data leakage and overfitting:
1. **Chronological Splitting:** Data was split strictly by `PurchDate` (Train/Val/Test) to ensure future auction data never leaked into past predictions.
2. **L1 Regularization (Lasso):** Applied an L1 penalty to the Logistic Regression model to perform embedded feature selection, automatically shrinking noisy, uninformative features to exactly zero.
3. **Non-Linear Features:** Engineered basic domain-specific ratios (e.g., `Odometer / VehicleAge` = Miles per Year) to help the linear model capture simple feature interactions.

---

## 🛠️ Tech Stack
* **Core:** Python, pandas, scikit-learn, numpy
* **Evaluation:** Optimized specifically for the Gini Coefficient (ROC-AUC) and PR-AUC.

---

## 📂 Project Structure

```text
├── src/                        # Modular, production-ready Python code
│   ├── preprocessing.py        # Custom FrequencyEncoder & NonLinearFeatureBuilder
│   ├── pipeline.py             # Sklearn ColumnTransformer & Pipeline setup
│   └── train.py                # End-to-end training & evaluation script
├── notebooks/                  # Visual storytelling and EDA
│   ├── 01_eda_and_feature_engineering.ipynb
│   └── 02_modeling_and_results.ipynb
├── data/                       # Data directory (CSVs not committed)
├── models/                     # Saved .joblib pipeline artifacts
└── requirements.txt            # Python dependencies
```

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/K-Margaret/car-auction-risk-prediction.git
   cd car-auction-risk-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add the dataset:**
   Download the Kaggle *Don't Get Kicked!* dataset and place `training.csv` inside the `data/raw/` directory.

4. **Run the training pipeline:**
   ```bash
   python src/train.py
   ```
   *This will execute the chronological split, train the pipeline, print the Gini/AUC metrics for all sets, and save the final model to the `models/` folder.*
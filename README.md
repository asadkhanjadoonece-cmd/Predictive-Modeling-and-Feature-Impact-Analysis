# 🩺 Predictive Modeling and Feature Impact Analysis for Hypertension Diagnosis

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)
![Machine Learning](https://img.shields.io/badge/Domain-Healthcare%20AI-green.svg)

## 📌 Project Overview
Hypertension (elevated blood pressure) affects an estimated 1.3 billion adults globally and is a leading contributor to stroke, heart attack, and premature death. Often called the "silent killer," it is frequently under-diagnosed until complications arise. 

This project investigates whether routinely collected demographic and lifestyle data can be used to build an accurate, interpretable classifier for hypertension risk. Beyond standard model comparison, this work places particular emphasis on **feature impact analysis**: how model accuracy and generalization change as features are incrementally added or removed.

**Author:** Asad Khan | Machine Learning Engineer  
**Contact:** Asadkhanjadoon.ece@gmail.com  

---

## 📊 Dataset Description
The analysis is built on `hypertension_dataset.csv`, containing ~1,900 patient records. Each row represents a single patient, combining demographic attributes, lifestyle behaviors, and clinical history.

### Feature Dictionary
| Feature | Type | Description |
| :--- | :--- | :--- |
| **Age** | Numeric | Patient age in years. |
| **Salt_Intake** | Numeric | Estimated daily dietary salt intake. |
| **Stress_Score** | Numeric | Self-reported psychological stress rating. |
| **BP_History** | Categorical | Prior blood-pressure classification/history (Multi-class). |
| **Sleep_Duration** | Numeric | Average hours of sleep per night. |
| **BMI** | Numeric | Body Mass Index. |
| **Medication** | Categorical | Current medication category (Multi-class). |
| **Family_History** | Binary | Family history of hypertension (Yes/No). |
| **Exercise_Level** | Categorical | Self-reported exercise frequency (Low/Moderate/High). |
| **Smoking_Status** | Binary | Smoker or Non-Smoker. |
| **Has_Hypertension** | Binary | **Target Variable:** Diagnosis of hypertension (Yes/No). |

---

## ⚙️ Data Preprocessing & Feature Engineering
To ensure robust model performance, the following preprocessing steps were applied:
1. **Handling Missing Values:** Records with null values in critical fields were removed rather than imputed to avoid introducing synthetic bias.
2. **Outlier Capping (Winsorization):** Continuous variables (`BMI`, `Sleep_Duration`) were screened using the Interquartile Range (IQR) method. Extreme values (beyond 1.5xIQR) were capped to prevent skewing distance-based models.
3. **Categorical Encoding:**
   * *Binary Encoding:* `Family_History` and `Smoking_Status` mapped directly to 1/0.
   * *One-Hot Encoding:* Multi-class variables (`BP_History`, `Medication`, `Exercise_Level`) transformed using `pd.get_dummies(drop_first=True)` to avoid multicollinearity.
4. **Feature Scaling:** `StandardScaler` was applied to numeric features for Logistic Regression and KNN to prevent features with larger scales (e.g., Age) from dominating distance calculations.

---

## 🤖 Methodology: Classification Algorithms
Three supervised learning algorithms were trained and evaluated under identical conditions (80/20 train-test split):

1. **Logistic Regression:** Used as the interpretable baseline linear model.
2. **K-Nearest Neighbors (KNN):** A non-parametric, instance-based algorithm sensitive to the curse of dimensionality.
3. **Random Forest:** An ensemble method aggregating decorrelated decision trees, inherently robust to outliers and non-linear relationships.

---

## 📈 Results and Analysis

### Baseline Performance (All Features)
| Model | Accuracy | Precision (avg) | Recall (avg) | F1-Score (avg) |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.9328** | 0.93 | 0.93 | 0.93 |
| **Logistic Regression** | 0.8782 | 0.88 | 0.88 | 0.88 |
| **K-Nearest Neighbors (KNN)** | 0.7773 | — | — | — |

*Figure 1: Model accuracy comparison using the full feature set.*
> `![Model Accuracy Comparison](link-to-your-accuracy-bar-chart.png)`

### 🔍 Feature Impact Analysis: The Bias-Variance Tradeoff
A core component of this project was incrementally adding and removing features to identify the optimal feature subset. 

*   **The Danger of Too FEW Features (Underfitting):** When the feature set was aggressively reduced to just 3 variables (Age, BMI, Sleep), accuracy collapsed to **0.59** (basically a coin flip). The model lacked the signal needed to learn.
*   **The Danger of Too MANY Features (Overfitting):** As more features were added, KNN and Logistic Regression began to degrade. KNN suffered from the *Curse of Dimensionality* (distance metrics become meaningless in high-dimensional space), and Logistic Regression suffered from multicollinearity.
*   **The Sweet Spot:** Empirically, **7 features** provided the maximum signal-to-noise ratio. 
*   **Random Forest Robustness:** Unlike the other models, Random Forest did not degrade as more features were added. It plateaued at its ceiling of ~93.3%, as the ensemble structure inherently ignores irrelevant features during tree splits.

*Figure 2: Bias-variance tradeoff - test accuracy vs. number of features added.*
> `![Bias-Variance Tradeoff](link-to-your-line-graph.png)`

### 🩺 Top Predictors of Hypertension (Logistic Regression Coefficients)
By interpreting the standardized coefficients, we identified the following clinical insights:
*   **Age (+0.461):** Increases hypertension likelihood.
*   **BMI (+0.327):** Increases hypertension likelihood.
*   **Sleep_Duration (-0.258):** Decreases hypertension likelihood (Protective factor).

---

## ⚠️ Limitations
While the models performed well, the following limitations must be acknowledged:
1. **Small Dataset Size:** (~1,900 rows) limits statistical power and increases the risk of sample-specific noise.
2. **Self-Reported Data:** Features like salt intake, stress, and sleep are self-reported, introducing potential reporting bias.
3. **Missing Genetic/Biometric Markers:** The absence of genetic risk scores or direct blood pressure readings means the model is an approximation, not a diagnostic replacement.

---

## 🚀 Future Work
*   Deploy the model as a web application for real-time risk scoring.
*   Test advanced models (XGBoost, LightGBM, Deep Learning) to determine if further accuracy gains are achievable.
*   Gather more patient data from multiple clinical sites to validate the 7-feature "sweet spot" across different populations.

## 🛠️ Installation & Usage
To run the code in this repository:

```bash
# Clone the repository
git clone https://github.com/asadkhanjadoonece-cmd/Predictive-Modeling-and-Feature-Impact-Analysis

# Navigate to the directory
cd hypertension-prediction

# Install required dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

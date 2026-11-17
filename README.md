# 🏦 Loan Approval Prediction Model

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Production--Ready-green)

## 📖 Project Overview
This project features a commercially viable **Decision Tree Classifier** machine learning model designed to predict whether a loan application will be **Approved (1)** or **Declined (0)**. 

By analyzing 13 key applicant features—including age, income, employment experience, credit score, loan amount, and previous default history—the model provides a reliable, data-driven risk assessment. The primary goal was to build a highly accurate classification engine that minimizes financial risk for lenders while maintaining fair approval rates.

---

## 🧠 Model Architecture
The core algorithm is the **Decision Tree Classifier** (`sklearn.tree.DecisionTreeClassifier`). 

A Decision Tree was chosen for its high interpretability, which is crucial in the financial sector. Lenders and regulators need to understand *why* a loan was declined. Decision Trees provide clear, rule-based logic (e.g., "If credit score < 600 AND loan_percent_income > 0.40, then Decline"), making it far more transparent than "black box" models.

---

## ⚙️ Data Preprocessing
Before training, the dataset (45,000 records) underwent critical preprocessing:
1. **Feature Selection**: Isolated 13 predictive features, excluding the target `loan_status`.
2. **Categorical Encoding**: Applied **One-Hot Encoding** (`pd.get_dummies`) to text-based features (`person_gender`, `person_education`, `person_home_ownership`, `loan_intent`, `previous_loan_defaults_on_file`), expanding the feature space to ensure the model could process categorical data mathematically.
3. **Train/Validation Split**: Split the data 80/20 (`random_state=1`) to ensure an honest, unbiased evaluation of model performance.

---

## 🔬 Hyperparameter Tuning & Optimization Process
To achieve a low margin of error and prevent overfitting, I implemented a systematic, step-by-step iterative tuning process. Rather than guessing, I isolated and tested each hyperparameter using a custom validation loop to measure **Accuracy**.

### Step 1: Controlling Tree Depth (`max_depth`)
*Tested:* `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, None]`
* **Observation:** Accuracy climbed steadily from 76.8% (depth 1) to 91.49% (depth 10). Beyond depth 10, the model began to overfit, and accuracy on the validation set dropped to 89.20% (`None`).
* **Selection:** `10`

### Step 2: Leaf Node Constraints (`min_samples_leaf`)
*Tested:* `[10, 20, 40, 50, 100, 150, 200, 250, 300, 350, 400, 500]`
* **Observation:** Forcing the tree to require more samples per leaf smoothed the model. `20` yielded the peak accuracy of **91.64%**, while higher values caused underfitting.
* **Selection:** `20`

### Step 3: Split Validation (`min_samples_split`)
*Tested:* `[2, 3, 4, 5, 6, 7, 8, 10]`
* **Observation:** The model was robust across low values, maintaining 91.64% accuracy. 
* **Selection:** `2` (The default, allowing maximum flexibility for early splits).

### Step 4: Tree Size Limitation (`max_leaf_nodes`)
*Tested:* `[5, 10, 15, 20, 50, 100, 150, 200, None]`
* **Observation:** Accuracy plateaued at 91.64% starting at 100 leaf nodes. 
* **Selection:** `100` (Prevents the tree from growing unnecessarily large and complex).

### Step 5: Handling Class Imbalance (`class_weight`)
*Tested:* `['balanced', None]`
* **Observation:** While `'balanced'` is often used for imbalanced data, it actually *decreased* overall accuracy to 88.92% in this specific dataset. The default `None` maintained the superior 91.64%.
* **Selection:** `None`

---

## 🏆 Final Model Configuration & Results
After the optimization process, the final production model was instantiated with the following parameters:

```python
DecisionTreeClassifier(
    random_state=1,
    max_depth=10,
    min_samples_leaf=20,
    min_samples_split=2,
    max_leaf_nodes=100,
    class_weight=None
)
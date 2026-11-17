# 🏦 Loan Approval Prediction Model

## 📖 Project Overview

This project builds a **Decision Tree Classifier** that predicts whether a loan application will be **Approved (1)** or **Declined (0)**.

Using 13 applicant and loan features (age, income, employment experience, credit score, loan amount, interest rate, previous default history and more), the model learns the patterns behind past approval decisions and applies them to new applications. The goal was to build an accurate, **explainable** classification model and to understand what its results mean for a lender.

**Final result: 91.64% validation accuracy**, compared with 89.20% for the untuned model and 76.87% for a naive "always decline" baseline.

| | |
|---|---|
| **Problem type** | Supervised binary classification |
| **Algorithm** | Decision Tree Classifier (scikit-learn) |
| **Data** | 45,000 loan applications, 13 features, no missing values |
| **Tools** | Python, pandas, scikit-learn, joblib, Jupyter |

---

## 🧠 Model Architecture

The core algorithm is the Decision Tree Classifier (`sklearn.tree.DecisionTreeClassifier`).

A decision tree makes predictions by asking a series of yes/no questions about the applicant, like a flowchart, until it reaches a final group (a leaf) that gives the outcome.

A Decision Tree was chosen for its **interpretability**, which matters in lending: staff and regulators need to understand why an application was declined. The logic is rule-based. For example (an illustration of the style of rule, not an actual rule from this model):

> *If there is a previous default on file AND loan_percent_income is high, then Decline.*

Trees are generally easier to explain than "black box" models such as neural networks, although a tree as deep as this one (up to 100 leaves) is much harder to read in full than a small one.

---

## ⚙️ Data Preprocessing

1. **Feature selection:** isolated 13 predictive features, with `loan_status` as the target (1 = approved, 0 = declined).
2. **Categorical encoding:** applied one-hot encoding (`pd.get_dummies`) to the text features (`person_gender`, `person_education`, `person_home_ownership`, `loan_intent`, `previous_loan_defaults_on_file`), because scikit-learn decision trees need numeric inputs.
3. **Train/validation split:** `train_test_split(random_state=1)` with scikit-learn's default 75/25 split, giving **33,750 training rows and 11,250 validation rows**.

**Class balance:** in the validation set, 76.9% of applications were declined and 23.1% approved. Because the classes are imbalanced, accuracy alone can mislead, so precision, recall and F1 are reported as well.

---

## 🔬 Hyperparameter Tuning & Optimization

To control overfitting, I tuned the model in a step-by-step process, testing one hyperparameter at a time with a custom validation loop and carrying the best value forward.

**Baseline:** an untuned tree (no limits) scored **89.20%**.

### Step 1: Tree depth (`max_depth`)
Tested: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, None]`

- **Observation:** accuracy climbed from 76.87% (depth 1, the same as always predicting "declined") to 91.49% at depth 10. Beyond that the tree began to overfit, and accuracy fell to 89.20% with no limit.
- **Selection:** `10`

### Step 2: Leaf size (`min_samples_leaf`)
Tested: `[10, 20, 40, 50, 100, 150, 200, 250, 300, 350, 400, 500]`

- **Observation:** requiring a minimum number of applications per leaf stops the tree making rules from a handful of cases. `20` gave the peak of 91.64%, and larger values caused underfitting (down to 90.25% at 500).
- **Selection:** `20`

### Step 3: Split size (`min_samples_split`)
Tested: `[2, 3, 4, 5, 6, 7, 8, 10]`

- **Observation:** every value gave 91.64%. This setting had no effect, because with `min_samples_leaf=20` a group already needs at least 40 applications to split into two valid leaves.
- **Selection:** `2` (the default)

### Step 4: Tree size (`max_leaf_nodes`)
Tested: `[5, 10, 15, 20, 50, 100, 150, 200, None]`

- **Observation:** accuracy rose from 90.00% (5 leaves) and plateaued at 91.64% from 100 leaves upward, so the other limits already keep the tree under that size.
- **Selection:** `100` (a safe ceiling that prevents unnecessary growth)

### Step 5: Class imbalance (`class_weight`)
Tested: `['balanced', None]`

- **Observation:** `'balanced'` lowered accuracy to 88.92%, while `None` kept 91.64%. This comparison used accuracy only. Balancing usually raises recall on the rarer class (approved), so a lender who cares more about catching good applicants may still want to test it.
- **Selection:** `None`

---

## 🏆 Final Model Configuration

```python
DecisionTreeClassifier(
    random_state=1,
    max_depth=10,
    min_samples_leaf=20,
    min_samples_split=2,
    max_leaf_nodes=100,
    class_weight=None
)
```

The configuration was evaluated on the validation set, then **refitted on all 45,000 records** and saved as `loan_approval_model.pkl` with joblib.

---

## 📊 Results

| Model | Accuracy |
|---|---|
| Always predict "declined" (naive baseline) | 76.87% |
| Untuned decision tree | 89.20% |
| **Tuned decision tree** | **91.64%** |

Tuning improved accuracy by **2.44 percentage points**, and the tuned model beats the naive baseline by almost 15 points.

### Classification report (11,250 validation applications)

| | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Declined (0) | 0.92 | 0.97 | 0.95 | 8,648 |
| Approved (1) | 0.88 | 0.74 | 0.80 | 2,602 |
| **Accuracy** | | | **0.92** | 11,250 |
| Macro average | 0.90 | 0.85 | 0.88 | 11,250 |
| Weighted average | 0.91 | 0.92 | 0.91 | 11,250 |

- **Precision (approved = 0.88):** of the applications predicted as approved, 88% really were.
- **Recall (approved = 0.74):** of the applications that really were approved, the model found 74%.

### Approximate confusion matrix
*(reconstructed from the rounded report values)*

| | Predicted declined | Predicted approved |
|---|---|---|
| **Actually declined** | about 8,385 | about 263 |
| **Actually approved** | about 677 | about 1,925 |

---

## 💼 Business Interpretation

### What the results mean

- **The model is a strong decline detector.** It catches about 97% of declined applications, and 92% of the applications it flags as declined really were declined.
- **When it predicts approval, it is usually right** (88% precision).
- **It is conservative.** It misses roughly 1 in 4 approvals (about 677 of 2,602), meaning it predicts "declined" for applications that were actually approved.

### The cost of each error

| Error | What happens | Business impact |
|---|---|---|
| **False decline** (about 677) | A good applicant is turned away | Lost interest income and a lost customer |
| **False approval** (about 263) | An application the original process declined is approved | Potential credit losses and collection costs |

For most lenders a false approval costs more than a false decline, so this model's cautious profile (high precision, lower recall for approvals) leans in a sensible direction. The right balance depends on real cost figures: the cut-off can be moved with `predict_proba`, or by testing `class_weight="balanced"`, to trade precision for recall.

### How a lender could use it

1. **Triage:** fast-track high-confidence approvals, quickly decline high-confidence declines, and send uncertain cases to a human underwriter.
2. **Decision support:** a consistent second opinion for credit officers, not a replacement for them.
3. **Consistency checks:** comparing human decisions to the model can flag unusual or inconsistent outcomes.
4. **Explainability:** the tree's rules can be traced to explain how a decision was reached.

### What the model is not

- **It predicts the recorded approval decision, not repayment.** The target is `loan_status`, so the model learns to imitate past decisions, including any biases in them. A true default-risk model would need data on how loans performed after approval.
- **It is not ready for production lending as it stands.** It has not been tested for fairness, validated on an independent test set, or checked against regulation.

### Fairness and compliance

The features include **gender, age and education**. Personal characteristics like these can create discrimination risk in lending and may be restricted by regulation, depending on the country. Before any real use: remove `person_gender`, review the other personal attributes with a compliance specialist, compare approval and error rates across groups, and keep a human in the loop.

---

## 🧾 Final Summary

| | |
|---|---|
| **Model** | Decision Tree Classifier, tuned |
| **Task** | Predict loan approval (1) or decline (0) |
| **Data** | 45,000 applications, 13 features, 77% declined / 23% approved |
| **Key settings** | `max_depth=10`, `min_samples_leaf=20`, `max_leaf_nodes=100` |
| **Accuracy** | **91.64%** (baseline 76.87%, untuned 89.20%) |
| **Declined class** | Precision 0.92, recall 0.97, F1 0.95 |
| **Approved class** | Precision 0.88, recall 0.74, F1 0.80 |
| **Strength** | Accurate and explainable, very reliable at spotting declines |
| **Weakness** | Misses about 26% of approvals; learns past decisions, not repayment |
| **Best use** | Triage and decision support with human review |
| **Next steps** | Test set and cross-validation, random forest or gradient boosting, feature importance, fairness audit |

---

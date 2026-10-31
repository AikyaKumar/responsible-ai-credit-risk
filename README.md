# Responsible AI on a Credit-Risk Classifier: leakage audit, fairness, explainability, red teaming

**TL;DR:** My first version of this model scored 99.2% accuracy. A feature ablation showed it was rebuilding the label from the inputs (**PR-AUC 0.975 → 0.082** once the leaking features were removed; base rate 0.046). This repo documents the full workflow: finding the leak, fixing the pipeline, auditing fairness by group, explaining predictions with SHAP and LIME, red-teaming the model, and documenting risks in a model card and a risk plan.

> **Important limitation:** the target is a *synthetic* rule on the inputs, not real default data. The numbers show how well a model re-learns that rule, not real credit risk.

## What's here
- End-to-end notebook: `responsible_ai.ipynb` (aims, ethics and risk, EDA, models, fairness, explainability, red team, model card, risk plan)
- Dataset: Kaggle *BankChurners* (10,127 customers; **not included**, see below)

## Key findings
**1. Target leakage.** Only one clause of the label rule fires (`Avg_Utilization_Ratio > 0.8`, 467 customers, 4.6%). `Avg_Open_To_Buy = Credit_Limit − Total_Revolving_Bal` exactly, so dropping only the target-defining columns is not enough.

| Feature set | Model | Accuracy | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| A: all features (leaky) | MLP | 0.992 | 0.975 | 0.999 |
| A: all features (leaky) | Logistic reg. | 0.984 | 0.998 | 1.000 |
| B: target-defining dropped | MLP | 0.968 | 0.649 | 0.976 |
| B: target-defining dropped | Logistic reg. | 0.904 | 0.940 | 0.997 |
| **C: + proxy dropped (clean)** | MLP | 0.900 | **0.082** | 0.718 |
| **C: + proxy dropped (clean)** | Logistic reg. | 0.661 | **0.095** | 0.788 |

Held-out test set: 3,039 rows, 140 high-risk (KNN is also in the notebook). On the clean set the models are only modestly better than chance (PR-AUC about 1.8× the 4.6% base rate), and MLP accuracy (0.900) is below an always-"low risk" baseline (0.954).

**2. Fairness (clean MLP).** Education: small gaps (flag-rate gap 0.032). Income: flag rates fall with income (9.9% → 1.1%) in line with the true high-risk rates, so the gap (0.088) mostly reflects group differences. Equal-opportunity gaps are unreliable (few true positives per group, wide bootstrap intervals).

**3. Explainability.** SHAP's top driver is `Credit_Limit`; education and income categories make up 3 of the top 5 features, a fairness flag worth reviewing before any real use. LIME explanations for a true positive, a false negative and a true negative are in the notebook.

**4. Red team.** Single-feature perturbations flipped at most 6.7% of decisions. Flipping 5% / 10% of training labels lowered PR-AUC from 0.082 to 0.064 / 0.062; a simple filtering defence made no clear difference (single run).

## Method
Split first; encoding, scaling and random oversampling live inside one pipeline fitted on training data only; 5-fold CV on the training set; the test set is evaluated once; fixed seed (42).

## Run it
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Download "BankChurners.csv" from Kaggle (Credit Card customers dataset) and place it next to the notebook.
jupyter notebook
```

## Limitations
Synthetic label; one dataset; fairness checked only on education and income; poisoning tests use random (not targeted) label flips and one seed; SHAP estimated on a sample of 150 test rows; explanations are approximations, not causal.

## Documents inside the notebook
Model card, EU AI Act high-risk mapping, risk register and plan, red-team playbook.

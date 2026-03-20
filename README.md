# Federated-and-Fairness-Aware-Learning-for-Rural-Healthcare-Risk-Prediction-Under-Data-Scarcity
Federated and fairness-aware machine learning framework for rural healthcare risk prediction, enabling privacy-preserving, interpretable, and equitable insights under data scarcity.
A privacy-preserving, fairness-conscious federated ensemble framework for district-level healthcare risk assessment under data scarcity and non-IID conditions.

---

## Overview

Centralized machine learning for healthcare risk prediction in rural India faces three critical barriers: fragmented data across districts, small sample sizes, and strict privacy regulations. This project proposes a federated and fairness-aware ensemble learning framework that enables collaborative model training across dispersed rural healthcare sites without exchanging raw data.

The framework is evaluated on NFHS-5 data spanning 707 Indian districts with 14 socioeconomic and healthcare indicators. It maintains approximately 98% of centralized AUC performance (0.865 vs. 0.879) while reducing socioeconomic disparity gaps and providing interpretable risk factor insights.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Explainability](#explainability)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Affiliation](#affiliation)

---

## Features

| Feature | Description |
|---|---|
| Privacy-Preserving | Laplace noise perturbation; only prediction updates are shared, no raw data |
| Fairness-Aware | TPR gap calibration across socioeconomic groups via threshold optimization |
| Federated Ensemble | LightGBM (65%) + XGBoost (35%) weighted ensemble across heterogeneous clients |
| Non-IID Simulation | Dirichlet allocation (alpha = 0.5) for realistic data fragmentation across clients |
| Explainability Suite | SHAP global importance, LIME local explanations, and partial dependence plots |
| Federated Evaluation | Heterogeneity via JS divergence, uncertainty estimation, feature stability, client attribution |

---

## Architecture

```
                         FEDERATED SERVER
           Sample-Size Weighted Prediction Aggregation
               Fairness Calibration | Explainability
                          |
          (prediction updates only — no raw data)
                          |
        ┌─────────────────┼─────────────────┐
        |                 |                 |
   Client 1          Client 2          Client K
   LightGBM          LightGBM          LightGBM
   Local Training    Local Training    Local Training
   + Laplace Noise   + Laplace Noise   + Laplace Noise
        |                 |                 |
   District Shard    District Shard    District Shard
   (Non-IID)         (Non-IID)         (Non-IID)
```

Global aggregation:

```
p_global = sum( (n_i / N) * p_i )  for i = 1 to K
```

where `n_i` is the client sample size and `N` is the total number of samples across all clients.

---

## Dataset

**Source:** National Family Health Survey (NFHS-5, 2019-21), International Institute for Population Sciences (IIPS)

| Category | Features |
|---|---|
| Infrastructure | Electricity access, Sanitation access, Clean fuel access, Water access |
| Education | Female literacy rate, School attendance rate |
| Demographics | Sex ratio, Population below age 15 |
| Healthcare | Public facility births, Iodized salt consumption, Antenatal care |

**Label Construction:** A weighted composite score from 6 indicators — institutional births, ANC 4+, postnatal care, health insurance coverage, teen marriage, and women's anemia. Districts in the upper 65th percentile are labeled `high-risk`.

- Districts: 707 across India
- Features: 14 socioeconomic and healthcare indicators
- Class balancing: Stratified split

---

## Methodology

### 1. Client Heterogeneity Simulation
Dirichlet allocation (alpha = 0.5) partitions the training data into up to K = 5 clients, with K = 4 effective clients after enforcing a minimum sample size of n >= 30. Inter-client heterogeneity is measured via Jensen-Shannon divergence.

### 2. Local Training
Each client trains a LightGBM model locally. A centralized XGBoost model serves as the baseline for comparison.

### 3. Privacy Layer
Calibrated Laplace noise is applied to local predictions before transmission. No raw features or labels are ever shared — only prediction updates leave the client.

### 4. Server Aggregation
Weighted ensemble combining 65% LightGBM and 35% XGBoost, tuned empirically on the validation set. Final aggregation is sample-size weighted across all clients.

### 5. Fairness Calibration
The optimal decision threshold is selected by maximizing F1 on aggregate validation predictions. A post-hoc TPR gap is then evaluated across low, medium, and high development district groups defined by tercile partitioning of a development composite score.

---

## Results

### Performance Comparison

| Model | AUC | F1-Score |
|---|---|---|
| Centralized Ensemble | 0.879 | 0.719 |
| Federated (No DP Noise) | 0.863 | 0.705 |
| Proposed Federated Fair | 0.865 | 0.705 |

Results averaged over 5 random seeds. AUC standard deviation = 0.01.

### Privacy and Stability

| Metric | Value |
|---|---|
| DP noise utility loss (delta AUC) | -0.014 |
| Membership inference proxy risk | 0.085 (avg) |
| AUC std dev across seeds | 0.01 |

### Ablation Study

| Configuration | AUC | F1 |
|---|---|---|
| Full System | 0.865 | 0.705 |
| No DP Noise | 0.863 | 0.705 |
| No Ensemble Weighting | ~0.847 | — |
| Single Client | ~0.825 | — |
| Fewer Rounds (10) | ~0.825 | — |
| Centralized Baseline | 0.879 | 0.719 |

---

## Explainability

### Top Risk Factors (Global SHAP)

1. `pop_below_15` — Population below age 15 (demographic pressure)
2. `births_public` — Public facility birth rate
3. `clean_fuel` — Clean fuel access
4. `infra_score` — Infrastructure composite score
5. `electricity` — Household electricity access

### Key Findings

- Better infrastructure and healthcare access correlates with lower district risk
- Higher youth population ratio increases district vulnerability
- Feature importance rankings remain stable across non-IID clients, irrespective of data distribution
- Privacy-preserving aggregation does not degrade instance-level interpretability, verified via LIME consistency

---

## Installation

```bash
git clone https://github.com/<your-username>/ffl-bhtc.git
cd ffl-bhtc

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Requirements

```
lightgbm>=3.3.0
xgboost>=1.6.0
scikit-learn>=1.1.0
numpy>=1.23.0
pandas>=1.4.0
shap>=0.41.0
lime>=0.2.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.9.0
jupyter>=1.0.0
```

---

## Usage

```bash
jupyter notebook ffl-bhtc_FINAL.ipynb
```

```python
# Load and preprocess NFHS-5 data
from src.data import load_nfhs_data
X, y = load_nfhs_data("data/nfhs5_districts.csv")

# Simulate federated clients via Dirichlet partitioning
from src.federation import dirichlet_partition
clients = dirichlet_partition(X, y, alpha=0.5, n_clients=5, min_samples=30)

# Train federated ensemble
from src.federated import FederatedFairEnsemble
model = FederatedFairEnsemble(lgbm_weight=0.65, xgb_weight=0.35, noise_scale=0.1)
model.fit(clients, rounds=20)

# Evaluate with fairness metrics
from src.evaluation import federated_eval_suite
results = federated_eval_suite(model, X_test, y_test, sensitive_attr="dev_score")
print(results)

# Generate SHAP explanations
from src.explainability import shap_global_analysis
shap_global_analysis(model.proxy_model, X_test)
```

---

## Project Structure

```
ffl-bhtc/
|
├── ffl-bhtc_FINAL.ipynb        # Main experiment notebook
├── FFL_BHTC2026.pdf            # Research paper
|
├── data/
|   ├── nfhs5_districts.csv     # NFHS-5 district-level data
|   └── processed/              # Preprocessed splits
|
├── src/
|   ├── data.py                 # Data loading and preprocessing
|   ├── federation.py           # Dirichlet partitioning and client simulation
|   ├── federated.py            # Federated ensemble training
|   ├── privacy.py              # Laplace noise mechanism
|   ├── fairness.py             # Threshold calibration and TPR gap
|   ├── evaluation.py           # Federated evaluation suite
|   └── explainability.py       # SHAP, LIME, and PDP analysis
|
├── results/
|   ├── figures/                # ROC curves, confusion matrices, SHAP plots
|   └── metrics/                # Performance tables
|
├── requirements.txt
└── README.md
```

---

## Future Work

- **Real-time deployment** on live district-level health data streams with adaptive client participation
- **Formal differential privacy guarantees** replacing the current empirical Laplace noise mechanism
- **Multimodal data integration** incorporating satellite imagery, facility census records, and longitudinal survey data
- **Adaptive client selection** to handle extreme stragglers and highly unbalanced client participation
- **Cross-state generalization** testing the framework on non-Indian rural healthcare datasets to validate transferability
- **Direct policy integration** with district health management systems for automated resource prioritization

---

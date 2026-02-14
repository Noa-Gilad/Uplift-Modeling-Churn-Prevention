# WellCo Churn Prevention - Uplift Modeling

Prioritized outreach list for WellCo by ranking members on predicted **uplift** (reduction in churn if contacted). The final ranked list is produced on the provided test set.

---

## Setup and run

### Prerequisites

- Python 3.10+ (or as in your environment)
- Virtual environment recommended (e.g. `venv` or `conda`)

### Install

From the project root:

```bash
pip install -r requirements.txt
```

### Data layout

Place the assignment data under a `files/` directory at the **project root** (sibling to `src/`):

- **Required for run**
  - `files/train/` — training CSVs: `web_visits.csv`, `app_usage.csv`, `claims.csv`, `churn_labels.csv`
  - `files/test/` — test CSVs: `test_web_visits.csv`, `test_app_usage.csv`, `test_claims.csv`, `test_members.csv`
  - `files/wellco_client_brief.txt` — client brief (reference)

After the EDA step, the pipeline expects:

- `files/engineered/train_engineered.csv`
- `files/engineered/test_engineered.csv`

The final deliverable is written to:

- **`files/top_n_members_outreach.csv`** — columns: `member_id`, `prioritization_score`, `rank` (sorted by priority, rank 1 = highest uplift).

### Run order

Run from the **`src/`** directory (so that paths resolve correctly).

1. **EDA and feature engineering**  
   Run `src/uplift_churn_EDA.ipynb` top to bottom.  
   This reads raw train/test CSVs and writes `files/engineered/train_engineered.csv` and `files/engineered/test_engineered.csv`.

2. **Uplift modeling and export**  
   Run `src/uplift_churn_prediction.ipynb` top to bottom.  
   This loads the engineered datasets, trains the chosen uplift model, scores the test set, and exports the top fraction to **`files/top_n_members_outreach.csv`** via `export_top_fraction(..., out_path=FILE_DIR / "top_n_members_outreach.csv")`.

---

## Approach (concise)

- **Objective:** Rank members by expected incremental retention from outreach and determine an optimal outreach size (*n*) under constant marginal cost.

- **Causal framing:** The outreach event is treated as a treatment variable. The task is formulated as estimating individual Conditional Average Treatment Effects (CATE) — the reduction in churn probability if contacted.

- **Modeling strategy:** Engineered behavioral and recency features → meta-learner (S- or T-learner with regularized base model) → stratified cross-validation using uplift-specific metrics (AUUC, Qini, Uplift@k) → hyperparameter tuning → final model trained on full training set.

- **Decision rule:** Members are ranked by predicted uplift (expected incremental retention). The recommended *n* is selected based on cross-validated uplift curves and decile stability, with the top 10% identified as the most reliable high-gain segment.

- **Output:** Test set scored by predicted uplift; top fraction exported to CSV for operational outreach.

---

## Assignment points addressed

- **Feature selection and engineering:**
  - **Final feature set (14 columns):** `days_since_signup`, `n_sessions`, `wellco_web_visits_count`, `session_today` / `session_last_week` / `session_older`, `wellco_web_today` / `wellco_web_last_week` / `wellco_web_older`, `claim_today` / `claim_last_week` / `claim_older`, `has_focus_icd`, `n_claims`. All preprocessing (scaling, transforms, binning) is fit on training data only; the same fitted parameters/boundaries are applied to test to avoid leakage.
  - **Web activity embedding (WellCo-relevant visits):** Raw web visits have free-text `title` and `description`. To count only visits relevant to WellCo (and avoid diluting signal with unrelated pages), **sentence embeddings** are used: the WellCo client brief is embedded once with a sentence-transformers model (`all-MiniLM-L6-v2`); each visit’s concatenated title+description is embedded; cosine similarity between visit and brief is computed; visits above a similarity threshold (default 0.2) are retained. From this filtered set `wellco_web_visits_count` and `days_since_last_wellco_web` are derived. **Rationale:** domain-aligned engagement signal without maintaining a keyword list. **Advantages:** reproducible, robust to wording, interpretable (brief defines “relevant”), and one threshold controls strictness.
  - **Dropped as redundant or low-variance (EDA):** *Multicollinearity* — `n_web_visits`, `n_distinct_titles` (high correlation with `wellco_web_visits_count`); `icd_distinct_count`, `n_focus_icd_claims` (strong correlation with `n_claims`). `wellco_web_visits_count` and `days_since_last_wellco_web` are retained for web signal, and `n_claims` and `has_focus_icd` for claims. *Low variance* — `has_app_usage`, `has_web_visits`, `has_claims` are dropped, they almost always 1 in the complete-case subset, so they add little discriminative power.
  - **Scaling and transforms (per feature), justified by EDA distributions:**
    - **days_since_signup:** Min-Max scaling → [0,1]. EDA: spread from 0 to ~600 days, fairly uniform. Min-Max preserves relative tenure on a bounded scale and avoids dominance by the wide range; tenure is a stable predictor of engagement and churn.
    - **n_sessions:** Standardization (Z-score). EDA: unimodal, peak around 8–10, mild right tail to ~20. Z-score is appropriate for this roughly symmetric, unimodal distribution; it centers session volume and puts engagement on a comparable scale.
    - **wellco_web_visits_count:** Log transform `log(x+1)`. EDA: zero-heavy (peak at 0), long right tail to ~60. Log compresses the long tail and reduces skew, making the feature more symmetric and stable for modeling while handling the zero mass.
    - **n_claims:** Min-Max scaling → [0,1]. EDA: right-skewed, peak around 5–7, tail to ~20. Min-Max keeps count granularity on a bounded scale without assuming normality; the skew is left to the model to capture.
    - **has_focus_icd:** Kept as-is (binary 0/1). EDA: mix of 0s and 1s, more balanced than other `has_*` variables; no scaling or binning is applied.
  - **Recency binning (0 | 1–7 | >7 days):** Applied to `days_since_last_session`, `days_since_last_wellco_web`, and `days_since_last_claim`. EDA: each recency variable shows a strong peak at 0 days, smaller bumps at 1–2 and ~5 days, then fast decay. Fixed boundaries **0** = active today, **1–7** = last week, **>7** = lapsed/stale align with this structure; each variable is one-hot encoded into three binaries (e.g. `session_today`, `session_last_week`, `session_older`). Binning is interpretable, captures “active vs. lapsed” without overfitting to raw day counts, and the same boundaries (fit from the training distribution) are applied to validation/test for consistency.
  
- **Model evaluation:**
  - **Metrics used and why:** **AUUC** (area under the uplift curve) measures ranking quality over the full curve—higher means the model orders users by incremental benefit better. **Qini coefficient** summarises the same in a single number and is compared to a random baseline (above random = model adds value). **Uplift@k** (e.g. uplift@10%, uplift@20%) is the average incremental effect when targeting the top k%—directly informs how much retention gain to expect from a given outreach size.
  - **Model and hyperparameter choice:** Meta-models and base learner were chosen via stratified K-fold CV. **T-learner + Ridge** was selected: highest mean AUUC, stable across folds, Qini curve above other models and above random. **Grid search** tuned the base learner (best: `alpha=0.001`; AUUC = 0.0336 ± 0.0162, Qini = 0.327 ± 0.275). The **final model** on the full training set (10,000 samples, 14 features) was trained with hyperparameters `{'alpha': 0.001}`.
  - **ParSHAP:** Partial correlation of SHAP with target on train vs holdout was computed. A non-zero overfitting gap was seen for some features (e.g. `days_since_signup`); for that feature the relationship is at least as strong on holdout, and tenure remains a plausible driver.
  - **Train (80%) vs holdout (20%) evaluation:** On the same 80/20 split used for diagnostic plots, AUUC, Qini, and uplift@k were computed for both partitions. **Train (80%):** AUUC 0.0440, Qini 0.6378, Uplift@5% 0.0897, Uplift@10% 0.0928, Uplift@15% 0.0765, Uplift@20% 0.0719. **Holdout (20%):** AUUC 0.0574, Qini 0.7098, Uplift@5% 0.1560, Uplift@10% 0.1330, Uplift@15% 0.1170, Uplift@20% 0.0958.
  - **Train (80%) vs holdout (20%) plots:** (1) **Qini curve**—stays above the random baseline, so the model adds value across the ranking. (2) **Cumulative uplift curve**—largest gains in the top fraction, so prioritising by score is justified. (3) **Realised uplift by decile**—top deciles (D1–D2) show strong, positive uplift on both splits, so the ranking is consistent and not just in-sample.
  - **Test set:**
    - **Four-quadrant segments** (Radcliffe-style, from predicted uplift and baseline): **Persuadables** (uplift > ε)—the only segment to target; **Sure Things** (|uplift| ≤ ε, low baseline churn)—would retain anyway; **Lost Causes** (|uplift| ≤ ε, high baseline churn)—will churn anyway; **Do-Not-Disturb** (uplift < −ε)—treatment has a negative effect, so they must be avoided. On the test set (10,000 members, 14 features): baseline churn risk median 0.2112, mean 0.2136. Segment distribution: **Persuadables** 69.8%, **Sure Things** 11.9%, **Lost Causes** 11.0%, **Do-Not-Disturb** 7.2%.
    - **SHAP:** On the full test set, the most influential features for predicted uplift are **claim_last_week**, **wellco_web_last_week**, and **session_today**—all show a strong positive correlation where the presence of recent activity (value 1) significantly increases the predicted treatment effect. Cumulative metrics **n_claims** and **wellco_web_visits_count** show that high historical volume negatively impacts uplift, likely identifying “Sure Things” who are already saturated. **days_since_signup** acts as a positive driver, indicating that long-term users who have remained active recently are the most responsive to treatment. For the **top 10% by predicted uplift**, the most influential features are **session_today**, **session_last_week**, and **wellco_web_visits_count**: the binary indicators (recent activity = 1) drive high predicted gain, while lower cumulative totals for wellco_web_visits_count and n_claims are associated with the highest uplift, confirming that the model isolates “Persuadables” who are not yet over-saturated. **days_since_signup** remains a positive driver (long-term tenure combined with active recent sessions).
    - **Business metrics (holdout-style summary on test):** At top 5%, 10%, and 20% targeting, churns prevented are 15.60%, 13.30%, and 9.58% respectively, with lift over random of 11.0×, 9.39×, and 6.76× at those fractions.

- **Selecting *n* (outreach size):**
  - **Objective:** The deliverable is a ranked list of the top *n* members for outreach. Outreach incurs a constant marginal cost, so *n* must maximise expected incremental retention, not simply predicted churn.
  - **Signals used (train and holdout):** Selection of *n* was based on three consistent signals across splits:
    - **Qini curves:** The curve remains above the random baseline across most targeting proportions, confirming positive incremental effect.
    - **Cumulative uplift curve:** The steepest gains occur within the top fraction of the ranked population; beyond that, marginal incremental benefit flattens.
    - **Realised uplift by decile:** Deciles D1–D2 show strong, positive uplift in both train and holdout; beyond ~20%, uplift becomes smaller and less stable.
  - **Why top 10%:** The top 10% was selected as the primary recommended outreach size because: (1) it lies within the region of highest marginal uplift on holdout (e.g. holdout Uplift@10% 0.1330 vs Uplift@20% 0.0958); (2) it generalises consistently from train to holdout; (3) it avoids the flattening region where incremental benefit diminishes; (4) it minimises exposure to segments with weak or potentially negative uplift. With constant marginal cost, optimal *n* is the largest prefix of the ranked list where expected incremental gain remains clearly positive and stable—the top 10% satisfies this and is reported as the primary targeting recommendation.
  - **Export:** The notebook exports a configurable top fraction (default 10%), allowing *n* to be adjusted for business cost assumptions or operational constraints.

- **Using outreach in modeling:** The outreach event (between observation and churn window) is the **treatment**. A T-learner is used: separate outcome models are fitted for control and treated units, and CATE is estimated as the difference between the two predictions. The model is trained on labeled data with treatment/control and predicts CATE; the prioritization score is the predicted uplift (retention gain), so the ranked list favors “persuadables.”

---

## Repository structure

```
├── files/                    # Data (create and populate)
│   ├── train/                # Training CSVs - not in git
│   ├── test/                 # Test CSVs - not in git
│   ├── engineered/           # Created by EDA notebook
│   ├── top_n_members_outreach.csv   # Final deliverable (created by uplift notebook)
│   └── wellco_client_brief.txt - not in git
├── src/
│   ├── uplift_churn_EDA.ipynb
│   ├── uplift_churn_prediction.ipynb
│   └── utils.py
├── requirements.txt
└── README.md
```

---

## Deliverables

- **This README** — setup, run instructions, and approach summary.
- **Final CSV** — `files/top_n_members_outreach.csv` (after running both notebooks).
- **Executive presentation** — will be added to the final submission.

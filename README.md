# Internal migration in Kazakhstan — Version 2

**Students:** Aruzhan Rysbayeva and Alua Sagyndyk · **Group:** SE-2403

We predict whether a region has a positive or negative internal migration balance. Version 2 compares the original **5 predictors** with **10 predictors**, adding education, healthcare and housing.

[Open the executed notebook in Google Colab](https://colab.research.google.com/github/aluasagyndyk2/kazakhstan-internal-migration-ml/blob/main/Assignment3_V2.ipynb)

## What we did

1. Combined twelve official Bureau of National Statistics (BNS) workbooks.
2. Prepared **164 eligible region-year observations for 2015–2025**: 40 positive, 124 negative.
3. Used predictor values from the preceding year.
4. Compared Logistic Regression, Decision Tree, Random Forest and Gradient Boosting, plus majority and previous-year-sign references.
5. Checked each next year after training on earlier years, for 2019–2025.

Both feature sets use exactly the same observations and rules. The 2025 model includes 2024 outcomes in training.

## Predictors

| Group | Predictors from the preceding year |
|---|---|
| Original five | Relative income, relative wage, unemployment, urban share, log population |
| Education | Schools and higher-education students per 10,000 residents |
| Healthcare | Doctors and hospital beds per 10,000 residents |
| Housing | Completed residential floor area, m² per resident |

Education years use the academic-year start (2014/15 → 2014). Housing values in the source sheet are **thousand m²**, converted to m² before population normalisation. Service ratios use average annual population; they are not direct measures of service quality. Split-affected regions are excluded before their data become comparable. No rows or historical values are invented.

## Main results

| Comparator | Core 2024–2025 accuracy | Extended accuracy | Core macro F1 | Extended macro F1 |
|---|---:|---:|---:|---:|
| Majority dummy | 0.775 | 0.775 | 0.437 | 0.437 |
| Logistic Regression | 0.850 | 0.900 | 0.785 | 0.844 |
| Decision Tree | 0.875 | 0.900 | 0.813 | 0.844 |
| Random Forest | 0.900 | 0.950 | 0.844 | 0.922 |
| Gradient Boosting | 0.900 | 0.950 | 0.844 | 0.922 |
| Previous-year sign | 0.950 | 0.950 | 0.922 | 0.922 |

Random Forest is selected for each feature set using **2019–2023 only**. Its mean yearly macro F1 across all seven origins is **0.882901 (core)** and **0.938785 (extended)**; persistence scores **0.937811**. The final slice gives 38/40 correct for extended Random Forest, matching persistence. These are descriptive results on a small repeated-region sample, not proof of a clear general advantage over persistence.

**Important:** 2024–2025 were already inspected in Version 1. Version 2 is an exploratory extension; this final slice is **not a fresh untouched test**. V1 used a fixed train-through-2023 split, while V2 expands training each year. Use the rerun core results above for the matched V2 comparison, not old V1 scores.

Education has the largest validation group permutation importance. Real SHAP explanations are included for 2023 out-of-time cases, using only earlier training rows as background. Neither method proves causal effects.

## Run in Colab

Open the notebook link, run the first setup cell, then **Runtime → Run all**. The setup downloads the complete repository; do not upload only the notebook. The analysis and SHAP calculation can take several minutes. A local run is verified; the notebook is prepared for Colab but no claim is made that it was executed on Google's servers.

## Run locally

Use Python 3.12 in a separate environment:

```bash
git clone https://github.com/aluasagyndyk2/kazakhstan-internal-migration-ml.git
cd kazakhstan-internal-migration-ml
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python build_dataset.py
python test_data.py
python test_analysis.py
python run_analysis.py
python verify_results.py
```

Raw sources are included, so data downloads are not required. Use the current `build_dataset.py` and `run_analysis.py`; the core extraction and model helpers are imported by those entry points.

## Files

- `source_manifest.json`: twelve direct source URLs, update/access dates and original file hashes.
- `data/raw/`: unchanged official Excel workbooks, including one original XLS file.
- `data/processed/model_dataset.csv`: matched dataset for both feature sets.
- `data/processed/feature_dictionary.csv`: ten predictors and groups.
- `data/processed/cell_lineage.csv`: every extracted source cell and value.
- `data/processed/excluded_rows.csv`: territorial exclusions.
- `results/`: scores, individual predictions, errors, importance, SHAP, configuration and verification.
- `Assignment3_V2.ipynb`: executed notebook with model code, data and results.
- `docs/`: simple English report and Russian defence notes.
- `source_pages/`: archived official source listings.

## Sources and dates

All twelve archived originals were accessed on **24 October 2026**. Source update dates describe the downloaded publication snapshot, not the years of observation. The feature years used are 2014–2024; school years are 2014/15–2024/25. Outcomes are 2015–2025. Exact sources are listed in `source_manifest.json` and the report.

## Versions and transparency

- `v1-economic-only`: original five-feature implementation and fixed split.
- `v2-extended-features`: extended ten-feature comparison and new rolling protocol.

Results were generated by real runs and independently checked. The code license does not relicense BNS source statistics; those retain their source terms.

# Regional internal migration in Kazakhstan

Assignment 3: Methodology and Core Practical Implementation.

**Students:** Aruzhan Rysbayeva and Alua Sagyndyk  
**Group:** SE-2403  

[Open notebook in Google Colab](https://colab.research.google.com/github/aluasagyndyk2/kazakhstan-internal-migration-ml/blob/main/Assignment3_Executed.ipynb)

## Research question

How does the predictive performance of interpretable machine-learning models compare with a statistical baseline for classifying the net internal migration balance of Kazakhstan's regions?

Each observation is a region-year. The label is 1 for positive internal net migration and 0 for negative balance. The study predicts the sign, not the exact number of migrants or causal effects.

## Data and sources

Seven original Bureau of National Statistics of Kazakhstan workbooks are archived in `data/raw/`: internal arrivals, internal departures, internal net migration, average annual population (including urban population), unemployment, per capita monetary income and wages.

- `source_manifest.json`: direct URLs, source-update dates, file sizes and SHA-256 hashes.
- `source_pages/`: frozen official source-listing pages.
- `data/processed/cell_lineage.csv`: source worksheet and cell for all 2,016 extracted values.
- `data/processed/data_dictionary.csv`: variable definitions and units.

The audit extraction has 240 rows for 2014-2025. The eligible modelling sample has **164 rows for 2015-2025**, with 40 positive and 124 negative labels. Fifty-six candidate rows are excluded conservatively because of the 2018/2022 territorial splits and incompatible earlier economic coverage. All 20 current regional units appear from outcome year 2023. See `excluded_rows.csv` for every exclusion.

Use `model_dataset.csv` for modelling. The audit table `regional_panel.csv` retains early source values whose economic geography may be incompatible; its compatibility flag must be respected.

## Models and validation

Five predictors from the preceding year: relative income, relative wage, unemployment, urban population share and log population. Current-year migration components and region names are excluded from the input features.

The four learned comparators are Logistic Regression, Decision Tree, Random Forest and Gradient Boosting. Additional benchmarks are a majority-class dummy and the previous-year migration sign. All learned models use the same features, training-only preprocessing and inverse-frequency training weights.

Five expanding validation years (2019-2023) select the model family using mean macro F1. The final held-out test is **2024-2025: 40 observations**. Hyperparameters are fixed; seed is 42; threshold is 0.5. Validation permutation importance uses 50 within-year shuffles per feature per fold. No SHAP analysis or synthetic data is used.

## Actual results

Random Forest was selected on validation, with mean macro F1 **0.898561**.

| Comparator | Test accuracy | Test macro F1 |
| --- | ---: | ---: |
| Majority dummy | 0.775 | 0.436620 |
| Logistic Regression | 0.875 | 0.813259 |
| Decision Tree | 0.825 | 0.758412 |
| Random Forest | 0.875 | 0.813259 |
| Gradient Boosting | 0.875 | 0.813259 |
| Previous-year sign | 0.950 | 0.921875 |

**H1 is not supported by this initial held-out comparison:** tree-based ML does not improve accuracy over Logistic Regression, and persistence performs better. Small samples, repeated regions and territorial exclusions limit generalisation. Revised statistics downloaded in 2026 and publication lags mean this is not evidence of a real-time forecast available on 1 January. Importance describes model reliance, not causation.

## Run in Google Colab

1. Use the **Open notebook in Google Colab** link above.
2. Run the first code cell: it downloads the complete repository, including the local Python modules and source tables.
3. Run the remaining cells in order, or choose **Runtime → Run all**.

Colab usually already contains the analysis libraries. If a third-party dependency is missing, run `%pip install pandas numpy scikit-learn matplotlib openpyxl`. `build_dataset` and `run_analysis` are local files, not pip packages. For exact original-version reproduction, use Python 3.12 and the pinned requirements in a separate environment; versions actually used are recorded by each run.

## Run locally

```bash
git clone https://github.com/aluasagyndyk2/kazakhstan-internal-migration-ml.git
cd kazakhstan-internal-migration-ml
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python build_dataset.py
python test_data.py
python run_analysis.py
python verify_results.py
```

Install `requirements-notebook.txt` if notebook execution support is needed. No data re-download is necessary for the local pipeline.

## Project files

```text
Assignment3_Executed.ipynb       Executed notebook and Colab setup
build_dataset.py               Extraction, provenance and eligible dataset
run_analysis.py                Models, temporal validation, evaluation and figures
test_data.py                   Official-source and temporal-integrity checks
verify_results.py              Independent metric and model-selection checks
requirements*.txt              Dependencies
source_manifest.json           Official source URLs and hashes
data/raw/                      Seven unchanged BNS workbooks
data/processed/                Dataset, dictionary, cell lineage and exclusions
results/                       Real metrics, predictions, errors, figures and logs
source_pages/                  Source-listing snapshots
docs/                          Assignment 3 English report
```

The revised annual BNS snapshot includes Akmola's +292 balance in 2025; earlier narrative releases can differ. The archived annual-table snapshot is the authority for this experiment.


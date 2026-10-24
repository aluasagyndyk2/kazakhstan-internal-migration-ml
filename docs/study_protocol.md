# Version 2 study protocol

Fixed before the new model run. Version 1 is archived separately.

- Task: classify the sign of regional internal net migration; retain the original research question and hypothesis.
- Data: twelve archived BNS workbooks; matched eligible region-year rows; no synthetic observations.
- Core: income ratio, wage ratio, unemployment, urban share and log population.
- Extended: core plus schools, students, doctors, beds per 10,000 residents and completed housing m2 per resident.
- Use preceding-year predictors. Education uses academic start years; housing thousand m2 is converted to m2.
- Preserve territorial exclusions for the 2018 and 2022 splits. Match official regional spelling variants.
- Fixed LR, Decision Tree, Random Forest and Gradient Boosting settings; seed 42, training class weights, probability threshold 0.5.
- Majority and previous-year-sign references. Identical rows for both feature sets.
- Expanding training years: train on all eligible earlier years, check 2019 through 2025 one year at a time. Thus 2024 outcomes enter 2025 training.
- Choose each learned family on mean macro F1 for 2019-2023 only. Report all seven yearly origins and the previously inspected 2024-2025 slice. No parameter search.
- 2024-2025 were already seen in V1. This is an exploratory extension, not an untouched confirmatory test.
- Validation permutation importance: individual features and joint groups, 50 repeats for each 2019-2023 year.
- SHAP: selected extended family, 2023 cases, 32 earlier training rows as background, 210 maximum evaluations, seed42; check additivity.
- Interpretation is predictive, not causal. Service counts do not measure quality; construction does not measure affordability.
- The report uses simple English, black text and 1200-1700 methodology words; all numerical claims must match saved predictions.

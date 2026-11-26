# Pulmonary dataset and evaluation plan

## Project question

Can readings that a finger-worn ring could plausibly collect support detection or monitoring of pulmonary disease?

A useful model needs matched inputs and labels: ring-compatible signals from the same participant and time period as a clinically confirmed pulmonary diagnosis or a timestamped pulmonary exacerbation. Public datasets found so far cover pieces of this question, but none covers the complete ring-to-disease task.

## Datasets and their valid roles

| Dataset | What it contains | Valid use here | Key limitation |
| --- | --- | --- | --- |
| [TOLIFE Clinical Study A v2.0](https://zenodo.org/records/23263785) | 86 people with COPD monitored across three sites; daily indices and an exacerbation register | Exploratory forecasting of a recorded moderate/severe COPD exacerbation among people already diagnosed with COPD | Multiple non-ring devices and sparse daily derived indices; not a new-disease diagnosis or ring validation |
| [HealthRing](https://www.nature.com/articles/s41597-026-07289-x) | 54 adults; finger-ring red/infrared PPG and 3-axis motion, with reference HR, RR, SpO₂ and blood pressure | Ring-vital estimation, motion-artifact handling, and later sensor-validation work | No COPD, asthma, pneumonia, or exacerbation diagnosis labels |
| [Ultra-Short PPG for COPD Screening](https://zenodo.org/records/18109876) | Six anonymized participant files in each of six duration archives; four COPD labels and two control labels, verified against each CSV | Separate, subject-held-out COPD-vs-control feasibility experiment using PPG waveform segments | All duration archives repeat the same six people. Window count does not increase the independent cohort. This is not a reliable diagnosis model or evidence that the proposed ring can detect COPD. |
| [HUSM and GMicro PPG peak-detection data](https://zenodo.org/records/8280127) | 66 minutes of PPG with systolic-peak annotations, collected from healthy participants and people with cardiopulmonary conditions | PPG peak-detection and signal-processing methods | The release and accompanying paper evaluate peak detection, not a pulmonary diagnosis target with patient-level disease labels; it cannot be added as COPD training data. |
| [Original PPG COPD study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10280226/) and its [Figshare feature matrix](https://doi.org/10.6084/m9.figshare.21280371.v2) | Paper reports 8 COPD and 6 controls; the shared feature matrix has 218,999 rows with 100 engineered features and a class label | Literature review and dataset-integrity audit | The shared feature matrix has no participant IDs, and the control row count (106,815) conflicts with the published per-person table (97,818). The published 50/50 split is by 2-second epochs, not held-out people. It is excluded from the primary evaluation rather than assigning guessed patient IDs. |
| [WestRo COPD and WestRo Porti COPD studies](https://doi.org/10.1002/advs.202203485) | Research reports 54 and 534 participants, with physiological signals and COPD-stage labels | High-priority external cohort to request for future, patient-separated validation | Patient-level data are available from the corresponding author on reasonable request, not as a public download. Access and use conditions must be resolved before training. |

### Do not merge these labels

- COPD diagnosis and future COPD exacerbation are different outcomes. A flare study among known COPD patients cannot diagnose COPD in a general population.
- HealthRing has true ring signals but no pulmonary labels. It can validate signal estimation, not teach a disease classifier.
- TOLIFE has COPD event labels but its channels come from smartwatch, mattress, phone/shoes, and spirometer-related measurements. Similarity to ring features is not proof of ring measurement equivalence.
- Lung-audio datasets require a microphone, which is not part of this ring. Hospital-record, imaging, and symptom-score datasets do not become ring datasets by relabeling or concatenating them.

No asthma, pneumonia, bronchitis, or pulmonary-infection diagnosis model is supported by these inputs and labels. Do not present a condition label unless a future matched cohort and independent validation support it.

## COPD classification from PPG: small feasibility experiment

The six Zenodo archives (`2s_db.zip` through `256s_db.zip`) each contain the same six named people: four files labeled COPD and two labeled control. The evaluator checks that the CSV label agrees with the file's class prefix, then tests each duration archive separately. It extracts scale-normalized waveform shape, derivative, peak, and normalized-spectrum summaries from PPG windows. It compares regularized logistic regression, an RBF support-vector machine, and Extra Trees.

Validation is leave-one-participant-out. All sampled windows from the held-out person stay out of model fitting, preprocessing is fitted inside each fold, and the held-out person's window scores are reduced to one median participant score before calculating metrics. The script deterministically samples up to 512 windows per person from each archive so prolific recordings cannot dominate. No patient-level scores or raw signals are written to the repository.

This is a **six-person feasibility audit**, not a working clinical detector. Its purpose is to add true disease-labeled PPG and test leakage-resistant methods; even favorable metrics would not establish performance in new people or PlumoRing hardware. The much larger Figshare feature matrix cannot repair the small sample because its participant IDs are absent and its class counts conflict with the paper's participant table. The original paper reports a random 50/50 epoch split, which can place a person's windows in both train and test; its reported accuracy is not treated as a participant-held-out result.

### Observed participant-held-out results

The reproducible run used up to 512 sampled windows per person for each archive. In the 2-second archive, regularized logistic regression, RBF SVM, and Extra Trees each scored participant-level ROC-AUC **0.000** and average precision **0.525**; the no-skill average-precision reference is **0.667**. All three methods ranked the six held-out participant scores in the opposite direction from the labels on this split. Across each of the other five duration variants, all three ROC-AUC values were also 0.000 and average precision remained 0.525. This is a failed feasibility result, not evidence of COPD detection. The cohort is too small to infer general performance, and no model is deployed.

## Larger datasets to pursue, without mixing tasks

- TOLIFE is currently the strongest open longitudinal COPD cohort used here for a separate exacerbation-forecast question. It contains 86 diagnosed participants, event records, and daily data from multiple non-ring devices. The grouped baseline failed its no-skill gate.
- WestRo is a much better prospect for a larger COPD-stage experiment, but its data require a request to the authors. It should be acquired with participant IDs and signal/diagnosis alignment intact, then split by person and clinic before any tuning.
- MIMIC-III-Ext-PPG is a large credentialed ICU PPG resource, but its published labels focus on heart rhythm and reference vital signs; a COPD diagnosis code is not a validated pulmonary PPG target. It is not used as a substitute COPD dataset.
- HealthRing is reserved for ring-vital estimation because it lacks disease labels. Keep its evaluation independent from COPD classification.

## TOLIFE exploratory evaluation

Source archive: TOLIFE Clinical Study A, v2.0, `index.zip`. The dataset's quality guidance says to use rows whose `index_validity` is 1. The script follows that rule and writes no raw rows into the repository.

### Target and inputs

- Population: participants already diagnosed with COPD.
- Observation: up to seven days of prior daily readings.
- Target: a clinically registered moderate/severe exacerbation 1 to 7 days after the observation day.
- Required coverage: at least two observed sensor families in the lookback.
- Input families: smartwatch pulse/HRV indices, mattress breathing-rate index, mobility indices from phone/shoes/watch, and the available spot SpO₂ value.
- Excluded: FEV1/PEF, derived CAT/CCQ/mMRC scores, audio, environmental variables, demographics, scheduled clinical visits, and future readings.
- Split: five-fold GroupKFold by participant kit. A participant never appears in both the training and held-out fold.

### Cohort counts

- 86 participants have reading files.
- 68 have eligible participant-days.
- 11 participants have at least one positive forecast day.
- 4,344 eligible participant-days, including 79 positive forecast days (1.82% prevalence).
- The register contains 80 events: 63 moderate and 17 severe.

### Baseline comparison

| Model | Grouped out-of-fold ROC-AUC | Grouped out-of-fold average precision |
| --- | ---: | ---: |
| Logistic regression | 0.4258 | 0.01448 |
| Histogram gradient boosting | 0.3544 | 0.01311 |
| Random forest | 0.4874 | 0.01694 |
| No-skill reference (positive-day prevalence) | — | 0.01819 |

The best average precision is below the no-skill reference, and every model fails the exploratory gate (average precision at least 1.25 times prevalence and ROC-AUC at least 0.60). This is a **failed feasibility result**. Do not export weights, show probabilities, rank patients, or drive lung color/severity from these outputs.

The grouped folds reduce cross-participant leakage. The 79 positive days are still concentrated in only 11 people, so day-level metrics may be unstable; a larger independent cohort and participant-level uncertainty analysis are required before any clinical interpretation.

## Model strategy

The current script compares regularized logistic regression, histogram gradient boosting, and a class-weighted random forest. This is a baseline audit, not the final deep-learning architecture. A neural sequence model is not justified by sparse daily summaries and 11 positive participants: it would add capacity without adding independent clinical evidence.

The decision gate is intentionally simple and fixed before interpreting the comparison. Passing it would still be only a research milestone. Any future model needs a patient-separated external cohort, calibration, sensitivity/specificity and positive predictive value at a declared alert rate, false alerts per patient-time, lead-time analysis, subgroup checks, and review by clinicians.

## Work needed for a ring-based pulmonary model

1. Collect or obtain a sufficiently large pulmonary cohort with the actual ring's raw red/infrared PPG, accelerometer, temperature/EDA if available, and timestamps.
2. Obtain spirometry-confirmed diagnoses for COPD/asthma and/or adjudicated, timestamped exacerbations. Keep diagnosis and flare forecasting as separate tasks.
3. Ensure labels are independent of model inputs; never feed derived symptom scores or post-event measurements into the predictor.
4. Split by participant and site before preprocessing, feature selection, tuning, and calibration. Keep a locked external test set.
5. Evaluate the exact ring hardware and measurement pipeline. A watch, in-ear PPG, spirometer, and mattress do not validate a finger ring.
6. Only after validation, connect the model result to the app. The lung surface may then show an explicitly labeled visualization of validated model output; until then it stays anatomical and non-diagnostic.

## Reproduction

Download TOLIFE v2.0 `index.zip` from the official Zenodo record into the ignored path `data/tolife/index.zip`, install `requirements-model.txt`, and run:

    python scripts/evaluate_pulmonary_study.py

The aggregate report is `models/pulmonary_feasibility_metrics.json`. The raw data, model weights, and any participant-level outputs must not be committed.

Download the six public COPD PPG duration archives from the [Zenodo record](https://zenodo.org/records/18109876) into the ignored `data/copd_ppg_zenodo/` directory, then run:

    python scripts/evaluate_copd_ppg_subject_holdout.py

This writes aggregate-only results to `models/copd_ppg_screening_metrics.json`. The six duration variants are evaluated separately; they share the same six people and are not independent cohorts.

# Pulmonary model integration status

## Current status

There is no deployed pulmonary disease model or patient-facing score. Two different research tracks are now evaluated separately:

1. COPD exacerbation forecasting on TOLIFE daily indices, among people already diagnosed with COPD. Its participant-held-out comparison failed the predeclared no-skill and ROC-AUC gates.
2. COPD-versus-control classification from public PPG segments, using the Zenodo archives with six identified people (four COPD, two controls). The script compares three classical models under leave-one-participant-out validation across six duration variants. The variants contain the same six people; results are feasibility evidence only and do not validate a disease detector.

No model weights are saved or loaded by the app. The app displays aggregate study evidence and keeps patient-facing inference disabled.

The PPG feasibility result is poor: all three candidates produced participant-level ROC-AUC 0.000 on the six-person split; the logistic average precision was 0.525 versus a 0.667 prevalence baseline. The original 14-person paper's Figshare feature matrix is not used because it lacks participant IDs and its healthy-row count conflicts with the paper's table. HealthRing has actual ring PPG/IMU but no pulmonary diagnosis labels. The data sources must remain separate.

## Local app

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-model.txt
    python server.py

Open http://127.0.0.1:5000. The Flask app binds to loopback.

- `GET /api/health` reports that inference is not ready, that no disease diagnosis is supported, and that the data are not ring-validated.
- `GET /api/pulmonary/metrics` serves aggregate evaluation metrics only.
- `GET /api/pulmonary/copd-ppg-metrics` serves aggregate COPD/PPG feasibility metrics only.
- `GET /models/pulmonary_feasibility_metrics.json` serves the same aggregate report.
- `GET /models/copd_ppg_screening_metrics.json` serves the separate aggregate PPG report.
- Legacy hospital replay and score routes return HTTP 410; they are not part of the pulmonary-only app.

The heart and both lung meshes remain interactive and animated. Their default motion values are illustrative. The lung surface has only a subtle non-diagnostic material texture; the app does not map readings or experimental scores to infection, inflammation, or severity colors.

## Reproduce the TOLIFE comparison

Download the official [TOLIFE v2.0 index archive](https://zenodo.org/records/23263785) and place it at `data/tolife/index.zip`. The local data directory is ignored by Git. Run:

    python scripts/evaluate_pulmonary_study.py

The evaluation uses quality-valid daily indices, a seven-day history, a future 1–7 day COPD exacerbation label, at least two observed sensor families, and five-fold participant-held-out GroupKFold. It compares logistic regression, histogram gradient boosting, and random forest. It writes only aggregate metrics; there is no saved deployment model.

The best average precision was 0.01694 against a no-skill prevalence reference of 0.01819; the best ROC-AUC was 0.4874. The model gate failed. These are not probabilities or evidence of disease detection.

## Reproduce the COPD/PPG feasibility comparison

Download all six duration archives from the official [Zenodo COPD PPG record](https://zenodo.org/records/18109876) into `data/copd_ppg_zenodo/`. Install the dependencies above and run:

    python scripts/evaluate_copd_ppg_subject_holdout.py

The script verifies each filename's COPD/control label against its rows, samples up to 512 windows per person, and evaluates each archive separately with leave-one-person-out folds. It compares regularized logistic regression, an RBF support-vector machine, and Extra Trees. Scores are aggregated to one median score per held-out person before calculating metrics. No saved model or individual score is produced.

Each archive has four COPD participants and two controls, and the same participants appear across all six duration variants. The subject count is still six, not 36 or the number of windows. The related 14-person paper's Figshare feature matrix is not used for validation: it has no participant IDs and its healthy-window count conflicts with the published per-person totals. The paper's random epoch split is not equivalent to holding out people.

## Why a deep model is not deployed

The event target is rare and positive days come from 11 participants. The dataset supplies daily indices from non-ring devices, not the raw continuous data stream of the proposed ring. The most relevant raw-ring dataset lacks pulmonary labels. A larger neural network cannot repair these label, device, and sample-size gaps; it would make an unsupported score look more precise.

## Minimum evidence before integration

A future disease model must use a matched cohort with the actual ring's signals and clinician-confirmed pulmonary labels. It must be split by participant and site, evaluated on a locked external cohort, and report calibration, sensitivity, specificity, precision, false-alert burden, lead time, and subgroup performance. The UI must receive a versioned model output with quality flags and uncertainty. Until then, it returns no disease result and the anatomy remains an illustration.

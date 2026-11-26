# PlumoRing AI

PlumoRing AI combines an interactive smart-ring concept with a pulmonary-sensing research workspace. The Gold, Onyx, and Doomsday ring assemblies remain product concepts; the heart and lung scene is an interactive anatomical illustration, not a patient scan.

## Research status

The intended research question is whether ring-compatible readings can identify or monitor pulmonary disease. **There is no validated ring-based pulmonary disease model in this repository.** The available public datasets do not provide enough matched ring signals and pulmonary labels to support that claim.

The app therefore reports evidence and model readiness, not a patient's diagnosis or disease score. It keeps three different research questions separate:

- **COPD flare forecasting:** TOLIFE provides clinical exacerbation records and daily indices from several devices. It is the largest relevant longitudinal cohort found, but it does not use the proposed ring and it covers flare forecasting among people already diagnosed with COPD, not COPD diagnosis.
- **COPD classification from PPG:** a public PPG release contains six identified people (four with COPD and two controls). Three classical models are compared with every person held out in turn across six window-duration variants. The variants reuse the same six people, so this is a feasibility audit, not a dependable disease detector.
- **Ring signal estimation:** HealthRing contains real ring PPG and motion with reference vital signs, but no pulmonary diagnosis labels. It can support future testing of vital-sign estimation, not disease classification.
- **Asthma worsening:** AAMOS-00 links weekly reports of increased systemic corticosteroid use with MiBand3 heart-rate/activity summaries. A nested participant-held-out evaluation was added; it is retrospective week-level classification and failed the sensitivity gate.
- **Apnea/hypopnea events:** OSASUD contains finger PPG plus physician-reviewed overnight PSG event annotations. A strict PPG-only Extra Trees baseline reached equal-participant ROC-AUC 0.780 and average precision 0.633 versus 0.313 prevalence across participant-held-out folds. This is exploratory event-window screening in hospital monitoring data, not sleep-apnea diagnosis or PlumoRing validation; model selection on the same grouped comparison makes the result optimistic.
- **Personalized COPD trend baseline:** a training-free 14-day per-person HR/RR/SpO₂ change score was compared on TOLIFE. It also fell below the no-skill reference on a sparse, reduced subset; no alert is enabled.
- **Best external evidence located:** a 2026 study reports six-month wearable monitoring in 220 people with COPD and 42 physician-validated exacerbations, but its patient data are restricted. The published score is the study authors' result and has not been tested on PlumoRing or our data.

These datasets are not merged. They differ in sensors, sampling, labels, and tasks. In particular, lung-sound recordings cannot stand in for ring PPG because the proposed ring has no microphone. The original 14-person study's Figshare feature matrix is excluded from grouped validation because it omits participant IDs and its healthy-row total conflicts with its participant table. Its random epoch split does not establish generalization to new people.

The COPD PPG grouped experiment failed: each of the three candidates scored participant-level ROC-AUC 0.000, and the logistic average precision was 0.525 against a 0.667 prevalence baseline. This is a negative feasibility result; the app exposes no COPD prediction.

## TOLIFE exploratory comparison

The reproducible evaluation uses TOLIFE v2.0, with quality-valid daily indices, a seven-day lookback, and a seven-day future COPD exacerbation target. It uses pulse-rate variability, heart rate, breathing-rate indices, movement, and a spot oxygen-saturation value. Spirometry values, derived symptom scores, audio, environmental variables, demographics, and clinical-visit fields are excluded. Participants are held out by kit in five folds.

| Model | Grouped ROC-AUC | Grouped average precision |
| --- | ---: | ---: |
| Logistic regression | 0.426 | 0.014 |
| Histogram gradient boosting | 0.354 | 0.013 |
| Random forest | 0.487 | 0.017 |
| No-skill average-precision reference | — | 0.018 |

The best candidate did not beat the event-prevalence reference. Only 11 participants contributed positive forecast days. No model weights are exported and the app does not enable patient-facing inference. These results are not evidence that the models detect COPD or any other lung disease.

## AAMOS-00 asthma worsening comparison

This analysis uses the public AAMOS-00 dataset (22 adults with asthma overall) and only MiBand3 heart-rate/activity readings from the seven days covered by the weekly questionnaire. The outcome is reported systemic corticosteroid use above usual, not a new asthma diagnosis or a future attack prediction. The nested participant-held-out model-selection result was ROC-AUC 0.610 and average precision 0.189 against a participant-weighted no-skill reference of 0.074; sensitivity was only 0.298 at a 0.931 specificity operating point. Only six participants contributed positive weeks, and this fails the predeclared sensitivity gate. Do not deploy or interpret these values as a patient's risk.

Reproduce with `python scripts/evaluate_aamos_asthma_worsening.py`. See [Dataset plan](DATASET_PLAN.md) for target coding, exclusions, and the dataset citation. AAMOS-00 is CC BY 4.0; cite the University of Edinburgh dataset and its Scientific Data paper.

## Personalized COPD trend comparison

Run `python scripts/evaluate_tolife_personalized_trend.py` to compare a training-free 14-day personal baseline against TOLIFE's recorded COPD flare windows. It used 1,015 eligible days from 36 people (only 3 with positive windows): equal-person ROC-AUC 0.444 and average precision 0.0064 versus a 0.0075 equal-person prevalence reference. It did not beat no-skill, and it is not deployed. See [Dataset plan](DATASET_PLAN.md) for its limits and the larger restricted cohort found in the literature review.

## OSASUD PPG event-window comparison

Run `python scripts/evaluate_osasud_ppg_events.py` to compare strict PPG-only baselines against annotated 30-second apnea/hypopnea windows. The run produced 27,627 quality-valid windows from 30 hospital participants. The best of three models reached equal-participant ROC-AUC 0.780 and average precision 0.633 versus 0.313 prevalence, but model selection on those same grouped folds makes this exploratory result optimistic. It is not sleep-apnea diagnosis or ring validation. The 3.1 GB source stays under ignored `data/`; the first run also writes an ignored local feature cache so later comparisons avoid re-reading and reprocessing the full pickle. The aggregate report is `models/osasud_ppg_event_metrics.json`.

## Open locally

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-model.txt
    python server.py

Visit http://127.0.0.1:5000. The server binds to loopback. It serves the interface, the aggregate pulmonary evaluation report, and the 3D anatomy assets. The interface works as a research presentation without downloading the datasets.

## Reproduce the feasibility evaluation

Obtain the TOLIFE v2.0 index archive from the [official Zenodo record](https://zenodo.org/records/23263785) and place it at data/tolife/index.zip. Raw public data stay local and are ignored by Git. Then run:

    python scripts/evaluate_pulmonary_study.py

The script compares three classical baselines with participant-level folds and writes aggregate metrics only to models/pulmonary_feasibility_metrics.json. It does not train or save a deployment model. See [Dataset plan](DATASET_PLAN.md) and [Model integration](MODEL_INTEGRATION.md) for details.

For the separate COPD-vs-control PPG feasibility track, download the six duration archives from the [official Zenodo record](https://zenodo.org/records/18109876) into `data/copd_ppg_zenodo/`, then run:

    python scripts/evaluate_copd_ppg_subject_holdout.py

This compares regularized logistic regression, an RBF support-vector machine, and Extra Trees using leave-one-participant-out evaluation. It writes aggregate metrics only to `models/copd_ppg_screening_metrics.json`. All six archive variants contain the same six people; no model is deployed. The results cannot be used to diagnose a person or claim the proposed ring detects COPD.

## Data handling and safety

This is a software research prototype, not a medical device. It has no live ring connection, no validated disease inference, and no clinical alerts. The anatomy animation uses illustrative default heart and breathing rates; it is not driven by patient data or disease severity. No personal readings are uploaded. See [Privacy](privacy.html).

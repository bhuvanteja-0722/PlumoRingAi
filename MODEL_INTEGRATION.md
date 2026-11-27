# Pulmonary model integration status

## Current status

There is no validated pulmonary disease model, patient-facing disease score, or live ring connection. The analysis area now presents two separate model tracks: our public-dataset evidence work and the supplied experimental telemetry network. Each dataset is kept in its own research track because the sensors, participants, labels, and tasks differ:

1. **COPD exacerbation forecasting:** TOLIFE daily indices and recorded moderate/severe exacerbations among people already diagnosed with COPD. Participant-held-out models failed the no-skill and ROC-AUC gates.
2. **COPD versus control from PPG:** the public Zenodo archive has six people (four COPD, two controls), repeated across six window-duration variants. All three candidates produced participant-level ROC-AUC 0.000, with logistic average precision 0.525 against a 0.667 prevalence reference. This is a failed feasibility experiment, not a diagnosis model.
3. **Asthma worsening:** AAMOS-00 MiBand3 heart-rate/activity data paired with weekly reports of systemic corticosteroid use above usual. This is retrospective classification in adults already diagnosed with asthma, not future-attack prediction or new-asthma diagnosis. Only six participants contributed positive weeks; the nested participant-held-out result fails the sensitivity gate.
4. **Apnea/hypopnea event windows:** OSASUD finger PPG and PPG-derived measures paired with physician-reviewed overnight PSG annotations. Its best strict-PPG exploratory baseline reached equal-participant ROC-AUC 0.780 and average precision 0.633 versus 0.313 prevalence on participant-held-out folds. The best of three models was selected on that same grouped comparison, so the result is optimistic and needs nested/external validation. It does not diagnose sleep apnea syndrome or validate PlumoRing hardware. Any expanded track with hospital ECG-derived heart rate or thoracic-impedance respiratory rate is explicitly not ring-only.
5. **Ring physiology estimation:** HealthRing has real finger-ring PPG and motion with reference vitals, but no pulmonary disease labels. It can support sensor estimation, not disease classification.
6. **Personalized COPD trend baseline:** a training-free 14-day per-person HR/RR/SpO₂ change score was separately evaluated on the sparse TOLIFE daily records. It had equal-participant ROC-AUC 0.444 and average precision 0.0064 versus a 0.0075 no-skill reference on only 1,015 days from 36 people, with just 3 people contributing positive days. It also failed and has no alert threshold.
7. **BIDMC PPG respiratory rate:** 53 ICU recordings and 832 thirty-second PPG-only windows with two manual breath readers. Nested participant-held-out model selection achieved participant-macro MAE 1.77 breaths/min. This is a vital-estimation task, not a disease label or ring evaluation.

The strongest external continuous-monitoring evidence located is a 2026 eMEUSE-SANTÉ study of 220 COPD patients monitored for six months, with 42 physician-validated exacerbations. The paper reports an AUC of 0.88 for its personalized HR/RR/SpO₂ score, but its underlying participant data are restricted due to re-identification risk and are available only through an approved research request. These published results are not PlumoRing results.

No datasets are concatenated to manufacture a larger disease cohort. Lung-audio, radiology, hospital-code, and symptom-only datasets are not treated as ring PPG labels. The original 14-person COPD PPG study's Figshare matrix is excluded because it lacks participant IDs and its healthy-row count conflicts with the paper's participant table.

Our public-dataset research tracks expose aggregate study evidence only. They do not return a disease probability, alert, or disease-driven anatomy color. The scripts compare research candidates and write aggregate metrics; they do not package fitted weights for patient-facing inference. The separate supplied telemetry network is described below.

## Supplied telemetry network (kept as a separate model)

The model from `trained_pulmonary_ring_app_updated (1).zip` is preserved at `models/temporal_telemetry_state_classifier.keras`. It is a Keras Functional network with input shape `(batch, 16, 8)` and five softmax outputs. The output order is copied from the supplied `app.py`: normal resting baseline, daily activity/exercise impact, sudden breathing trouble/stress, fever/lung-infection alert, and loose-ring/bad-reading. The last two labels are the model author's class names; they are not validated disease labels.

The adapter in `telemetry_state_model.py` matches the eight normalization formulas in the supplied `app.py`, checks the model's input/output shapes, and loads the `.keras` file with Keras safe mode enabled and compilation disabled. Model 1 runs only the saved network; it does not execute the supplied Flask app or its medication, diet, yoga, or care-advice code. A separate `rule_hypothesis` field uses hand-written input thresholds, independent of Model 1's class, to show illustrative asthma-like, COPD-like, infection-like, or cause-unclear patterns. This is not a learned disease output; it has no probability and is labeled unvalidated. The original package did not include the training dataset, per-class sample counts, train/test split, or held-out evaluation report, so the network output is also experimental and unvalidated. Raw softmax values are shown as raw class scores, not calibrated probabilities.

The analysis panel's slider preview sends one snapshot and repeats it over 16 steps to match the supplied adapter. The API also accepts an actual `sequence` with shape `(16, 8)` for a sampled history. The network returns telemetry-state scores; the interface displays them as percentages for readability, but they are not calibrated probabilities of disease. The separate rule preview evaluates only the latest row and returns a textual pattern label with observed inputs; it never returns a disease probability, severity, or care recommendation. At motion above 2.2 g, it withholds a named pulmonary pattern and displays a high-motion message; this is only a demo safeguard, not a validated signal-quality test. These threshold labels do not mean Model 1 detects asthma, COPD, pneumonia, bronchitis, or any other disease. Our public-dataset evaluation tracks remain separate.

Run locally with the Python environment created from `requirements-model.txt`, which installs TensorFlow. The saved weights are loaded only when the user presses **Run Model 1**.

## Local app

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-model.txt
    python server.py

Open http://127.0.0.1:5000. Flask binds to loopback.

- `GET /api/health` reports that inference is not ready and that the evidence is not ring-validated.
- `GET /api/pulmonary/metrics` serves aggregate TOLIFE results.
- `GET /api/pulmonary/copd-ppg-metrics` serves aggregate COPD/PPG feasibility results.
- `GET /api/pulmonary/aamos-metrics` serves aggregate AAMOS-00 asthma-worsening results.
- `GET /api/pulmonary/osasud-metrics` serves aggregate OSASUD apnea/hypopnea event results.
- `GET /api/pulmonary/tolife-trend-metrics` serves aggregate personalized-trend baseline results.
- `GET /api/pulmonary/healthring-vitals-metrics` serves HealthRing RR/SpO₂ physiology estimates and partial-archive coverage.
- `GET /api/pulmonary/bidmc-rr-metrics` serves PPG-only BIDMC respiratory-rate results and the separate cross-device HealthRing evaluation.
- `GET /api/pulmonary/tolife-associations` serves aggregate pre-visit clinical-index association coverage.
- `GET /api/pulmonary/research-snapshot` returns an aggregate-only summary of COPD flare, COPD PPG, asthma-worsening, apnea-event, and personalized COPD-trend evaluations. It marks pneumonia/bronchitis/infection as unavailable because no matched labeled dataset is included. The UI can save this snapshot in a separate browser-local research history; it is not personal reading history or a per-patient prediction.
- `GET /api/models/temporal-telemetry/status` reports whether the Model 1 file and TensorFlow runtime are available.
- `POST /api/models/temporal-telemetry/predict` runs Model 1 and its separate rule preview. Send either `{"sensors": [hr, spo2, rr, wheeze, temp_delta, imu, eda, pm25]}` for a single snapshot or `{"sequence": [[... eight values ...], ...]}` with exactly 16 rows. The response includes uncalibrated class scores and a non-probabilistic `rule_hypothesis`; neither is a diagnosis.
- Equivalent aggregate reports are served under `/models/` when available.
- Legacy hospital replay and scoring endpoints return HTTP 410.

The heart and both lung meshes remain interactive and animated. Their rates are illustrative defaults. The anatomy does not display patient results or experimental disease severity.

## Reproduce the studies

Download the public datasets into the ignored `data/` directory, install `requirements-model.txt`, and run the matching script:

| Track | Script | Aggregate report |
| --- | --- | --- |
| TOLIFE COPD flare | `python scripts/evaluate_pulmonary_study.py` | `models/pulmonary_feasibility_metrics.json` |
| COPD PPG | `python scripts/evaluate_copd_ppg_subject_holdout.py` | `models/copd_ppg_screening_metrics.json` |
| AAMOS-00 asthma worsening | `python scripts/evaluate_aamos_asthma_worsening.py` | `models/aamos_asthma_worsening_metrics.json` |
| OSASUD PPG event windows | `python scripts/evaluate_osasud_ppg_events.py` | `models/osasud_ppg_event_metrics.json` |
| TOLIFE personalized trend score | `python scripts/evaluate_tolife_personalized_trend.py` | `models/tolife_personalized_trend_metrics.json` |
| HealthRing ring-vital estimation | `python scripts/evaluate_healthring_ring_vitals.py` | `models/healthring_ring_vitals_metrics.json` |
| BIDMC PPG respiratory-rate and ring transfer | `python scripts/evaluate_bidmc_ppg_respiratory_rate.py` | `models/bidmc_ppg_respiratory_rate_metrics.json` |
| TOLIFE clinical-index associations | `python scripts/analyze_tolife_clinical_associations.py` | `models/tolife_clinical_associations.json` |

The AAMOS-00 archive is available from [Edinburgh DataShare](https://datashare.ed.ac.uk/items/8478e384-fd1b-4a37-9555-0c6e1218e90b), under CC BY 4.0. Its primary paper reports 22 adults and 2,054 patient-days. The evaluator uses only smartwatch HR/activity over the reported week, and holds out whole participants. Its target is retrospective self-reported steroid use above usual.

The OSASUD archive is available on [Figshare](https://springernature.figshare.com/articles/dataset/Main_dataset_file/19214289). It is a large pickle, so first load can take several minutes. The evaluator extracts quality-checked 30-second windows and reports PPG-only results separately from hospital ECG/impedance-derived scalar references. Raw data and participant-level outputs are not committed.

## Why this is not a clinical model

The disease-labeled cohorts remain small or mismatched to a ring. TOLIFE's COPD flare result failed; the six-person COPD PPG study failed; AAMOS-00 has only six positive participants in the watch-covered subset and misses the sensitivity gate. OSASUD is an overnight hospital event cohort of 30 stroke-unit patients. No current result supports diagnosis of asthma, COPD, pneumonia, bronchitis, respiratory infection, or sleep apnea using this ring.

The current datasets do not justify the supplied high-capacity sequence architecture for deployment. A larger neural model cannot compensate for too few independent patients, labels without reliable event timing, different sensor hardware, or absent external validation. Use classical, regularized baselines for these feasibility checks; keep the patient-facing model disabled.

Before future integration, collect a matched cohort with the actual ring's raw red/infrared PPG, movement and relevant channels; use clinician-confirmed pulmonary labels; split by participant and site before feature selection and calibration; and reserve an independent external cohort. Evaluate calibration, sensitivity, specificity, precision, false alerts per patient-time, lead time, subgroup performance, and device-specific signal quality. Only then consider a versioned inference endpoint and a carefully labeled visual response.

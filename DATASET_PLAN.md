# Pulmonary dataset and evaluation plan

## Project question

Can readings that a finger-worn ring could plausibly collect support detection or monitoring of pulmonary disease?

A useful model needs matched inputs and labels: ring-compatible signals from the same participant and time period as a clinically confirmed pulmonary diagnosis or a timestamped pulmonary exacerbation. Public datasets found so far cover pieces of this question, but none covers the complete ring-to-disease task.

## Datasets and their valid roles

| Dataset | What it contains | Valid use here | Key limitation |
| --- | --- | --- | --- |
| [TOLIFE Clinical Study A v2.0](https://zenodo.org/records/23263785) | 86 people with COPD monitored across three sites; daily indices and an exacerbation register | Exploratory forecasting of a recorded moderate/severe COPD exacerbation among people already diagnosed with COPD | Multiple non-ring devices and sparse daily derived indices; not a new-disease diagnosis or ring validation |
| [eMEUSE-SANTÉ wearable COPD study](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0001405) | Six months of HR, breathing rate, SpO₂ and activity from 220 COPD participants; 42 physician-validated exacerbations in 39 people | Strong external reference for personalized continuous COPD-exacerbation monitoring methods | Patient-level data are restricted due re-identification risk; eligible research access requires approval and a data-sharing agreement through the MARS/HP2 repository |
| [AAMOS-00 asthma study](https://datashare.ed.ac.uk/items/8478e384-fd1b-4a37-9555-0c6e1218e90b) | 22 adults with asthma; 2,054 patient-days overall; MiBand3 heart-rate/activity and weekly asthma questionnaires | Separate exploratory recognition of a reported past-week increase in systemic corticosteroid use | Only 20 people have watch data; the outcome is retrospective and self-reported, and inputs are smartwatch summaries, not raw ring PPG |
| [OSASUD stroke-unit study](https://springernature.figshare.com/articles/dataset/Main_dataset_file/19214289) | 30 hospitalized stroke-unit patients with one-second finger PPG, PPG-derived oxygen/perfusion, and physician-tagged PSG apnea/hypopnea events | Separate overnight apnea/hypopnea event-window detection feasibility study using strict PPG-only inputs | One small, older, hospitalized cohort; it evaluates annotated events, not a diagnosis of sleep apnea syndrome or daytime pulmonary disease |
| [HealthRing](https://www.nature.com/articles/s41597-026-07289-x) | 54 adults; finger-ring red/infrared PPG and 3-axis motion, with reference HR, RR, SpO₂ and blood pressure | Ring-vital estimation, motion-artifact handling, and later sensor-validation work | No COPD, asthma, pneumonia, or exacerbation diagnosis labels |
| [BIDMC PPG and Respiration](https://physionet.org/content/bidmc/1.0.0/) | 53 eight-minute ICU records; 125 Hz pulse-oximeter PPG and two manual breath-annotation streams | Separate PPG-only respiratory-rate estimation benchmark and cross-device transfer check to HealthRing | ICU pulse-oximeter data, not ring hardware or a pulmonary-diagnosis cohort; ODC-BY 1.0 attribution required |
| [Oura COVID ring studies](https://pmc.ncbi.nlm.nih.gov/articles/PMC8891385/) | Oura ring physiology paired with participant illness reports; one TemPredict analysis reports 63,153 participants and 704 possible self-reported COVID cases | Literature evidence that ring signals can change around acute infection; a candidate for a future data-access request | Participant-level data are not public under the study's data-use terms, so they cannot be used for this free-public-data model. Illness reporting is not equivalent to clinician-adjudicated pulmonary diagnosis. |
| [Ultra-Short PPG for COPD Screening](https://zenodo.org/records/18109876) | Six participant files in each of six duration archives; four COPD labels and two controls, verified against each CSV | Separate, subject-held-out COPD-vs-control feasibility experiment using PPG waveform segments | All duration archives repeat the same six people. The public README does not specify ring hardware or sensor placement, so this is generic PPG evidence, not proof of ring compatibility. Window count does not increase the independent cohort. |
| [HUSM and GMicro PPG peak-detection data](https://zenodo.org/records/8280127) | 66 minutes of PPG with systolic-peak annotations, collected from healthy participants and people with cardiopulmonary conditions | PPG peak-detection and signal-processing methods | The release and accompanying paper evaluate peak detection, not a pulmonary diagnosis target with patient-level disease labels; it cannot be added as COPD training data. |
| [Original PPG COPD study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10280226/) and its [Figshare feature matrix](https://doi.org/10.6084/m9.figshare.21280371.v2) | Paper reports 8 COPD and 6 controls; the shared feature matrix has 218,999 rows with 100 engineered features and a class label | Literature review and dataset-integrity audit | The shared feature matrix has no participant IDs, and the control row count (106,815) conflicts with the published per-person table (97,818). The published 50/50 split is by 2-second epochs, not held-out people. It is excluded from the primary evaluation rather than assigning guessed patient IDs. |
| [WestRo COPD and WestRo Porti COPD studies](https://doi.org/10.1002/advs.202203485) | Research reports 54 and 534 participants, with physiological signals and COPD-stage labels | High-priority external cohort to request for future, patient-separated validation | Patient-level data are available from the corresponding author on reasonable request, not as a public download. Access and use conditions must be resolved before training. |

### Do not merge these labels

- COPD diagnosis and future COPD exacerbation are different outcomes. A flare study among known COPD patients cannot diagnose COPD in a general population.
- HealthRing has true ring signals but no pulmonary labels. It can validate signal estimation, not teach a disease classifier.
- Published Oura studies show illness-associated ring physiology, but their individual-level data are restricted; an open access route was not found. The public COVID-RED data are from a different wearable system and do not become ring evidence.
- The six-person COPD PPG release is useful only as a small generic-PPG experiment because its public documentation does not establish that the sensor was a finger ring. Its participant-held-out results fail to support reliable COPD screening.
- TOLIFE has COPD event labels but its channels come from smartwatch, mattress, phone/shoes, and spirometer-related measurements. Similarity to ring features is not proof of ring measurement equivalence.
- Lung-audio datasets require a microphone, which is not part of this ring. Hospital-record, imaging, and symptom-score datasets do not become ring datasets by relabeling or concatenating them.

No asthma, pneumonia, bronchitis, or pulmonary-infection diagnosis model is supported by these inputs and labels. The AAMOS target is asthma worsening within a known-asthma cohort, not new asthma diagnosis. OSASUD labels apnea/hypopnea events during overnight clinical monitoring, not an OSAS diagnosis from a ring. Do not present a condition label unless a future matched cohort and independent validation support it.

## Ring-vital physiology benchmark: HealthRing and BIDMC

These experiments estimate a vital sign; neither classifies pulmonary disease.

### HealthRing: actual ring PPG and motion

The official HealthRing v2.1 release provides finger-worn red/infrared PPG and accelerometer signals with reference physiology, but no pulmonary disease diagnoses. The downloaded source file's MD5 matches the Zenodo record, yet its ZIP directory is absent. Local ZIP repair recovered 56 readable per-person tables from 29 unique participants (3,497 windows); one recovered table was empty or unreadable. No treadmill cohort was recovered. All reported estimates below therefore use this partial subset, not the paper's full 54-person cohort.

The evaluator extracts only ring PPG and motion-derived predictors; reference HR, RR, SpO₂, ECG, oximeter waveforms and respiratory instrumentation are targets or excluded channels, never predictors. Ring 1 and Ring 2 are evaluated independently. Five outer participant-grouped folds hold out each participant, with model-family choice nested inside the training participants for RR.

| Ring task | Nested / grouped result | Interpretation |
| --- | ---: | --- |
| Ring 1 respiratory rate | Nested participant-macro MAE **2.99 breaths/min** (95% participant bootstrap CI **2.64–3.40**); 1,752 windows / 27 participants | Extra Trees was selected inside all five outer folds; exploratory estimation on the recovered subset |
| Ring 2 respiratory rate | Nested participant-macro MAE **3.48 breaths/min** (95% CI **2.99–3.99**); 1,670 windows / 27 participants | The nested candidate is only marginally better than the training-mean baseline; not a robust ring-general result |
| Ring 2 SpO₂ | Best same-comparison Extra Trees participant-macro MAE **1.64 percentage points**; 1,595 windows / 28 participants | Model selection and scoring use the same grouped comparison; optimistic, weakly correlated estimates and no clinical validation |

The data provide no test of COPD, asthma, pneumonia, infection, exacerbation, or disease severity. The broken source also prevented the planned treadmill transfer test. Reproduce after downloading the official archive; if the file again has no central directory, repair a local copy with `zip -FF RingDatasetV2.1_submission.zip --out RingDatasetV2.1_repaired.zip`, then run `python scripts/evaluate_healthring_ring_vitals.py`. Raw signals and repaired archives remain local under ignored `data/`.

### BIDMC: separate PPG-only breathing-rate benchmark

The [BIDMC dataset](https://physionet.org/content/bidmc/1.0.0/) contains 53 eight-minute ICU recordings with PPG, respiration, ECG and 1 Hz monitor values. The study describes two manual breath annotators who marked breaths from the impedance-respiration signal. The evaluator uses only the 125 Hz PLETH waveform as input and the mean of both manual readers as its 30-second target; it excludes the respiration waveform, ECG, SpO₂, monitor numerics and demographics from predictors.

The analysis produced **832 30-second windows from all 53 recordings**. The two manual readers differed by a mean **0.33 breaths/min** across retained windows (Pearson *r* **0.945**). Nested five-fold participant-held-out model selection chose Random Forest in all five outer folds and produced participant-macro MAE **1.77 breaths/min** (95% participant bootstrap CI **1.32–2.32**; pooled MAE **1.62**). This is a respiratory-rate result on critically ill ICU patients, not disease detection.

A separate source-to-ring transfer trained on BIDMC PPG only, selected its model using participant-held-out BIDMC folds, then evaluated on HealthRing Ring 1 without fitting on ring labels. Its MAE was **3.79 breaths/min** across 1,752 windows from 27 ring participants, worse than the fixed BIDMC training-mean baseline of **3.55 breaths/min**; *R*² was **−0.129**. That negative result shows the ICU pulse-oximeter model does not transfer to the recovered ring cohort. It is not a PlumoRing performance estimate.

Reproduce with `python scripts/evaluate_bidmc_ppg_respiratory_rate.py`. The script reads the ignored local archive at `data/bidmc/bidmc.zip` and writes aggregate-only metrics to `models/bidmc_ppg_respiratory_rate_metrics.json`. BIDMC is distributed under the Open Data Commons Attribution License v1.0; cite PhysioNet and the dataset DOI when reusing the study.

### TOLIFE clinical-index association audit

TOLIFE also contains actual COPD clinic outcomes and daily non-ring indices. A separate descriptive analysis compared each measured clinic outcome with valid sensor-index averages from the seven complete days before that visit. It included **374 visits from 86 participants**. To avoid false precision, an association was only summarized if at least five participants had three or more informative repeated visits for that sensor/outcome pair. Every tested pair fell below that coverage threshold; no stable within-person estimate was reported. Same-day/future readings, model-estimated symptom scores, and spirometry predictors were excluded. This remains non-ring association research and does not validate a ring disease model. Reproduce with `python scripts/analyze_tolife_clinical_associations.py`.

## Clinical targets determine what can be learned

- **COPD diagnosis** needs a verified diagnostic label. GOLD states that post-bronchodilator spirometry with FEV₁/FVC below 0.7 is required to establish COPD; pulse rate, breathing rate, SpO₂ and motion can be context but are not equivalent diagnostic labels ([GOLD spirometry guide](https://goldcopd.org/spirometry-quick-guide/)).
- **COPD exacerbation monitoring** is a separate task among people already diagnosed with COPD. It needs a timestamped, clinically defined exacerbation and readings that precede the event. TOLIFE supplies a separate non-ring feasibility target; its models failed the no-skill comparison.
- **Asthma diagnosis** depends on characteristic variable respiratory symptoms plus evidence of variable expiratory airflow, commonly from spirometry or peak-flow testing. A ring's HR/activity summary is not that diagnostic measurement ([GINA 2025 summary](https://ginasthma.org/wp-content/uploads/2025/11/GINA-Summary-Guide-2025-WEB_FINAL-WMS.pdf)). AAMOS-00 instead asks whether a wearable summary relates to a reported past-week treatment increase in people already diagnosed with asthma.
- **Obstructive sleep apnea diagnosis** is not equivalent to detecting a few annotated PPG event windows. AASM recommends polysomnography or technically adequate home sleep-apnea testing for diagnosis; the OSASUD model stays an event-window research result ([AASM guideline](https://aasm.org/resources/clinicalguidelines/diagnostic-testing-osa.pdf)).
- **Pneumonia, bronchitis, and respiratory infection** have no matched ring dataset in this project. Generic changes in HR, RR, temperature or SpO₂ are non-specific. Without patient-level clinical labels and independent validation, the app must not convert them into an infection name or disease-colored lung.

The practical model architecture is therefore task-specific: ring-only physiology regression on HealthRing; PPG respiratory-rate estimation and transfer on BIDMC; and separate disease-related feasibility audits on TOLIFE, COPD PPG, AAMOS-00 and OSASUD. Do not concatenate their rows or train one disease classifier across incompatible devices and targets.

## AAMOS-00 asthma worsening feasibility evaluation

The public [AAMOS-00 dataset](https://datashare.ed.ac.uk/items/8478e384-fd1b-4a37-9555-0c6e1218e90b) contains MiBand3 heart rate, steps, activity and questionnaire data. The paper describes 22 adults with asthma and 2,054 patient-days; the wearable data are from smartwatch hardware rather than a PlumoRing. The dataset is licensed CC BY 4.0, and attribution is required.

### Target and evaluation

- Target: questionnaire code `weekly_oral == 3`, which the data dictionary defines as systemic corticosteroid use **above the participant's usual amount during the past week**. Codes 1 (no increase) and 2 (yes, no more than usual) are the negative class; unsure/missing responses are excluded.
- Inputs: only smartwatch heart-rate and activity/step summaries from the same seven-day reporting window. Symptoms, the outcome field, inhaler use, peak flow, environment, and future readings are excluded.
- This is retrospective week-level classification, not attack forecasting. The outcome is not timestamped within the week, so the model cannot claim continuous detection or lead time.
- A week is included only with at least three days of watch data. Evaluation uses five-fold StratifiedGroupKFold by participant. Model choice and an operating threshold are selected inside each training fold (nested participant-held-out evaluation).
- Compared a participant-prevalence baseline, regularized logistic regression, RBF SVM, histogram gradient boosting, random forest, and Extra Trees. Preprocessing is fitted within folds.

### Counts and result

- 324 weekly questionnaire rows; 224 labeled weeks with at least three watch days (26 positive, 198 negative); 18 participants in this analysis, with positive weeks from 6 participants. Ninety-nine weeks were excluded for insufficient watch coverage and one for a missing outcome.
- Nested model selection most often chose histogram gradient boosting. Equal-participant-weight ROC-AUC was **0.610**, average precision **0.189** versus a weighted no-skill reference of **0.074**.
- At an inner-training-selected threshold targeting 90% specificity, held-out sensitivity was **0.298**, specificity **0.931**, and equal-participant-weight PPV **0.255**.
- The predeclared research gate requires AP at least 1.25 times prevalence, AUC at least 0.60, and sensitivity at least 0.50 at the 90% specificity target. The nested result **fails** because sensitivity is too low. Six positive participants also make estimates unstable.

This is not a deployable asthma model. It cannot identify asthma in an undiagnosed person, and it does not support pneumonia, infection, COPD, or an always-on ring alert. No score, probability, alert, model weight, or disease-driven lung visualization is exposed in the app.

### Reproduction

Download the CC BY 4.0 archive from Edinburgh DataShare into ignored `data/aamos00/aamos00.zip`, extract it to `data/aamos00/raw/`, install `requirements-model.txt`, then run:

    python scripts/evaluate_aamos_asthma_worsening.py

The aggregate-only report is `models/aamos_asthma_worsening_metrics.json`. Raw participant files remain in ignored `data/` and must not be committed.

## OSASUD apnea/hypopnea event-window feasibility

The public [OSASUD study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9018698/) provides nearly one million one-second rows from 30 hospitalized stroke-unit participants. It pairs finger PPG and PPG-derived oxygen/perfusion measurements with physician-reviewed polysomnography event annotations. This is an overnight event-window task, not an outpatient sleep-apnea diagnosis, pulmonary-infection diagnosis, or ring validation.

The evaluator builds quality-filtered 30-second windows and uses only finger PPG waveform summaries, PPG-derived pulse timing/variability, SpO₂ and perfusion for its strict PPG-only track. ECG waveforms, thoracic-impedance respiratory rate, airflow, effort, snoring audio, body position, and event metadata are excluded as predictors. A separate expanded-vitals comparison is reported for analysis only and is not eligible for ring inference. All model folds hold out whole participants.

Across 27,627 quality-valid windows (8,640 with at least one apnea/hypopnea second), Extra Trees was the best of three strict PPG-only baselines in the same grouped out-of-fold comparison: equal-participant ROC-AUC **0.780** and average precision **0.633** versus **0.313** prevalence. Because the best model was selected using that same comparison, this is an optimistic feasibility result, not an unbiased final performance estimate. Independent nested or external validation is required. Even a sound event-window detector would not diagnose sleep apnea syndrome, and this clinical finger sensor is not PlumoRing hardware. No result is deployed.

The local source archive is large (about 3.1 GB) and remains in ignored `data/`. The aggregate-only report is `models/osasud_ppg_event_metrics.json`; reproduce with `python scripts/evaluate_osasud_ppg_events.py`.

## COPD classification from PPG: small feasibility experiment

The six Zenodo archives (`2s_db.zip` through `256s_db.zip`) each contain the same six named people: four files labeled COPD and two labeled control. The evaluator checks that the CSV label agrees with the file's class prefix, then tests each duration archive separately. It extracts scale-normalized waveform shape, derivative, peak, and normalized-spectrum summaries from PPG windows. It compares regularized logistic regression, an RBF support-vector machine, and Extra Trees.

Validation is leave-one-participant-out. All sampled windows from the held-out person stay out of model fitting, preprocessing is fitted inside each fold, and the held-out person's window scores are reduced to one median participant score before calculating metrics. The script deterministically samples up to 512 windows per person from each archive so prolific recordings cannot dominate. No patient-level scores or raw signals are written to the repository.

This is a **six-person feasibility audit**, not a working clinical detector. Its purpose is to add true disease-labeled PPG and test leakage-resistant methods; even favorable metrics would not establish performance in new people or PlumoRing hardware. The much larger Figshare feature matrix cannot repair the small sample because its participant IDs are absent and its class counts conflict with the paper's participant table. The original paper reports a random 50/50 epoch split, which can place a person's windows in both train and test; its reported accuracy is not treated as a participant-held-out result.

### Observed participant-held-out results

The reproducible run used up to 512 sampled windows per person for each archive. In the 2-second archive, regularized logistic regression, RBF SVM, and Extra Trees each scored participant-level ROC-AUC **0.000** and average precision **0.525**; the no-skill average-precision reference is **0.667**. All three methods ranked the six held-out participant scores in the opposite direction from the labels on this split. Across each of the other five duration variants, all three ROC-AUC values were also 0.000 and average precision remained 0.525. This is a failed feasibility result, not evidence of COPD detection. The cohort is too small to infer general performance, and no model is deployed.

## Larger datasets to pursue, without mixing tasks

- TOLIFE is currently the strongest open longitudinal COPD cohort used here for a separate exacerbation-forecast question. It contains 86 diagnosed participants, event records, and daily data from multiple non-ring devices. The grouped baseline failed its no-skill gate.
- The 2026 eMEUSE-SANTÉ paper is the best-matched external monitoring evidence found: 220 people with COPD wore a medical wristband for six months, with HR, breathing rate and SpO₂ and 42 physician-validated exacerbations. Its published personalized BVS³ score reported AUC 0.88, 74% sensitivity at 85% specificity, and a mean 4.4-day lead time. These are the authors' results, not PlumoRing results. The underlying data are not public because of re-identification risk; access is by research request and institutional approval. The paper directs data requests to `contact.mars@chu-grenoble.fr`.
- A separate [2024 McGill pilot](https://www.frontiersin.org/journals/signal-processing/articles/10.3389/frsip.2024.1362754/full) collected Oura-ring and wristband data for 21 days from nine people already experiencing a severe COPD exacerbation. Only ring heart rate was near-continuous; the ring's breathing rate and oxygen readings were nightly averages. Raw patient records are not published and require an author request, and the cohort is too small for disease-model training. The study is useful for device feasibility, not validation of an alert.
- The larger [telEPOC COPD telemonitoring study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11868553/) reports 149 patients and 159,719 daily submissions, but its analyzed records are available from the corresponding author on reasonable request. Its red-alarm outcome is partly constructed from the same daily symptom/vital questionnaire fields, so a clean reading-only model needs careful target reconstruction before any use.
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

### Personalized multi-vital trend baseline

Alongside the supervised baselines, a training-free method was evaluated on the same public TOLIFE archive. It compares each day's HR, respiratory-rate and SpO₂ indices with that participant's previous 14 calendar days, applies the expected adverse direction (HR/RR rise, SpO₂ fall), and combines available standardized changes. It requires at least four earlier valid readings for each used channel and at least two current channels. It is an exploratory adaptation for sparse daily data, not an exact reproduction of the published BVS³ pipeline, which used much denser hourly monitoring.

Only 1,015 participant-days from 36 participants met these history and channel requirements; just 3 participants contributed positive forecast days. Equal-participant-weight ROC-AUC was **0.444** and average precision was **0.0064**, below the equal-participant no-skill reference of **0.0075**. Pooled average precision was **0.0248**, also below the subset's pooled positive prevalence (**0.0276**). This method also failed; no threshold or alert is enabled. Results are in `models/tolife_personalized_trend_metrics.json`.

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

### Public ring disease-data search

The publicly documented ring dataset located for model development is HealthRing, which has synchronized ring PPG/motion and reference vitals but healthy-volunteer signal-estimation labels rather than pulmonary diagnosis labels. The large Oura COVID-19 studies establish a relevant infection-monitoring research direction, but their underlying participant-level records are not publicly downloadable under the cited study's data-use policy. Thus, no open dataset found in this review supports a ring-only, participant-held-out COPD/asthma/pneumonia diagnosis model. This is a data-availability limit, not a modeling gap that can be repaired by concatenating unrelated datasets.

## Reproduction

Download TOLIFE v2.0 `index.zip` from the official Zenodo record into the ignored path `data/tolife/index.zip`, install `requirements-model.txt`, and run:

    python scripts/evaluate_pulmonary_study.py

Run the independent personalized-change baseline with:

    python scripts/evaluate_tolife_personalized_trend.py

The aggregate report is `models/pulmonary_feasibility_metrics.json`. The raw data, model weights, and any participant-level outputs must not be committed.

Download the six public COPD PPG duration archives from the [Zenodo record](https://zenodo.org/records/18109876) into the ignored `data/copd_ppg_zenodo/` directory, then run:

    python scripts/evaluate_copd_ppg_subject_holdout.py

This writes aggregate-only results to `models/copd_ppg_screening_metrics.json`. The six duration variants are evaluated separately; they share the same six people and are not independent cohorts.

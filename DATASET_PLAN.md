# PlumoRing AI: public-data study design

PlumoRing AI is a software research prototype. No public dataset currently validates this ring design as a disease detector. We therefore keep ring-measurement research and hospital disease-risk research as separate, clearly named tracks.

## Track A — ring signal estimation

[HealthRing](https://doi.org/10.5281/zenodo.18426864) is a public dataset of ring PPG and inertial measurements from 54 adults, aligned with reference HR, respiration rate, SpO₂ and blood-pressure readings. Its collection includes rest, ordinary activities, treadmill movement and a controlled low-oxygen simulation. The archive is about 2.6 GB. Its authors report error benchmarks; those are published results, not PlumoRing AI results. It has no disease diagnosis labels, so it can support measurement estimation and motion-quality studies, not infection classification.

The dataset is released under CC BY 4.0. Cite Tang et al., *HealthRing: Physiology Dataset for Health Sensing on Rings*, Scientific Data (2026), DOI [10.1038/s41597-026-07289-x](https://doi.org/10.1038/s41597-026-07289-x). The companion implementation and documented record layout are in the authors' [RingTool repository](https://github.com/thuhci/RingTool).

The current repository prepares a compatible ring-data track but does not redistribute the 2.6 GB archive or claim a locally trained ring estimator. We can train it as a follow-on after selecting compute and a subject-held-out protocol; the archive size and 54-person cohort warrant careful evaluation.

## Track B — experimental hospital sepsis-risk sequence model

We use [PhysioNet/CinC Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/) for an experimental 16-hour hospital sepsis-risk benchmark. The challenge has 40,336 de-identified ICU patient records from two publicly released hospital systems, with hourly rows and 40 possible clinical input fields (vitals, labs, demographics, and ICU context), plus the challenge's six-hour-shifted SepsisLabel. The label is sepsis-related, not pneumonia or another lung infection. This is hospital data, not ring data.

Dataset attribution: *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019*, PhysioNet, DOI [10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857), released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The local model, metrics, and replay are derived from sampled challenge records; the replay omits source record identifiers.

- Set A is split by patient into development and validation partitions.
- Set B is held out for external evaluation and is never used for weight selection.
- Imputation is forward-only within each record. The model receives explicit missingness masks for all 40 challenge features. No future rows or outcome fields enter the input. Lab and demographic values are included because this is an ICU clinical-data benchmark; they are not ring readings.
- PhysioNet labels are used as published; they mark the challenge sepsis-risk window, not a pulmonary infection diagnosis.
- Set A is sampled to include at least one window from every training/validation patient, then extra hourly windows are added within the local 8 GB memory budget. Set B evaluation likewise includes every patient before extra windows are sampled. Patient-cluster bootstrap intervals accompany the external estimates.
- The sequence network is compared with logistic regression and histogram gradient boosting on the same sampled windows. Selection uses set-A validation average precision; set B is used once for the reported external comparison. The report records exact counts, ROC-AUC, average precision, patient-level bootstrap intervals, and a fixed-threshold confusion matrix. Scores are research outputs and are not calibrated clinical probabilities.
- Current run: 16,269 development patients and 4,067 validation patients from set A; all 20,000 set-B patients represented. It sampled 120,000 training, 50,000 validation, and 50,000 external windows. On set B, temporal ROC-AUC / average precision were 0.756 / 0.0600; logistic regression 0.700 / 0.0521; histogram gradient boosting 0.743 / 0.0635. The boosted model was selected on validation average precision. Its patient-bootstrap 95% AP interval (0.0569–0.0754) overlaps the temporal model's interval (0.0534–0.0689), so the small point-estimate advantage is uncertain.

Download with `python scripts/download_physionet_sepsis.py`, then train with `python scripts/train_sepsis_risk.py`. Raw data stays in the ignored `data/` folder. Only model weights, a compact de-identified replay, and aggregate metrics are generated for the project.

## Other candidates and why they are separate

| Dataset | Useful study | Why it is not joined into the sepsis/ring model |
|---|---|---|
| [BIDMC PPG and Respiration](https://physionet.org/content/bidmc/1.0.0/) | 53 short ICU recordings with PPG, reference HR/RR/SpO₂ and manually annotated breaths; useful for signal/respiration research. | No disease labels, no ring hardware, and a short critically ill cohort. |
| [ICBHI 2017 respiratory sounds](https://bhichallenge.med.auth.gr/) | Crackle/wheeze respiratory-cycle labels and diagnosis metadata; potential audio research. | It is lung audio, not a signal captured by the proposed ring. |
| [WESAD](https://archive.ics.uci.edu/dataset/465/wesad) | Wrist/chest physiology, motion and stress/affect protocol labels. | No pulmonary disease targets; different body location and restricted academic/non-commercial terms. |

These datasets are not concatenated into one classifier. HealthRing supports ring-vital estimation; BIDMC supports pulse-derived breathing research; PhysioNet supports hospital sepsis-risk research; ICBHI supports respiratory-audio research. Their samples, sensors, and target labels do not line up, so treating them as a single disease dataset would create invalid labels and false performance claims.

Do not align unrelated participants across datasets, fill absent channels with invented values, or present hospital-risk model output as a ring diagnosis. A future disease study needs an appropriate disease-labelled cohort, a pre-registered patient-level split, external validation, calibration, clinical review and regulatory analysis before use beyond research.

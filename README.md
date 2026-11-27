# PlumoRing AI

PlumoRing AI combines an interactive smart-ring concept with a pulmonary-sensing research workspace. The Gold, Onyx, and Doomsday ring assemblies remain product concepts; the heart and lung scene is an interactive anatomical illustration, not a patient scan.

## Research status

The intended research question is whether ring-compatible readings can identify or monitor pulmonary disease. **There is no validated ring-based pulmonary disease model in this repository.** The available public datasets do not provide enough matched ring signals and pulmonary labels to support that claim.

The app keeps our public-dataset research tracks and **Model 1 · Temporal Telemetry State Model** side by side. Model 1 returns five experimental sensor-state classes. A separate demo rules panel can display labels such as asthma or infection from the supplied bundle's hand-written thresholds, but those labels are not neural-network disease predictions and have no calibrated probabilities. Neither track is a validated pulmonary disease diagnosis.

- **COPD flare forecasting:** TOLIFE provides clinical exacerbation records and daily indices from several devices. It is the largest relevant longitudinal cohort found, but it does not use the proposed ring and it covers flare forecasting among people already diagnosed with COPD, not COPD diagnosis.
- **COPD classification from PPG:** a public PPG release contains six identified people (four with COPD and two controls). Three classical models are compared with every person held out in turn across six window-duration variants. The variants reuse the same six people, so this is a feasibility audit, not a dependable disease detector.
- **Ring signal estimation:** HealthRing contains real ring PPG and motion with reference vital signs, but no pulmonary diagnosis labels. A partial participant-held-out analysis yielded nested RR MAE of 2.99 breaths/min on Ring 1 and 3.48 on Ring 2. The downloaded archive is structurally incomplete despite matching the published checksum; only 29 participants and 3,497 windows were recovered.
- **Respiratory-rate estimation:** a separate PPG-only model was evaluated on 53 BIDMC ICU recordings and 832 thirty-second windows using two manual breath readers. Nested participant-held-out participant-macro MAE was 1.77 breaths/min. A model trained on BIDMC did not transfer to HealthRing better than a mean baseline (3.79 vs 3.55 breaths/min), so no ring inference is enabled.
- **Model 1 · Temporal Telemetry State Model:** the provided 4.7 MB Keras model is included and can be run from the analysis panel. It accepts eight readings over 16 steps and returns the five class scores in the supplied app's label order. The supplied bundle did not include its training dataset, split, or validation report. A single-reading demo repeats one snapshot over all 16 steps to match the supplied adapter; this is a shape-compatible preview, not continuous monitoring. Its class scores are uncalibrated and do not mean disease probabilities. A separate, clearly marked rule preview shows textual condition-pattern labels only; it is not a disease model.
- **Asthma worsening:** AAMOS-00 links weekly reports of increased systemic corticosteroid use with MiBand3 heart-rate/activity summaries. A nested participant-held-out evaluation was added; it is retrospective week-level classification and failed the sensitivity gate.
- **Apnea/hypopnea events:** OSASUD contains finger PPG plus physician-reviewed overnight PSG event annotations. A strict PPG-only Extra Trees baseline reached equal-participant ROC-AUC 0.780 and average precision 0.633 versus 0.313 prevalence across participant-held-out folds. This is exploratory event-window screening in hospital monitoring data, not sleep-apnea diagnosis or PlumoRing validation; model selection on the same grouped comparison makes the result optimistic.
- **Personalized COPD trend baseline:** a training-free 14-day per-person HR/RR/SpO₂ change score was compared on TOLIFE. It also fell below the no-skill reference on a sparse, reduced subset; no alert is enabled.
- **Best external evidence located:** a 2026 study reports six-month wearable monitoring in 220 people with COPD and 42 physician-validated exacerbations, but its patient data are restricted. The published score is the study authors' result and has not been tested on PlumoRing or our data.

These datasets are not merged. They differ in sensors, sampling, labels, and tasks. In particular, lung-sound recordings cannot stand in for ring PPG because the proposed ring has no microphone. The original 14-person study's Figshare feature matrix is excluded from grouped validation because it omits participant IDs and its healthy-row total conflicts with its participant table. Its random epoch split does not establish generalization to new people.

TOLIFE's separate clinical-index association audit used 374 visits from 86 people but found too few repeated measurements for every sensor/outcome pair to report a stable within-person association. See [Dataset plan](DATASET_PLAN.md) for measured results, target requirements, and reproduction commands.

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

Separate physiology and association tracks use their own public files:

| Track | Required local archive | Command | Aggregate report |
| --- | --- | --- | --- |
| HealthRing ring RR/SpO₂ estimation | `data/healthring/RingDatasetV2.1_submission.zip` (repaired local copy may be required) | `python scripts/evaluate_healthring_ring_vitals.py` | `models/healthring_ring_vitals_metrics.json` |
| BIDMC PPG-only RR and HealthRing transfer | `data/bidmc/bidmc.zip` | `python scripts/evaluate_bidmc_ppg_respiratory_rate.py` | `models/bidmc_ppg_respiratory_rate_metrics.json` |
| TOLIFE pre-visit clinical associations | `data/tolife/index.zip` | `python scripts/analyze_tolife_clinical_associations.py` | `models/tolife_clinical_associations.json` |

These are research comparisons; the physiology models are not exposed as live inference and do not identify disease. The separate supplied Keras network is available for an explicitly unvalidated telemetry-state preview only.

For the separate COPD-vs-control PPG feasibility track, download the six duration archives from the [official Zenodo record](https://zenodo.org/records/18109876) into `data/copd_ppg_zenodo/`, then run:

    python scripts/evaluate_copd_ppg_subject_holdout.py

This compares regularized logistic regression, an RBF support-vector machine, and Extra Trees using leave-one-participant-out evaluation. It writes aggregate metrics only to `models/copd_ppg_screening_metrics.json`. All six archive variants contain the same six people; no model is deployed. The results cannot be used to diagnose a person or claim the proposed ring detects COPD.

## Data handling and safety

This is a software research prototype, not a medical device. It has no live ring connection, no validated disease inference, and no clinical alerts. Model 1 runs locally and returns five uncalibrated state-class scores; it does not diagnose disease. A separate demo rules panel is illustrative only. Model 1 checks and chatbot conversations are stored in this browser with export and clear controls. Aggregate public-dataset result snapshots have their own separate research history; they are not personal readings. Optional AI summaries/chat contact the selected NVIDIA or OpenAI provider only after the user clicks the relevant action; chat readings are sent only when the user checks the sharing option. The API key stays on the local server in an ignored `.env` file. Voice input/output use browser speech features and depend on browser/device support. See [Privacy](privacy.html).

### Optional AI summaries and chatbot

The chatbot remains available without a key in a limited local reference mode for common pulmonary topics. Those replies use curated educational references and are English only; they do not send messages to an external API. For broader replies in the selected language and reading summaries, copy `.env.example` to `.env`, add an NVIDIA API key to `NVIDIA_API_KEY`, and restart the local server. `AI_PROVIDER=auto` prefers NVIDIA when configured and falls back to OpenAI; `NVIDIA_MODEL` selects the NVIDIA NIM model. The default `qwen/qwen3-next-80b-a3b-instruct` uses NVIDIA's OpenAI-compatible chat-completions endpoint. Qwen reports Qwen3 support for 119 languages and dialects, including many Indian languages, but output quality is not guaranteed ([NVIDIA endpoint details](https://docs.api.nvidia.com/nim/reference/qwen-qwen3-next-80b-a3b-instruct-infer), [Qwen language list](https://qwenlm.github.io/blog/qwen3/)). The NVIDIA chat endpoint can be changed with `NVIDIA_CHAT_URL`. For OpenAI, set `AI_PROVIDER=openai`, add `OPENAI_API_KEY`, and choose `OPENAI_MODEL`. Keep `.env` private and never commit the real key. Chat sends the user's message and up to 12 recent conversation turns only after Send; saved readings are included only when the user checks the box. Summary requests send at most 12 saved snapshots after an explicit click. Requests use the selected provider; the OpenAI Responses API request disables response storage. Provider usage may be subject to account limits or charges. The language menu includes English and India's 22 scheduled languages; response quality depends on the selected AI model and varies by language. All chatbot output is educational, not diagnosis or treatment. A reading history stays in the current browser and can be exported or cleared there.

The chatbot's **MIC** button uses the browser's SpeechRecognition implementation, when available, to fill in the question. It does not submit automatically; review the transcript and press Send. Depending on browser and operating-system settings, speech recognition may be handled by a browser/platform speech service. The app does not record or store microphone audio. Assistant replies have a **READ ALOUD** control that uses a matching voice installed on the device; voice availability varies by language. Microphone input and speech output are separate from the NVIDIA/OpenAI text API.

The supplied [public-apis catalog](https://github.com/public-apis/public-apis) is a directory of API links rather than a disease-name dataset. Its Health section lists NHS Health A-Z with API-key authentication and Open Disease for COVID-19/influenza case data; neither is a comprehensive, unauthenticated pulmonary-condition taxonomy. The chat therefore exposes a bounded educational topic list (asthma, COPD, bronchitis, pneumonia, and sleep apnea) with cited public-health references. These names are not Model 1 classes or detected diagnoses.

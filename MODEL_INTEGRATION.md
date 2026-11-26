# Research model and local app

## Run the frontend and API

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-model.txt
python server.py
```

Open `http://127.0.0.1:5000`. The ring studio, preserved 3D heart and lungs, and the precomputed public-data replay load without a live sensor. `GET /api/health` reports whether local weights are available; `GET /api/replay` serves the compact, de-identified held-out timeline. The API binds only to loopback.

## Rebuild from public data

```bash
python scripts/download_physionet_sepsis.py
python scripts/train_sepsis_risk.py
```

The downloader fetches training set A and B from the public PhysioNet S3 mirror. Training uses patient-separated set-A development/validation and holds set B for external evaluation. The generated files are `models/sepsis_risk.pt`, `models/sepsis_metrics.json`, and `assets/research_replay.json`. Raw patient files remain under ignored `data/` and must not be committed.

## Model and input contract

The temporal model is a compact multi-scale temporal convolution + squeeze/excitation + bidirectional GRU + self-attention classifier. Its input has shape `(batch, 16 hourly steps, 80 channels)`: the 40 PhysioNet clinical fields, normalized using training-set statistics, plus one observed/missing mask per field. Within each patient, missing values are carried forward only from earlier hours; the model receives no future rows and no SepsisLabel input. It predicts the benchmark's six-hour-shifted SepsisLabel from hospital data.

The temporal network is compared with a logistic-regression baseline and a histogram-gradient-boosting baseline. Selection uses set-A patient-held-out average precision; final comparisons use set B, which contains a separate hospital cohort. Each evaluation sample includes at least one window per patient and extra windows are added within the machine's memory budget. External metrics include patient-cluster bootstrap intervals to reflect repeated windows from the same person.

`POST /api/score-window` accepts `{"sequence": [<16 chronological hourly clinical rows>]}`. Each row may be an object with the published 40 feature names or a 40-value array in that exact order; unmeasured fields may be `null`. Alias keys `hr`, `spo2`, and `resp` are supported for the matching fields. Validation rejects sequences that only contain ring-compatible readings or omit the wider clinical context. The model output is a research benchmark score, not a calibrated probability, diagnosis, severity estimate, alert, or care recommendation.

## Ring model status

HealthRing is the most relevant public ring-sensor dataset found: it includes raw ring PPG/IMU and reference HR/RR/SpO₂ measurements, but no disease labels. It is a separate estimation/quality track and is not fused with hospital sepsis records. The current app's ring finish is a product concept; its sensors and placements have not been verified against PlumoRing CAD. Consequently, this project cannot currently claim ring-based disease detection; it can study ring-vital estimation and, separately, hospital sepsis-risk classification.

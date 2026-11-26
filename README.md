# PlumoRing AI

PlumoRing AI is a responsive product and research interface for a proposed health-sensing ring. The top experience presents the Gold, Onyx, and Doomsday ring concepts as interactive 3D assemblies. The component explorer explains proposed parts; it is illustrative and not production CAD.

Below the product experience, the project separates continuous monitoring from deep-learning research:

- **Monitoring:** a de-identified, hourly replay from a held-out hospital dataset, with pulse/oxygen/breathing trends, change markers, clinical-input coverage, and synchronized heart/lung motion. It is a recorded dataset replay, not a live ring stream.
- **Deep learning:** a temporal Conv + BiGRU + attention model is compared with logistic regression and histogram gradient boosting on the public PhysioNet Challenge 2019 sepsis benchmark. The hospital models can use up to 40 hourly clinical fields (vitals, labs, demographics and context); they are not smart-ring models and do not diagnose lung infection.
- **Ring sensing research:** the public HealthRing dataset provides real ring PPG/IMU and reference HR/RR/SpO₂ for a separate signal-estimation study. It has no disease diagnosis labels and is not joined to the hospital cohort.

The original 3D heart and both lungs remain in the analysis view. Their motion follows the recorded HR and respiration readings; the physiology-only surface preview responds to oxygen and breathing values. It is not an image or severity measurement and never encodes a disease result.

## Open locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-model.txt
python server.py
```

Visit `http://127.0.0.1:5000`. The included replay and trained model files are served locally by Flask; the API exposes health and hospital-sequence scoring routes. Its scoring route rejects ring-only inputs because that model was trained on hospital clinical records. No account or cloud service is required.

## Download and train

```bash
python scripts/download_physionet_sepsis.py
python scripts/train_sepsis_risk.py
```

See [Dataset plan](DATASET_PLAN.md) and [Model integration](MODEL_INTEGRATION.md) for data provenance, patient-level split design, model comparison, API details, and limits. Raw datasets are stored locally under `data/` and are ignored by Git.

## Safety and data handling

This is a software research prototype, not a medical device. The replay uses de-identified public hospital data at one-hour steps. Its risk scores are experimental and uncalibrated; they must not guide diagnosis or care. Nothing in the page connects to a physical ring or continuously collects a person's vital signs. See [Privacy](privacy.html).

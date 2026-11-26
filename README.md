# Intelligent IoT-Based Energy Monitoring and Optimization

> **PFE project — Treetronix**

## Overview

This project explores an IoT-based energy-monitoring solution that collects electrical measurements from a **Schneider PowerLogic PM2200** meter, transfers them through an **RS485-to-Ethernet gateway** to a **Raspberry Pi**, stores measurements for analysis, and experiments with **LSTM-based machine-learning models** for energy forecasting and anomaly detection. The original project material also documents a TagoIO dashboard and notification experiments.

The repository groups the deliverables, datasets, model artifacts, experiments, screenshots and final report in a maintainable layout. Some scripts are exploratory versions and may require path or dataset adjustments before execution (see [Known limitations](#known-limitations)).

## System architecture

```text
Electrical installation
        │
        ▼
Schneider PM2200 energy meter
        │  Modbus RTU / RS485
        ▼
RS485–Ethernet gateway
        │  Modbus TCP / Ethernet
        ▼
Raspberry Pi 4
        │
        ├── Data acquisition and local storage (SQLite, as documented)
        └── Data forwarding to TagoIO
                    │
                    ▼
             Dashboard / monitoring

Historical energy data ──► Python data preparation
                         └► LSTM experiments
                              ├─ Forecasting
                              └─ Anomaly detection
```

*The diagram summarizes the documented project workflow; it does not imply that every component is fully automated by the scripts included in this repository.*

## Main technologies

- **IoT and hardware:** Raspberry Pi 4, Schneider PM2200, RS485/Ethernet gateway
- **Communication:** Modbus RTU, Modbus TCP, PyModbus
- **Data and monitoring:** Python, Pandas, NumPy, SQLite, TagoIO
- **Machine learning:** TensorFlow/Keras LSTM models, LSTM autoencoder, scikit-learn preprocessing and metrics
- **Notifications:** email and SMS workflow screenshots

## Repository structure

```text
PFE_Treetronix/
├── data/
│   ├── training/       # Training dataset(s)
│   ├── testing/        # Test datasets
│   └── experiments/    # Datasets used by earlier experiments
├── src/
│   ├── training/       # Model training scripts
│   ├── evaluation/     # Model evaluation / prediction scripts
│   └── experiments/    # Earlier LSTM and classification experiments
├── models/             # Saved Keras models, scalers and thresholds
├── docs/
│   ├── screenshots/    # Implementation and dashboard screenshots
│   └── report/         # Final PFE report (PDF)
├── demo/               # Final project video
├── README.md
├── requirements.txt
└── .gitignore
```

## Machine-learning workflow

1. Prepare and sort the energy measurements chronologically (or by night identifier for night-based experiments).
2. Aggregate energy readings by night where the selected experiment requires it.
3. Normalize input values using `MinMaxScaler`.
4. Build time sequences for LSTM models.
5. Train forecasting or reconstruction models with TensorFlow/Keras.
6. Save model artifacts and associated preprocessing/threshold files in `models/`.
7. Evaluate predictions or reconstruction errors against test data.

The repository contains several experimental approaches, including sequence forecasting, an LSTM autoencoder and classification experiments. They are preserved as project history rather than presented as one unified production pipeline.

## Setup

Python **3.10–3.11** is a reasonable starting point for the TensorFlow/Keras scripts. Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Running the scripts

Run commands from the repository root. For example:

```bash
python src/training/train.py
python src/training/autoencoder_train.py
python src/training/forecasting.py
```

**Important:** these are the original training experiments reorganized into folders. Before running a script, verify that its expected CSV filename and columns match the dataset you intend to use. Some scripts save artifacts to their current working directory; move or configure those outputs under `models/` if you adapt the scripts. Evaluation scripts also contain machine-specific Windows paths that must be replaced with relative paths.

## Model artifacts

The `models/` directory contains the saved `.keras` models and serialized preprocessing/threshold artifacts supplied with the project. Keep each model paired with the scaler and threshold generated for that model; do not assume that similarly named artifacts from different experiments are interchangeable. Only load trusted `.save`/joblib files.

## Documentation and demo

- Final report: [`docs/report/final_pfe_report.pdf`](docs/report/final_pfe_report.pdf)
- Implementation screenshots: [`docs/screenshots/`](docs/screenshots/)
- Project video: [`demo/final_project_demo.mp4`](demo/final_project_demo.mp4)

## Known limitations

- The supplied scripts represent multiple development iterations; they are not yet consolidated into a single configurable application.
- Some multivariate experiment scripts refer to `high_performance_energy_dataset.csv`, which is not present under that exact name in the supplied archive.
- Some evaluation scripts use absolute paths from the original Windows workstation. Update these paths before execution.
- A few CSV files in the source archive contain headers only or may be intermediate datasets. Check row counts and column names before training.
- The repository does not include a complete Raspberry Pi acquisition service or cloud deployment configuration as source code; the hardware and dashboard stages are documented mainly through the report and screenshots.

## Security and privacy

Do not commit API keys, TagoIO tokens, passwords, private IP details that should not be public, or screenshots exposing credentials. The source archive contained an image named `secret key`; it has intentionally been left out of this GitHub-ready copy. Review all screenshots and report pages before making the repository public. Store credentials in environment variables or a local `.env` file, and never commit `.env`.

## License

No license was provided with the source materials. Add a license only after confirming that you have the right to publish the code, datasets, report, screenshots and video.

---

**Project:** PFE Treetronix — IoT energy monitoring and AI-based optimization.

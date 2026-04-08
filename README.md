# Traffic Sign Detection: YOLOv8n on TT100K

Compact traffic-sign detection project for an Automated Driving System capstone.  
The model is trained on TT100K (50 classes) with YOLOv8n and exported to ONNX for Raspberry Pi CPU inference.

## Features
- YOLOv8n training pipeline for TT100K
- Config-driven workflow via YAML
- ONNX export script for edge deployment
- Clean repo layout for research and production iteration
- Basic CI + tests scaffolding

## Project Structure
```text
traffic-sign-detection-model/
├── .github/workflows/ci.yml
├── configs/
│   ├── default.yaml
│   └── kaggle.yaml
├── notebooks/
│   └── README.md
├── scripts/
│   └── export_onnx.py
├── src/
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── train.py
│   └── utils/
│       ├── __init__.py
│       └── io.py
├── tests/
│   └── test_config.py
├── .gitignore
├── LICENSE
├── requirements.txt
└── run.py
```

## Dataset
Expected TT100K path (Kaggle):
- `/kaggle/input/datasets/braunge/tt100k/mydata`

Dataset layout:
```text
mydata/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

## Quick Start
1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Train:
```bash
python run.py --config configs/kaggle.yaml --mode train
```

3) Export ONNX:
```bash
python run.py --config configs/kaggle.yaml --mode export
```

## Notes
- For demonstration, set `epochs: 5`; for production, use `30-50`.
- The model is YOLOv8n for a speed/accuracy trade-off on edge devices.

## License
Apache-2.0. See `LICENSE`.

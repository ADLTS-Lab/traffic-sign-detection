# Traffic Sign Detection: YOLOv8n on TT100K

Compact traffic-sign detection project for an Automated Driving System capstone.  
The model is trained on TT100K (50 classes) with YOLOv8n and exported to ONNX.

## Features
- YOLOv8n training pipeline for TT100K (50 classes)
- Config-driven workflow via YAML
- Validation and single-image inference commands
- ONNX export script for Raspberry Pi CPU deployment
- Kaggle notebook + modular Python pipeline
- CI + tests scaffolding

## Project Structure
```text
traffic-sign-detection-model/
├── .github/workflows/ci.yml
├── configs/
│   ├── default.yaml
│   └── kaggle.yaml
├── data/
├── notebooks/
├── scripts/
│   ├── export_onnx.py
│   └── infer_image.py
├── src/
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── evaluate.py
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
For local training, place the dataset under:
- `data/tt100k/mydata`

Expected layout:
```text
data/tt100k/mydata/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

If you want to download the Kaggle version manually, use the Kaggle API and unzip it into `data/tt100k/mydata`. The Kaggle notebook can still use the Kaggle-hosted path directly through `configs/kaggle.yaml`.

Kaggle path used by the notebook:
```text
/kaggle/input/datasets/braunge/tt100k/mydata
```

## Quick Start
1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Download or copy the dataset so it exists at `data/tt100k/mydata`.

3) Train locally:
```bash
python run.py --config configs/default.yaml --mode train
```

4) Validate:
```bash
python run.py --config configs/default.yaml --mode validate
```

5) Export ONNX:
```bash
python run.py --config configs/default.yaml --mode export
```

6) Inference on one image:
```bash
python run.py --config configs/default.yaml --mode infer --image /path/to/image.jpg --conf 0.25
```

## Notes
- The default config auto-selects the device when possible, so it works on GPU machines and CPU-only machines.
- If you are on Kaggle, use `configs/kaggle.yaml` instead of the local default.
- Training resumes from the latest checkpoint in `runs/detect/tt100k_training/yolov8n_30e/weights/last.pt` when that file exists.

## License
Apache-2.0. See `LICENSE`.

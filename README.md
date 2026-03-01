# Mini Project 7: Blood Cell Detection using YOLOv26

> COMP 9130 — Applied Artificial Intelligence | Mini Project VII

## Problem Description

MedScan Diagnostics is a healthcare technology company developing AI-assisted screening tools for pathology labs. This project builds a proof-of-concept object detection system for **automated blood cell detection** in blood smear images using the YOLOv26 model.

The goal is to detect and classify **red blood cells (RBC)**, **white blood cells (WBC)**, and **platelets** to support automated complete blood count (CBC) analysis for clinical laboratories processing thousands of samples daily.

## Dataset

- **Name:** BCCD (Blood Cell Count and Detection)
- **Source:** [Roboflow Universe](https://universe.roboflow.com/) — BCCD Dataset
- **Classes:** 3 (RBC, WBC, Platelet)
- **Size:** ~360 images with bounding box annotations
- **Format:** YOLOv8/YOLO format (bounding box annotations)

### How to Download the Dataset

<!-- TODO: Add specific Roboflow download instructions or API snippet -->

```bash
# Option 1: Download via Roboflow API
# pip install roboflow
# from roboflow import Roboflow
# rf = Roboflow(api_key="YOUR_API_KEY")
# project = rf.workspace().project("bccd-xxxxxxx")
# dataset = project.version(X).download("yolov8")

# Option 2: Manual download
# 1. Visit the BCCD dataset page on Roboflow Universe
# 2. Select YOLOv8 format
# 3. Download and extract to data/raw/
```

## Environment Setup

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended) or CPU

### Installation

```bash
# Clone the repository
git clone https://github.com/<username>/MiniProject7-BloodCellDetection.git
cd MiniProject7-BloodCellDetection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## How to Run

```bash
# 1. Download the dataset (see instructions above) and place in data/raw/

# 2. Open and run the training notebook
jupyter notebook notebooks/

# 3. Follow the cells in order:
#    - Data exploration & preprocessing
#    - Model training (YOLOv26)
#    - Evaluation & metrics
#    - Visualization of predictions
```

## Project Structure

```
MiniProject7-BloodCellDetection/
├── README.md                  # Project documentation (this file)
├── requirements.txt           # Python dependencies
├── notebooks/                 # Jupyter notebooks
│   └── ...                    # Training, evaluation, and analysis notebooks
├── data/
│   ├── raw/                   # Original dataset (not tracked by git)
│   └── processed/             # Preprocessed data (if any)
├── models/                    # Saved model weights (not tracked by git)
├── results/
│   ├── figures/               # Prediction visualizations, training curves
│   └── metrics/               # Evaluation metrics, confusion matrices
└── docs/                      # Additional documentation or report assets
```

## Results Summary

<!-- TODO: Fill in after training is complete -->

| Metric        | Value |
|---------------|-------|
| mAP@50        | —     |
| mAP@50-95     | —     |
| Precision     | —     |
| Recall        | —     |

### Per-Class Performance

| Class    | Precision | Recall | mAP@50 |
|----------|-----------|--------|--------|
| RBC      | —         | —      | —      |
| WBC      | —         | —      | —      |
| Platelet | —         | —      | —      |

### Sample Predictions

<!-- TODO: Add prediction visualization images -->
<!-- ![Prediction Example](results/figures/prediction_example.png) -->

### Training Curves

<!-- TODO: Add training loss and mAP curves -->
<!-- ![Training Curves](results/figures/training_curves.png) -->

## Key Findings

<!-- TODO: Fill in after analysis is complete -->

- **Class Balance:** —
- **Confidence Threshold Recommendation:** —
- **Most Challenging Class:** —
- **False Positives vs. False Negatives:** —
- **Business Recommendation:** —

## Team Member Contributions

| Member | Contributions |
|--------|---------------|
| <!-- Name 1 --> | <!-- e.g., Data preprocessing, model training, README --> |
| <!-- Name 2 --> | <!-- e.g., Evaluation, analysis, report writing --> |

## References

- Ultralytics YOLOv26: https://docs.ultralytics.com/
- BCCD Dataset on Roboflow Universe: https://universe.roboflow.com/
- Original BCCD Dataset: https://github.com/Shenggan/BCCD_Dataset

## Disclaimer

This is an educational exercise for COMP 9130 — Applied Artificial Intelligence. Real medical AI systems require extensive validation, regulatory approval, and clinical trials before deployment.

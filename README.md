# 🩺 Retinal Fundus Image Analysis Platform

> **Digital Image Processing (DIP) Lab Project**  
> Blood Vessel Segmentation & Hard-Exudate Candidate Extraction using Classical Computer Vision.

---

## 📌 Project Overview

This project implements an end-to-end **Classical Digital Image Processing** system to analyze digital retinal fundus photographs. Without relying on deep neural networks, it applies deterministic mathematical transforms to extract key anatomical features for diabetic retinopathy screening.

### Core DIP Techniques Used

- **Green Channel Extraction:** Isolates the spectrum (540–570 nm) where blood vessels exhibit maximum contrast against the retinal background.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Enhances local micro-contrast without over-amplifying background sensor noise.
- **Gabor Filter Bank (12 Orientations):** Uses directional bandpass filters and Maximum Intensity Projection (MIP) to detect curvilinear blood vessels across all possible angles.
- **White Top-Hat Transform:** Subtracts the morphological opening of an image from the original to isolate small, bright, and localized lesions (hard exudates).
- **Morphological Operations & Otsu Thresholding:** Uses automated variance-based binarization, followed by morphological opening and closing to remove noise and bridge gap artifacts.

---

## 🔬 Processing Pipeline

```text
Input Retinal Image (RGB)
│
├── Field of View (FOV) Mask
│
├──────────────────────────────────────────────┐
│                                              │
▼                                              ▼

Path A: Vessel Extraction                Path B: Exudate Detection

1. Extract Green Channel                 1. Convert to CIE-L*a*b*
2. CLAHE Enhancement                     2. Extract L (Luminance) Channel
3. Gaussian Background Correction        3. CLAHE on L Channel
4. 12-Orientation Gabor Bank             4. White Top-Hat Transform
5. Maximum Intensity Projection          5. Connected Components Filtering
6. Otsu Threshold + Morphology           6. Region Geometry Analysis

│                                              │
└──────────────────────────────────────────────┘
                       │
                       ▼

      Interactive Streamlit Dashboard
      • Live Parameter Tuning
      • Visual Overlays
      • Performance Metrics
```

---

## 📁 Repository Structure

```text
retinal_dip_analysis/
├── README.md                # Project documentation
├── requirements.txt         # Python package dependencies
├── app.py                   # Interactive Streamlit web application
├── main.py                  # Command-line interface
├── sample_images/           # Test images and GT masks
└── src/
    ├── __init__.py
    ├── fov.py               # Field of View extraction
    ├── preprocessing.py     # Green channel, CLAHE, illumination correction
    ├── gabor.py             # Gabor filter bank & MIP
    ├── vessels.py           # Vessel segmentation pipeline
    ├── exudates.py          # Hard exudate detector
    ├── visualization.py     # Diagnostic overlays
    └── metrics.py           # Dice, IoU, Sensitivity, Specificity
```

---

## ⚙️ Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/retinal-dip-analysis.git
cd retinal-dip-analysis
```

### 2. Create and Activate a Virtual Environment

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Method 1: Interactive Streamlit Web Application

The recommended mode for visualizing intermediate DIP stages and tuning hyperparameters dynamically.

```bash
streamlit run app.py
```

The application launches at:

```text
http://localhost:8501
```

---

### Method 2: Command-Line Interface (CLI)

Run the full pipeline directly from the terminal.

#### Basic Run

```bash
python main.py --image sample_images/test_image.png
```

#### Evaluation Using Ground Truth

```bash
python main.py \
    --image sample_images/test_image.png \
    --gt sample_images/ground_truth.png
```

#### Generated Outputs

```text
output_overlay.png
output_vessels.png
output_exudates.png
```

---

## 📊 Evaluation Metrics

When a manually annotated ground-truth vessel mask (**G**) is available, the system evaluates the predicted mask (**P**) using the following metrics:

| Metric | Formula | Purpose |
|----------|-----------|-----------|
| Dice Coefficient (F1) | `2TP / (2TP + FP + FN)` | Measures overlap between prediction and ground truth |
| IoU (Jaccard Index) | `TP / (TP + FP + FN)` | Measures intersection relative to union |
| Sensitivity (Recall) | `TP / (TP + FN)` | Fraction of vessel pixels correctly detected |
| Specificity | `TN / (TN + FP)` | Fraction of background pixels correctly rejected |

Where:

- **TP** = True Positives
- **FP** = False Positives
- **TN** = True Negatives
- **FN** = False Negatives

---

## 🎛️ Streamlit Dashboard Features

### Vessel Analysis

- Green-channel enhancement
- CLAHE contrast boosting
- Gabor filter vessel detection
- Vessel density estimation

### Exudate Analysis

- CIE-Lab color space processing
- White Top-Hat lesion enhancement
- Candidate extraction
- Connected-component measurements

### Interactive Outputs

- Original retinal image
- Vessel segmentation mask
- Composite diagnostic overlay
- Intermediate DIP stages
- Lesion candidate geometry table
- Ground-truth performance evaluation

---

## 🛠️ Technology Stack

### Programming Language

- Python 3.11.17

### Core DIP Libraries

- OpenCV (`cv2`)
- NumPy
- Pillow

### UI Framework

- Streamlit




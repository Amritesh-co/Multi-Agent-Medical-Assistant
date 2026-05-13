# Srujan — Musculoskeletal, Breast & Extended Dermatology Agents

## Role
You own three agents across different imaging modalities:
1. **FRACTURE_AGENT** — Musculoskeletal X-ray abnormality detection (DenseNet169, MURA dataset)
2. **MAMMOGRAPHY_AGENT** — Breast finding classification (EfficientNet, CBIS-DDSM)
3. **HAM10000_AGENT** — 7-class skin lesion dermatology classification (EfficientNet, HAM10000)

You deliver working inference wrappers, trained checkpoints, and training scripts to Amritesh for integration. You do NOT modify `config.py`, `__init__.py`, or `agent_decision.py` — Amritesh owns those.

---

## Technology Stack

- Python 3.10+
- PyTorch + torchvision
- `albumentations` for augmentations
- `sklearn.metrics` for evaluation
- `pandas` for CBIS-DDSM CSV metadata handling

---

## Strict Output Contracts — Read This First

**Fracture:**
```python
{
    "label":            str,    # "Normal" or "Abnormal"
    "confidence":       float,
    "body_part":        str,    # e.g., "WRIST", "SHOULDER", "ELBOW", etc.
    "needs_validation": True,
    "explanation":      str
}
```

**Mammography:**
```python
{
    "label":            str,    # "BENIGN" or "MALIGNANT"
    "confidence":       float,
    "finding_type":     str,    # "MASS" or "CALCIFICATION"
    "needs_validation": True,
    "explanation":      str
}
```

**HAM10000:**
```python
{
    "label":            str,    # one of 7 class abbreviations
    "confidence":       float,
    "needs_validation": True,
    "explanation":      str
}
```

All `predict()` take only `image_path: str`. Fracture takes an additional `body_part: str` argument (see below).

---

## Agent 1: FRACTURE_AGENT

### Folder Structure
```
agents/image_analysis_agent/fracture_agent/
├── fracture_inference.py
├── train.py
├── models/
│   ├── fracture_densenet.pth
│   └── label_map.json
```

### Dataset

- **Source**: MURA (Musculoskeletal Radiographs) — Stanford ML Group
  - Download: [MURA on Kaggle](https://www.kaggle.com/datasets/cjinny/mura-v11)
  - OR official: [MURA — Stanford AIMI](https://stanfordaimi.azurewebsites.net/datasets/20c1e8a9-f8c6-4b2b-8db0-e7b3c7b53e67)
- **Body parts**: XR_SHOULDER, XR_ELBOW, XR_FINGER, XR_HAND, XR_HUMERUS, XR_FOREARM, XR_WRIST
- **Labels**: `positive` (abnormal — fracture or other pathology) and `negative` (normal)
- **Size**: ~40,561 multi-view radiograph images across 7 body parts

### Data Preparation

MURA's directory structure is:
```
MURA-v1.1/
├── train/
│   ├── XR_SHOULDER/
│   │   ├── patient00001/
│   │   │   ├── study1_positive/   ← contains image(s)
│   │   │   └── study1_negative/
│   │   └── ...
│   ├── XR_ELBOW/ ...
│   └── ...
└── valid/
    └── (same structure)
```

Flatten into:
```
data/fracture/
├── train/
│   ├── Normal/       ← all negative studies from all body parts
│   └── Abnormal/     ← all positive studies from all body parts
├── val/
│   ├── Normal/
│   └── Abnormal/
└── test/
    ├── Normal/
    └── Abnormal/
```

**Important**: Preserve body part information in the filename or in a sidecar CSV so inference can report which body part the image came from. Rename files as:
```
<body_part>_<patient_id>_<study_id>_<image_index>.png
e.g., WRIST_patient00123_study1_img001.png
```

This way, at inference time, the `predict()` method can extract `body_part` from the filename or the caller can pass it directly.

### label_map.json
```json
{"0": "Normal", "1": "Abnormal"}
```

### Preprocessing Constants
```python
IMAGE_SIZE = 320
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
```

### train.py

Required constants:
```python
DATASET_SOURCE  = "MURA v1.1 — Stanford AIMI (https://stanfordaimi.azurewebsites.net/)"
RANDOM_SEED     = 42
CLASS_NAMES     = ['Normal', 'Abnormal']
IMAGE_SIZE      = 320
BATCH_SIZE      = 32
EPOCHS          = 30
LR              = 1e-4
CHECKPOINT_PATH = "./agents/image_analysis_agent/fracture_agent/models/fracture_densenet.pth"
LABEL_MAP_PATH  = "./agents/image_analysis_agent/fracture_agent/models/label_map.json"
```

**Model (DenseNet169 as specified in the plan):**
```python
import torchvision.models as models
import torch.nn as nn

model = models.densenet169(weights="IMAGENET1K_V1")
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 2)   # binary
```

**Class imbalance:** MURA is roughly balanced overall but varies per body part. Use WeightedRandomSampler.

**Loss:** `nn.CrossEntropyLoss()`

**Important — Multi-image studies:** MURA studies have multiple images per patient visit. The standard MURA evaluation averages predictions across all images in a study. For simplicity in this project, train on individual images and predict per image. Note this in your report.

**Augmentations:**
```python
import albumentations as A
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.CLAHE(clip_limit=2.0, p=0.3),    # CLAHE helps with X-ray bone detail
    A.Rotate(limit=10, p=0.4),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])
```

**Required metrics:** Accuracy, AUC, F1, Precision, Recall — per-body-part breakdown if possible (keep a CSV tracking body_part per test sample)

### fracture_inference.py — Exact Class Spec

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)

CLASS_NAMES = ['Normal', 'Abnormal']
IMAGE_SIZE  = 320
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Body part extraction from filename prefix (matches renamed file convention)
BODY_PARTS = ['SHOULDER', 'ELBOW', 'FINGER', 'HAND', 'HUMERUS', 'FOREARM', 'WRIST']

EXPLANATION_MAP = {
    ('Normal',   True):  "No fracture or abnormality detected in this X-ray.",
    ('Abnormal', True):  "Abnormality detected — possible fracture, dislocation, or other pathology. Orthopedic consultation required.",
    ('Normal',   False): "No fracture or abnormality detected.",
    ('Abnormal', False): "Abnormality detected. Orthopedic consultation required."
}

class FractureClassification:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = self._build_model()
        self._load_weights()
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

    def _build_model(self):
        model = models.densenet169(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
        return model.to(self.device)

    def _load_weights(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            logger.info(f"Fracture model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load fracture model: {e}")
            raise

    def _extract_body_part(self, image_path: str) -> str:
        """Extract body part from filename prefix if file was renamed per convention."""
        filename = os.path.basename(image_path).upper()
        for bp in BODY_PARTS:
            if filename.startswith(bp):
                return bp
        return "UNKNOWN"

    def predict(self, image_path: str) -> dict:
        """
        Returns:
            {"label": str, "confidence": float, "body_part": str,
             "needs_validation": True, "explanation": str}
        """
        try:
            body_part = self._extract_body_part(image_path)
            image = Image.open(image_path).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs  = torch.softmax(logits, dim=1)
                conf, idx = torch.max(probs, dim=1)
            label = CLASS_NAMES[idx.item()]
            explanation = (
                f"Analysis of {body_part.title()} X-ray: "
                + ("Abnormality detected — possible fracture or pathology. Orthopedic consultation required."
                   if label == "Abnormal"
                   else "No significant abnormality detected.")
            )
            return {
                "label":            label,
                "confidence":       round(conf.item(), 4),
                "body_part":        body_part,
                "needs_validation": True,
                "explanation":      explanation
            }
        except Exception as e:
            logger.error(f"Fracture inference error: {e}")
            return {"label": "unknown", "confidence": 0.0, "body_part": "UNKNOWN",
                    "needs_validation": True, "explanation": str(e)}
```

---

## Agent 2: MAMMOGRAPHY_AGENT

### Folder Structure
```
agents/image_analysis_agent/mammography_agent/
├── mammography_inference.py
├── train.py
├── models/
│   ├── mammography_efficientnet.pth
│   └── label_map.json
```

### Dataset

- **Source**: CBIS-DDSM (Curated Breast Imaging Subset of DDSM)
  - Download: [CBIS-DDSM on TCIA](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)
  - OR Kaggle: search "CBIS-DDSM" on Kaggle — cropped ROI version is most convenient
  - Download command: `kaggle datasets download -d awsaf49/cbis-ddsm-breast-cancer-image-dataset`
- **Structure**: Mass cases + Calcification cases, each labeled BENIGN or MALIGNANT
- **Task**: Binary classification — BENIGN vs MALIGNANT — with metadata about finding_type (MASS or CALCIFICATION)

### Understanding the Data

CBIS-DDSM provides:
- Full mammogram images (DICOM) — too large for patch-level training
- **ROI crop images** — recommended for this project
- A CSV with: `patient_id`, `image_file_path`, `pathology` (BENIGN/MALIGNANT), `abnormality_type` (MASS/CALC)

Use the ROI cropped images. Both MASS and CALCIFICATION cases should be used together in a single binary classifier.

### Data Preparation

Organize by pathology:
```
data/mammography/
├── train/
│   ├── BENIGN/     ← includes both benign mass and benign calc ROI crops
│   └── MALIGNANT/  ← includes both malignant mass and malignant calc ROI crops
├── val/
│   ├── BENIGN/
│   └── MALIGNANT/
└── test/
    ├── BENIGN/
    └── MALIGNANT/
```

**Preserve finding_type in filename:** Rename files as `MASS_<original_name>.png` or `CALC_<original_name>.png` so inference can extract finding_type from the filename.

### label_map.json
```json
{"0": "BENIGN", "1": "MALIGNANT"}
```

### Preprocessing Constants
```python
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
```

### DICOM to PNG Conversion (if working from DICOM files)
```python
import pydicom
import numpy as np
from PIL import Image

def dcm_to_png(dcm_path, output_path):
    dcm = pydicom.dcmread(dcm_path)
    arr = dcm.pixel_array.astype(float)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    img.save(output_path)
```

### train.py

Required constants:
```python
DATASET_SOURCE  = "CBIS-DDSM — TCIA (https://www.cancerimagingarchive.net/collection/cbis-ddsm/)"
RANDOM_SEED     = 42
CLASS_NAMES     = ['BENIGN', 'MALIGNANT']
IMAGE_SIZE      = 224
BATCH_SIZE      = 32
EPOCHS          = 30
LR              = 1e-4
CHECKPOINT_PATH = "./agents/image_analysis_agent/mammography_agent/models/mammography_efficientnet.pth"
LABEL_MAP_PATH  = "./agents/image_analysis_agent/mammography_agent/models/label_map.json"
```

**Model:**
```python
model = models.efficientnet_b3(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
```

**Class imbalance:** CBIS-DDSM tends to have more benign than malignant cases. Use WeightedRandomSampler.

**Loss:** `nn.CrossEntropyLoss()` with class weights

**Augmentations (mammography-appropriate — no vertical flip, mammograms have an orientation):**
```python
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),         # horizontal flip is valid for mammograms
    A.RandomBrightnessContrast(p=0.4),
    A.CLAHE(clip_limit=3.0, p=0.4),  # CLAHE helps micro-calcification visibility
    A.Rotate(limit=15, p=0.3),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])
# NO VerticalFlip — mammogram orientation matters
```

**Required metrics:** Accuracy, AUC, F1, Precision, Recall

### mammography_inference.py — Exact Class Spec

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)

CLASS_NAMES   = ['BENIGN', 'MALIGNANT']
IMAGE_SIZE    = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

EXPLANATION_MAP = {
    'BENIGN':    'Finding classified as BENIGN. Routine follow-up recommended per clinical protocol. Radiologist review required.',
    'MALIGNANT': 'Finding classified as MALIGNANT. Urgent biopsy and oncology referral recommended. Do not delay.'
}

class MammographyClassification:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = self._build_model()
        self._load_weights()
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

    def _build_model(self):
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        return model.to(self.device)

    def _load_weights(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            logger.info(f"Mammography model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load mammography model: {e}")
            raise

    def _extract_finding_type(self, image_path: str) -> str:
        """Extract MASS or CALCIFICATION from filename prefix."""
        fname = os.path.basename(image_path).upper()
        if fname.startswith("MASS"):
            return "MASS"
        elif fname.startswith("CALC"):
            return "CALCIFICATION"
        return "UNKNOWN"

    def predict(self, image_path: str) -> dict:
        """
        Returns:
            {"label": str, "confidence": float, "finding_type": str,
             "needs_validation": True, "explanation": str}
        """
        try:
            finding_type = self._extract_finding_type(image_path)
            image = Image.open(image_path).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs  = torch.softmax(logits, dim=1)
                conf, idx = torch.max(probs, dim=1)
            label = CLASS_NAMES[idx.item()]
            return {
                "label":            label,
                "confidence":       round(conf.item(), 4),
                "finding_type":     finding_type,
                "needs_validation": True,
                "explanation":      EXPLANATION_MAP.get(label, "")
            }
        except Exception as e:
            logger.error(f"Mammography inference error: {e}")
            return {"label": "unknown", "confidence": 0.0, "finding_type": "UNKNOWN",
                    "needs_validation": True, "explanation": str(e)}
```

---

## Agent 3: HAM10000_AGENT

### Folder Structure
```
agents/image_analysis_agent/ham10000_agent/
├── ham10000_inference.py
├── train.py
├── models/
│   ├── ham10000_efficientnet.pth
│   └── label_map.json
```

### Dataset

- **Source**: HAM10000 (Human Against Machine with 10000 training images) / Skin Cancer MNIST
  - Download: `kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection`
  - OR: `kaggle competitions download -c siim-isic-melanoma-classification` (overlapping)
  - Official: [ISIC Archive HAM10000](https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main)
- **Size**: 10,015 dermoscopic images across 7 classes
- **Classes**:

| Abbreviation | Full Name                          |
|--------------|------------------------------------|
| mel          | Melanoma                           |
| nv           | Melanocytic nevi                   |
| bcc          | Basal cell carcinoma               |
| akiec        | Actinic keratoses / Bowen's disease|
| bkl          | Benign keratosis-like lesions      |
| df           | Dermatofibroma                     |
| vasc         | Vascular lesions                   |

### label_map.json
```json
{
  "0": "akiec",
  "1": "bcc",
  "2": "bkl",
  "3": "df",
  "4": "mel",
  "5": "nv",
  "6": "vasc"
}
```

### Data Preparation

HAM10000 comes as two ZIP files of images + a metadata CSV. The CSV maps `image_id` to `dx` (diagnosis abbreviation).

Organize as:
```
data/ham10000/
├── train/
│   ├── mel/ ├── nv/ ├── bcc/ ├── akiec/ ├── bkl/ ├── df/ └── vasc/
├── val/
│   └── (same 7 subdirs)
└── test/
    └── (same 7 subdirs)
```

Use the metadata CSV to copy images into subdirectories. Split: 80% train, 10% val, 10% test — stratified by class.

### Critical Class Imbalance Note

HAM10000 is severely imbalanced:
- `nv` (nevi): ~6,700 images — vastly dominant
- `mel` (melanoma): ~1,113 images
- `vasc`: ~142 images — minority class

You MUST address this:
```python
# Option 1: WeightedRandomSampler (recommended)
from torch.utils.data import WeightedRandomSampler

# Option 2: Oversampling minorities with augmentation
# Option 3: Class-weighted CrossEntropyLoss

# Use WeightedRandomSampler — best practice for this dataset
```

### Preprocessing Constants
```python
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
```

### train.py

Required constants:
```python
DATASET_SOURCE  = "HAM10000 — ISIC Archive (https://www.isic-archive.com/)"
RANDOM_SEED     = 42
CLASS_NAMES     = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
IMAGE_SIZE      = 224
BATCH_SIZE      = 32
EPOCHS          = 35
LR              = 1e-4
CHECKPOINT_PATH = "./agents/image_analysis_agent/ham10000_agent/models/ham10000_efficientnet.pth"
LABEL_MAP_PATH  = "./agents/image_analysis_agent/ham10000_agent/models/label_map.json"
```

**Model:**
```python
model = models.efficientnet_b3(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
```

**Loss:** `nn.CrossEntropyLoss()` with class weights:
```python
class_counts = [count per class in sorted(CLASS_NAMES) order]
class_weights = torch.FloatTensor(1.0 / np.array(class_counts)).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Augmentations (dermoscopy-appropriate):**
```python
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.4),
    A.GaussianBlur(p=0.2),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])
```

**Required metrics:** Per-class AUC (very important for imbalanced medical data), macro AUC, weighted F1, macro F1

### ham10000_inference.py — Exact Class Spec

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
IMAGE_SIZE  = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

EXPLANATION_MAP = {
    'mel':   'Melanoma detected — most dangerous form of skin cancer. Immediate dermatology referral and biopsy required.',
    'nv':    'Melanocytic nevi (common mole) — typically benign. Monitor for ABCDE changes; routine follow-up advised.',
    'bcc':   'Basal cell carcinoma — most common skin cancer. Generally slow-growing; dermatologist evaluation required for treatment.',
    'akiec': 'Actinic keratosis / Bowen\'s disease — precancerous lesion. Dermatologist evaluation recommended for removal.',
    'bkl':   'Benign keratosis-like lesion (seborrheic keratosis or lichen planus) — benign. Monitoring advised.',
    'df':    'Dermatofibroma — benign fibrous nodule. Usually harmless; monitoring recommended.',
    'vasc':  'Vascular lesion (angioma or pyogenic granuloma) — usually benign. Dermatologist evaluation recommended.'
}

class HAM10000Classification:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = self._build_model()
        self._load_weights()
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

    def _build_model(self):
        model = models.efficientnet_b3(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
        return model.to(self.device)

    def _load_weights(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            logger.info(f"HAM10000 model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load HAM10000 model: {e}")
            raise

    def predict(self, image_path: str) -> dict:
        """
        Returns:
            {"label": str, "confidence": float, "needs_validation": True, "explanation": str}
        """
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs  = torch.softmax(logits, dim=1)
                conf, idx = torch.max(probs, dim=1)
            label = CLASS_NAMES[idx.item()]
            return {
                "label":            label,
                "confidence":       round(conf.item(), 4),
                "needs_validation": True,
                "explanation":      EXPLANATION_MAP.get(label, "")
            }
        except Exception as e:
            logger.error(f"HAM10000 inference error: {e}")
            return {"label": "unknown", "confidence": 0.0, "needs_validation": True, "explanation": str(e)}
```

### Standalone Tests

```python
# Fracture
from agents.image_analysis_agent.fracture_agent.fracture_inference import FractureClassification
clf = FractureClassification("./agents/image_analysis_agent/fracture_agent/models/fracture_densenet.pth")
result = clf.predict("./data/fracture/test/Abnormal/WRIST_patient00123_study1_img001.png")
assert "body_part" in result
assert result["needs_validation"] == True
print("Fracture:", result)

# Mammography
from agents.image_analysis_agent.mammography_agent.mammography_inference import MammographyClassification
clf = MammographyClassification("./agents/image_analysis_agent/mammography_agent/models/mammography_efficientnet.pth")
result = clf.predict("./data/mammography/test/MALIGNANT/MASS_patient001_study1.png")
assert "finding_type" in result
assert result["needs_validation"] == True
print("Mammography:", result)

# HAM10000
from agents.image_analysis_agent.ham10000_agent.ham10000_inference import HAM10000Classification
clf = HAM10000Classification("./agents/image_analysis_agent/ham10000_agent/models/ham10000_efficientnet.pth")
result = clf.predict("./data/ham10000/test/mel/some_image.jpg")
assert result["label"] in ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
assert result["needs_validation"] == True
print("HAM10000:", result)
```

---

## Integration Handoff — What You Give Amritesh

```
SRUJAN INTEGRATION KEY DOCUMENT
================================

AGENT 1: FRACTURE_AGENT
------------------------
Config key (model path):     self.fracture_model_path = "./agents/image_analysis_agent/fracture_agent/models/fracture_densenet.pth"
Config validation flag:      "FRACTURE_AGENT": True
Class name:                  FractureClassification
Import statement:            from .fracture_agent.fracture_inference import FractureClassification
ImageAnalysisAgent method:   classify_fracture(self, image_path: str) -> dict
Agent node name:             FRACTURE_AGENT
Modality in image_classifier: MSK X-RAY
Checkpoint path:             agents/image_analysis_agent/fracture_agent/models/fracture_densenet.pth
Checkpoint size:             <fill in MB>
Test Accuracy:               <fill in>
Test AUC:                    <fill in>
Per-body-part breakdown:     <provide table if available>

AGENT 2: MAMMOGRAPHY_AGENT
---------------------------
Config key (model path):     self.mammography_model_path = "./agents/image_analysis_agent/mammography_agent/models/mammography_efficientnet.pth"
Config validation flag:      "MAMMOGRAPHY_AGENT": True
Class name:                  MammographyClassification
Import statement:            from .mammography_agent.mammography_inference import MammographyClassification
ImageAnalysisAgent method:   classify_mammography(self, image_path: str) -> dict
Agent node name:             MAMMOGRAPHY_AGENT
Modality in image_classifier: MAMMOGRAM
Checkpoint path:             agents/image_analysis_agent/mammography_agent/models/mammography_efficientnet.pth
Checkpoint size:             <fill in MB>
Test Accuracy:               <fill in>
Test AUC:                    <fill in>

AGENT 3: HAM10000_AGENT
------------------------
Config key (model path):     self.ham10000_model_path = "./agents/image_analysis_agent/ham10000_agent/models/ham10000_efficientnet.pth"
Config validation flag:      "HAM10000_AGENT": True
Class name:                  HAM10000Classification
Import statement:            from .ham10000_agent.ham10000_inference import HAM10000Classification
ImageAnalysisAgent method:   classify_ham10000(self, image_path: str) -> dict
Agent node name:             HAM10000_AGENT
Modality in image_classifier: SKIN LESION (LLM routes based on query context: classification vs segmentation)
Checkpoint path:             agents/image_analysis_agent/ham10000_agent/models/ham10000_efficientnet.pth
Checkpoint size:             <fill in MB>
Test macro AUC:              <fill in>
Test weighted F1:            <fill in>
Per-class AUC table:         <provide 7 rows>
```

---

## Your Final Report — Mandatory Format

Write `Work/Srujan_Report.md` with the following sections:

### Section 1: Fracture Agent
- Dataset: total images, images per body part, train/val/test split
- Filename renaming strategy for body_part extraction (confirm filenames follow the `<BODYPART>_*` convention)
- Model: DenseNet169, number of parameters
- Class imbalance strategy
- Training: LR, batch size, epochs, CLAHE applied yes/no
- Test metrics: Accuracy, AUC, F1, Precision, Recall
- Per-body-part performance table (if computed): Body Part | Accuracy | AUC
- Checkpoint path and file size

### Section 2: Mammography Agent
- Dataset: CBIS-DDSM ROI crops vs full images (which did you use?), images per class, DICOM conversion approach
- Filename renaming strategy for finding_type extraction
- Class imbalance ratio and how it was handled
- Training: LR, batch size, epochs, augmentation strategy
- Test metrics: Accuracy, AUC, F1, Precision, Recall — separately for MASS and CALCIFICATION subsets if possible
- Checkpoint path and file size

### Section 3: HAM10000 Agent
- Dataset: exact images per class, imbalance ratios
- Class weighting / sampling strategy used
- Training: LR, batch size, epochs, loss function with weights
- Test metrics:
  - Per-class AUC table (7 rows: Class | AUC)
  - Macro AUC, Weighted AUC
  - Per-class F1 table
  - Macro F1, Weighted F1
  - Overall Accuracy
- Checkpoint path and file size

### Section 4: Integration Key Document (completed)
- Fill in all `<fill in>` fields above

### Section 5: Standalone Test Results
- Copy-paste terminal output from all 3 standalone tests

### Section 6: Known Issues / Limitations
- CBIS-DDSM access and DICOM conversion challenges
- HAM10000 class imbalance impact on performance
- MURA multi-image study aggregation note

---

## Completion Test Checklist

**The job is not complete until every checkbox below is ticked. Each item is a pass/fail gate.**

### T1 — Fracture Agent (`FractureClassification`)

- [ ] `FractureClassification("path/to/fracture_densenet.pth")` instantiates without error
- [ ] Model is in `eval()` mode after `__init__` returns (`model.training == False`)
- [ ] `predict("valid_xray.png", "WRIST")` returns a Python `dict`
- [ ] Returned dict contains exactly these keys: `label`, `confidence`, `body_part`, `needs_validation`, `explanation`
- [ ] `result["label"]` is exactly `"Normal"` or `"Abnormal"` (title-case, exact match)
- [ ] `result["confidence"]` is a Python `float` with `0.0 <= result["confidence"] <= 1.0`, rounded to 4 dp
- [ ] `result["body_part"]` is the `body_part` string that was passed as the second argument to `predict()` — it must be echoed through unchanged
- [ ] `result["needs_validation"]` is the boolean `True`
- [ ] `result["explanation"]` is a non-empty string
- [ ] `predict("nonexistent.jpg", "SHOULDER")` does NOT raise — returns error dict with `"confidence": 0.0`
- [ ] `WeightedRandomSampler` is present in `train.py` to handle the 3× class imbalance in MURA
- [ ] MURA data has been flattened from nested `XR_<BODYPART>/patient/study_positive/` into a flat `Normal/` and `Abnormal/` structure before training
- [ ] `models/fracture_densenet.pth` exists on disk
- [ ] `models/label_map.json` exists and equals `{"0": "Normal", "1": "Abnormal"}`
- [ ] `models/metrics.json` exists and contains: `accuracy`, `precision`, `recall`, `f1`, `auc`
- [ ] `metrics.json` AUC >= 0.88 (if below, document the gap)
- [ ] Run standalone test:

  ```python
  from agents.image_analysis_agent.fracture_agent.fracture_inference import FractureClassification
  clf = FractureClassification("./agents/image_analysis_agent/fracture_agent/models/fracture_densenet.pth")
  result = clf.predict("./data/fracture/test/Abnormal/WRIST_patient00001_study1_image1.png", "WRIST")
  assert isinstance(result, dict)
  assert result["label"] in {"Normal", "Abnormal"}
  assert result["body_part"] == "WRIST"
  assert result["needs_validation"] == True
  assert isinstance(result["confidence"], float)
  print(result)
  ```

  **Must print valid dict without AssertionError.**

### T2 — Mammography Agent (`MammographyClassification`)

- [ ] `MammographyClassification("path/to/mammography_efficientnet.pth")` instantiates without error
- [ ] Model is in `eval()` mode after `__init__` returns
- [ ] `predict("valid_mammo.dcm_or_png")` returns a Python `dict`
- [ ] Returned dict contains exactly these keys: `label`, `confidence`, `finding_type`, `needs_validation`, `explanation`
- [ ] `result["label"]` is exactly `"BENIGN"` or `"MALIGNANT"` (uppercase, exact match)
- [ ] `result["confidence"]` is `float`, `0.0 <= x <= 1.0`, rounded to 4 dp
- [ ] `result["finding_type"]` is exactly `"MASS"` or `"CALCIFICATION"` (uppercase, exact match)
- [ ] `result["needs_validation"]` is boolean `True`
- [ ] `result["explanation"]` is a non-empty string
- [ ] `predict("nonexistent.dcm")` does NOT raise — returns error dict
- [ ] DICOM loading via `pydicom` is tested on at least one DICOM file — pixel array normalised to uint8 range `[0, 255]` before PIL conversion
- [ ] `finding_type` is extracted from the image filename inside `predict()` using the pattern: `"MASS" if "Mass" in image_path else "CALCIFICATION"`
- [ ] `models/mammography_efficientnet.pth` exists on disk
- [ ] `models/label_map.json` exists and equals `{"0": "BENIGN", "1": "MALIGNANT"}`
- [ ] `models/metrics.json` exists and contains: `accuracy`, `precision`, `recall`, `f1`, `auc`
- [ ] `metrics.json` AUC >= 0.80 (if below, document the gap)
- [ ] Run standalone test:

  ```python
  from agents.image_analysis_agent.mammography_agent.mammography_inference import MammographyClassification
  clf = MammographyClassification("./agents/image_analysis_agent/mammography_agent/models/mammography_efficientnet.pth")
  result = clf.predict("./data/mammography/test/Mass-Training_P_00001_LEFT_CC.png")
  assert isinstance(result, dict)
  assert result["label"] in {"BENIGN", "MALIGNANT"}
  assert result["finding_type"] in {"MASS", "CALCIFICATION"}
  assert result["needs_validation"] == True
  assert isinstance(result["confidence"], float)
  print(result)
  ```

  **Must print valid dict without AssertionError.**

### T3 — HAM10000 Agent (`HAM10000Classification`)

- [ ] `HAM10000Classification("path/to/ham10000_efficientnet.pth")` instantiates without error
- [ ] Model is in `eval()` mode after `__init__` returns
- [ ] `predict("valid_dermoscopy.jpg")` returns a Python `dict`
- [ ] Returned dict contains exactly these keys: `label`, `confidence`, `needs_validation`, `explanation`
- [ ] `result["label"]` is one of: `"akiec"`, `"bcc"`, `"bkl"`, `"df"`, `"mel"`, `"nv"`, `"vasc"` (lowercase, exact match)
- [ ] `result["confidence"]` is `float`, `0.0 <= x <= 1.0`, rounded to 4 dp
- [ ] `result["needs_validation"]` is boolean `True`
- [ ] `result["explanation"]` includes the full class name (e.g., `"Melanoma"`) and a clinical note — not just the abbreviation
- [ ] `predict("nonexistent.jpg")` does NOT raise — returns error dict
- [ ] BOTH `WeightedRandomSampler` AND `CrossEntropyLoss(weight=class_weights)` are used in `train.py` — verify both are present
- [ ] Class weights are computed as: `n_samples / (n_classes * class_counts)` — verify the formula in `train.py`
- [ ] `models/ham10000_efficientnet.pth` exists on disk
- [ ] `models/label_map.json` exists and has exactly 7 entries
- [ ] `models/metrics.json` exists and contains: `accuracy`, `macro_f1`, `weighted_f1`, per-class `auc` for all 7 classes, `macro_auc`, `weighted_auc`
- [ ] `metrics.json` macro AUC >= 0.85 (if below, document the gap)
- [ ] Run standalone test:

  ```python
  from agents.image_analysis_agent.ham10000_agent.ham10000_inference import HAM10000Classification
  clf = HAM10000Classification("./agents/image_analysis_agent/ham10000_agent/models/ham10000_efficientnet.pth")
  result = clf.predict("./data/ham10000/test/mel/ISIC_0024310.jpg")
  assert isinstance(result, dict)
  assert result["label"] in {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
  assert result["needs_validation"] == True
  assert isinstance(result["confidence"], float)
  print(result)
  ```

  **Must print valid dict without AssertionError.**

### T4 — Integration Readiness (Amritesh's pre-wire checklist)

Before handing off, verify ALL of the following:

- [ ] `agents/image_analysis_agent/fracture_agent/fracture_inference.py` exists
- [ ] `agents/image_analysis_agent/fracture_agent/models/fracture_densenet.pth` exists
- [ ] `agents/image_analysis_agent/fracture_agent/models/label_map.json` exists
- [ ] `agents/image_analysis_agent/fracture_agent/models/metrics.json` exists
- [ ] `agents/image_analysis_agent/mammography_agent/mammography_inference.py` exists
- [ ] `agents/image_analysis_agent/mammography_agent/models/mammography_efficientnet.pth` exists
- [ ] `agents/image_analysis_agent/mammography_agent/models/label_map.json` exists
- [ ] `agents/image_analysis_agent/mammography_agent/models/metrics.json` exists
- [ ] `agents/image_analysis_agent/ham10000_agent/ham10000_inference.py` exists
- [ ] `agents/image_analysis_agent/ham10000_agent/models/ham10000_efficientnet.pth` exists
- [ ] `agents/image_analysis_agent/ham10000_agent/models/label_map.json` exists (7 entries)
- [ ] `agents/image_analysis_agent/ham10000_agent/models/metrics.json` exists
- [ ] No class contains a hardcoded model path — all paths come from `__init__(self, model_path: str)`
- [ ] Class names are exactly: `FractureClassification`, `MammographyClassification`, `HAM10000Classification`
- [ ] `from agents.image_analysis_agent.fracture_agent.fracture_inference import FractureClassification` works
- [ ] `from agents.image_analysis_agent.mammography_agent.mammography_inference import MammographyClassification` works
- [ ] `from agents.image_analysis_agent.ham10000_agent.ham10000_inference import HAM10000Classification` works
- [ ] `IMAGE_SIZE`, `MEAN`, `STD` are defined as module-level constants in all 3 inference files
- [ ] `IMAGE_SIZE` in each inference file matches `IMAGE_SIZE` in the corresponding `train.py`
- Recommendations for improvement

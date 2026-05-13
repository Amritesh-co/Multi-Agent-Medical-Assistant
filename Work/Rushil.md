# Rushil — Ophthalmology Imaging Agents

## Role
You own all retinal and ophthalmic image analysis agents:
1. **DIABETIC_RETINOPATHY_AGENT** — Fundus photograph grading (EfficientNet-B4, 5 grades)
2. **OCT_AGENT** — Optical coherence tomography 4-class classification (EfficientNet-B3)

You deliver working inference wrappers, trained checkpoints, and training scripts to Amritesh for integration. You do NOT modify `config.py`, `__init__.py`, or `agent_decision.py` — Amritesh owns those. You only create the agent folders and the models.

---

## Technology Stack

- Python 3.10+
- PyTorch + torchvision
- `efficientnet_pytorch` or `torchvision.models`
- `albumentations` for augmentations
- `sklearn.metrics` for AUC/classification metrics
- `PIL` / `cv2` for image processing

---

## Strict Output Contracts — Read This First

Every `predict()` method you write must return exactly these shapes.

**Diabetic Retinopathy:**
```python
{
    "grade":            int,    # 0, 1, 2, 3, or 4
    "severity_text":    str,    # "No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"
    "confidence":       float,  # 0.0 to 1.0
    "needs_validation": True,
    "explanation":      str
}
```

**OCT:**
```python
{
    "label":            str,    # "CNV", "DME", "DRUSEN", or "NORMAL"
    "confidence":       float,  # 0.0 to 1.0
    "needs_validation": True,
    "explanation":      str
}
```

Both `predict()` methods take exactly one argument: `image_path: str`

---

## Agent 1: DIABETIC_RETINOPATHY_AGENT

### Folder Structure
```
agents/image_analysis_agent/diabetic_retinopathy_agent/
├── diabetic_retinopathy_inference.py
├── train.py
├── models/
│   ├── diabetic_retinopathy_efficientnet.pth
│   └── label_map.json
```

### Dataset

- **Primary**: APTOS 2019 Blindness Detection (Kaggle)
  - Download: `kaggle competitions download -c aptos2019-blindness-detection`
  - Size: ~3,662 training images, 5 classes (0 through 4)
  - Format: JPEG fundus photographs of varying quality
- **Supplementary** (optional if more data needed): EyePACS
  - Download: `kaggle competitions download -c diabetic-retinopathy-detection`
  - Much larger (~35,000 images) — sample from it if APTOS alone underperforms

### Grade Mapping

| Grade | Severity Text       | Medical Meaning                                      |
|-------|---------------------|------------------------------------------------------|
| 0     | No DR               | No diabetic retinopathy detected                     |
| 1     | Mild DR             | Microaneurysms only                                  |
| 2     | Moderate DR         | More than just microaneurysms but less than severe   |
| 3     | Severe DR           | More than 20 hemorrhages; venous beading; IRMA       |
| 4     | Proliferative DR    | Neovascularization or vitreous/preretinal hemorrhage |

### Data Preparation

Organize as:
```
data/diabetic_retinopathy/
├── train/
│   ├── 0/   ← grade 0 images
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   └── 4/
├── val/
│   ├── 0/ ... 4/
└── test/
    ├── 0/ ... 4/
```

APTOS provides a CSV with `id_code` and `diagnosis` columns. Write a script to copy files into this directory structure.

### label_map.json
```json
{
  "0": "No DR",
  "1": "Mild DR",
  "2": "Moderate DR",
  "3": "Severe DR",
  "4": "Proliferative DR"
}
```

### Critical Preprocessing: Ben Graham Preprocessing

Fundus photographs have uneven illumination and large black borders. You MUST apply:

**Step 1 — Circular crop (remove black borders):**
```python
import cv2
import numpy as np

def crop_fundus(img):
    """Remove black border from fundus image."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return img[y:y+h, x:x+w]
```

**Step 2 — Ben Graham contrast enhancement:**
```python
def ben_graham(img, sigmaX=10):
    """Subtract Gaussian blur for local contrast enhancement."""
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

**Step 3 — Resize to EfficientNet-B4 native size:**
```python
IMAGE_SIZE = 380   # EfficientNet-B4 optimal input
```

Apply preprocessing before your transforms. The order is: load → crop → ben_graham → resize → normalize.

### Preprocessing Constants
```python
IMAGE_SIZE = 380
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
```

### train.py

Required constants at the top:
```python
DATASET_SOURCE  = "https://www.kaggle.com/competitions/aptos2019-blindness-detection"
RANDOM_SEED     = 42
CLASS_NAMES     = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
IMAGE_SIZE      = 380
BATCH_SIZE      = 16   # EfficientNet-B4 is larger
EPOCHS          = 30
LR              = 1e-4
CHECKPOINT_PATH = "./agents/image_analysis_agent/diabetic_retinopathy_agent/models/diabetic_retinopathy_efficientnet.pth"
LABEL_MAP_PATH  = "./agents/image_analysis_agent/diabetic_retinopathy_agent/models/label_map.json"
```

**Model:**
```python
import torchvision.models as models
model = models.efficientnet_b4(weights="IMAGENET1K_V1")
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 5)   # 5-class
```

**Class imbalance:** APTOS is imbalanced (grade 0 dominates). Use:
```python
from torch.utils.data import WeightedRandomSampler
class_counts = [count_of_each_grade_in_train_set]
weights = 1.0 / np.array(class_counts)
sample_weights = [weights[label] for label in all_train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
# Use sampler instead of shuffle=True in DataLoader
```

**Loss:** `nn.CrossEntropyLoss()` (standard) OR `nn.MSELoss()` with float targets for ordinal regression. CrossEntropy is recommended for simplicity.

**Augmentations:**
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])
```

**Required metrics:** Accuracy, Quadratic Weighted Kappa (the competition metric), per-class F1, per-class AUC

**Save checkpoint as:**
```python
torch.save(model.state_dict(), CHECKPOINT_PATH)
```

**Save label map after training, with correct mapping from your dataset's class_to_idx.**

### diabetic_retinopathy_inference.py — Exact Class Spec

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

CLASS_NAMES  = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
IMAGE_SIZE   = 380
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

EXPLANATION_MAP = {
    0: "No diabetic retinopathy detected. Continue routine annual screening.",
    1: "Mild diabetic retinopathy (microaneurysms). Recommend follow-up in 12 months.",
    2: "Moderate diabetic retinopathy. Recommend follow-up in 6–12 months with ophthalmologist.",
    3: "Severe diabetic retinopathy. Urgent ophthalmology referral recommended within 1 month.",
    4: "Proliferative diabetic retinopathy. Urgent specialist intervention required — risk of severe vision loss."
}

def _crop_fundus(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_rgb
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return img_rgb[y:y+h, x:x+w]

def _ben_graham(img_rgb, sigmaX=10):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.addWeighted(img_bgr, 4, cv2.GaussianBlur(img_bgr, (0, 0), sigmaX), -4, 128)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


class DiabeticRetinopathyClassification:
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
        model = models.efficientnet_b4(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
        return model.to(self.device)

    def _load_weights(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            logger.info(f"DR model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load DR model: {e}")
            raise

    def predict(self, image_path: str) -> dict:
        """
        Returns:
            {"grade": int, "severity_text": str, "confidence": float,
             "needs_validation": True, "explanation": str}
        """
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = _crop_fundus(img_rgb)
            img_rgb = _ben_graham(img_rgb)
            pil_img = Image.fromarray(img_rgb)
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs  = torch.softmax(logits, dim=1)
                conf, idx = torch.max(probs, dim=1)
            grade = idx.item()
            return {
                "grade":            grade,
                "severity_text":    CLASS_NAMES[grade],
                "confidence":       round(conf.item(), 4),
                "needs_validation": True,
                "explanation":      EXPLANATION_MAP.get(grade, "")
            }
        except Exception as e:
            logger.error(f"DR inference error: {e}")
            return {"grade": -1, "severity_text": "unknown", "confidence": 0.0,
                    "needs_validation": True, "explanation": str(e)}
```

### Standalone Test for DR
```python
from agents.image_analysis_agent.diabetic_retinopathy_agent.diabetic_retinopathy_inference import DiabeticRetinopathyClassification
clf = DiabeticRetinopathyClassification("./agents/image_analysis_agent/diabetic_retinopathy_agent/models/diabetic_retinopathy_efficientnet.pth")
result = clf.predict("./data/diabetic_retinopathy/test/2/some_image.png")
assert isinstance(result, dict)
assert "grade" in result
assert result["needs_validation"] == True
print(result)
```

---

## Agent 2: OCT_AGENT

### Folder Structure
```
agents/image_analysis_agent/oct_agent/
├── oct_inference.py
├── train.py
├── models/
│   ├── oct_efficientnet.pth
│   └── label_map.json
```

### Dataset

- **Source**: Kermany 2018 OCT dataset (Kaggle)
  - Download: `kaggle datasets download -d paultimothymooney/chest-xray-pneumonia` — NO, wrong one.
  - Correct: search "OCT Kermany 2018" on Kaggle OR use: `kaggle datasets download -d paultimothymooney/kermany2018`
  - OR direct: [Mendeley Data — Labeled Optical Coherence Tomography](https://data.mendeley.com/datasets/rscbjbr9sj/3)
- **Classes**: CNV, DME, DRUSEN, NORMAL
- **Size**: ~84,484 training images (train) + 968 test images across 4 classes

### Class Meanings

| Class  | Full Name                              | Notes                                            |
|--------|----------------------------------------|--------------------------------------------------|
| CNV    | Choroidal Neovascularization           | Abnormal blood vessel growth; sign of wet AMD    |
| DME    | Diabetic Macular Edema                 | Fluid accumulation in macula due to diabetes     |
| DRUSEN | Drusen deposits                        | Yellow deposits; early sign of AMD               |
| NORMAL | Normal OCT                             | No pathological findings                         |

### Data Preparation

The Kermany dataset already comes in train/val/test splits with class subdirectories:
```
data/oct/
├── train/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
├── val/
│   ├── CNV/ ... NORMAL/
└── test/
    ├── CNV/ ... NORMAL/
```

Preserve the official splits exactly.

### label_map.json
```json
{"0": "CNV", "1": "DME", "2": "DRUSEN", "3": "NORMAL"}
```

### Preprocessing Constants
```python
IMAGE_SIZE = 300    # EfficientNet-B3 native
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
```

OCT scans are grayscale — convert to RGB for the model:
```python
image = Image.open(path).convert("RGB")   # replicate channel 3x
```

### train.py

Required constants:
```python
DATASET_SOURCE  = "https://data.mendeley.com/datasets/rscbjbr9sj/3 (Kermany 2018 OCT)"
RANDOM_SEED     = 42
CLASS_NAMES     = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
IMAGE_SIZE      = 300
BATCH_SIZE      = 32
EPOCHS          = 25
LR              = 1e-4
CHECKPOINT_PATH = "./agents/image_analysis_agent/oct_agent/models/oct_efficientnet.pth"
LABEL_MAP_PATH  = "./agents/image_analysis_agent/oct_agent/models/label_map.json"
```

**Model:**
```python
model = models.efficientnet_b3(weights="IMAGENET1K_V1")
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 4)   # 4-class
```

**Class imbalance:** NORMAL class is much larger in Kermany. Use WeightedRandomSampler (same approach as DR agent above).

**Loss:** `nn.CrossEntropyLoss()`

**Augmentations:**
```python
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.2),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])
```

**Required metrics:** Accuracy, Precision (macro), Recall (macro), F1 (macro), AUC (one-vs-rest per class)

**Save checkpoint:** `torch.save(model.state_dict(), CHECKPOINT_PATH)`

### oct_inference.py — Exact Class Spec

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)

CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
IMAGE_SIZE  = 300
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

EXPLANATION_MAP = {
    'CNV':    'Choroidal neovascularization detected — abnormal blood vessel growth consistent with wet AMD. Urgent retina specialist referral recommended.',
    'DME':    'Diabetic macular edema detected — fluid accumulation in the macula. Ophthalmologist evaluation required; anti-VEGF treatment may be indicated.',
    'DRUSEN': 'Drusen deposits detected in the macula — early sign of age-related macular degeneration (AMD). Monitoring and lifestyle modification advised.',
    'NORMAL': 'No pathological findings detected in this OCT scan. Retinal layers appear intact.'
}

class OCTClassification:
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
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
        return model.to(self.device)

    def _load_weights(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            logger.info(f"OCT model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load OCT model: {e}")
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
            logger.error(f"OCT inference error: {e}")
            return {"label": "unknown", "confidence": 0.0, "needs_validation": True, "explanation": str(e)}
```

### Standalone Test for OCT
```python
from agents.image_analysis_agent.oct_agent.oct_inference import OCTClassification
clf = OCTClassification("./agents/image_analysis_agent/oct_agent/models/oct_efficientnet.pth")
result = clf.predict("./data/oct/test/CNV/some_image.jpeg")
assert isinstance(result, dict)
assert result["needs_validation"] == True
assert result["label"] in ["CNV", "DME", "DRUSEN", "NORMAL"]
print(result)
```

---

## Integration Handoff — What You Give Amritesh

```
RUSHIL INTEGRATION KEY DOCUMENT
================================

AGENT 1: DIABETIC_RETINOPATHY_AGENT
-------------------------------------
Config key (model path):     self.diabetic_retinopathy_model_path = "./agents/image_analysis_agent/diabetic_retinopathy_agent/models/diabetic_retinopathy_efficientnet.pth"
Config validation flag:      "DIABETIC_RETINOPATHY_AGENT": True
Class name:                  DiabeticRetinopathyClassification
Import statement:            from .diabetic_retinopathy_agent.diabetic_retinopathy_inference import DiabeticRetinopathyClassification
ImageAnalysisAgent method:   classify_diabetic_retinopathy(self, image_path: str) -> dict
Agent node name:             DIABETIC_RETINOPATHY_AGENT
Modality in image_classifier: FUNDUS RETINA
Checkpoint path:             agents/image_analysis_agent/diabetic_retinopathy_agent/models/diabetic_retinopathy_efficientnet.pth
Checkpoint size:             <fill in MB>
Test Accuracy:               <fill in>
Quadratic Weighted Kappa:    <fill in>
Mean AUC:                    <fill in>

AGENT 2: OCT_AGENT
-------------------
Config key (model path):     self.oct_model_path = "./agents/image_analysis_agent/oct_agent/models/oct_efficientnet.pth"
Config validation flag:      "OCT_AGENT": True
Class name:                  OCTClassification
Import statement:            from .oct_agent.oct_inference import OCTClassification
ImageAnalysisAgent method:   classify_oct(self, image_path: str) -> dict
Agent node name:             OCT_AGENT
Modality in image_classifier: OCT
Checkpoint path:             agents/image_analysis_agent/oct_agent/models/oct_efficientnet.pth
Checkpoint size:             <fill in MB>
Test Accuracy:               <fill in>
Test F1 (macro):             <fill in>
Test AUC (mean):             <fill in>
```

---

## Your Final Report — Mandatory Format

After completing both agents, write `Work/Rushil_Report.md` with the following sections:

### Section 1: Diabetic Retinopathy Agent
- Dataset: images per grade (0–4), total size, train/val/test split sizes
- Preprocessing: describe crop_fundus and ben_graham application (was it applied consistently in training and inference?)
- Model: EfficientNet-B4, number of parameters, pretrained weights source
- Class imbalance strategy: describe sampler configuration
- Training: LR, batch size, epochs, loss function used, optimizer, scheduler
- Metrics on test set:
  - Per-class Accuracy
  - Per-class F1
  - Per-class AUC
  - Quadratic Weighted Kappa (main metric for APTOS)
  - Overall Accuracy
- Checkpoint path and file size
- Deviations from this document (if any)

### Section 2: OCT Agent
- Dataset: images per class (CNV, DME, DRUSEN, NORMAL), train/val/test split used
- Preprocessing: grayscale-to-RGB conversion confirmed
- Model: EfficientNet-B3, pretrained weights source
- Training: LR, batch size, epochs, loss, optimizer
- Metrics on test set:
  - Per-class Accuracy
  - Per-class F1
  - Per-class AUC
  - Overall Accuracy (macro F1)
- Checkpoint path and file size

### Section 3: Integration Key Document (completed)
- Fill in all `<fill in>` values above

### Section 4: Standalone Test Results
- For each agent: copy-paste of the terminal output from the standalone test script showing the exact dict returned

### Section 5: Preprocessing Verification
- Confirm that the preprocessing applied in `train.py` EXACTLY matches what is in `predict()` in the inference file
- If there is any discrepancy, list it explicitly

### Section 6: Known Issues / Limitations
- Note any dataset download issues
- Note any class imbalance observations

---

## Completion Test Checklist

**The job is not complete until every checkbox below is ticked. Each item is a pass/fail gate.**

### T1 — Diabetic Retinopathy Agent (`DiabeticRetinopathyClassification`)

- [ ] `DiabeticRetinopathyClassification("path/to/diabetic_retinopathy_efficientnet.pth")` instantiates without error
- [ ] Model is in `eval()` mode after `__init__` returns (`model.training == False`)
- [ ] `predict("valid_fundus.jpg")` returns a Python `dict`
- [ ] Returned dict contains exactly these keys: `grade`, `severity_text`, `confidence`, `needs_validation`, `explanation`
- [ ] `result["grade"]` is a Python `int` with value in `{0, 1, 2, 3, 4}` (not a string, not a float)
- [ ] `result["severity_text"]` is one of: `"No DR"`, `"Mild DR"`, `"Moderate DR"`, `"Severe DR"`, `"Proliferative DR"`
- [ ] `result["severity_text"]` matches `SEVERITY_MAP[result["grade"]]` exactly
- [ ] `result["confidence"]` is a Python `float` with `0.0 <= result["confidence"] <= 1.0`, rounded to 4 dp
- [ ] `result["needs_validation"]` is the boolean `True`
- [ ] `result["explanation"]` is a non-empty string
- [ ] `predict("nonexistent.jpg")` does NOT raise — returns error dict with `"confidence": 0.0`
- [ ] Ben Graham preprocessing `ben_graham_preprocess()` is defined and called in BOTH `train.py` AND `predict()` — verify this is identical in both places
- [ ] `WeightedRandomSampler` is present in `train.py` DataLoader for handling class imbalance
- [ ] `models/diabetic_retinopathy_efficientnet.pth` exists on disk
- [ ] `models/label_map.json` exists and has exactly 5 entries mapping `"0"` through `"4"` to severity strings
- [ ] `models/metrics.json` exists and contains: `accuracy`, `precision`, `recall`, `f1`, `quadratic_weighted_kappa`
- [ ] `metrics.json` Quadratic Weighted Kappa (QWK) >= 0.80 (if below, document the gap)
- [ ] Run standalone test:

  ```python
  from agents.image_analysis_agent.diabetic_retinopathy_agent.diabetic_retinopathy_inference import DiabeticRetinopathyClassification
  clf = DiabeticRetinopathyClassification("./agents/image_analysis_agent/diabetic_retinopathy_agent/models/diabetic_retinopathy_efficientnet.pth")
  result = clf.predict("./data/diabetic_retinopathy/test/0/some_fundus.jpeg")
  assert isinstance(result, dict)
  assert isinstance(result["grade"], int)
  assert result["grade"] in {0, 1, 2, 3, 4}
  assert result["needs_validation"] == True
  assert isinstance(result["confidence"], float)
  print(result)
  ```

  **Must print valid dict without AssertionError.**

- [ ] Preprocessing consistency check: apply `ben_graham_preprocess` to the same image in training mode and inference mode, then compare the resulting numpy arrays — they must be pixel-identical

### T2 — OCT Agent (`OCTClassification`)

- [ ] `OCTClassification("path/to/oct_efficientnet.pth")` instantiates without error
- [ ] Model is in `eval()` mode after `__init__` returns
- [ ] `predict("valid_oct.jpeg")` returns a Python `dict`
- [ ] Returned dict contains exactly these keys: `label`, `confidence`, `needs_validation`, `explanation`
- [ ] `result["label"]` is one of: `"CNV"`, `"DME"`, `"DRUSEN"`, `"NORMAL"` (uppercase, exact match)
- [ ] `result["confidence"]` is `float`, `0.0 <= x <= 1.0`, rounded to 4 dp
- [ ] `result["needs_validation"]` is boolean `True`
- [ ] `result["explanation"]` is a non-empty string
- [ ] `predict("nonexistent.jpg")` does NOT raise — returns error dict
- [ ] Grayscale OCT images are converted to 3-channel RGB before inference (verify `Image.open(...).convert("RGB")` or equivalent)
- [ ] `models/oct_efficientnet.pth` exists on disk
- [ ] `models/label_map.json` exists and equals `{"0": "CNV", "1": "DME", "2": "DRUSEN", "3": "NORMAL"}`
- [ ] `models/metrics.json` exists and contains: `accuracy`, per-class `f1`, per-class `auc`, `macro_f1`
- [ ] `metrics.json` overall accuracy >= 90% (Kermany dataset is high-quality; if below, investigate)
- [ ] Run standalone test:

  ```python
  from agents.image_analysis_agent.oct_agent.oct_inference import OCTClassification
  clf = OCTClassification("./agents/image_analysis_agent/oct_agent/models/oct_efficientnet.pth")
  result = clf.predict("./data/oct/test/CNV/some_image.jpeg")
  assert isinstance(result, dict)
  assert result["label"] in {"CNV", "DME", "DRUSEN", "NORMAL"}
  assert result["needs_validation"] == True
  assert isinstance(result["confidence"], float)
  print(result)
  ```

  **Must print valid dict without AssertionError.**

### T3 — Integration Readiness (Amritesh's pre-wire checklist)

Before handing off, verify ALL of the following:

- [ ] `agents/image_analysis_agent/diabetic_retinopathy_agent/diabetic_retinopathy_inference.py` exists
- [ ] `agents/image_analysis_agent/diabetic_retinopathy_agent/models/diabetic_retinopathy_efficientnet.pth` exists
- [ ] `agents/image_analysis_agent/diabetic_retinopathy_agent/models/label_map.json` exists
- [ ] `agents/image_analysis_agent/diabetic_retinopathy_agent/models/metrics.json` exists
- [ ] `agents/image_analysis_agent/oct_agent/oct_inference.py` exists
- [ ] `agents/image_analysis_agent/oct_agent/models/oct_efficientnet.pth` exists
- [ ] `agents/image_analysis_agent/oct_agent/models/label_map.json` exists
- [ ] `agents/image_analysis_agent/oct_agent/models/metrics.json` exists
- [ ] No class contains a hardcoded model path — all paths come from `__init__(self, model_path: str)`
- [ ] Class names are exactly: `DiabeticRetinopathyClassification`, `OCTClassification`
- [ ] `from agents.image_analysis_agent.diabetic_retinopathy_agent.diabetic_retinopathy_inference import DiabeticRetinopathyClassification` works at project root
- [ ] `from agents.image_analysis_agent.oct_agent.oct_inference import OCTClassification` works at project root
- [ ] `IMAGE_SIZE`, `MEAN`, `STD` are defined as module-level constants in BOTH inference files
- [ ] The `IMAGE_SIZE` used in `predict()` transform matches the `IMAGE_SIZE` used in `train.py` transform (300 for EfficientNet-B3, 380 for EfficientNet-B4)
- Recommendations for further improvement

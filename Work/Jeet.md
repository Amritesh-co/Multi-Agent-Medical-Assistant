# Jeet — Pathology, GI & Wound Imaging Agents

## Role
You own three distinct imaging modality agents:
1. **HISTOPATHOLOGY_AGENT** — H&E stained tissue patch classification (EfficientNet, 9 classes, NCT-CRC-HE-100K)
2. **COLON_POLYP_AGENT** — Colonoscopy polyp segmentation (U-Net, Kvasir-SEG)
3. **DIABETIC_FOOT_ULCER_AGENT** — Wound classification (EfficientNet, 4 classes)

You deliver working inference wrappers, trained checkpoints, and training scripts to Amritesh for integration. You do NOT modify `config.py`, `__init__.py`, or `agent_decision.py` — Amritesh owns those.

---

## Technology Stack

- Python 3.10+
- PyTorch + torchvision
- `segmentation_models_pytorch` (for U-Net — Colon Polyp)
- `albumentations` for augmentations
- `staintools` or `torchstain` for stain normalization (Histopathology)
- `sklearn.metrics` for evaluation

Install if not present:
```bash
pip install segmentation-models-pytorch albumentations torchstain
```

---

## Strict Output Contracts — Read This First

**Histopathology:**
```python
{
    "label":            str,    # tissue class name
    "confidence":       float,
    "needs_validation": True,
    "explanation":      str
}
```

**Colon Polyp (segmentation):**
```python
{
    "label":            str,    # "polyp_detected" or "no_polyp"
    "confidence":       float,
    "mask_path":        str,    # path to saved binary mask PNG
    "overlay_path":     str,    # path to overlay PNG
    "needs_validation": True,
    "explanation":      str
}
```

The colon polyp `predict()` signature is:
```python
def predict(self, image_path: str, output_path: str) -> dict:
```

**Diabetic Foot Ulcer:**
```python
{
    "label":            str,    # "Ulcer", "Infection", "Normal", "Gangrene"
    "confidence":       float,
    "needs_validation": True,
    "explanation":      str
}
```

All `predict()` for classification take only `image_path: str`.

---

## Agent 1: HISTOPATHOLOGY_AGENT

### Folder Structure
```
agents/image_analysis_agent/histopathology_agent/
├── histopathology_inference.py
├── train.py
├── models/
│   ├── histopathology_efficientnet.pth
│   └── label_map.json
```

### Dataset

- **Source**: NCT-CRC-HE-100K (Colorectal Cancer Histology)
  - Download: `kaggle datasets download -d kmader/colorectal-histology-mnist` (Kather Colon dataset, same data)
  - OR direct: [Zenodo NCT-CRC-HE-100K](https://zenodo.org/record/1214456)
- **Size**: 100,000 non-overlapping image patches of human colorectal cancer histology, 224×224 pixels, H&E stained
- **Classes (9)**:

| Abbrev | Full Name               |
|--------|-------------------------|
| ADI    | Adipose                 |
| BACK   | Background              |
| DEB    | Debris                  |
| LYM    | Lymphocytes             |
| MUC    | Mucus                   |
| MUS    | Smooth muscle           |
| NORM   | Normal colon mucosa     |
| STR    | Cancer-associated stroma|
| TUM    | Colorectal adenocarcinoma epithelium |

### Data Preparation

NCT-CRC-HE-100K patches are already 224×224. Organize as:
```
data/histopathology/
├── train/
│   ├── ADI/ ├── BACK/ ├── DEB/ ├── LYM/ ├── MUC/
│   ├── MUS/ ├── NORM/ ├── STR/ └── TUM/
├── val/
│   └── (same 9 subdirs)
└── test/
    └── (same 9 subdirs)
```

Split: 80% train, 10% val, 10% test. Use stratified split.

### label_map.json
```json
{
  "0": "ADI",
  "1": "BACK",
  "2": "DEB",
  "3": "LYM",
  "4": "MUC",
  "5": "MUS",
  "6": "NORM",
  "7": "STR",
  "8": "TUM"
}
```

### Critical Preprocessing: Stain Normalization

H&E images from different labs have different color profiles. Apply Macenko stain normalization at inference time:

```python
from torchstain.normalizers import MacenkoNormalizer
import torch

normalizer = MacenkoNormalizer(backend='torch')
# Fit on a representative 'target' image from the dataset (one clean patch)
# target = PIL.Image.open("path/to/reference_patch.jpg")
# normalizer.fit(T.ToTensor()(target).unsqueeze(0) * 255)
```

Note: stain normalization can be applied in `train.py` as part of preprocessing, but it is OPTIONAL for inference since the model should generalize. However, you MUST document whether you applied it during training. If you applied it during training, you MUST apply it during inference in `predict()`.

**Simple alternative (recommended for consistency):** Skip stain normalization, rely on heavy augmentation with ColorJitter during training. Document your choice.

### Preprocessing Constants
```python
IMAGE_SIZE = 224    # patches are already 224x224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
```

### train.py

Required constants:
```python
DATASET_SOURCE  = "https://zenodo.org/record/1214456 (NCT-CRC-HE-100K)"
RANDOM_SEED     = 42
CLASS_NAMES     = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
IMAGE_SIZE      = 224
BATCH_SIZE      = 64   # patches are small, can use larger batch
EPOCHS          = 25
LR              = 1e-4
CHECKPOINT_PATH = "./agents/image_analysis_agent/histopathology_agent/models/histopathology_efficientnet.pth"
LABEL_MAP_PATH  = "./agents/image_analysis_agent/histopathology_agent/models/label_map.json"
```

**Model:**
```python
import torchvision.models as models
model = models.efficientnet_b3(weights="IMAGENET1K_V1")
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 9)   # 9-class
```

**Loss:** `nn.CrossEntropyLoss()` — NCT-CRC is relatively balanced

**Augmentations:**
```python
import albumentations as A
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),  # important for H&E
    A.GaussianBlur(p=0.2),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])
```

**Required metrics:** Accuracy, per-class Precision, per-class Recall, per-class F1, macro F1, AUC (one-vs-rest)

### histopathology_inference.py — Exact Class Spec

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)

CLASS_NAMES = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
IMAGE_SIZE  = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

EXPLANATION_MAP = {
    'ADI':  'Adipose tissue detected. Normal fat cells — no malignancy indicators.',
    'BACK': 'Background or glass region — no tissue in this patch.',
    'DEB':  'Cellular debris detected. Necrotic material — may indicate tissue death.',
    'LYM':  'Lymphocyte-rich region. Dense immune cell infiltration — may indicate inflammatory response.',
    'MUC':  'Mucus region detected.',
    'MUS':  'Smooth muscle tissue detected.',
    'NORM': 'Normal colon mucosa. No malignancy detected in this patch.',
    'STR':  'Cancer-associated stroma detected. Desmoplastic stromal reaction — pathologist review recommended.',
    'TUM':  'Colorectal adenocarcinoma epithelium detected. Malignant tissue — urgent pathologist confirmation required.'
}

class HistopathologyClassification:
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
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 9)
        return model.to(self.device)

    def _load_weights(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            logger.info(f"Histopathology model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load histopathology model: {e}")
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
            logger.error(f"Histopathology inference error: {e}")
            return {"label": "unknown", "confidence": 0.0, "needs_validation": True, "explanation": str(e)}
```

---

## Agent 2: COLON_POLYP_AGENT

### Folder Structure
```
agents/image_analysis_agent/colon_polyp_agent/
├── colon_polyp_inference.py
├── train.py
├── models/
│   └── colon_polyp_unet.pth
```

### Dataset

- **Primary**: Kvasir-SEG
  - Download: [Kvasir-SEG on Simula](https://datasets.simula.no/kvasir-seg/) — direct download available
  - Size: 1,000 colonoscopy images with corresponding polyp segmentation masks
  - Format: JPEG images + PNG masks (polyp region = white/255, background = black/0)
- **Supplementary**: CVC-ClinicDB
  - Download: [CVC-ClinicDB on Kaggle](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)
  - Size: 612 images with masks
- **Combined**: ~1,612 polyp images — augmentation is critical with this small dataset

### Data Preparation

```
data/colon_polyp/
├── train/
│   ├── images/     ← JPEG colonoscopy frames
│   └── masks/      ← PNG binary masks (255=polyp, 0=background)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Merge Kvasir-SEG and CVC-ClinicDB into combined train/val/test split:
- 80% train, 10% val, 10% test — stratify by dataset source

### Preprocessing Constants
```python
IMAGE_SIZE = 352    # standard for polyp segmentation (divisible by many encoder strides)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
```

### train.py

Required constants:
```python
DATASET_SOURCE  = "Kvasir-SEG (https://datasets.simula.no/kvasir-seg/) + CVC-ClinicDB (Kaggle)"
RANDOM_SEED     = 42
IMAGE_SIZE      = 352
BATCH_SIZE      = 8
EPOCHS          = 50   # small dataset — more epochs
LR              = 1e-4
CHECKPOINT_PATH = "./agents/image_analysis_agent/colon_polyp_agent/models/colon_polyp_unet.pth"
```

**Model:**
```python
import segmentation_models_pytorch as smp
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
)
```

**Loss:** Dice + BCE combination (same as Pneumothorax agent):
```python
from segmentation_models_pytorch.losses import DiceLoss
bce_loss  = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss(mode='binary', from_logits=True)
loss = 0.5 * bce_loss(pred, mask) + 0.5 * dice_loss(pred, mask)
```

**Augmentations (heavy — small dataset):**
```python
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, p=0.5),
    A.ElasticTransform(p=0.3),
    A.GridDistortion(p=0.2),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
], additional_targets={'mask': 'mask'})   # <-- CRITICAL: apply same transform to mask
```

Note the `additional_targets` — when you apply augmentations, you must apply them to BOTH image and mask simultaneously (same random transform):
```python
augmented = train_transform(image=np.array(img), mask=np.array(mask))
img_t  = augmented["image"]
mask_t = augmented["mask"]
```

**Required metrics:** Dice coefficient, IoU, pixel precision, pixel recall on test set

### colon_polyp_inference.py — Exact Class Spec

```python
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import logging
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)

IMAGE_SIZE = 352
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

class ColonPolypSegmentation:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = self._build_model()
        self._load_weights()
        self.model.eval()

    def _build_model(self):
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None
        )
        return model.to(self.device)

    def _load_weights(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            logger.info(f"Colon polyp model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load colon polyp model: {e}")
            raise

    def predict(self, image_path: str, output_path: str) -> dict:
        """
        Returns:
            {
                "label": str, "confidence": float,
                "mask_path": str, "overlay_path": str,
                "needs_validation": True, "explanation": str
            }
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
            for i, (m, s) in enumerate(zip(MEAN, STD)):
                img_resized[:, :, i] = (img_resized[:, :, i] - m) / s
            tensor = torch.tensor(img_resized).permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = torch.sigmoid(self.model(tensor)).squeeze().cpu().numpy()

            mask = (pred > 0.5).astype(np.uint8)
            confidence = float(np.mean(pred))
            mask_resized = cv2.resize(mask, (w, h))

            # Save binary mask
            mask_path = output_path.replace(".png", "_mask.png")
            cv2.imwrite(mask_path, mask_resized * 255)

            # Save overlay (green highlight for polyp)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.axis("off")
            ax.imshow(img_rgb)
            overlay = np.zeros_like(img_rgb)
            overlay[mask_resized == 1] = [0, 255, 0]
            ax.imshow(overlay, alpha=0.4)
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()

            label = "polyp_detected" if mask.sum() > 50 else "no_polyp"
            explanation = (
                "Colon polyp segmented and highlighted in green overlay. Gastroenterology review required."
                if label == "polyp_detected"
                else "No colon polyp detected in this colonoscopy frame."
            )
            return {
                "label":            label,
                "confidence":       round(confidence, 4),
                "mask_path":        mask_path,
                "overlay_path":     output_path,
                "needs_validation": True,
                "explanation":      explanation
            }
        except Exception as e:
            logger.error(f"Colon polyp inference error: {e}")
            return {"label": "unknown", "confidence": 0.0, "mask_path": None, "overlay_path": None,
                    "needs_validation": True, "explanation": str(e)}
```

---

## Agent 3: DIABETIC_FOOT_ULCER_AGENT

### Folder Structure
```
agents/image_analysis_agent/diabetic_foot_ulcer_agent/
├── diabetic_foot_ulcer_inference.py
├── train.py
├── models/
│   ├── dfu_efficientnet.pth
│   └── label_map.json
```

### Dataset

- **Primary**: DFU (Diabetic Foot Ulcer) Kaggle dataset
  - Search Kaggle: "Diabetic Foot Ulcer Classification"
  - OR: [DFUC2021 Grand Challenge](https://dfu2021.grand-challenge.org/) — Detection + Classification dataset
  - OR: Search "wound classification dataset" on Kaggle for related datasets
- **Classes (4)**: Ulcer, Infection, Normal, Gangrene
- **Note**: DFU datasets vary in class definitions. Use whichever dataset gives you these 4 classes or remap classes from a broader wound classification dataset.
- **Alternative if DFUC2021 not available**: Use the AZH Wound Care Center dataset from Kaggle which has wound classification labels.

### Data Preparation

```
data/diabetic_foot_ulcer/
├── train/
│   ├── Ulcer/
│   ├── Infection/
│   ├── Normal/
│   └── Gangrene/
├── val/
│   └── (same 4 subdirs)
└── test/
    └── (same 4 subdirs)
```

### label_map.json
```json
{"0": "Ulcer", "1": "Infection", "2": "Normal", "3": "Gangrene"}
```

### Critical Preprocessing Note

Wound images are clinical photographs — very high color variation, different lighting, different zoom levels. Heavy color augmentation is essential.

### Preprocessing Constants
```python
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
```

### train.py

Required constants:
```python
DATASET_SOURCE  = "DFUC2021 / DFU Kaggle dataset (diabetic foot ulcer classification)"
RANDOM_SEED     = 42
CLASS_NAMES     = ['Ulcer', 'Infection', 'Normal', 'Gangrene']
IMAGE_SIZE      = 224
BATCH_SIZE      = 32
EPOCHS          = 35
LR              = 1e-4
CHECKPOINT_PATH = "./agents/image_analysis_agent/diabetic_foot_ulcer_agent/models/dfu_efficientnet.pth"
LABEL_MAP_PATH  = "./agents/image_analysis_agent/diabetic_foot_ulcer_agent/models/label_map.json"
```

**Model:**
```python
model = models.efficientnet_b3(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
```

**Class imbalance:** Use WeightedRandomSampler (Normal class is typically overrepresented)

**Loss:** `nn.CrossEntropyLoss()` with class weights

**Augmentations (heavy color augmentation):**
```python
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.8),
    A.RandomShadow(p=0.3),
    A.GaussianBlur(p=0.2),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])
```

**Required metrics:** Accuracy, per-class F1, macro F1, AUC

### diabetic_foot_ulcer_inference.py — Exact Class Spec

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)

CLASS_NAMES = ['Ulcer', 'Infection', 'Normal', 'Gangrene']
IMAGE_SIZE  = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

EXPLANATION_MAP = {
    'Ulcer':     'Open diabetic foot ulcer detected. Wound care protocol should be initiated. Podiatry or vascular surgery referral recommended.',
    'Infection': 'Signs of wound infection detected — erythema, pus, or necrotic tissue. Antibiotic therapy and urgent diabetic care team review required.',
    'Normal':    'No wound or ulceration detected in this foot image.',
    'Gangrene':  'Gangrene detected — tissue death with blackened/necrotic appearance. Emergency surgical evaluation required immediately.'
}

class DiabeticFootUlcerClassification:
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
            logger.info(f"DFU model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load DFU model: {e}")
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
            logger.error(f"DFU inference error: {e}")
            return {"label": "unknown", "confidence": 0.0, "needs_validation": True, "explanation": str(e)}
```

### Standalone Tests

```python
# Histopathology
from agents.image_analysis_agent.histopathology_agent.histopathology_inference import HistopathologyClassification
clf = HistopathologyClassification("./agents/image_analysis_agent/histopathology_agent/models/histopathology_efficientnet.pth")
result = clf.predict("./data/histopathology/test/TUM/some_patch.tif")
assert result["needs_validation"] == True
print("Histo:", result)

# Colon Polyp
from agents.image_analysis_agent.colon_polyp_agent.colon_polyp_inference import ColonPolypSegmentation
seg = ColonPolypSegmentation("./agents/image_analysis_agent/colon_polyp_agent/models/colon_polyp_unet.pth")
result = seg.predict("./data/colon_polyp/test/images/some_frame.jpg", "./uploads/colon_polyp_output/segmentation_plot.png")
assert result["overlay_path"] is not None
print("Polyp:", result)

# DFU
from agents.image_analysis_agent.diabetic_foot_ulcer_agent.diabetic_foot_ulcer_inference import DiabeticFootUlcerClassification
clf = DiabeticFootUlcerClassification("./agents/image_analysis_agent/diabetic_foot_ulcer_agent/models/dfu_efficientnet.pth")
result = clf.predict("./data/diabetic_foot_ulcer/test/Ulcer/some_image.jpg")
assert result["needs_validation"] == True
print("DFU:", result)
```

---

## Integration Handoff — What You Give Amritesh

```
JEET INTEGRATION KEY DOCUMENT
==============================

AGENT 1: HISTOPATHOLOGY_AGENT
------------------------------
Config key (model path):     self.histopathology_model_path = "./agents/image_analysis_agent/histopathology_agent/models/histopathology_efficientnet.pth"
Config validation flag:      "HISTOPATHOLOGY_AGENT": True
Class name:                  HistopathologyClassification
Import statement:            from .histopathology_agent.histopathology_inference import HistopathologyClassification
ImageAnalysisAgent method:   classify_histopathology(self, image_path: str) -> dict
Agent node name:             HISTOPATHOLOGY_AGENT
Modality in image_classifier: HISTOPATHOLOGY PATCH
Checkpoint path:             agents/image_analysis_agent/histopathology_agent/models/histopathology_efficientnet.pth
Checkpoint size:             <fill in MB>
Test Accuracy:               <fill in>
Test macro F1:               <fill in>

AGENT 2: COLON_POLYP_AGENT
---------------------------
Config key (model path):     self.colon_polyp_model_path = "./agents/image_analysis_agent/colon_polyp_agent/models/colon_polyp_unet.pth"
Config key (output path):    self.colon_polyp_output_path = "./uploads/colon_polyp_output/segmentation_plot.png"
Config validation flag:      "COLON_POLYP_AGENT": True
Class name:                  ColonPolypSegmentation
Import statement:            from .colon_polyp_agent.colon_polyp_inference import ColonPolypSegmentation
ImageAnalysisAgent method:   segment_colon_polyp(self, image_path: str) -> dict  [Amritesh passes output_path]
Agent node name:             COLON_POLYP_AGENT
Modality in image_classifier: COLONOSCOPY FRAME
Checkpoint path:             agents/image_analysis_agent/colon_polyp_agent/models/colon_polyp_unet.pth
Checkpoint size:             <fill in MB>
Test Dice:                   <fill in>
Test IoU:                    <fill in>

AGENT 3: DIABETIC_FOOT_ULCER_AGENT
------------------------------------
Config key (model path):     self.diabetic_foot_ulcer_model_path = "./agents/image_analysis_agent/diabetic_foot_ulcer_agent/models/dfu_efficientnet.pth"
Config validation flag:      "DIABETIC_FOOT_ULCER_AGENT": True
Class name:                  DiabeticFootUlcerClassification
Import statement:            from .diabetic_foot_ulcer_agent.diabetic_foot_ulcer_inference import DiabeticFootUlcerClassification
ImageAnalysisAgent method:   classify_diabetic_foot_ulcer(self, image_path: str) -> dict
Agent node name:             DIABETIC_FOOT_ULCER_AGENT
Modality in image_classifier: FOOT ULCER IMAGE
Checkpoint path:             agents/image_analysis_agent/diabetic_foot_ulcer_agent/models/dfu_efficientnet.pth
Checkpoint size:             <fill in MB>
Test Accuracy:               <fill in>
Test macro F1:               <fill in>
```

---

## Your Final Report — Mandatory Format

Write `Work/Jeet_Report.md` with the following sections:

### Section 1: Histopathology Agent
- Dataset source, total images, images per class
- Stain normalization decision (applied or not, and why)
- Model: EfficientNet-B3, number of parameters
- Training: LR, batch size, epochs, augmentations applied
- Test metrics: Per-class table (Class | Precision | Recall | F1 | AUC), macro F1, overall accuracy
- Checkpoint path and file size

### Section 2: Colon Polyp Agent
- Dataset: Kvasir-SEG + CVC-ClinicDB image counts, final merged counts, split sizes
- Augmentation approach (especially joint image+mask transforms)
- Model: smp.Unet encoder details
- Training: loss combo used, batch size, epochs
- Test metrics: Dice coefficient, IoU, pixel precision, pixel recall
- Sample overlay output (include path to an example saved during testing)
- Checkpoint path and file size

### Section 3: Diabetic Foot Ulcer Agent
- Dataset source, images per class, class imbalance ratio
- Preprocessing: color augmentation strategy
- Training: LR, epochs, class balancing method
- Test metrics: per-class Accuracy, F1, macro F1, AUC
- Checkpoint path and file size

### Section 4: Integration Key Document (completed)
- Fill in all `<fill in>` fields above

### Section 5: Standalone Test Results
- Copy-paste terminal output from all 3 standalone tests showing dict shapes

### Section 6: Known Issues / Limitations
- DFU dataset variability notes
- Histopathology stain normalization trade-offs

---

## Completion Test Checklist

**The job is not complete until every checkbox below is ticked. Each item is a pass/fail gate.**

### T1 — Histopathology Agent (`HistopathologyClassification`)

- [ ] `HistopathologyClassification("path/to/histopathology_efficientnet.pth")` instantiates without error
- [ ] Model is in `eval()` mode after `__init__` returns (`model.training == False`)
- [ ] `predict("valid_patch.png")` returns a Python `dict`
- [ ] Returned dict contains exactly these keys: `label`, `confidence`, `needs_validation`, `explanation`
- [ ] `result["label"]` is one of: `"ADI"`, `"BACK"`, `"DEB"`, `"LYM"`, `"MUC"`, `"MUS"`, `"NORM"`, `"STR"`, `"TUM"` (uppercase, exact match)
- [ ] `result["confidence"]` is a Python `float` with `0.0 <= result["confidence"] <= 1.0`, rounded to 4 dp
- [ ] `result["needs_validation"]` is the boolean `True`
- [ ] `result["explanation"]` is a non-empty string with the full tissue class name included
- [ ] `predict("nonexistent.jpg")` does NOT raise — returns error dict with `"confidence": 0.0`
- [ ] `models/histopathology_efficientnet.pth` exists on disk
- [ ] `models/label_map.json` exists and has exactly 9 entries mapping index strings to class abbreviations
- [ ] `models/metrics.json` exists and contains: `accuracy`, `macro_f1`, per-class table with `precision`, `recall`, `f1`, `auc` for all 9 classes
- [ ] `metrics.json` macro F1 >= 0.90 (NCT-CRC-HE-100K is clean; if below, investigate augmentation or stain normalization)
- [ ] Run standalone test:

  ```python
  from agents.image_analysis_agent.histopathology_agent.histopathology_inference import HistopathologyClassification
  clf = HistopathologyClassification("./agents/image_analysis_agent/histopathology_agent/models/histopathology_efficientnet.pth")
  result = clf.predict("./data/histopathology/test/TUM/some_patch.png")
  assert isinstance(result, dict)
  assert result["label"] in {"ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"}
  assert result["needs_validation"] == True
  assert isinstance(result["confidence"], float)
  print(result)
  ```

  **Must print valid dict without AssertionError.**

### T2 — Colon Polyp Agent (`ColonPolypSegmentation`)

- [ ] `ColonPolypSegmentation("path/to/colon_polyp_unet.pth")` instantiates without error
- [ ] Model is in `eval()` mode after `__init__` returns
- [ ] `predict("valid_colonoscopy.jpg", "output/overlay.png")` returns a Python `dict`
- [ ] Returned dict contains exactly these keys: `label`, `confidence`, `mask_path`, `overlay_path`, `needs_validation`, `explanation`
- [ ] `result["label"]` is exactly `"polyp_detected"` or `"no_polyp"` (lowercase, underscored, exact match)
- [ ] `result["confidence"]` is `float`, `0.0 <= x <= 1.0`, rounded to 4 dp
- [ ] `result["needs_validation"]` is boolean `True`
- [ ] `result["overlay_path"]` equals the `output_path` argument passed to `predict()`
- [ ] After calling `predict(...)`, the overlay PNG file actually exists at `output_path` on disk
- [ ] After calling `predict(...)`, the mask PNG file actually exists at `result["mask_path"]` on disk
- [ ] Labeling logic: `mask.max() > 0.5` → `"polyp_detected"`, otherwise `"no_polyp"`
- [ ] Joint augmentation in `train.py` uses `A.Compose([...], additional_targets={'mask': 'mask'})` — verify the `additional_targets` kwarg is present
- [ ] The same `aug(image=img, mask=msk)` call returns transformed `image` and `mask` with identical spatial transforms applied to both
- [ ] `models/colon_polyp_unet.pth` exists on disk
- [ ] `models/metrics.json` exists and contains: `dice`, `iou`, `pixel_precision`, `pixel_recall`
- [ ] `metrics.json` Dice >= 0.80 (if below, document the gap)
- [ ] Run standalone test:

  ```python
  import os
  from agents.image_analysis_agent.colon_polyp_agent.colon_polyp_inference import ColonPolypSegmentation
  seg = ColonPolypSegmentation("./agents/image_analysis_agent/colon_polyp_agent/models/colon_polyp_unet.pth")
  result = seg.predict("./data/colon_polyp/test/images/some_frame.jpg", "./test_polyp_overlay.png")
  assert isinstance(result, dict)
  assert result["label"] in {"polyp_detected", "no_polyp"}
  assert result["needs_validation"] == True
  assert os.path.exists(result["overlay_path"])
  assert os.path.exists(result["mask_path"])
  print(result)
  ```

  **Must complete without AssertionError and both PNG files must exist on disk.**

### T3 — Diabetic Foot Ulcer Agent (`DiabeticFootUlcerClassification`)

- [ ] `DiabeticFootUlcerClassification("path/to/diabetic_foot_ulcer_efficientnet.pth")` instantiates without error
- [ ] Model is in `eval()` mode after `__init__` returns
- [ ] `predict("valid_wound.jpg")` returns a Python `dict`
- [ ] Returned dict contains exactly these keys: `label`, `confidence`, `needs_validation`, `explanation`
- [ ] `result["label"]` is one of: `"Ulcer"`, `"Infection"`, `"Normal"`, `"Gangrene"` (title-case, exact match)
- [ ] `result["confidence"]` is `float`, `0.0 <= x <= 1.0`, rounded to 4 dp
- [ ] `result["needs_validation"]` is boolean `True`
- [ ] `result["explanation"]` is a non-empty string
- [ ] `predict("nonexistent.jpg")` does NOT raise — returns error dict
- [ ] `models/diabetic_foot_ulcer_efficientnet.pth` exists on disk
- [ ] `models/label_map.json` exists and has exactly 4 entries
- [ ] `models/metrics.json` exists and contains: `accuracy`, `macro_f1`, per-class `f1`, `auc`
- [ ] Run standalone test:

  ```python
  from agents.image_analysis_agent.diabetic_foot_ulcer_agent.diabetic_foot_ulcer_inference import DiabeticFootUlcerClassification
  clf = DiabeticFootUlcerClassification("./agents/image_analysis_agent/diabetic_foot_ulcer_agent/models/diabetic_foot_ulcer_efficientnet.pth")
  result = clf.predict("./data/diabetic_foot_ulcer/test/Ulcer/some_image.jpg")
  assert isinstance(result, dict)
  assert result["label"] in {"Ulcer", "Infection", "Normal", "Gangrene"}
  assert result["needs_validation"] == True
  assert isinstance(result["confidence"], float)
  print(result)
  ```

  **Must print valid dict without AssertionError.**

### T4 — Integration Readiness (Amritesh's pre-wire checklist)

Before handing off, verify ALL of the following:

- [ ] `agents/image_analysis_agent/histopathology_agent/histopathology_inference.py` exists
- [ ] `agents/image_analysis_agent/histopathology_agent/models/histopathology_efficientnet.pth` exists
- [ ] `agents/image_analysis_agent/histopathology_agent/models/label_map.json` exists (9 entries)
- [ ] `agents/image_analysis_agent/histopathology_agent/models/metrics.json` exists
- [ ] `agents/image_analysis_agent/colon_polyp_agent/colon_polyp_inference.py` exists
- [ ] `agents/image_analysis_agent/colon_polyp_agent/models/colon_polyp_unet.pth` exists
- [ ] `agents/image_analysis_agent/colon_polyp_agent/models/metrics.json` exists
- [ ] `agents/image_analysis_agent/diabetic_foot_ulcer_agent/diabetic_foot_ulcer_inference.py` exists
- [ ] `agents/image_analysis_agent/diabetic_foot_ulcer_agent/models/diabetic_foot_ulcer_efficientnet.pth` exists
- [ ] `agents/image_analysis_agent/diabetic_foot_ulcer_agent/models/label_map.json` exists (4 entries)
- [ ] `agents/image_analysis_agent/diabetic_foot_ulcer_agent/models/metrics.json` exists
- [ ] No class contains a hardcoded model path — all paths come from `__init__(self, model_path: str)`
- [ ] Class names are exactly: `HistopathologyClassification`, `ColonPolypSegmentation`, `DiabeticFootUlcerClassification`
- [ ] `from agents.image_analysis_agent.histopathology_agent.histopathology_inference import HistopathologyClassification` works
- [ ] `from agents.image_analysis_agent.colon_polyp_agent.colon_polyp_inference import ColonPolypSegmentation` works
- [ ] `from agents.image_analysis_agent.diabetic_foot_ulcer_agent.diabetic_foot_ulcer_inference import DiabeticFootUlcerClassification` works
- [ ] `IMAGE_SIZE`, `MEAN`, `STD` are defined as module-level constants in all 3 inference files
- [ ] The `IMAGE_SIZE` in each inference file matches the `IMAGE_SIZE` in its corresponding `train.py`
- Any dataset labeling issues encountered

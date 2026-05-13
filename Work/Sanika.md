# Sanika — Pulmonary & Chest Imaging Agents

## Role
You own all chest and pulmonary X-ray agents beyond the existing COVID classifier. You are responsible for three agents:
1. **TB_AGENT** — Tuberculosis detection (binary, DenseNet121)
2. **PNEUMOTHORAX_AGENT** — Pneumothorax segmentation (U-Net)
3. **MULTI_LABEL_CXR_AGENT** — 14-finding multi-label classification (EfficientNet-B4, CheXpert)

You deliver working inference wrappers, trained checkpoints, and training scripts to Amritesh for integration. You do NOT modify `config.py`, `__init__.py`, or `agent_decision.py` — Amritesh owns those. You only create the agent folders and the models.

---

## Technology Stack

- Python 3.10+
- PyTorch + torchvision
- `segmentation_models_pytorch` (for U-Net with pretrained encoder)
- `efficientnet_pytorch` or `torchvision.models.efficientnet_b4`
- `albumentations` for augmentations
- `sklearn.metrics` for evaluation

Install if not present:
```bash
pip install segmentation-models-pytorch albumentations
```

---

## Strict Output Contracts — Read This First

Every `predict()` method you write must return exactly these shapes. Amritesh's integration code depends on these keys. Do not add extra keys, do not rename keys.

**Classification agents (TB, Multi-label):**
```python
# TB — binary classification
{
    "label":            str,   # "TB" or "Normal"
    "confidence":       float, # 0.0 to 1.0
    "needs_validation": True,
    "explanation":      str
}

# Multi-label CXR
{
    "findings": [
        {"label": str, "confidence": float, "status": str}  # status: "positive"|"negative"|"uncertain"
    ],
    "needs_validation": True,
    "explanation": str
}
```

**Segmentation agent (Pneumothorax):**
```python
{
    "label":            str,    # "pneumothorax_detected" or "no_pneumothorax"
    "confidence":       float,
    "mask_path":        str,    # absolute or relative path to saved binary mask PNG
    "overlay_path":     str,    # absolute or relative path to overlay PNG
    "needs_validation": True,
    "explanation":      str
}
```

The `predict()` signature for segmentation is:
```python
def predict(self, image_path: str, output_path: str) -> dict:
```
`output_path` is the overlay image path. Amritesh passes this from `config.medical_cv.pneumothorax_output_path`.

---

## Agent 1: TB_AGENT

### Folder Structure
```
agents/image_analysis_agent/tb_agent/
├── tb_inference.py
├── train.py
├── models/
│   ├── tb_densenet121.pth
│   └── label_map.json
```

### Dataset

- **Primary**: Shenzhen TB Chest X-Ray + Montgomery TB Chest X-Ray (both public)
  - Shenzhen: ~662 images (336 TB, 326 Normal)
  - Montgomery: ~138 images (58 TB, 80 Normal)
- **Combined total**: ~800 images
- **Download**:
  - Shenzhen: [NIH/NLM Shenzhen dataset](https://openi.nlm.nih.gov/faq) — search "Shenzhen" on OpenI
  - Montgomery: same OpenI portal — search "Montgomery"
  - Alternative Kaggle mirror: search "Shenzhen Montgomery tuberculosis" on Kaggle
- **Supplementary**: NIH Chest X-ray14 (if you need more Normal class data, sample Normal images from it)

### Data Preparation

After download, organize exactly as:
```
data/tb/
├── train/
│   ├── TB/
│   └── Normal/
├── val/
│   ├── TB/
│   └── Normal/
└── test/
    ├── TB/
    └── Normal/
```

Split: 70% train, 15% val, 15% test. Use stratified split (keep class ratio).

### label_map.json
```json
{"0": "Normal", "1": "TB"}
```

### Preprocessing Constants
```python
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
```

Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) during training:
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])
```

Because images are grayscale X-rays but DenseNet expects 3-channel input, convert grayscale to RGB:
```python
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)   # replicate channel 3 times
```

### train.py

Required constants at the top:
```python
DATASET_SOURCE  = "Shenzhen TB + Montgomery TB (NLM OpenI)"
RANDOM_SEED     = 42
CLASS_NAMES     = ['Normal', 'TB']
IMAGE_SIZE      = 224
BATCH_SIZE      = 32
EPOCHS          = 40
LR              = 1e-4
CHECKPOINT_PATH = "./agents/image_analysis_agent/tb_agent/models/tb_densenet121.pth"
LABEL_MAP_PATH  = "./agents/image_analysis_agent/tb_agent/models/label_map.json"
```

**Model:**
```python
import torchvision.models as models
import torch.nn as nn

model = models.densenet121(weights="IMAGENET1K_V1")
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 2)   # binary
```

**Class imbalance handling:**
```python
from torch.utils.data import WeightedRandomSampler
# Compute class weights from training set label distribution
# Use WeightedRandomSampler to oversample the minority class
```

**Loss:** `nn.CrossEntropyLoss()` with class weights OR `nn.BCEWithLogitsLoss()` (binary)

**Optimizer:** `Adam(model.parameters(), lr=LR)`

**Scheduler:** `ReduceLROnPlateau(optimizer, patience=5, factor=0.5)`

**Required output:**
- Save best val AUC checkpoint to `CHECKPOINT_PATH`
- Print per-epoch: epoch, train_loss, val_loss, val_accuracy, val_AUC
- After training, run test set evaluation and print: Accuracy, Precision, Recall, F1, AUC
- Save test metrics to `agents/image_analysis_agent/tb_agent/models/metrics.json`
- Save label_map.json

### tb_inference.py — Exact Class Spec

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)

CLASS_NAMES = ['Normal', 'TB']
IMAGE_SIZE  = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

EXPLANATION_MAP = {
    'TB':     'Chest X-ray findings are consistent with active tuberculosis. Characteristic features include upper-lobe infiltrates and cavitation. Requires immediate clinical confirmation and isolation protocol.',
    'Normal': 'No radiographic findings suggestive of tuberculosis. Lung fields appear clear.'
}

class TBClassification:
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
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
        return model.to(self.device)

    def _load_weights(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            logger.info(f"TB model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load TB model: {e}")
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
            logger.error(f"TB inference error: {e}")
            return {"label": "unknown", "confidence": 0.0, "needs_validation": True, "explanation": str(e)}
```

### Standalone Test for TB
```python
from agents.image_analysis_agent.tb_agent.tb_inference import TBClassification
clf = TBClassification("./agents/image_analysis_agent/tb_agent/models/tb_densenet121.pth")
result = clf.predict("./data/tb/test/TB/some_image.png")
assert isinstance(result, dict)
assert result["needs_validation"] == True
print(result)
```

---

## Agent 2: PNEUMOTHORAX_AGENT

### Folder Structure
```
agents/image_analysis_agent/pneumothorax_agent/
├── pneumothorax_inference.py
├── train.py
├── models/
│   └── pneumothorax_unet.pth
```

### Dataset

- **Source**: SIIM-ACR Pneumothorax Segmentation (Kaggle 2019)
- **Download**: `kaggle competitions download -c siim-acr-pneumothorax-segmentation`
- **Format**: DICOM images + run-length encoded (RLE) masks
- **Size**: ~12,000 training images; ~3,000 have pneumothorax; rest are normal
- **Task**: Binary segmentation — produce a mask where pneumothorax pixels are 1, background is 0

### Data Preparation

Convert DICOM to PNG and decode RLE masks:
```python
import pydicom
import numpy as np
from PIL import Image

def dicom_to_png(dcm_path):
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(float)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255
    return Image.fromarray(img.astype(np.uint8)).convert("RGB")

def rle_to_mask(rle_str, height=1024, width=1024):
    """Decode run-length encoding to binary mask."""
    if rle_str == "-1" or rle_str == "":
        return np.zeros((height, width), dtype=np.uint8)
    nums = list(map(int, rle_str.split()))
    starts, lengths = nums[0::2], nums[1::2]
    mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1
    return mask.reshape((height, width), order='F')
```

Organize as:
```
data/pneumothorax/
├── train/images/   ← PNG files
├── train/masks/    ← binary mask PNGs (0=background, 255=pneumothorax)
├── val/images/
├── val/masks/
└── test/images/    ← masks not needed for inference test
```

Split: use the provided train/test split. Create a 90/10 val split from training data.

### Preprocessing Constants
```python
IMAGE_SIZE = 512    # U-Net input
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
```

### train.py

Required constants:
```python
DATASET_SOURCE  = "https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation"
RANDOM_SEED     = 42
IMAGE_SIZE      = 512
BATCH_SIZE      = 8   # U-Net is memory intensive
EPOCHS          = 30
LR              = 1e-4
CHECKPOINT_PATH = "./agents/image_analysis_agent/pneumothorax_agent/models/pneumothorax_unet.pth"
```

**Model (use segmentation_models_pytorch):**
```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None   # raw logits; apply sigmoid manually
)
```

**Loss:** Dice loss + Binary Cross-Entropy combined:
```python
from segmentation_models_pytorch.losses import DiceLoss
bce_loss  = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss(mode='binary', from_logits=True)
total_loss = 0.5 * bce_loss(pred, mask) + 0.5 * dice_loss(pred, mask)
```

**Augmentations:**
```python
import albumentations as A
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=10, p=0.3),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2()
])
```

**Required metrics:** Dice score, IoU (Jaccard), pixel precision, pixel recall on test set.

**Checkpoint format:** Save model state_dict as:
```python
torch.save(model.state_dict(), CHECKPOINT_PATH)
```

### pneumothorax_inference.py — Exact Class Spec

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

IMAGE_SIZE = 512
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

class PneumothoraxSegmentation:
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
            logger.info(f"Pneumothorax model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load pneumothorax model: {e}")
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
            img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
            for i, (m, s) in enumerate(zip(MEAN, STD)):
                img_resized[:, :, i] = (img_resized[:, :, i] - m) / s
            tensor = torch.tensor(img_resized).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = torch.sigmoid(self.model(tensor)).squeeze().cpu().numpy()
            mask = (pred > 0.5).astype(np.uint8)
            confidence = float(np.mean(pred))

            # Resize mask to original image size
            h, w = img_rgb.shape[:2]
            mask_resized = cv2.resize(mask, (w, h))

            # Save overlay
            mask_path = output_path.replace(".png", "_mask.png")
            cv2.imwrite(mask_path, mask_resized * 255)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.axis("off")
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            overlay = np.zeros_like(img_rgb)
            overlay[mask_resized == 1] = [255, 0, 0]
            ax.imshow(overlay, alpha=0.4)
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()

            label = "pneumothorax_detected" if mask.sum() > 100 else "no_pneumothorax"
            explanation = (
                "Pneumothorax detected — collapsed lung region highlighted in overlay. "
                "Immediate clinical evaluation required."
                if label == "pneumothorax_detected"
                else "No significant pneumothorax detected in this chest X-ray."
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
            logger.error(f"Pneumothorax inference error: {e}")
            return {"label": "unknown", "confidence": 0.0, "mask_path": None, "overlay_path": None,
                    "needs_validation": True, "explanation": str(e)}
```

### Standalone Test for Pneumothorax
```python
from agents.image_analysis_agent.pneumothorax_agent.pneumothorax_inference import PneumothoraxSegmentation
seg = PneumothoraxSegmentation("./agents/image_analysis_agent/pneumothorax_agent/models/pneumothorax_unet.pth")
result = seg.predict("./data/pneumothorax/test/images/test_img.png", "./uploads/pneumothorax_output/segmentation_plot.png")
assert isinstance(result, dict)
assert result["needs_validation"] == True
assert result["overlay_path"] is not None
print(result)
```

---

## Agent 3: MULTI_LABEL_CXR_AGENT

### Folder Structure
```
agents/image_analysis_agent/multi_label_cxr_agent/
├── multi_label_cxr_inference.py
├── train.py
├── models/
│   ├── multi_label_cxr_efficientnet.pth
│   └── label_map.json
```

### Dataset

- **Source**: CheXpert (Stanford)
- **Download**: [CheXpert — Stanford](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2) — requires registration
- **Labels (14)**: No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices
- **Size**: ~224,000 frontal X-rays
- **Important**: CheXpert has uncertain labels (−1 = uncertain, 0 = negative, 1 = positive)

### Uncertainty Handling

For uncertain labels (−1), use the "U-Zeros" policy (treat as 0/negative):
```python
# In dataset loading:
labels = np.clip(labels, 0, 1)   # map -1 → 0
```

This is the standard approach from the CheXpert paper.

### label_map.json
```json
{
  "0":  "No Finding",
  "1":  "Enlarged Cardiomediastinum",
  "2":  "Cardiomegaly",
  "3":  "Lung Opacity",
  "4":  "Lung Lesion",
  "5":  "Edema",
  "6":  "Consolidation",
  "7":  "Pneumonia",
  "8":  "Atelectasis",
  "9":  "Pneumothorax",
  "10": "Pleural Effusion",
  "11": "Pleural Other",
  "12": "Fracture",
  "13": "Support Devices"
}
```

### Preprocessing Constants
```python
IMAGE_SIZE = 320
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
NUM_CLASSES = 14
THRESHOLD   = 0.5    # per-label sigmoid threshold for positive/uncertain/negative
```

### train.py

Required constants:
```python
DATASET_SOURCE  = "https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2"
RANDOM_SEED     = 42
IMAGE_SIZE      = 320
BATCH_SIZE      = 32
EPOCHS          = 20
LR              = 1e-4
NUM_CLASSES     = 14
CHECKPOINT_PATH = "./agents/image_analysis_agent/multi_label_cxr_agent/models/multi_label_cxr_efficientnet.pth"
LABEL_MAP_PATH  = "./agents/image_analysis_agent/multi_label_cxr_agent/models/label_map.json"
```

**Model:**
```python
import torchvision.models as models
model = models.efficientnet_b4(weights="IMAGENET1K_V1")
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)  # no activation here — use sigmoid in loss
```

**Loss:** `BCEWithLogitsLoss()` — applied per label, averaged over batch

**Evaluation:**
```python
from sklearn.metrics import roc_auc_score
# Per-label AUC
# Micro-average AUC
# Macro-average AUC
```

**Required metrics output:** Per-label AUC table + mean AUC printed and saved to `metrics.json`

### multi_label_cxr_inference.py — Exact Class Spec

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import logging
import json, os

logger = logging.getLogger(__name__)

LABEL_NAMES = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]
NUM_CLASSES = 14
IMAGE_SIZE  = 320
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
THRESHOLD   = 0.5

class MultiLabelCXRClassification:
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
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        return model.to(self.device)

    def _load_weights(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            logger.info(f"Multi-label CXR model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load multi-label CXR model: {e}")
            raise

    def predict(self, image_path: str) -> dict:
        """
        Returns:
            {
                "findings": [{"label": str, "confidence": float, "status": str}],
                "needs_validation": True,
                "explanation": str
            }
        """
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs  = torch.sigmoid(logits).squeeze().cpu().numpy()
            findings = []
            positives = []
            for i, (label, conf) in enumerate(zip(LABEL_NAMES, probs)):
                if conf >= THRESHOLD:
                    status = "positive"
                    positives.append(label)
                elif conf >= 0.3:
                    status = "uncertain"
                else:
                    status = "negative"
                findings.append({
                    "label":      label,
                    "confidence": round(float(conf), 4),
                    "status":     status
                })
            explanation = (
                f"Positive findings: {', '.join(positives)}. Requires radiologist review."
                if positives else
                "No significant findings detected. Clinical correlation recommended."
            )
            return {
                "findings":         findings,
                "needs_validation": True,
                "explanation":      explanation
            }
        except Exception as e:
            logger.error(f"Multi-label CXR inference error: {e}")
            return {"findings": [], "needs_validation": True, "explanation": str(e)}
```

### Standalone Test for Multi-label CXR
```python
from agents.image_analysis_agent.multi_label_cxr_agent.multi_label_cxr_inference import MultiLabelCXRClassification
clf = MultiLabelCXRClassification("./agents/image_analysis_agent/multi_label_cxr_agent/models/multi_label_cxr_efficientnet.pth")
result = clf.predict("./data/chexpert/test/some_image.jpg")
assert isinstance(result, dict)
assert "findings" in result
assert isinstance(result["findings"], list)
assert result["needs_validation"] == True
print(result)
```

---

## Integration Handoff — What You Give Amritesh

After completing all three agents, give Amritesh the following exact document (fill in the blanks):

```
SANIKA INTEGRATION KEY DOCUMENT
================================

AGENT 1: TB_AGENT
-----------------
Config key (model path):     self.tb_model_path = "./agents/image_analysis_agent/tb_agent/models/tb_densenet121.pth"
Config validation flag:      "TB_AGENT": True
Class name:                  TBClassification
Import statement:            from .tb_agent.tb_inference import TBClassification
ImageAnalysisAgent method:   classify_tb(self, image_path: str) -> dict
Agent node name (in graph):  TB_AGENT
Modality classifier maps to: CHEST X-RAY (LLM routes based on query context)
Checkpoint path:             agents/image_analysis_agent/tb_agent/models/tb_densenet121.pth
Checkpoint size:             <fill in MB>
Test set AUC:                <fill in>
Test set F1:                 <fill in>

AGENT 2: PNEUMOTHORAX_AGENT
----------------------------
Config key (model path):     self.pneumothorax_model_path = "./agents/image_analysis_agent/pneumothorax_agent/models/pneumothorax_unet.pth"
Config key (output path):    self.pneumothorax_output_path = "./uploads/pneumothorax_output/segmentation_plot.png"
Config validation flag:      "PNEUMOTHORAX_AGENT": True
Class name:                  PneumothoraxSegmentation
Import statement:            from .pneumothorax_agent.pneumothorax_inference import PneumothoraxSegmentation
ImageAnalysisAgent method:   segment_pneumothorax(self, image_path: str) -> dict  [Amritesh passes output_path internally]
Agent node name:             PNEUMOTHORAX_AGENT
Modality classifier maps to: CHEST X-RAY
Checkpoint path:             agents/image_analysis_agent/pneumothorax_agent/models/pneumothorax_unet.pth
Checkpoint size:             <fill in MB>
Test set Dice:               <fill in>
Test set IoU:                <fill in>

AGENT 3: MULTI_LABEL_CXR_AGENT
-------------------------------
Config key (model path):     self.multi_label_cxr_model_path = "./agents/image_analysis_agent/multi_label_cxr_agent/models/multi_label_cxr_efficientnet.pth"
Config validation flag:      "MULTI_LABEL_CXR_AGENT": True
Class name:                  MultiLabelCXRClassification
Import statement:            from .multi_label_cxr_agent.multi_label_cxr_inference import MultiLabelCXRClassification
ImageAnalysisAgent method:   classify_multi_label_cxr(self, image_path: str) -> dict
Agent node name:             MULTI_LABEL_CXR_AGENT
Modality classifier maps to: CHEST X-RAY
Checkpoint path:             agents/image_analysis_agent/multi_label_cxr_agent/models/multi_label_cxr_efficientnet.pth
Checkpoint size:             <fill in MB>
Mean AUC (14 labels):        <fill in>
```

---

## Your Final Report — Mandatory Format

After completing all three agents, write `Work/Sanika_Report.md` with the following sections:

### Section 1: TB Agent
- Dataset: images per class, total size, preprocessing applied (CLAHE yes/no, normalization values)
- Model: architecture name, number of parameters, pretrained source
- Training: learning rate, batch size, epochs, optimizer, scheduler, class balancing strategy used
- Test metrics: Accuracy, AUC, Precision, Recall, F1
- Checkpoint path and file size in MB
- Deviations from this document (if any, justify them)

### Section 2: Pneumothorax Agent
- Dataset: total images, mask coverage percentage, split sizes
- DICOM conversion approach used
- Model: smp.Unet encoder name, encoder pretrained source
- Training: loss function combination, batch size, epochs
- Test metrics: Dice coefficient, IoU, pixel precision, pixel recall
- Checkpoint path and file size
- Sample overlay image (include path to a saved example)

### Section 3: Multi-label CXR Agent
- Dataset: total images used, uncertainty handling strategy
- Model: EfficientNet-B4, number of output heads
- Training: loss function, batch size, epochs, threshold used
- Per-label AUC table (14 rows: label name | AUC value)
- Mean AUC, Micro AUC, Macro AUC
- Checkpoint path and file size

### Section 4: Integration Key Document (completed)
- Fill in all `<fill in>` fields in the Integration Handoff section above

### Section 5: Standalone Test Results
- For each agent: command you ran + output printed to console confirming the dict shape
- Screenshot or copy-paste of the terminal output

### Section 6: Known Issues / Limitations
- Any dataset issues encountered (corrupted files, class imbalance, labeling issues)
- Any model performance concerns

---

## Completion Test Checklist

**The job is not complete until every checkbox below is ticked. Each item is a pass/fail gate.**

### T1 — TB Agent (`TBClassification`)

- [ ] `TBClassification("path/to/tb_densenet121.pth")` instantiates without error
- [ ] Model is in `eval()` mode after `__init__` returns (`model.training == False`)
- [ ] `predict("valid_xray.png")` returns a Python `dict`
- [ ] Returned dict contains exactly these keys: `label`, `confidence`, `needs_validation`, `explanation`
- [ ] `result["label"]` is exactly `"TB"` or exactly `"Normal"` (case-sensitive)
- [ ] `result["confidence"]` is a Python `float` with `0.0 <= result["confidence"] <= 1.0`
- [ ] `result["confidence"]` is rounded to 4 decimal places (`round(x, 4)`)
- [ ] `result["needs_validation"]` is the boolean `True` — not the string `"True"`, not `1`
- [ ] `result["explanation"]` is a non-empty string
- [ ] `predict("nonexistent_path.jpg")` does NOT raise — returns error dict with `"label": "unknown"`, `"confidence": 0.0`
- [ ] CLAHE augmentation is applied in `train.py` transforms (verify the `A.CLAHE(...)` line exists)
- [ ] Grayscale → RGB conversion is in `train.py` data loading (`cv2.COLOR_GRAY2RGB`)
- [ ] `WeightedRandomSampler` is used in `train.py` DataLoader
- [ ] `models/tb_densenet121.pth` exists on disk at the correct relative path
- [ ] `models/label_map.json` exists and equals `{"0": "Normal", "1": "TB"}`
- [ ] `models/metrics.json` exists and contains: `accuracy`, `precision`, `recall`, `f1`, `auc`
- [ ] `metrics.json` AUC >= 0.90 (if below, document the gap with explanation)
- [ ] Run standalone test:
  ```python
  from agents.image_analysis_agent.tb_agent.tb_inference import TBClassification
  clf = TBClassification("./agents/image_analysis_agent/tb_agent/models/tb_densenet121.pth")
  result = clf.predict("./data/tb/test/TB/some_image.png")
  assert isinstance(result, dict)
  assert result["needs_validation"] == True
  assert isinstance(result["confidence"], float)
  ```
  → **must print valid dict without AssertionError**

### T2 — Pneumothorax Agent (`PneumothoraxSegmentation`)

- [ ] `PneumothoraxSegmentation("path/to/pneumothorax_unet.pth")` instantiates without error
- [ ] Model is in `eval()` mode after `__init__` returns
- [ ] `predict("valid_xray.dcm_or_png", "output/overlay.png")` returns a Python `dict`
- [ ] Returned dict contains exactly these keys: `label`, `confidence`, `mask_path`, `overlay_path`, `needs_validation`, `explanation`
- [ ] `result["label"]` is exactly `"pneumothorax_detected"` or `"no_pneumothorax"` (case-sensitive)
- [ ] `result["confidence"]` is `float`, `0.0 <= x <= 1.0`, rounded to 4 dp
- [ ] `result["needs_validation"]` is boolean `True`
- [ ] `result["overlay_path"]` equals the `output_path` argument that was passed in
- [ ] After calling `predict(...)`, the file at `output_path` actually exists on disk (PNG created)
- [ ] After calling `predict(...)`, the file at `result["mask_path"]` exists on disk
- [ ] Labeling logic: mask with no positive pixels (`mask.max() <= 0.5`) → label is `"no_pneumothorax"`
- [ ] Labeling logic: mask with positive pixels (`mask.max() > 0.5`) → label is `"pneumothorax_detected"`
- [ ] `rle_decode()` function exists in `train.py` or a shared utility file and correctly decodes SIIM RLE strings
- [ ] DICOM loading via `pydicom` is tested on at least one DICOM file from the dataset
- [ ] `models/pneumothorax_unet.pth` exists on disk
- [ ] `models/metrics.json` exists and contains: `dice`, `iou`, `pixel_precision`, `pixel_recall`
- [ ] `metrics.json` Dice >= 0.75 (if below, document the gap)
- [ ] Run standalone test:
  ```python
  from agents.image_analysis_agent.pneumothorax_agent.pneumothorax_inference import PneumothoraxSegmentation
  seg = PneumothoraxSegmentation("./agents/image_analysis_agent/pneumothorax_agent/models/pneumothorax_unet.pth")
  result = seg.predict("./data/pneumothorax/test/some_image.png", "./test_overlay.png")
  assert isinstance(result, dict)
  assert result["needs_validation"] == True
  assert "overlay_path" in result
  import os; assert os.path.exists(result["overlay_path"])
  ```
  → **must complete without AssertionError and overlay PNG must exist**

### T3 — Multi-Label CXR Agent (`MultiLabelCXRClassification`)

- [ ] `MultiLabelCXRClassification("path/to/multi_label_cxr_efficientnet.pth")` instantiates without error
- [ ] Model is in `eval()` mode after `__init__` returns
- [ ] `predict("valid_cxr.jpg")` returns a Python `dict`
- [ ] Returned dict contains exactly these keys: `findings`, `needs_validation`, `explanation`
- [ ] `result["findings"]` is a Python `list`
- [ ] `len(result["findings"]) == 14` (exactly 14 entries)
- [ ] Each element in `result["findings"]` is a dict with exactly 3 keys: `label`, `confidence`, `status`
- [ ] Each `label` value is one of the 14 CheXpert finding names (case must match FINDINGS list exactly)
- [ ] Each `confidence` is float `0.0 <= x <= 1.0`, rounded to 4 dp
- [ ] Each `status` is exactly `"positive"` or `"negative"` (lowercase, no other values)
- [ ] `result["needs_validation"]` is boolean `True`
- [ ] `result["explanation"]` is a non-empty string
- [ ] U-Zeros policy: verify in `train.py` that `-1` values in label CSV are replaced with `0` before training
- [ ] Sigmoid is applied in `predict()` (not softmax — this is multi-label)
- [ ] Threshold 0.5 is used to determine `status`: `>= 0.5` → `"positive"`, `< 0.5` → `"negative"`
- [ ] `models/multi_label_cxr_efficientnet.pth` exists on disk
- [ ] `models/metrics.json` exists and contains: `mean_auc`, `micro_auc`, `macro_auc` plus per-label AUC for all 14 labels
- [ ] `metrics.json` mean AUC >= 0.80 (if below, document the gap)
- [ ] Run standalone test:
  ```python
  from agents.image_analysis_agent.multi_label_cxr_agent.multi_label_cxr_inference import MultiLabelCXRClassification
  clf = MultiLabelCXRClassification("./agents/image_analysis_agent/multi_label_cxr_agent/models/multi_label_cxr_efficientnet.pth")
  result = clf.predict("./data/chexpert/test/some_xray.jpg")
  assert isinstance(result, dict)
  assert len(result["findings"]) == 14
  assert result["needs_validation"] == True
  assert all(f["status"] in ("positive", "negative") for f in result["findings"])
  ```
  → **must complete without AssertionError**

### T4 — Integration Readiness (Amritesh's pre-wire checklist)

Before handing off, verify ALL of the following:

- [ ] `agents/image_analysis_agent/tb_agent/tb_inference.py` exists
- [ ] `agents/image_analysis_agent/tb_agent/models/tb_densenet121.pth` exists
- [ ] `agents/image_analysis_agent/tb_agent/models/label_map.json` exists
- [ ] `agents/image_analysis_agent/tb_agent/models/metrics.json` exists
- [ ] `agents/image_analysis_agent/pneumothorax_agent/pneumothorax_inference.py` exists
- [ ] `agents/image_analysis_agent/pneumothorax_agent/models/pneumothorax_unet.pth` exists
- [ ] `agents/image_analysis_agent/pneumothorax_agent/models/metrics.json` exists
- [ ] `agents/image_analysis_agent/multi_label_cxr_agent/multi_label_cxr_inference.py` exists
- [ ] `agents/image_analysis_agent/multi_label_cxr_agent/models/multi_label_cxr_efficientnet.pth` exists
- [ ] `agents/image_analysis_agent/multi_label_cxr_agent/models/label_map.json` exists (14 entries)
- [ ] `agents/image_analysis_agent/multi_label_cxr_agent/models/metrics.json` exists
- [ ] No class contains a hardcoded model path string — all paths come from `__init__(self, model_path: str)`
- [ ] Class names are exactly: `TBClassification`, `PneumothoraxSegmentation`, `MultiLabelCXRClassification`
- [ ] `from agents.image_analysis_agent.tb_agent.tb_inference import TBClassification` works
- [ ] `from agents.image_analysis_agent.pneumothorax_agent.pneumothorax_inference import PneumothoraxSegmentation` works
- [ ] `from agents.image_analysis_agent.multi_label_cxr_agent.multi_label_cxr_inference import MultiLabelCXRClassification` works
- Recommendations for improvement

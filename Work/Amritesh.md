# Amritesh — Integration Lead + Brain Tumor Agent

## Role
You own the entire system architecture. You are responsible for:
1. Finishing the Brain Tumor Agent (currently `# TBD`)
2. Retrofitting the two existing agents (chest xray, skin lesion) to return structured output contracts
3. Expanding `config.py` with every new model path and validation flag
4. Expanding `image_classifier.py` to cover all 12 modalities
5. Expanding `agents/image_analysis_agent/__init__.py` to register all new agents
6. Expanding `agents/agent_decision.py` to route all new agents in the LangGraph
7. Fixing `app.py` to use a generic artifact response mechanism instead of the current hardcoded skin-lesion branch
8. Receiving deliveries from Sanika, Rushil, Jeet, Srujan — verifying their contracts — wiring each one in

You write the most code. You are the final integration gate. Nothing ships without your approval.

---

## Technology Stack

- Python 3.10+
- PyTorch + torchvision
- EfficientNet via `efficientnet_pytorch` or `torchvision.models`
- LangGraph (already in use)
- FastAPI (already in use)
- All LLM config already handled in `config.py` via NVIDIA NIM

---

## Assignment 1: Brain Tumor Agent

### 1.1 Folder Structure to Create

```
agents/image_analysis_agent/brain_tumor_agent/
├── brain_tumor_inference.py     ← replace the current "# TBD" file
├── train.py
├── models/
│   └── brain_tumor_efficientnet.pth   ← trained checkpoint goes here
└── label_map.json
```

### 1.2 Dataset

- **Source**: [Brain Tumor MRI Dataset — Kaggle (Masoud Nickparvar)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes**: `glioma`, `meningioma`, `pituitary`, `no_tumor`
- **Size**: ~7,000 MRI images across 4 classes
- **Download**: `kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset`

After download, organize as:
```
data/brain_tumor/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
├── val/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
└── test/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── no_tumor/
```

The Kaggle dataset provides a train/test split. Use 80/20 on the provided train set for train/val.

### 1.3 label_map.json

```json
{
  "0": "glioma",
  "1": "meningioma",
  "2": "no_tumor",
  "3": "pituitary"
}
```

Save this at `agents/image_analysis_agent/brain_tumor_agent/models/label_map.json`.

### 1.4 Preprocessing Constants (MUST match training and inference)

```python
IMAGE_SIZE = 300          # EfficientNet-B3 native size
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
```

### 1.5 train.py

File location: `agents/image_analysis_agent/brain_tumor_agent/train.py`

The script must do the following in this exact order:

```python
# REQUIRED HEADER — log provenance
DATASET_SOURCE = "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"
DATASET_VERSION = "1.0"
RANDOM_SEED = 42
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
IMAGE_SIZE = 300
CHECKPOINT_PATH = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_efficientnet.pth"
LABEL_MAP_PATH  = "./agents/image_analysis_agent/brain_tumor_agent/models/label_map.json"
```

**Step 1 — Seed everything:**
```python
import random, numpy as np, torch
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
```

**Step 2 — DataLoaders:**
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])
train_dataset = datasets.ImageFolder("./data/brain_tumor/train", transform=train_transform)
val_dataset   = datasets.ImageFolder("./data/brain_tumor/val",   transform=val_transform)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4)
val_loader    = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4)
```

**Step 3 — Model:**
```python
import torchvision.models as models
import torch.nn as nn

model = models.efficientnet_b3(weights="IMAGENET1K_V1")
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(CLASS_NAMES))
model = model.to(device)
```

**Step 4 — Training loop:**
- Optimizer: `Adam(model.parameters(), lr=1e-4)`
- Loss: `CrossEntropyLoss()`
- Epochs: 30
- Save best val accuracy checkpoint to `CHECKPOINT_PATH`
- Print epoch, train_loss, train_acc, val_loss, val_acc each epoch

**Step 5 — Save label map:**
```python
import json
label_map = {str(v): k for k, v in train_dataset.class_to_idx.items()}
with open(LABEL_MAP_PATH, "w") as f:
    json.dump(label_map, f, indent=2)
```

**Step 6 — Final metrics on test set (required):**
- Accuracy, Precision (macro), Recall (macro), F1 (macro), AUC (one-vs-rest)
- Print and save to `agents/image_analysis_agent/brain_tumor_agent/models/metrics.json`

### 1.6 brain_tumor_inference.py (Exact Class Spec)

Replace the current `# TBD` with this exact structure:

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
IMAGE_SIZE  = 300
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

SEVERITY_MAP = {
    'glioma':      'High-grade malignant brain tumor requiring urgent specialist review.',
    'meningioma':  'Usually benign tumor arising from meninges; requires neurosurgeon review.',
    'pituitary':   'Pituitary gland tumor; may affect hormone levels — endocrinology referral recommended.',
    'no_tumor':    'No tumor detected in this MRI scan.'
}

class BrainTumorClassification:
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
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(CLASS_NAMES))
        return model.to(self.device)

    def _load_weights(self):
        try:
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            logger.info(f"Brain tumor model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load brain tumor model: {e}")
            raise

    def predict(self, image_path: str) -> dict:
        """
        Returns:
            {
                "label": str,           # one of CLASS_NAMES
                "confidence": float,    # 0.0 to 1.0
                "needs_validation": True,
                "explanation": str
            }
        """
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs  = torch.softmax(logits, dim=1)
                conf, idx = torch.max(probs, dim=1)
            label      = CLASS_NAMES[idx.item()]
            confidence = round(conf.item(), 4)
            return {
                "label":            label,
                "confidence":       confidence,
                "needs_validation": True,
                "explanation":      SEVERITY_MAP.get(label, "")
            }
        except Exception as e:
            logger.error(f"Brain tumor inference error: {e}")
            return {
                "label":            "unknown",
                "confidence":       0.0,
                "needs_validation": True,
                "explanation":      f"Inference failed: {str(e)}"
            }
```

### 1.7 Standalone Test

Before integration, test the inference wrapper directly:
```python
from agents.image_analysis_agent.brain_tumor_agent.brain_tumor_inference import BrainTumorClassification
clf = BrainTumorClassification("./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_efficientnet.pth")
result = clf.predict("./data/brain_tumor/test/glioma/some_image.jpg")
assert isinstance(result, dict)
assert "label" in result
assert "confidence" in result
assert result["needs_validation"] == True
print(result)
```

---

## Assignment 2: Retrofit Existing Agents to Structured Output

The two existing agents return raw values, not dicts. You must update them BEFORE doing any integration work so that the run_*_agent functions in agent_decision.py have a consistent interface.

### 2.1 chest_xray_inference.py Changes

In `agents/image_analysis_agent/chest_xray_agent/covid_chest_xray_inference.py`, update the `predict()` method:

**Current return (line 76):** `return pred_class`

**Replace with:**
```python
EXPLANATION_MAP = {
    'covid19': 'The chest X-ray shows bilateral peripheral ground-glass opacities consistent with COVID-19 pneumonia. Requires clinical correlation and PCR confirmation.',
    'normal':  'No abnormalities detected in this chest X-ray. Lung fields appear clear.'
}

# Inside predict(), replace "return pred_class" with:
confidence = float(torch.softmax(out, dim=1).max().item())
return {
    "label":            pred_class,
    "confidence":       round(confidence, 4),
    "needs_validation": True,
    "explanation":      EXPLANATION_MAP.get(pred_class, "")
}
```

Also add the softmax confidence calculation. The full updated predict():
```python
def predict(self, img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0)
        input_tensor = Variable(image_tensor).to(self.device)
        with torch.no_grad():
            out = self.model(input_tensor)
            probs = torch.softmax(out, dim=1)
            _, preds = torch.max(probs, 1)
            idx = preds.cpu().numpy()[0]
            pred_class = self.class_names[idx]
            confidence = round(float(probs.max().item()), 4)
        self.logger.info(f"Predicted Class: {pred_class}, Confidence: {confidence}")
        return {
            "label":            pred_class,
            "confidence":       confidence,
            "needs_validation": True,
            "explanation":      EXPLANATION_MAP.get(pred_class, "")
        }
    except Exception as e:
        self.logger.error(f"Error during prediction: {str(e)}")
        return {"label": "unknown", "confidence": 0.0, "needs_validation": True, "explanation": str(e)}
```

### 2.2 skin_lesion_inference.py Changes

In `agents/image_analysis_agent/skin_lesion_agent/skin_lesion_inference.py`, update `predict()`:

**Current return:** `return self._overlay_mask(img, generated_mask_resized, output_path)` which returns `True`

**Replace with:**
```python
def predict(self, image_path: str, output_path: str) -> dict:
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img_resized = cv2.resize(img, (256, 256))
        img_tensor = torch.Tensor(img_resized).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            generated_mask = self.model(img_tensor).squeeze().cpu().numpy()
        generated_mask_resized = cv2.resize(generated_mask, (img.shape[1], img.shape[0]))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._overlay_mask(img, generated_mask_resized, output_path)
        mask_confidence = float(np.mean(generated_mask_resized > 0.5))
        return {
            "label":            "lesion_segmented",
            "confidence":       round(mask_confidence, 4),
            "mask_path":        output_path,
            "overlay_path":     output_path,
            "needs_validation": True,
            "explanation":      "Segmentation mask generated. High-confidence lesion regions shown in overlay. Dermatologist review required."
        }
    except Exception as e:
        logger.error(f"Error during segmentation: {e}")
        return {
            "label":            "unknown",
            "confidence":       0.0,
            "mask_path":        None,
            "overlay_path":     None,
            "needs_validation": True,
            "explanation":      f"Segmentation failed: {str(e)}"
        }
```

---

## Assignment 3: config.py — Full Expansion

Open `config.py`. Make the following changes exactly.

### 3.1 MedicalCVConfig — Add All New Model Paths

In `class MedicalCVConfig`, after the existing 3 paths, add:

```python
# New agents — model checkpoints
self.tb_model_path                  = "./agents/image_analysis_agent/tb_agent/models/tb_densenet121.pth"
self.pneumothorax_model_path        = "./agents/image_analysis_agent/pneumothorax_agent/models/pneumothorax_unet.pth"
self.multi_label_cxr_model_path     = "./agents/image_analysis_agent/multi_label_cxr_agent/models/multi_label_cxr_efficientnet.pth"
self.diabetic_retinopathy_model_path= "./agents/image_analysis_agent/diabetic_retinopathy_agent/models/diabetic_retinopathy_efficientnet.pth"
self.mammography_model_path         = "./agents/image_analysis_agent/mammography_agent/models/mammography_efficientnet.pth"
self.histopathology_model_path      = "./agents/image_analysis_agent/histopathology_agent/models/histopathology_efficientnet.pth"
self.fracture_model_path            = "./agents/image_analysis_agent/fracture_agent/models/fracture_densenet.pth"
self.oct_model_path                 = "./agents/image_analysis_agent/oct_agent/models/oct_efficientnet.pth"
self.diabetic_foot_ulcer_model_path = "./agents/image_analysis_agent/diabetic_foot_ulcer_agent/models/dfu_efficientnet.pth"
self.colon_polyp_model_path         = "./agents/image_analysis_agent/colon_polyp_agent/models/colon_polyp_unet.pth"
self.ham10000_model_path            = "./agents/image_analysis_agent/ham10000_agent/models/ham10000_efficientnet.pth"

# Segmentation output paths (one per segmentation agent)
self.pneumothorax_output_path       = "./uploads/pneumothorax_output/segmentation_plot.png"
self.colon_polyp_output_path        = "./uploads/colon_polyp_output/segmentation_plot.png"
```

### 3.2 ValidationConfig — Add All New Agents

In `class ValidationConfig`, extend `require_validation`:

```python
self.require_validation = {
    "CONVERSATION_AGENT":           False,
    "RAG_AGENT":                    False,
    "WEB_SEARCH_AGENT":             False,
    "BRAIN_TUMOR_AGENT":            True,
    "CHEST_XRAY_AGENT":             True,
    "SKIN_LESION_AGENT":            True,
    # New agents — all high-risk
    "TB_AGENT":                     True,
    "PNEUMOTHORAX_AGENT":           True,
    "MULTI_LABEL_CXR_AGENT":        True,
    "DIABETIC_RETINOPATHY_AGENT":   True,
    "MAMMOGRAPHY_AGENT":            True,
    "HISTOPATHOLOGY_AGENT":         True,
    "FRACTURE_AGENT":               True,
    "OCT_AGENT":                    True,
    "DIABETIC_FOOT_ULCER_AGENT":    True,
    "COLON_POLYP_AGENT":            True,
    "HAM10000_AGENT":               True,
}
```

---

## Assignment 4: image_classifier.py — Full Expansion

Open `agents/image_analysis_agent/image_classifier.py`. Replace the `vision_prompt` inside `classify_image()`.

### 4.1 New Prompt (replace lines 43–51 entirely)

```python
CONFIDENCE_THRESHOLD = 0.65

vision_prompt = [
    {"role": "system", "content": "You are an expert in medical imaging. Respond ONLY with valid JSON, no other text."},
    {"role": "user", "content": [
        {"type": "text", "text": (
            "Classify this image into exactly one of the following categories. "
            "If confidence is below 0.65, return 'OTHER'. "
            "Categories:\n"
            "- BRAIN MRI SCAN: axial/coronal/sagittal MRI slices of the brain\n"
            "- CHEST X-RAY: posteroanterior or anteroposterior chest X-ray\n"
            "- SKIN LESION: dermoscopic or clinical photograph of skin lesion\n"
            "- FUNDUS RETINA: fundus photograph of the retina (eye)\n"
            "- OCT: optical coherence tomography retinal scan (layered cross-section)\n"
            "- MAMMOGRAM: mammographic X-ray of breast tissue\n"
            "- HISTOPATHOLOGY PATCH: H&E stained tissue patch under microscope\n"
            "- MSK X-RAY: musculoskeletal X-ray of bone/joint (shoulder, elbow, wrist, hand, hip, knee, finger)\n"
            "- COLONOSCOPY FRAME: colonoscopy video frame showing colon interior\n"
            "- FOOT ULCER IMAGE: clinical photograph of diabetic foot wound/ulcer\n"
            "- OTHER: medical image that does not match any category above, or confidence is low\n"
            "- NON-MEDICAL: not a medical image\n\n"
            f"{format_instructions}"
        )},
        {"type": "image_url", "image_url": {"url": self.local_image_to_data_url(image_path)}}
    ]}
]
```

### 4.2 Add Confidence Threshold Logic

After the JSON parsing (at the end of `classify_image()`), before `return`, add:

```python
# If confidence is below threshold, override to OTHER
parsed = ...  # whatever variable holds the parsed result
if isinstance(parsed, dict) and parsed.get("confidence", 1.0) < CONFIDENCE_THRESHOLD:
    parsed["image_type"] = "OTHER"
    parsed["reasoning"] = f"Low confidence ({parsed['confidence']:.2f}) — returning OTHER."
return parsed
```

Apply this to both the direct-parse path and the fallback path.

---

## Assignment 5: agents/image_analysis_agent/__init__.py — Full Expansion

Replace the entire file with the following (follow existing patterns exactly):

```python
from .image_classifier import ImageClassifier
from .chest_xray_agent.covid_chest_xray_inference        import ChestXRayClassification
from .brain_tumor_agent.brain_tumor_inference             import BrainTumorClassification
from .skin_lesion_agent.skin_lesion_inference             import SkinLesionSegmentation
# Sanika's agents
from .tb_agent.tb_inference                              import TBClassification
from .pneumothorax_agent.pneumothorax_inference          import PneumothoraxSegmentation
from .multi_label_cxr_agent.multi_label_cxr_inference   import MultiLabelCXRClassification
# Rushil's agents
from .diabetic_retinopathy_agent.diabetic_retinopathy_inference import DiabeticRetinopathyClassification
from .oct_agent.oct_inference                            import OCTClassification
# Jeet's agents
from .histopathology_agent.histopathology_inference      import HistopathologyClassification
from .colon_polyp_agent.colon_polyp_inference            import ColonPolypSegmentation
from .diabetic_foot_ulcer_agent.diabetic_foot_ulcer_inference import DiabeticFootUlcerClassification
# Srujan's agents
from .fracture_agent.fracture_inference                  import FractureClassification
from .mammography_agent.mammography_inference            import MammographyClassification
from .ham10000_agent.ham10000_inference                  import HAM10000Classification


class ImageAnalysisAgent:
    def __init__(self, config):
        cv = config.medical_cv
        self.image_classifier       = ImageClassifier(vision_model=cv.vision_llm)
        # Existing agents
        self.chest_xray_agent       = ChestXRayClassification(model_path=cv.chest_xray_model_path)
        self.brain_tumor_agent      = BrainTumorClassification(model_path=cv.brain_tumor_model_path)
        self.skin_lesion_agent      = SkinLesionSegmentation(model_path=cv.skin_lesion_model_path)
        self.skin_lesion_output     = cv.skin_lesion_segmentation_output_path
        # Sanika's agents
        self.tb_agent               = TBClassification(model_path=cv.tb_model_path)
        self.pneumothorax_agent     = PneumothoraxSegmentation(model_path=cv.pneumothorax_model_path)
        self.pneumothorax_output    = cv.pneumothorax_output_path
        self.multi_label_cxr_agent  = MultiLabelCXRClassification(model_path=cv.multi_label_cxr_model_path)
        # Rushil's agents
        self.diabetic_retinopathy_agent = DiabeticRetinopathyClassification(model_path=cv.diabetic_retinopathy_model_path)
        self.oct_agent              = OCTClassification(model_path=cv.oct_model_path)
        # Jeet's agents
        self.histopathology_agent   = HistopathologyClassification(model_path=cv.histopathology_model_path)
        self.colon_polyp_agent      = ColonPolypSegmentation(model_path=cv.colon_polyp_model_path)
        self.colon_polyp_output     = cv.colon_polyp_output_path
        self.diabetic_foot_ulcer_agent = DiabeticFootUlcerClassification(model_path=cv.diabetic_foot_ulcer_model_path)
        # Srujan's agents
        self.fracture_agent         = FractureClassification(model_path=cv.fracture_model_path)
        self.mammography_agent      = MammographyClassification(model_path=cv.mammography_model_path)
        self.ham10000_agent         = HAM10000Classification(model_path=cv.ham10000_model_path)

    def analyze_image(self, image_path: str) -> dict:
        return self.image_classifier.classify_image(image_path)

    # Existing
    def classify_chest_xray(self, image_path: str) -> dict:
        return self.chest_xray_agent.predict(image_path)

    def classify_brain_tumor(self, image_path: str) -> dict:
        return self.brain_tumor_agent.predict(image_path)

    def segment_skin_lesion(self, image_path: str) -> dict:
        return self.skin_lesion_agent.predict(image_path, self.skin_lesion_output)

    # Sanika's
    def classify_tb(self, image_path: str) -> dict:
        return self.tb_agent.predict(image_path)

    def segment_pneumothorax(self, image_path: str) -> dict:
        return self.pneumothorax_agent.predict(image_path, self.pneumothorax_output)

    def classify_multi_label_cxr(self, image_path: str) -> dict:
        return self.multi_label_cxr_agent.predict(image_path)

    # Rushil's
    def classify_diabetic_retinopathy(self, image_path: str) -> dict:
        return self.diabetic_retinopathy_agent.predict(image_path)

    def classify_oct(self, image_path: str) -> dict:
        return self.oct_agent.predict(image_path)

    # Jeet's
    def classify_histopathology(self, image_path: str) -> dict:
        return self.histopathology_agent.predict(image_path)

    def segment_colon_polyp(self, image_path: str) -> dict:
        return self.colon_polyp_agent.predict(image_path, self.colon_polyp_output)

    def classify_diabetic_foot_ulcer(self, image_path: str) -> dict:
        return self.diabetic_foot_ulcer_agent.predict(image_path)

    # Srujan's
    def classify_fracture(self, image_path: str) -> dict:
        return self.fracture_agent.predict(image_path)

    def classify_mammography(self, image_path: str) -> dict:
        return self.mammography_agent.predict(image_path)

    def classify_ham10000(self, image_path: str) -> dict:
        return self.ham10000_agent.predict(image_path)
```

---

## Assignment 6: agents/agent_decision.py — Full Expansion

### 6.1 Add artifact_path to AgentState

In the `AgentState` class, add one new field:

```python
class AgentState(MessagesState):
    agent_name: Optional[str]
    current_input: Optional[Union[str, Dict]]
    has_image: bool
    image_type: Optional[str]
    output: Optional[str]
    needs_human_validation: bool
    retrieval_confidence: float
    bypass_routing: bool
    insufficient_info: bool
    artifact_path: Optional[str]   # ← NEW: path to segmentation overlay image
```

Also update `init_agent_state()`:
```python
def init_agent_state() -> AgentState:
    return {
        ...existing fields...,
        "artifact_path": None   # ← ADD THIS
    }
```

### 6.2 Update DECISION_SYSTEM_PROMPT

Replace the `Available agents:` section in `DECISION_SYSTEM_PROMPT` with:

```
Available agents:
1.  CONVERSATION_AGENT               - General chat, greetings, non-medical questions.
2.  RAG_AGENT                        - Medical knowledge from literature (brain tumor, COVID, chest X-ray topics).
3.  WEB_SEARCH_PROCESSOR_AGENT       - Recent medical developments, current outbreaks, time-sensitive information.
4.  BRAIN_TUMOR_AGENT                - Brain MRI image analysis: glioma, meningioma, pituitary tumor, no tumor.
5.  CHEST_XRAY_AGENT                 - Chest X-ray: COVID-19 vs normal binary classification.
6.  SKIN_LESION_AGENT                - Skin lesion dermoscopy: U-Net segmentation of lesion boundary.
7.  TB_AGENT                         - Chest X-ray: tuberculosis vs normal binary classification.
8.  PNEUMOTHORAX_AGENT               - Chest X-ray: pneumothorax segmentation (collapsed lung detection).
9.  MULTI_LABEL_CXR_AGENT            - Chest X-ray: multi-label classification of 14 CheXpert findings.
10. DIABETIC_RETINOPATHY_AGENT       - Fundus retina image: diabetic retinopathy grading 0–4.
11. MAMMOGRAPHY_AGENT                - Mammogram: benign vs malignant breast finding classification.
12. HISTOPATHOLOGY_AGENT             - Histopathology patch (H&E stain): 9-class tissue classification.
13. FRACTURE_AGENT                   - Musculoskeletal X-ray: normal vs abnormal (fracture detection).
14. OCT_AGENT                        - OCT retinal scan: CNV, DME, DRUSEN, NORMAL classification.
15. DIABETIC_FOOT_ULCER_AGENT        - Foot ulcer photograph: Ulcer, Infection, Normal, Gangrene classification.
16. COLON_POLYP_AGENT                - Colonoscopy frame: colon polyp segmentation.
17. HAM10000_AGENT                   - Skin lesion: 7-class HAM10000 dermatology classification.

Routing rules for images:
- If image_type is BRAIN MRI SCAN → route to BRAIN_TUMOR_AGENT
- If image_type is CHEST X-RAY AND query mentions tuberculosis/TB → route to TB_AGENT
- If image_type is CHEST X-RAY AND query mentions pneumothorax/collapsed lung → route to PNEUMOTHORAX_AGENT
- If image_type is CHEST X-RAY AND query mentions multiple findings/CheXpert → route to MULTI_LABEL_CXR_AGENT
- If image_type is CHEST X-RAY (default/COVID) → route to CHEST_XRAY_AGENT
- If image_type is SKIN LESION AND query mentions classification/type/diagnosis → route to HAM10000_AGENT
- If image_type is SKIN LESION (default/segmentation) → route to SKIN_LESION_AGENT
- If image_type is FUNDUS RETINA → route to DIABETIC_RETINOPATHY_AGENT
- If image_type is OCT → route to OCT_AGENT
- If image_type is MAMMOGRAM → route to MAMMOGRAPHY_AGENT
- If image_type is HISTOPATHOLOGY PATCH → route to HISTOPATHOLOGY_AGENT
- If image_type is MSK X-RAY → route to FRACTURE_AGENT
- If image_type is COLONOSCOPY FRAME → route to COLON_POLYP_AGENT
- If image_type is FOOT ULCER IMAGE → route to DIABETIC_FOOT_ULCER_AGENT
```

### 6.3 Retrofit run_brain_tumor_agent (currently a stub)

Replace the current `run_brain_tumor_agent` stub with:

```python
def run_brain_tumor_agent(state: AgentState) -> AgentState:
    print("Selected agent: BRAIN_TUMOR_AGENT")
    image_path = state["current_input"].get("image", None)
    result = AgentConfig.image_analyzer.classify_brain_tumor(image_path)
    label      = result.get("label", "unknown")
    confidence = result.get("confidence", 0.0)
    explanation= result.get("explanation", "")
    content = (
        f"**Brain Tumor MRI Analysis**\n\n"
        f"**Detected:** {label.replace('_', ' ').title()} "
        f"(Confidence: {confidence:.1%})\n\n"
        f"{explanation}"
    )
    return {
        **state,
        "output": AIMessage(content=content),
        "needs_human_validation": True,
        "agent_name": "BRAIN_TUMOR_AGENT",
        "artifact_path": None
    }
```

### 6.4 Retrofit run_chest_xray_agent (use structured return)

```python
def run_chest_xray_agent(state: AgentState) -> AgentState:
    print("Selected agent: CHEST_XRAY_AGENT")
    image_path = state["current_input"].get("image", None)
    result = AgentConfig.image_analyzer.classify_chest_xray(image_path)
    label      = result.get("label", "unknown")
    confidence = result.get("confidence", 0.0)
    explanation= result.get("explanation", "")
    if label == "covid19":
        content = f"**COVID-19 Chest X-Ray Analysis**\n\n**Result: POSITIVE for COVID-19** (Confidence: {confidence:.1%})\n\n{explanation}"
    elif label == "normal":
        content = f"**COVID-19 Chest X-Ray Analysis**\n\n**Result: NORMAL** (Confidence: {confidence:.1%})\n\n{explanation}"
    else:
        content = "Could not classify the uploaded chest X-ray image."
    return {
        **state,
        "output": AIMessage(content=content),
        "needs_human_validation": True,
        "agent_name": "CHEST_XRAY_AGENT",
        "artifact_path": None
    }
```

### 6.5 Retrofit run_skin_lesion_agent (use structured return + artifact_path)

```python
def run_skin_lesion_agent(state: AgentState) -> AgentState:
    print("Selected agent: SKIN_LESION_AGENT")
    image_path = state["current_input"].get("image", None)
    result = AgentConfig.image_analyzer.segment_skin_lesion(image_path)
    overlay = result.get("overlay_path")
    explanation = result.get("explanation", "")
    if overlay:
        content = f"**Skin Lesion Segmentation**\n\nSegmentation complete. {explanation}"
    else:
        content = "Could not segment the uploaded skin lesion image."
    return {
        **state,
        "output": AIMessage(content=content),
        "needs_human_validation": True,
        "agent_name": "SKIN_LESION_AGENT",
        "artifact_path": overlay
    }
```

### 6.6 Add All 11 New run_*_agent Functions

Add these after the existing agent functions. Each follows the same pattern.

**TB Agent:**
```python
def run_tb_agent(state: AgentState) -> AgentState:
    print("Selected agent: TB_AGENT")
    image_path = state["current_input"].get("image", None)
    result = AgentConfig.image_analyzer.classify_tb(image_path)
    label, confidence, explanation = result.get("label","unknown"), result.get("confidence",0.0), result.get("explanation","")
    content = f"**Tuberculosis (TB) Chest X-Ray Analysis**\n\n**Result: {label.upper()}** (Confidence: {confidence:.1%})\n\n{explanation}"
    return {**state, "output": AIMessage(content=content), "needs_human_validation": True, "agent_name": "TB_AGENT", "artifact_path": None}
```

**Pneumothorax Agent:**
```python
def run_pneumothorax_agent(state: AgentState) -> AgentState:
    print("Selected agent: PNEUMOTHORAX_AGENT")
    image_path = state["current_input"].get("image", None)
    result = AgentConfig.image_analyzer.segment_pneumothorax(image_path)
    overlay = result.get("overlay_path")
    explanation = result.get("explanation", "")
    content = f"**Pneumothorax Segmentation Analysis**\n\nSegmentation complete. {explanation}"
    return {**state, "output": AIMessage(content=content), "needs_human_validation": True, "agent_name": "PNEUMOTHORAX_AGENT", "artifact_path": overlay}
```

**Multi-label CXR Agent:**
```python
def run_multi_label_cxr_agent(state: AgentState) -> AgentState:
    print("Selected agent: MULTI_LABEL_CXR_AGENT")
    image_path = state["current_input"].get("image", None)
    result = AgentConfig.image_analyzer.classify_multi_label_cxr(image_path)
    findings = result.get("findings", [])
    explanation = result.get("explanation", "")
    positive = [f"**{f['label']}** ({f['confidence']:.1%})" for f in findings if f.get("status") == "positive"]
    summary = "Positive findings: " + ", ".join(positive) if positive else "No positive findings detected."
    content = f"**Multi-label Chest X-Ray Analysis (CheXpert)**\n\n{summary}\n\n{explanation}"
    return {**state, "output": AIMessage(content=content), "needs_human_validation": True, "agent_name": "MULTI_LABEL_CXR_AGENT", "artifact_path": None}
```

**Diabetic Retinopathy Agent:**
```python
def run_diabetic_retinopathy_agent(state: AgentState) -> AgentState:
    print("Selected agent: DIABETIC_RETINOPATHY_AGENT")
    image_path = state["current_input"].get("image", None)
    result = AgentConfig.image_analyzer.classify_diabetic_retinopathy(image_path)
    grade = result.get("grade", "N/A"), result.get("severity_text", ""), result.get("confidence", 0.0), result.get("explanation","")
    content = f"**Diabetic Retinopathy Analysis**\n\n**Grade {grade[0]} — {grade[1]}** (Confidence: {grade[2]:.1%})\n\n{grade[3]}"
    return {**state, "output": AIMessage(content=content), "needs_human_validation": True, "agent_name": "DIABETIC_RETINOPATHY_AGENT", "artifact_path": None}
```

**OCT Agent:**
```python
def run_oct_agent(state: AgentState) -> AgentState:
    print("Selected agent: OCT_AGENT")
    image_path = state["current_input"].get("image", None)
    result = AgentConfig.image_analyzer.classify_oct(image_path)
    label, confidence, explanation = result.get("label","unknown"), result.get("confidence",0.0), result.get("explanation","")
    content = f"**OCT Retinal Scan Analysis**\n\n**Condition: {label.upper()}** (Confidence: {confidence:.1%})\n\n{explanation}"
    return {**state, "output": AIMessage(content=content), "needs_human_validation": True, "agent_name": "OCT_AGENT", "artifact_path": None}
```

**Histopathology Agent:**
```python
def run_histopathology_agent(state: AgentState) -> AgentState:
    print("Selected agent: HISTOPATHOLOGY_AGENT")
    image_path = state["current_input"].get("image", None)
    result = AgentConfig.image_analyzer.classify_histopathology(image_path)
    label, confidence, explanation = result.get("label","unknown"), result.get("confidence",0.0), result.get("explanation","")
    content = f"**Histopathology Tissue Analysis**\n\n**Tissue Class: {label.replace('_',' ').title()}** (Confidence: {confidence:.1%})\n\n{explanation}"
    return {**state, "output": AIMessage(content=content), "needs_human_validation": True, "agent_name": "HISTOPATHOLOGY_AGENT", "artifact_path": None}
```

**Colon Polyp Agent:**
```python
def run_colon_polyp_agent(state: AgentState) -> AgentState:
    print("Selected agent: COLON_POLYP_AGENT")
    image_path = state["current_input"].get("image", None)
    result = AgentConfig.image_analyzer.segment_colon_polyp(image_path)
    overlay, explanation = result.get("overlay_path"), result.get("explanation", "")
    content = f"**Colon Polyp Segmentation**\n\nSegmentation complete. {explanation}"
    return {**state, "output": AIMessage(content=content), "needs_human_validation": True, "agent_name": "COLON_POLYP_AGENT", "artifact_path": overlay}
```

**Diabetic Foot Ulcer Agent:**
```python
def run_diabetic_foot_ulcer_agent(state: AgentState) -> AgentState:
    print("Selected agent: DIABETIC_FOOT_ULCER_AGENT")
    image_path = state["current_input"].get("image", None)
    result = AgentConfig.image_analyzer.classify_diabetic_foot_ulcer(image_path)
    label, confidence, explanation = result.get("label","unknown"), result.get("confidence",0.0), result.get("explanation","")
    content = f"**Diabetic Foot Ulcer Analysis**\n\n**Classification: {label.upper()}** (Confidence: {confidence:.1%})\n\n{explanation}"
    return {**state, "output": AIMessage(content=content), "needs_human_validation": True, "agent_name": "DIABETIC_FOOT_ULCER_AGENT", "artifact_path": None}
```

**Fracture Agent:**
```python
def run_fracture_agent(state: AgentState) -> AgentState:
    print("Selected agent: FRACTURE_AGENT")
    image_path = state["current_input"].get("image", None)
    result = AgentConfig.image_analyzer.classify_fracture(image_path)
    label, confidence, body_part, explanation = result.get("label","unknown"), result.get("confidence",0.0), result.get("body_part","unknown"), result.get("explanation","")
    content = f"**Fracture Detection — {body_part.title()}**\n\n**Result: {label.upper()}** (Confidence: {confidence:.1%})\n\n{explanation}"
    return {**state, "output": AIMessage(content=content), "needs_human_validation": True, "agent_name": "FRACTURE_AGENT", "artifact_path": None}
```

**Mammography Agent:**
```python
def run_mammography_agent(state: AgentState) -> AgentState:
    print("Selected agent: MAMMOGRAPHY_AGENT")
    image_path = state["current_input"].get("image", None)
    result = AgentConfig.image_analyzer.classify_mammography(image_path)
    label, confidence, finding_type, explanation = result.get("label","unknown"), result.get("confidence",0.0), result.get("finding_type","unknown"), result.get("explanation","")
    content = f"**Mammography Analysis**\n\n**Finding Type: {finding_type.title()}** — **{label.upper()}** (Confidence: {confidence:.1%})\n\n{explanation}"
    return {**state, "output": AIMessage(content=content), "needs_human_validation": True, "agent_name": "MAMMOGRAPHY_AGENT", "artifact_path": None}
```

**HAM10000 Agent:**
```python
def run_ham10000_agent(state: AgentState) -> AgentState:
    print("Selected agent: HAM10000_AGENT")
    image_path = state["current_input"].get("image", None)
    result = AgentConfig.image_analyzer.classify_ham10000(image_path)
    label, confidence, explanation = result.get("label","unknown"), result.get("confidence",0.0), result.get("explanation","")
    content = f"**Dermatology Classification (HAM10000)**\n\n**Lesion Type: {label.replace('_',' ').title()}** (Confidence: {confidence:.1%})\n\n{explanation}"
    return {**state, "output": AIMessage(content=content), "needs_human_validation": True, "agent_name": "HAM10000_AGENT", "artifact_path": None}
```

### 6.7 Add All New Nodes and Edges to Workflow Graph

After the existing `workflow.add_node(...)` calls, add:
```python
workflow.add_node("TB_AGENT",                   run_tb_agent)
workflow.add_node("PNEUMOTHORAX_AGENT",          run_pneumothorax_agent)
workflow.add_node("MULTI_LABEL_CXR_AGENT",       run_multi_label_cxr_agent)
workflow.add_node("DIABETIC_RETINOPATHY_AGENT",  run_diabetic_retinopathy_agent)
workflow.add_node("MAMMOGRAPHY_AGENT",           run_mammography_agent)
workflow.add_node("HISTOPATHOLOGY_AGENT",        run_histopathology_agent)
workflow.add_node("FRACTURE_AGENT",              run_fracture_agent)
workflow.add_node("OCT_AGENT",                   run_oct_agent)
workflow.add_node("DIABETIC_FOOT_ULCER_AGENT",   run_diabetic_foot_ulcer_agent)
workflow.add_node("COLON_POLYP_AGENT",           run_colon_polyp_agent)
workflow.add_node("HAM10000_AGENT",              run_ham10000_agent)
```

Extend the `add_conditional_edges` routing table for `route_to_agent`:
```python
{
    "CONVERSATION_AGENT":           "CONVERSATION_AGENT",
    "RAG_AGENT":                    "RAG_AGENT",
    "WEB_SEARCH_PROCESSOR_AGENT":   "WEB_SEARCH_PROCESSOR_AGENT",
    "BRAIN_TUMOR_AGENT":            "BRAIN_TUMOR_AGENT",
    "CHEST_XRAY_AGENT":             "CHEST_XRAY_AGENT",
    "SKIN_LESION_AGENT":            "SKIN_LESION_AGENT",
    "TB_AGENT":                     "TB_AGENT",
    "PNEUMOTHORAX_AGENT":           "PNEUMOTHORAX_AGENT",
    "MULTI_LABEL_CXR_AGENT":        "MULTI_LABEL_CXR_AGENT",
    "DIABETIC_RETINOPATHY_AGENT":   "DIABETIC_RETINOPATHY_AGENT",
    "MAMMOGRAPHY_AGENT":            "MAMMOGRAPHY_AGENT",
    "HISTOPATHOLOGY_AGENT":         "HISTOPATHOLOGY_AGENT",
    "FRACTURE_AGENT":               "FRACTURE_AGENT",
    "OCT_AGENT":                    "OCT_AGENT",
    "DIABETIC_FOOT_ULCER_AGENT":    "DIABETIC_FOOT_ULCER_AGENT",
    "COLON_POLYP_AGENT":            "COLON_POLYP_AGENT",
    "HAM10000_AGENT":               "HAM10000_AGENT",
    "needs_validation":             "RAG_AGENT"
}
```

Add edges for all new nodes (connect to check_validation, same as existing):
```python
workflow.add_edge("TB_AGENT",                   "check_validation")
workflow.add_edge("PNEUMOTHORAX_AGENT",          "check_validation")
workflow.add_edge("MULTI_LABEL_CXR_AGENT",       "check_validation")
workflow.add_edge("DIABETIC_RETINOPATHY_AGENT",  "check_validation")
workflow.add_edge("MAMMOGRAPHY_AGENT",           "check_validation")
workflow.add_edge("HISTOPATHOLOGY_AGENT",        "check_validation")
workflow.add_edge("FRACTURE_AGENT",              "check_validation")
workflow.add_edge("OCT_AGENT",                   "check_validation")
workflow.add_edge("DIABETIC_FOOT_ULCER_AGENT",   "check_validation")
workflow.add_edge("COLON_POLYP_AGENT",           "check_validation")
workflow.add_edge("HAM10000_AGENT",              "check_validation")
```

Also update `perform_human_validation` to preserve `artifact_path`:
```python
def perform_human_validation(state: AgentState) -> AgentState:
    validation_prompt = f"{state['output'].content}\n\n**Human Validation Required:**\n- Healthcare professional: Please validate. Select **Yes** or **No**. If No, provide comments.\n- Patient: Click Yes to confirm."
    return {
        **state,
        "output": AIMessage(content=validation_prompt),
        "agent_name": f"{state['agent_name']}, HUMAN_VALIDATION",
        "artifact_path": state.get("artifact_path")   # preserve it
    }
```

---

## Assignment 7: app.py — Generic Artifact Mechanism

### 7.1 Current Problem (lines 120–125 and 188–193)

Both `/chat` and `/upload` handlers have:
```python
if response_data["agent_name"] == "SKIN_LESION_AGENT, HUMAN_VALIDATION":
    segmentation_path = os.path.join(SKIN_LESION_OUTPUT, "segmentation_plot.png")
    if os.path.exists(segmentation_path):
        result["result_image"] = f"/uploads/skin_lesion_output/segmentation_plot.png"
```

This must be replaced in BOTH handlers.

### 7.2 Add New Output Directories

In the directory setup block at the top of `app.py`, add:
```python
PNEUMOTHORAX_OUTPUT   = "uploads/pneumothorax_output"
COLON_POLYP_OUTPUT    = "uploads/colon_polyp_output"

for directory in [UPLOAD_FOLDER, FRONTEND_UPLOAD_FOLDER, SKIN_LESION_OUTPUT,
                  SPEECH_DIR, PNEUMOTHORAX_OUTPUT, COLON_POLYP_OUTPUT]:
    os.makedirs(directory, exist_ok=True)
```

### 7.3 Replace Both Hardcoded Blocks

Replace the skin-lesion check in `/chat` AND `/upload` with the same generic block:

```python
# Generic artifact path from any segmentation agent
artifact_path = response_data.get("artifact_path")
if artifact_path and os.path.exists(artifact_path):
    # Convert filesystem path to URL path: strip leading "./" or "/"
    url_path = "/" + artifact_path.lstrip("./").lstrip("/")
    result["result_image"] = url_path
```

Remove the `SKIN_LESION_OUTPUT` constant usage in handler logic (it's still created as a directory but the URL is now derived from `artifact_path` directly).

---

## Integration Process (For Receiving Deliveries from Team)

### Step 1 — Receive delivery from each person

Each person gives you:
- Their agent folder(s) placed under `agents/image_analysis_agent/`
- Their trained `.pth` checkpoint under `agents/image_analysis_agent/<agent>_agent/models/`
- Their `label_map.json`
- Their integration key document (see template in their respective work files)

### Step 2 — Verify the contract before wiring

Run this for each new agent:
```python
# Quick contract check — run from project root
from agents.image_analysis_agent.<person>_agent.<file> import <ClassName>
agent = <ClassName>(model_path="<path>")
result = agent.predict("<test_image_path>")
assert isinstance(result, dict), "predict() must return a dict"
assert "label" in result or "findings" in result, "missing label/findings"
assert "confidence" in result or "findings" in result, "missing confidence"
assert result.get("needs_validation") == True, "needs_validation must be True"
print("CONTRACT OK:", result)
```

### Step 3 — Wire per the assignments above (Assignments 3–7)

Wire in this order:
1. config.py model paths + validation flags (Assignment 3)
2. __init__.py imports + methods (Assignment 5)  
3. agent_decision.py nodes + edges + run_* functions (Assignment 6)
4. Test via /upload endpoint with a real image
5. Verify artifact_path flows correctly for segmentation agents

### Step 4 — End-to-end test

```bash
# Start the server
uvicorn app:app --host 0.0.0.0 --port 8001

# Test each agent via curl
curl -X POST http://localhost:8001/upload \
  -F "image=@./data/brain_tumor/test/glioma/some_image.jpg" \
  -F "text=analyze this brain MRI"
```

---

## Your Final Report — Mandatory Format

After completing all assignments, write `Work/Amritesh_Report.md` with the following sections:

### Section 1: Brain Tumor Agent
- Training dataset details (number of images per class, split sizes)
- Model architecture (EfficientNet-B3 + head)
- Training hyperparameters (lr, batch size, epochs, optimizer)
- Final test metrics: Accuracy, Precision, Recall, F1, AUC per class
- Checkpoint path and file size
- Any preprocessing deviations from the plan

### Section 2: Architecture Changes
- List every file modified and what changed
- Before/after for the app.py artifact mechanism
- Before/after for the agent_decision.py routing table
- Screenshot or curl output demonstrating the brain tumor agent working end-to-end

### Section 3: Integration Status per Person
For each of Sanika / Rushil / Jeet / Srujan:
- Date received
- Contract check result (pass/fail)
- Wiring status (done/pending)
- Any contract violations found and how they were resolved

### Section 4: System Test Results
- Table: Agent Name | Test Image Used | Predicted Output | Pass/Fail
- For segmentation agents: verify `result_image` appears in the /upload response JSON

### Section 5: Deliverables Checklist
- [ ] brain_tumor_inference.py — real implementation
- [ ] brain_tumor_efficientnet.pth — trained checkpoint
- [ ] brain_tumor train.py — runnable training script
- [ ] metrics.json — test set metrics
- [ ] config.py — all 14 agents wired
- [ ] image_classifier.py — 12 modalities + threshold
- [ ] __init__.py — all 15 agents registered
- [ ] agent_decision.py — full routing + all nodes/edges
- [ ] app.py — generic artifact mechanism

---

## Completion Test Checklist

**The job is not complete until every checkbox below is ticked. Each item is a pass/fail gate.**

### T1 — Brain Tumor Agent

- [ ] `BrainTumorClassification("path/to/brain_tumor_efficientnet.pth")` instantiates without error
- [ ] Model is in `eval()` mode after `__init__` returns (`model.training == False`)
- [ ] `predict("valid_image.jpg")` returns a Python `dict` (use `assert isinstance(result, dict)`)
- [ ] Returned dict contains exactly these keys: `label`, `confidence`, `needs_validation`, `explanation`
- [ ] `result["label"]` is one of: `"glioma"`, `"meningioma"`, `"no_tumor"`, `"pituitary"`
- [ ] `result["confidence"]` is a Python `float` with `0.0 <= result["confidence"] <= 1.0`
- [ ] `result["needs_validation"]` is the boolean `True` (not the string `"True"`, not `1`)
- [ ] `result["explanation"]` is a non-empty string
- [ ] `predict("nonexistent_path.jpg")` does NOT raise an exception — returns error dict with `confidence=0.0`
- [ ] `predict("corrupt_file.bin")` does NOT raise an exception — returns error dict
- [ ] Run predict on one image per class (4 images total); all return valid dicts
- [ ] `models/brain_tumor_efficientnet.pth` exists on disk
- [ ] `models/label_map.json` exists and contains exactly 4 entries
- [ ] `models/metrics.json` exists and contains: `accuracy`, `precision`, `recall`, `f1`, `auc`
- [ ] `metrics.json` accuracy >= 85% (if below, retrain or document the gap)
- [ ] `train.py` runs from scratch without error (smoke test: 1 epoch, small subset)

### T2 — Retrofit: covid_chest_xray_inference.py

- [ ] `ChestXRayClassification.predict(image_path)` returns a `dict` (previously returned a `str`)
- [ ] Returned dict keys: `label`, `confidence`, `needs_validation`, `explanation`
- [ ] `result["label"]` is one of: `"covid19"`, `"normal"`
- [ ] `result["confidence"]` is `float`, `0.0 <= x <= 1.0`
- [ ] `result["needs_validation"]` is boolean `True`
- [ ] `EXPLANATION_MAP` is defined as a module-level constant above the class
- [ ] No other logic in `predict()` was changed — only the return statement

### T3 — Retrofit: skin_lesion_inference.py

- [ ] `SkinLesionSegmentation.predict(image_path, output_path)` returns a `dict` (previously returned `bool`)
- [ ] Returned dict keys: `label`, `confidence`, `mask_path`, `overlay_path`, `needs_validation`, `explanation`
- [ ] `result["label"]` is `"lesion_segmented"` or `"no_lesion_detected"`
- [ ] `result["mask_path"]` and `result["overlay_path"]` are strings (not empty on success)
- [ ] `result["confidence"]` is float `0.0 <= x <= 1.0`
- [ ] `result["needs_validation"]` is boolean `True`
- [ ] A PNG file is actually written to `output_path` when segmentation succeeds
- [ ] predict() does NOT raise on error — returns error dict

### T4 — config.py

- [ ] `MedicalCVConfig` has exactly these 14 model path fields (verify attribute names match exactly):
  - `brain_tumor_model_path`, `chest_xray_model_path`, `skin_lesion_model_path`
  - `tb_model_path`, `pneumothorax_model_path`, `multi_label_cxr_model_path`
  - `diabetic_retinopathy_model_path`, `oct_model_path`
  - `histopathology_model_path`, `colon_polyp_model_path`, `diabetic_foot_ulcer_model_path`
  - `fracture_model_path`, `mammography_model_path`, `ham10000_model_path`
- [ ] `MedicalCVConfig` has exactly 3 output path fields: `skin_lesion_output_path`, `pneumothorax_output_path`, `colon_polyp_output_path`
- [ ] `ValidationConfig.require_validation` dict has exactly 17 entries
- [ ] `from config import config` works at top level without error
- [ ] `config.medical_cv.brain_tumor_model_path` resolves to the correct relative path string

### T5 — image_classifier.py

- [ ] `classify_image()` prompt lists exactly 12 modality names (count them in the prompt string)
- [ ] `CONFIDENCE_THRESHOLD = 0.65` is defined as module-level constant
- [ ] When LLM returns confidence < 0.65, `image_type` is overridden to `"OTHER"`
- [ ] `classify_image()` returns a dict with keys: `image_type`, `reasoning`, `confidence`
- [ ] `image_type` is one of the 12 valid strings or `"OTHER"`
- [ ] JSON extraction fallback (curly-brace regex) still works — do not remove it

### T6 — `__init__.py`

- [ ] All 15 inference classes are imported at the top of the file
- [ ] `ImageAnalysisAgent.__init__()` instantiates all 15 agents using `config.medical_cv.*` paths
- [ ] All 14 method names exist on `ImageAnalysisAgent` (list from Section 4.6 in this doc)
- [ ] Each method returns the dict from `predict()` without modification
- [ ] `from agents.image_analysis_agent import ImageAnalysisAgent` works without error
- [ ] `ImageAnalysisAgent` constructor does NOT crash if a checkpoint file is missing — it should log an error and raise, not silently fail

### T7 — agent_decision.py

- [ ] `AgentState` has `artifact_path: Optional[str]` field
- [ ] `DECISION_SYSTEM_PROMPT` lists all 17 agent names (grep for each name)
- [ ] `run_brain_tumor_agent` function exists
- [ ] `run_tb_agent` function exists
- [ ] `run_pneumothorax_agent` function exists
- [ ] `run_multi_label_cxr_agent` function exists
- [ ] `run_diabetic_retinopathy_agent` function exists
- [ ] `run_oct_agent` function exists
- [ ] `run_histopathology_agent` function exists
- [ ] `run_colon_polyp_agent` function exists
- [ ] `run_diabetic_foot_ulcer_agent` function exists
- [ ] `run_fracture_agent` function exists
- [ ] `run_mammography_agent` function exists
- [ ] `run_ham10000_agent` function exists
- [ ] `workflow.add_node(...)` call exists for all 14 image agent nodes
- [ ] `routing_map` dict in the conditional function has exactly 17 keys
- [ ] `perform_human_validation` does NOT clear `state["artifact_path"]` — pass-through confirmed
- [ ] For all 3 segmentation agents, the run_*_agent function sets `state["artifact_path"]`
- [ ] LangGraph `workflow.compile()` runs without error

### T8 — app.py

- [ ] Search `app.py` for the string `"SKIN_LESION_AGENT, HUMAN_VALIDATION"` — it must NOT appear anywhere in the file
- [ ] Generic `artifact_path` block exists in the `/chat` handler
- [ ] Generic `artifact_path` block exists in the `/upload` handler
- [ ] Both blocks use `response_data.get("artifact_path")` (not dict key access without `.get()`)
- [ ] Both blocks check `os.path.exists(artifact_path)` before building the URL
- [ ] `os.makedirs(config.medical_cv.pneumothorax_output_path, exist_ok=True)` is in startup
- [ ] `os.makedirs(config.medical_cv.colon_polyp_output_path, exist_ok=True)` is in startup

### T9 — End-to-End System Tests

Run `uvicorn app:app --reload` and test each of the following via curl or the browser UI:

- [ ] POST `/upload` with a brain MRI image → response JSON contains `analysis_result.label` in `{glioma, meningioma, no_tumor, pituitary}`
- [ ] POST `/upload` with a chest X-ray image → response JSON contains `analysis_result.label` in `{covid19, normal}`
- [ ] POST `/upload` with a skin lesion image → response JSON contains `result_image` URL pointing to overlay PNG
- [ ] POST `/upload` skin lesion → GET the `result_image` URL → HTTP 200 and valid PNG returned
- [ ] POST `/chat` with `"What is the treatment for pneumonia?"` → returns text response (RAG or web)
- [ ] POST `/upload` with any image where `artifact_path` is empty → `result_image` key is NOT present in response JSON (no null key)
- [ ] Server starts without error even if no GPU is present (CPU fallback)
- [ ] Server starts without error even before new agent checkpoints are present (lazy loading or graceful error log)
- [ ] All team deliveries integrated and tested

# Multi-Agent Medical Assistant — Work Division Document

**Version**: 1.0  
**Date**: 2026-05-13  
**Audience**: AI coding agents and human developers  
**Purpose**: Master reference for the full extension of the Medical Assistant system from 3 agents to 17 agents. Every person's responsibilities, deliverable contracts, file paths, class names, and integration handoff procedures are defined here with zero ambiguity.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Team and Role Assignment Summary](#2-team-and-role-assignment-summary)
3. [Global Conventions — Mandatory for All Contributors](#3-global-conventions--mandatory-for-all-contributors)
4. [Amritesh — Core System & Brain Tumor Agent](#4-amritesh--core-system--brain-tumor-agent)
5. [Sanika — Pulmonary & Chest Imaging Agents](#5-sanika--pulmonary--chest-imaging-agents)
6. [Rushil — Ophthalmology Imaging Agents](#6-rushil--ophthalmology-imaging-agents)
7. [Jeet — Pathology, GI & Wound Imaging Agents](#7-jeet--pathology-gi--wound-imaging-agents)
8. [Srujan — Musculoskeletal, Breast & Extended Dermatology Agents](#8-srujan--musculoskeletal-breast--extended-dermatology-agents)
9. [Integration Protocol — How All Pieces Connect](#9-integration-protocol--how-all-pieces-connect)
10. [Final Report Requirements](#10-final-report-requirements)
11. [Agent Routing Table — Complete Reference](#11-agent-routing-table--complete-reference)
12. [Output Contract Reference — All 17 Agents](#12-output-contract-reference--all-17-agents)

---

## 1. System Architecture Overview

The system is a FastAPI web application backed by a LangGraph multi-agent pipeline.

### Entry Points

- **`app.py`** — FastAPI server. Two upload endpoints: `/chat` (text) and `/upload` (image + text). Both call `process_query()` and return JSON to the frontend.
- **`agents/agent_decision.py`** — LangGraph `StateGraph`. The `process_query()` function lives here. It routes user input to the correct agent node based on LLM decision and image classification.
- **`config.py`** — `MedicalCVConfig` dataclass holds all model checkpoint paths and output directories. `ValidationConfig` holds per-agent validation flags. All agents read their model paths from here — they do NOT hardcode paths.

### LangGraph Data Flow

```
User Input (text + optional image)
       ↓
ImageClassifier (vision LLM, 12-class modality detection)
       ↓
AgentState populated: {messages, image_path, agent_name, analysis_result, artifact_path, ...}
       ↓
Decision Node (LLM picks agent name from 17 possible values)
       ↓
Agent Node (runs inference, writes result to AgentState)
       ↓
[Optional] Human Validation Node (if needs_validation == True)
       ↓
process_query() returns response_data dict to app.py
       ↓
app.py reads artifact_path generically, returns JSON to frontend
```

### AgentState Fields (after Amritesh's extension)

```python
class AgentState(MessagesState):
    user_query:      str
    image_path:      Optional[str]
    agent_name:      str
    analysis_result: Optional[dict]    # structured output from predict()
    artifact_path:   Optional[str]     # overlay/mask image path (segmentation agents only)
    web_search:      bool
    rag_results:     Optional[list]
    validation_required: bool
```

### Directory Structure (full project)

```
Multi-Agent-Medical-Assistant/
├── app.py
├── config.py
├── requirements.txt
├── agents/
│   ├── agent_decision.py
│   ├── guardrails/
│   │   └── local_guardrails.py
│   ├── image_analysis_agent/
│   │   ├── __init__.py                           ← ImageAnalysisAgent class
│   │   ├── image_classifier.py                   ← 12-modality ImageClassifier
│   │   ├── brain_tumor_agent/
│   │   │   ├── brain_tumor_inference.py
│   │   │   ├── train.py
│   │   │   └── models/
│   │   │       ├── brain_tumor_efficientnet.pth
│   │   │       ├── label_map.json
│   │   │       └── metrics.json
│   │   ├── chest_xray_agent/
│   │   │   └── covid_chest_xray_inference.py     ← retrofit to structured output
│   │   ├── skin_lesion_agent/
│   │   │   └── skin_lesion_inference.py          ← retrofit to structured output
│   │   ├── tb_agent/                             ← Sanika
│   │   ├── pneumothorax_agent/                   ← Sanika
│   │   ├── multi_label_cxr_agent/                ← Sanika
│   │   ├── diabetic_retinopathy_agent/           ← Rushil
│   │   ├── oct_agent/                            ← Rushil
│   │   ├── histopathology_agent/                 ← Jeet
│   │   ├── colon_polyp_agent/                    ← Jeet
│   │   ├── diabetic_foot_ulcer_agent/            ← Jeet
│   │   ├── fracture_agent/                       ← Srujan
│   │   ├── mammography_agent/                    ← Srujan
│   │   └── ham10000_agent/                       ← Srujan
│   ├── rag_agent/
│   └── web_search_processor_agent/
├── templates/
│   └── index.html
├── uploads/
│   ├── skin_lesion_output/
│   ├── pneumothorax_output/                      ← create at startup
│   └── colon_polyp_output/                       ← create at startup
└── Work/
    ├── WORK_DIVISION.md                          ← this file
    ├── Amritesh.md
    ├── Sanika.md
    ├── Rushil.md
    ├── Jeet.md
    └── Srujan.md
```

---

## 2. Team and Role Assignment Summary

| Person    | Role                                 | Agents Owned                                           | Files to Create/Modify                                                                                                          |
|-----------|--------------------------------------|--------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| Amritesh  | Core System & Brain Tumor Agent      | BRAIN_TUMOR_AGENT + system wiring                      | `brain_tumor_inference.py`, `train.py`, retrofit `covid_chest_xray_inference.py` + `skin_lesion_inference.py`, `config.py`, `image_classifier.py`, `__init__.py`, `agent_decision.py`, `app.py` |
| Sanika    | Pulmonary & Chest Imaging            | TB_AGENT, PNEUMOTHORAX_AGENT, MULTI_LABEL_CXR_AGENT   | `tb_agent/`, `pneumothorax_agent/`, `multi_label_cxr_agent/` folders + inference + train scripts                               |
| Rushil    | Ophthalmology Imaging                | DIABETIC_RETINOPATHY_AGENT, OCT_AGENT                  | `diabetic_retinopathy_agent/`, `oct_agent/` folders + inference + train scripts                                                 |
| Jeet      | Pathology, GI & Wound Imaging        | HISTOPATHOLOGY_AGENT, COLON_POLYP_AGENT, DIABETIC_FOOT_ULCER_AGENT | `histopathology_agent/`, `colon_polyp_agent/`, `diabetic_foot_ulcer_agent/` folders + inference + train scripts  |
| Srujan    | Musculoskeletal, Breast & Dermatology| FRACTURE_AGENT, MAMMOGRAPHY_AGENT, HAM10000_AGENT       | `fracture_agent/`, `mammography_agent/`, `ham10000_agent/` folders + inference + train scripts                                  |

**Integration ownership**: Amritesh owns `config.py`, `__init__.py`, `agent_decision.py`, and `app.py`. No other contributor modifies those 4 files.

**Deliverable ownership**: Sanika, Rushil, Jeet, and Srujan each own their agent sub-folders only. They deliver self-contained inference wrappers and checkpoints. Amritesh wires them in.

---

## 3. Global Conventions — Mandatory for All Contributors

These rules apply to every file written by every contributor. Violations will block integration.

### 3.1 Python Version

Python 3.10+. Use `from __future__ import annotations` if you use PEP 604 union types (`X | Y`).

### 3.2 Device Handling

Every inference class must use:
```python
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
Never hardcode `"cpu"` or `"cuda"`.

### 3.3 Model Loading

Every `__init__` must call `self.model.eval()` after loading weights. Never leave the model in training mode at inference time.

Weight loading must use `map_location=self.device` to support CPU-only machines:
```python
self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
```

### 3.4 Inference — No Gradient

All inference code must be wrapped in `torch.no_grad()`:
```python
with torch.no_grad():
    output = self.model(tensor)
```

### 3.5 Output Contracts — Non-Negotiable

Every `predict()` method must return a Python `dict`. The exact keys depend on the agent type (classification vs. segmentation). See Section 12 for the complete reference. The following rules apply universally:

- `"confidence"` must be a Python `float` in range `[0.0, 1.0]`, rounded to 4 decimal places: `round(float(value), 4)`
- `"needs_validation"` must be the boolean literal `True` — not a string, not `1`
- `"label"` must be a Python `str`
- The `predict()` method must never raise an exception to the caller. Wrap the body in `try/except` and return an error dict with `"confidence": 0.0` and `"needs_validation": True`

### 3.6 Logging

Use Python's built-in `logging` module. Do not use `print()` in production inference files.
```python
import logging
logger = logging.getLogger(__name__)
```

### 3.7 Model Checkpoint Paths

Checkpoints are stored in `models/` subdirectories within each agent folder. Paths are always passed via `__init__(self, model_path: str)` — never hardcoded inside the class.

### 3.8 Preprocessing Constants

Define `IMAGE_SIZE`, `MEAN`, and `STD` as module-level constants in every inference file. The constants used in training must exactly match those used in inference. A mismatch will silently corrupt predictions.

Standard ImageNet normalization:
```python
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
```

### 3.9 Training Script Requirements

Every `train.py` must include at the top:
```python
DATASET_SOURCE  = "<URL or citation>"
RANDOM_SEED     = 42
```
And must call:
```python
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
```

Every `train.py` must save a `metrics.json` at the end with at minimum: `accuracy`, `precision`, `recall`, `f1`, `auc` (where applicable).

### 3.10 Folder Structure per Agent

Every agent folder must follow this exact layout:
```
agents/image_analysis_agent/<agent_name>_agent/
├── <agent_name>_inference.py      ← inference wrapper class
├── train.py                       ← training script
└── models/
    ├── <checkpoint>.pth           ← trained PyTorch checkpoint
    ├── label_map.json             ← {index_string: class_name} mapping
    └── metrics.json               ← final test set metrics
```

### 3.11 Standalone Test

Before handing off to Amritesh, every inference class must pass this test pattern:
```python
clf = YourClassName("./path/to/model.pth")
result = clf.predict("./path/to/test_image.jpg")
assert isinstance(result, dict), "predict() must return dict"
assert result["needs_validation"] == True, "needs_validation must be True"
assert isinstance(result["confidence"], float), "confidence must be float"
assert 0.0 <= result["confidence"] <= 1.0, "confidence out of range"
print(result)
```

---

## 4. Amritesh — Core System & Brain Tumor Agent

### 4.1 Scope

Amritesh is responsible for:
1. Implementing `BRAIN_TUMOR_AGENT` from scratch (currently `# TBD`)
2. Retrofitting the two existing agents to return structured output dicts
3. Expanding `config.py` with all 11 new model paths and output directories
4. Expanding `image_classifier.py` from 5 to 12 modalities
5. Expanding `agents/image_analysis_agent/__init__.py` to register all 15 agents
6. Expanding `agents/agent_decision.py` to route all 17 agents in the LangGraph graph
7. Fixing `app.py` to use a generic `artifact_path` mechanism
8. Receiving deliverables from Sanika, Rushil, Jeet, Srujan and wiring each one into the system

### 4.2 Assignment A: Brain Tumor Agent

**Target directory**: `agents/image_analysis_agent/brain_tumor_agent/`

**Dataset**: Brain Tumor MRI Dataset — Kaggle (Masoud Nickparvar)  
`kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset`

**Classes (4)**: `glioma`, `meningioma`, `no_tumor`, `pituitary`

**Model architecture**: EfficientNet-B3 (torchvision)  
`IMAGE_SIZE = 300`

**Checkpoint path**: `agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_efficientnet.pth`

**label_map.json**:
```json
{"0": "glioma", "1": "meningioma", "2": "no_tumor", "3": "pituitary"}
```

**Class to implement** in `brain_tumor_inference.py`:

```python
class BrainTumorClassification:
    def __init__(self, model_path: str): ...
    def _build_model(self) -> nn.Module: ...
    def _load_weights(self) -> None: ...
    def predict(self, image_path: str) -> dict:
        # Returns:
        # {
        #     "label":            str,    # one of CLASS_NAMES
        #     "confidence":       float,
        #     "needs_validation": True,
        #     "explanation":      str
        # }
```

**Training**: EfficientNet-B3, 30 epochs, Adam(lr=1e-4), CrossEntropyLoss, batch=32.  
Train on 80% of Kaggle train set; validate on 20%. Save best val accuracy checkpoint.

### 4.3 Assignment B: Retrofit Existing Agents

#### B1 — `agents/image_analysis_agent/chest_xray_agent/covid_chest_xray_inference.py`

Current `predict()` returns a raw string `pred_class`.  
Replace the `return pred_class` line (currently line ~76) with:

```python
EXPLANATION_MAP = {
    'covid19': 'Bilateral peripheral ground-glass opacities consistent with COVID-19 pneumonia. Requires clinical correlation and PCR confirmation.',
    'normal':  'No abnormalities detected. Lung fields appear clear.'
}

# Replace return statement:
confidence = float(torch.softmax(out, dim=1).max().item())
return {
    "label":            pred_class,
    "confidence":       round(confidence, 4),
    "needs_validation": True,
    "explanation":      EXPLANATION_MAP.get(pred_class, "")
}
```

Also add `EXPLANATION_MAP` as a module-level constant above the class.

#### B2 — `agents/image_analysis_agent/skin_lesion_agent/skin_lesion_inference.py`

Current `predict(image_path, output_path)` returns `True` or `False`.  
Replace to return a dict:

```python
def predict(self, image_path: str, output_path: str) -> dict:
    try:
        # ... existing segmentation logic ...
        mask_confidence = float(mask.mean())
        overlay_success = self._overlay_mask(image_path, mask, output_path)
        label = "lesion_segmented" if overlay_success else "no_lesion_detected"
        return {
            "label":            label,
            "confidence":       round(mask_confidence, 4),
            "mask_path":        output_path,
            "overlay_path":     output_path,
            "needs_validation": True,
            "explanation":      "Skin lesion segmentation complete. Overlay saved. Requires dermatologist review."
        }
    except Exception as e:
        logger.error(f"Skin lesion inference error: {e}")
        return {
            "label":            "unknown",
            "confidence":       0.0,
            "mask_path":        "",
            "overlay_path":     "",
            "needs_validation": True,
            "explanation":      f"Inference failed: {str(e)}"
        }
```

### 4.4 Assignment C: config.py Extensions

In `config.py`, expand `MedicalCVConfig` dataclass to include all new model paths:

```python
@dataclass
class MedicalCVConfig:
    # Existing
    brain_tumor_model_path: str = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_efficientnet.pth"
    chest_xray_model_path:  str = "./agents/image_analysis_agent/chest_xray_agent/models/covid_densenet.pth"
    skin_lesion_model_path: str = "./agents/image_analysis_agent/skin_lesion_agent/models/skin_lesion_unet.pth"

    # New — Sanika
    tb_model_path:               str = "./agents/image_analysis_agent/tb_agent/models/tb_densenet121.pth"
    pneumothorax_model_path:     str = "./agents/image_analysis_agent/pneumothorax_agent/models/pneumothorax_unet.pth"
    multi_label_cxr_model_path:  str = "./agents/image_analysis_agent/multi_label_cxr_agent/models/multi_label_cxr_efficientnet.pth"

    # New — Rushil
    diabetic_retinopathy_model_path: str = "./agents/image_analysis_agent/diabetic_retinopathy_agent/models/diabetic_retinopathy_efficientnet.pth"
    oct_model_path:              str = "./agents/image_analysis_agent/oct_agent/models/oct_efficientnet.pth"

    # New — Jeet
    histopathology_model_path:   str = "./agents/image_analysis_agent/histopathology_agent/models/histopathology_efficientnet.pth"
    colon_polyp_model_path:      str = "./agents/image_analysis_agent/colon_polyp_agent/models/colon_polyp_unet.pth"
    diabetic_foot_ulcer_model_path: str = "./agents/image_analysis_agent/diabetic_foot_ulcer_agent/models/diabetic_foot_ulcer_efficientnet.pth"

    # New — Srujan
    fracture_model_path:         str = "./agents/image_analysis_agent/fracture_agent/models/fracture_densenet.pth"
    mammography_model_path:      str = "./agents/image_analysis_agent/mammography_agent/models/mammography_efficientnet.pth"
    ham10000_model_path:         str = "./agents/image_analysis_agent/ham10000_agent/models/ham10000_efficientnet.pth"

    # Output paths for segmentation agents
    skin_lesion_output_path:     str = "./uploads/skin_lesion_output"
    pneumothorax_output_path:    str = "./uploads/pneumothorax_output"
    colon_polyp_output_path:     str = "./uploads/colon_polyp_output"
```

Expand `ValidationConfig.require_validation` dict to include all 17 agents:
```python
require_validation: dict = field(default_factory=lambda: {
    "BRAIN_TUMOR_AGENT":             True,
    "CHEST_XRAY_AGENT":              True,
    "SKIN_LESION_AGENT":             True,
    "TB_AGENT":                      True,
    "PNEUMOTHORAX_AGENT":            True,
    "MULTI_LABEL_CXR_AGENT":         True,
    "DIABETIC_RETINOPATHY_AGENT":    True,
    "OCT_AGENT":                     True,
    "HISTOPATHOLOGY_AGENT":          True,
    "COLON_POLYP_AGENT":             True,
    "DIABETIC_FOOT_ULCER_AGENT":     True,
    "FRACTURE_AGENT":                True,
    "MAMMOGRAPHY_AGENT":             True,
    "HAM10000_AGENT":                True,
    "RAG_AGENT":                     False,
    "WEB_SEARCH_AGENT":              False,
    "GENERAL_AGENT":                 False,
})
```

### 4.5 Assignment D: image_classifier.py Extension

In `agents/image_analysis_agent/image_classifier.py`, update the vision LLM prompt inside `classify_image()` to list all 12 modalities. The current prompt lists only 5.

**New `CONFIDENCE_THRESHOLD = 0.65`** — if returned confidence is below this, override to `"OTHER"`.

New modality list for the prompt:
```
1. BRAIN_MRI       — MRI scan of the brain/head
2. CHEST_XRAY      — Chest radiograph (X-ray)
3. SKIN_LESION     — Dermatoscopy or clinical skin lesion photo
4. FUNDUS          — Fundus/retinal photograph
5. OCT             — Optical coherence tomography cross-section
6. HISTOPATHOLOGY  — H&E stained tissue section microscopy
7. COLONOSCOPY     — Colonoscopy/endoscopy video frame
8. FOOT_WOUND      — Diabetic foot ulcer or wound photograph
9. MUSCULOSKELETAL — Musculoskeletal X-ray (hand, wrist, elbow, shoulder, etc.)
10. MAMMOGRAPHY    — Mammography image (breast)
11. SKIN_DERM      — Dermoscopy image for multi-class lesion classification
12. OTHER          — None of the above
```

After JSON extraction, add:
```python
if result.get("confidence", 1.0) < CONFIDENCE_THRESHOLD:
    result["image_type"] = "OTHER"
```

### 4.6 Assignment E: `__init__.py` Extension

In `agents/image_analysis_agent/__init__.py`, expand `ImageAnalysisAgent` to instantiate and expose all 15 agents (existing 3 + 12 new ones). The class must:

1. Import all 15 inference classes in `__init__`
2. Instantiate each using the corresponding path from `config.medical_cv`
3. Expose one method per agent (e.g., `classify_brain_tumor(image_path)`, `classify_tb(image_path)`, `segment_pneumothorax(image_path, output_path)`, etc.)
4. Each method calls `agent.predict(...)` and returns the dict directly

Full method list:
```python
def classify_brain_tumor(self, image_path: str) -> dict
def classify_chest_xray(self, image_path: str) -> dict         # existing — now returns dict
def segment_skin_lesion(self, image_path: str, output_path: str) -> dict  # existing — now returns dict
def classify_tb(self, image_path: str) -> dict
def segment_pneumothorax(self, image_path: str, output_path: str) -> dict
def classify_multi_label_cxr(self, image_path: str) -> dict
def classify_diabetic_retinopathy(self, image_path: str) -> dict
def classify_oct(self, image_path: str) -> dict
def classify_histopathology(self, image_path: str) -> dict
def segment_colon_polyp(self, image_path: str, output_path: str) -> dict
def classify_diabetic_foot_ulcer(self, image_path: str) -> dict
def classify_fracture(self, image_path: str, body_part: str) -> dict
def classify_mammography(self, image_path: str) -> dict
def classify_ham10000(self, image_path: str) -> dict
```

### 4.7 Assignment F: `agent_decision.py` Extension

#### F1 — AgentState

Add `artifact_path: Optional[str] = None` to `AgentState`:
```python
class AgentState(MessagesState):
    user_query:          str
    image_path:          Optional[str]
    agent_name:          str
    analysis_result:     Optional[dict]
    artifact_path:       Optional[str]    # ← ADD THIS
    web_search:          bool
    rag_results:         Optional[list]
    validation_required: bool
```

#### F2 — DECISION_SYSTEM_PROMPT

The prompt must list all 17 valid agent names with routing rules. Include these explicit disambiguation rules:

- If query mentions TB, tuberculosis → `TB_AGENT`
- If query mentions pneumothorax, collapsed lung → `PNEUMOTHORAX_AGENT`
- If query mentions multiple findings, multi-label, CheXpert, general chest → `MULTI_LABEL_CXR_AGENT`
- If query mentions COVID, pneumonia (no other qualifier) → `CHEST_XRAY_AGENT`
- If query mentions diabetic retinopathy, fundus, retinal → `DIABETIC_RETINOPATHY_AGENT`
- If query mentions OCT, optical coherence → `OCT_AGENT`
- If query mentions histopathology, tissue, H&E, biopsy, colorectal → `HISTOPATHOLOGY_AGENT`
- If query mentions colonoscopy, polyp, colon → `COLON_POLYP_AGENT`
- If query mentions foot ulcer, wound, diabetic foot → `DIABETIC_FOOT_ULCER_AGENT`
- If query mentions fracture, MURA, musculoskeletal, bone → `FRACTURE_AGENT`
- If query mentions mammography, breast finding, BIRADS → `MAMMOGRAPHY_AGENT`
- If query mentions skin type, melanoma, lesion classification (7-class) → `HAM10000_AGENT`
- If skin image with no classification context → `SKIN_LESION_AGENT`
- If query is general medical text, no image → `RAG_AGENT`
- If query needs recent data, guidelines → `WEB_SEARCH_AGENT`
- Default for no match → `GENERAL_AGENT`

#### F3 — New run_*_agent Functions

Add one function per new agent. Pattern for classification agents:
```python
def run_<agent_name>_agent(state: AgentState) -> AgentState:
    image_path = state.get("image_path")
    if not image_path:
        state["messages"].append(AIMessage(content="No image provided for <AGENT_NAME>."))
        return state
    result = AgentConfig.image_analyzer.<method_name>(image_path)
    state["analysis_result"] = result
    state["messages"].append(AIMessage(content=str(result)))
    return state
```

Pattern for segmentation agents (adds `artifact_path`):
```python
def run_<agent_name>_agent(state: AgentState) -> AgentState:
    image_path = state.get("image_path")
    if not image_path:
        state["messages"].append(AIMessage(content="No image provided."))
        return state
    output_path = os.path.join(config.medical_cv.<output_path_field>, f"{uuid.uuid4()}_overlay.png")
    result = AgentConfig.image_analyzer.<segment_method>(image_path, output_path)
    state["analysis_result"] = result
    state["artifact_path"]   = result.get("overlay_path", "")
    state["messages"].append(AIMessage(content=str(result)))
    return state
```

#### F4 — LangGraph Node Registration

For every new `run_*_agent` function, add:
```python
workflow.add_node("<AGENT_NAME>", run_<agent_name>_agent)
```

#### F5 — Routing Table Extension

The routing conditional must map all 17 agent names:
```python
routing_map = {
    "BRAIN_TUMOR_AGENT":             "BRAIN_TUMOR_AGENT",
    "CHEST_XRAY_AGENT":              "CHEST_XRAY_AGENT",
    "SKIN_LESION_AGENT":             "SKIN_LESION_AGENT",
    "TB_AGENT":                      "TB_AGENT",
    "PNEUMOTHORAX_AGENT":            "PNEUMOTHORAX_AGENT",
    "MULTI_LABEL_CXR_AGENT":         "MULTI_LABEL_CXR_AGENT",
    "DIABETIC_RETINOPATHY_AGENT":    "DIABETIC_RETINOPATHY_AGENT",
    "OCT_AGENT":                     "OCT_AGENT",
    "HISTOPATHOLOGY_AGENT":          "HISTOPATHOLOGY_AGENT",
    "COLON_POLYP_AGENT":             "COLON_POLYP_AGENT",
    "DIABETIC_FOOT_ULCER_AGENT":     "DIABETIC_FOOT_ULCER_AGENT",
    "FRACTURE_AGENT":                "FRACTURE_AGENT",
    "MAMMOGRAPHY_AGENT":             "MAMMOGRAPHY_AGENT",
    "HAM10000_AGENT":                "HAM10000_AGENT",
    "RAG_AGENT":                     "rag_retrieval",
    "WEB_SEARCH_AGENT":              "web_search",
    "GENERAL_AGENT":                 "general_response",
}
```

#### F6 — Human Validation Node

The existing `perform_human_validation` node must propagate `artifact_path` unchanged:
```python
def perform_human_validation(state: AgentState) -> AgentState:
    # ... existing validation message logic ...
    # Do NOT clear artifact_path — pass it through:
    # state["artifact_path"] remains unchanged
    return state
```

### 4.8 Assignment G: app.py Fix

Replace both hardcoded skin-lesion blocks in `/chat` and `/upload` handlers.

**Current (REMOVE)**:
```python
if response_data["agent_name"] == "SKIN_LESION_AGENT, HUMAN_VALIDATION":
    result["result_image"] = "/uploads/skin_lesion_output/segmentation_plot.png"
```

**Replace with (in BOTH handlers)**:
```python
artifact_path = response_data.get("artifact_path")
if artifact_path and os.path.exists(artifact_path):
    url_path = "/" + artifact_path.lstrip("./").lstrip("/")
    result["result_image"] = url_path
```

Also add directory creation at startup for new output paths:
```python
os.makedirs(config.medical_cv.pneumothorax_output_path, exist_ok=True)
os.makedirs(config.medical_cv.colon_polyp_output_path,  exist_ok=True)
```

---

## 5. Sanika — Pulmonary & Chest Imaging Agents

### 5.1 Scope

Sanika delivers 3 agents: `TB_AGENT`, `PNEUMOTHORAX_AGENT`, `MULTI_LABEL_CXR_AGENT`.  
Sanika does NOT modify `config.py`, `__init__.py`, `agent_decision.py`, or `app.py`.

### 5.2 Agent 1: TB_AGENT

**Folder**: `agents/image_analysis_agent/tb_agent/`

**Class**: `TBClassification` in `tb_inference.py`

**Dataset**: Shenzhen TB + Montgomery TB (NLM OpenI)  
~800 combined images (336+58 TB, 326+80 Normal)

**Model**: DenseNet121, pretrained ImageNet, replace classifier:
```python
model = models.densenet121(weights="IMAGENET1K_V1")
model.classifier = nn.Linear(model.classifier.in_features, 2)
```

**Preprocessing**: CLAHE via albumentations. Grayscale → RGB conversion required before transforms.
```python
IMAGE_SIZE = 224
```

**Class imbalance**: Use `WeightedRandomSampler` on the training DataLoader.

**Training**: 40 epochs, Adam(lr=1e-4), ReduceLROnPlateau(patience=5, factor=0.5).  
Save best val AUC checkpoint to `tb_densenet121.pth`.

**Output contract**:
```python
{
    "label":            str,    # "TB" or "Normal"
    "confidence":       float,
    "needs_validation": True,
    "explanation":      str
}
```

**Checkpoint path**: `agents/image_analysis_agent/tb_agent/models/tb_densenet121.pth`  
**label_map.json**: `{"0": "Normal", "1": "TB"}`

### 5.3 Agent 2: PNEUMOTHORAX_AGENT

**Folder**: `agents/image_analysis_agent/pneumothorax_agent/`

**Class**: `PneumothoraxSegmentation` in `pneumothorax_inference.py`

**Dataset**: SIIM-ACR Pneumothorax Segmentation (Kaggle 2019)  
`kaggle competitions download -c siim-acr-pneumothorax-segmentation`  
Format: DICOM images + RLE masks. ~12,000 images; ~3,000 positive.

**Model**: `segmentation_models_pytorch.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1)`

**DICOM reading**: Use `pydicom`. Convert pixel array to 3-channel numpy uint8.

**RLE decode**: Implement `rle_decode(rle_str, shape) -> np.ndarray` (standard SIIM implementation).

**Preprocessing**:
```python
IMAGE_SIZE = 512  # preserve resolution for segmentation
```

**predict() signature**:
```python
def predict(self, image_path: str, output_path: str) -> dict:
```

**Output contract**:
```python
{
    "label":            str,    # "pneumothorax_detected" or "no_pneumothorax"
    "confidence":       float,  # mean mask activation
    "mask_path":        str,    # path to binary mask PNG
    "overlay_path":     str,    # path to overlay PNG (= output_path)
    "needs_validation": True,
    "explanation":      str
}
```

**Threshold for label**: if `mask.max() > 0.5` → `"pneumothorax_detected"`, else `"no_pneumothorax"`.

**Checkpoint path**: `agents/image_analysis_agent/pneumothorax_agent/models/pneumothorax_unet.pth`

### 5.4 Agent 3: MULTI_LABEL_CXR_AGENT

**Folder**: `agents/image_analysis_agent/multi_label_cxr_agent/`

**Class**: `MultiLabelCXRClassification` in `multi_label_cxr_inference.py`

**Dataset**: CheXpert (Stanford) — 14 labels, ~224,316 training images  
`kaggle datasets download -d ashery/chexpert`  
Format: JPEG + CSV with label columns.

**CheXpert U-Zeros policy**: Replace all `-1` (uncertain) values with `0` (negative) in the label CSV.

**Labels (14)**:
```python
FINDINGS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]
```

**Model**: EfficientNet-B4, replace classifier:
```python
model = models.efficientnet_b4(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 14)
```

**Loss**: `nn.BCEWithLogitsLoss()` — multi-label binary cross-entropy.

**Inference**: Apply sigmoid per logit. Threshold = 0.5 per label.

**IMAGE_SIZE = 380** (EfficientNet-B4 native)

**predict() signature**:
```python
def predict(self, image_path: str) -> dict:
```

**Output contract**:
```python
{
    "findings": [
        {"label": str, "confidence": float, "status": str}
        # status: "positive" if conf >= 0.5, else "negative"
    ],
    "needs_validation": True,
    "explanation": str   # list positive findings in sentence form
}
```

**Checkpoint path**: `agents/image_analysis_agent/multi_label_cxr_agent/models/multi_label_cxr_efficientnet.pth`

---

## 6. Rushil — Ophthalmology Imaging Agents

### 6.1 Scope

Rushil delivers 2 agents: `DIABETIC_RETINOPATHY_AGENT`, `OCT_AGENT`.  
Rushil does NOT modify `config.py`, `__init__.py`, `agent_decision.py`, or `app.py`.

### 6.2 Agent 1: DIABETIC_RETINOPATHY_AGENT

**Folder**: `agents/image_analysis_agent/diabetic_retinopathy_agent/`

**Class**: `DiabeticRetinopathyClassification` in `diabetic_retinopathy_inference.py`

**Dataset**: APTOS 2019 Blindness Detection (Kaggle)  
`kaggle competitions download -c aptos2019-blindness-detection`  
~3,662 images, 5 classes (grades 0–4).

**Grade mapping**:
```python
SEVERITY_MAP = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}
```

**Model**: EfficientNet-B4, replace classifier head for 5-class:
```python
model = models.efficientnet_b4(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
```

**IMAGE_SIZE = 380** (EfficientNet-B4 native)

**Ben Graham preprocessing** (apply during training AND inference):
```python
def ben_graham_preprocess(image: np.ndarray, sigmaX: int = 10) -> np.ndarray:
    # Circular crop: keep only the circular fundus region
    # Subtract Gaussian blur to normalize lighting
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image
```

**Class imbalance**: Use `WeightedRandomSampler`. Grades 0 and 2 dominate; grades 1, 3, 4 are rare.

**Primary metric**: Quadratic Weighted Kappa (QWK). Report in `metrics.json`.

**predict() signature**:
```python
def predict(self, image_path: str) -> dict:
```

**Output contract**:
```python
{
    "grade":            int,    # 0, 1, 2, 3, or 4
    "severity_text":    str,    # from SEVERITY_MAP
    "confidence":       float,
    "needs_validation": True,
    "explanation":      str
}
```

**Checkpoint path**: `agents/image_analysis_agent/diabetic_retinopathy_agent/models/diabetic_retinopathy_efficientnet.pth`  
**label_map.json**: `{"0": "No DR", "1": "Mild DR", "2": "Moderate DR", "3": "Severe DR", "4": "Proliferative DR"}`

### 6.3 Agent 2: OCT_AGENT

**Folder**: `agents/image_analysis_agent/oct_agent/`

**Class**: `OCTClassification` in `oct_inference.py`

**Dataset**: Kermany 2018 OCT Dataset (Kaggle)  
`kaggle datasets download -d paultimothymooney/kermany2018`  
~84,495 images, 4 classes.

**Classes (4)**: `CNV`, `DME`, `DRUSEN`, `NORMAL`

**Model**: EfficientNet-B3, replace classifier:
```python
model = models.efficientnet_b3(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
```

**IMAGE_SIZE = 300** (EfficientNet-B3 native)

**predict() signature**:
```python
def predict(self, image_path: str) -> dict:
```

**Output contract**:
```python
{
    "label":            str,    # "CNV", "DME", "DRUSEN", or "NORMAL"
    "confidence":       float,
    "needs_validation": True,
    "explanation":      str
}
```

**Checkpoint path**: `agents/image_analysis_agent/oct_agent/models/oct_efficientnet.pth`  
**label_map.json**: `{"0": "CNV", "1": "DME", "2": "DRUSEN", "3": "NORMAL"}`

---

## 7. Jeet — Pathology, GI & Wound Imaging Agents

### 7.1 Scope

Jeet delivers 3 agents: `HISTOPATHOLOGY_AGENT`, `COLON_POLYP_AGENT`, `DIABETIC_FOOT_ULCER_AGENT`.  
Jeet does NOT modify `config.py`, `__init__.py`, `agent_decision.py`, or `app.py`.

### 7.2 Agent 1: HISTOPATHOLOGY_AGENT

**Folder**: `agents/image_analysis_agent/histopathology_agent/`

**Class**: `HistopathologyClassification` in `histopathology_inference.py`

**Dataset**: NCT-CRC-HE-100K (Zenodo or Kaggle)  
`kaggle datasets download -d kmader/colorectal-histology-mnist`  
100,000 patches, 224×224, 9 classes.

**Classes (9)**: `ADI`, `BACK`, `DEB`, `LYM`, `MUC`, `MUS`, `NORM`, `STR`, `TUM`

**Model**: EfficientNet-B3, replace classifier:
```python
model = models.efficientnet_b3(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 9)
```

**IMAGE_SIZE = 224** (patches are native 224×224)

**Stain normalization** (apply during training, optional at inference via flag):  
Use MacenkoNormalizer from `torchstain`:
```python
from torchstain import MacenkoNormalizer
normalizer = MacenkoNormalizer(backend='torch')
normalizer.fit(reference_image_tensor)
normalized, _, _ = normalizer.normalize(image_tensor)
```

**predict() signature**:
```python
def predict(self, image_path: str) -> dict:
```

**Output contract**:
```python
{
    "label":            str,    # tissue class abbreviation
    "confidence":       float,
    "needs_validation": True,
    "explanation":      str
}
```

**Checkpoint path**: `agents/image_analysis_agent/histopathology_agent/models/histopathology_efficientnet.pth`

### 7.3 Agent 2: COLON_POLYP_AGENT

**Folder**: `agents/image_analysis_agent/colon_polyp_agent/`

**Class**: `ColonPolypSegmentation` in `colon_polyp_inference.py`

**Dataset**: Kvasir-SEG + CVC-ClinicDB (combined)  
- Kvasir-SEG: 1,000 colonoscopy images + masks (Zenodo)
- CVC-ClinicDB: 612 images + masks (Kaggle)

**Model**: `segmentation_models_pytorch.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)`

**Joint augmentation** (image + mask must transform identically):
```python
import albumentations as A
aug = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
], additional_targets={'mask': 'mask'})
```

**IMAGE_SIZE = 256**

**predict() signature**:
```python
def predict(self, image_path: str, output_path: str) -> dict:
```

**Output contract**:
```python
{
    "label":            str,    # "polyp_detected" or "no_polyp"
    "confidence":       float,  # mean mask activation
    "mask_path":        str,
    "overlay_path":     str,    # = output_path
    "needs_validation": True,
    "explanation":      str
}
```

**Threshold**: if `mask.max() > 0.5` → `"polyp_detected"`, else `"no_polyp"`.

**Checkpoint path**: `agents/image_analysis_agent/colon_polyp_agent/models/colon_polyp_unet.pth`

### 7.4 Agent 3: DIABETIC_FOOT_ULCER_AGENT

**Folder**: `agents/image_analysis_agent/diabetic_foot_ulcer_agent/`

**Class**: `DiabeticFootUlcerClassification` in `diabetic_foot_ulcer_inference.py`

**Dataset**: DFUC2021 (Diabetic Foot Ulcer Challenge 2021)  
Download from the official challenge page or Kaggle.  
~2,000 wound images, 4 classes.

**Classes (4)**: `Ulcer`, `Infection`, `Normal`, `Gangrene`

**Model**: EfficientNet-B3:
```python
model = models.efficientnet_b3(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
```

**IMAGE_SIZE = 300**

**predict() signature**:
```python
def predict(self, image_path: str) -> dict:
```

**Output contract**:
```python
{
    "label":            str,    # "Ulcer", "Infection", "Normal", "Gangrene"
    "confidence":       float,
    "needs_validation": True,
    "explanation":      str
}
```

**Checkpoint path**: `agents/image_analysis_agent/diabetic_foot_ulcer_agent/models/diabetic_foot_ulcer_efficientnet.pth`

---

## 8. Srujan — Musculoskeletal, Breast & Extended Dermatology Agents

### 8.1 Scope

Srujan delivers 3 agents: `FRACTURE_AGENT`, `MAMMOGRAPHY_AGENT`, `HAM10000_AGENT`.  
Srujan does NOT modify `config.py`, `__init__.py`, `agent_decision.py`, or `app.py`.

### 8.2 Agent 1: FRACTURE_AGENT

**Folder**: `agents/image_analysis_agent/fracture_agent/`

**Class**: `FractureClassification` in `fracture_inference.py`

**Dataset**: MURA v1.1 (Stanford)  
`kaggle datasets download -d cjinny/mura-v11`  
~40,561 images, 7 body parts, binary label (Normal / Abnormal).

**MURA body parts**: `XR_SHOULDER`, `XR_ELBOW`, `XR_FINGER`, `XR_HAND`, `XR_HUMERUS`, `XR_FOREARM`, `XR_WRIST`

**Data flattening**: Extract body_part from MURA directory path. Flatten to:
```
data/fracture/train/Normal/   (combine all body parts)
data/fracture/train/Abnormal/
```

**Model**: DenseNet169:
```python
model = models.densenet169(weights="IMAGENET1K_V1")
model.classifier = nn.Linear(model.classifier.in_features, 2)
```

**IMAGE_SIZE = 320**

**Class imbalance**: Use `WeightedRandomSampler` — MURA has 3× more normal studies.

**predict() signature**:
```python
def predict(self, image_path: str, body_part: str = "UNKNOWN") -> dict:
```

`body_part` is extracted from the image file path by the caller (Amritesh) using the prefix convention:
```python
# Pattern: "XR_WRIST_patientXXXXX_studyY_imageZ.png"
body_part = image_path.split("_")[1] if "_" in image_path else "UNKNOWN"
```

**Output contract**:
```python
{
    "label":            str,    # "Normal" or "Abnormal"
    "confidence":       float,
    "body_part":        str,    # e.g., "WRIST", "SHOULDER"
    "needs_validation": True,
    "explanation":      str
}
```

**Checkpoint path**: `agents/image_analysis_agent/fracture_agent/models/fracture_densenet.pth`  
**label_map.json**: `{"0": "Normal", "1": "Abnormal"}`

### 8.3 Agent 2: MAMMOGRAPHY_AGENT

**Folder**: `agents/image_analysis_agent/mammography_agent/`

**Class**: `MammographyClassification` in `mammography_inference.py`

**Dataset**: CBIS-DDSM (Curated Breast Imaging Subset of DDSM)  
`kaggle datasets download -d awsaf49/cbis-ddsm-breast-cancer-image-dataset`  
Format: DICOM images + CSV metadata files.

**Labels (2)**: `BENIGN`, `MALIGNANT`  
**Finding types (2)**: `MASS`, `CALCIFICATION` (extracted from CSV metadata / filename)

**DICOM handling**: Use `pydicom`. Normalize pixel array to uint8 range before PIL conversion.

**Model**: EfficientNet-B3:
```python
model = models.efficientnet_b3(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
```

**IMAGE_SIZE = 300**

**predict() signature**:
```python
def predict(self, image_path: str) -> dict:
```

`finding_type` is extracted from the image filename by the wrapper:
```python
# CBIS-DDSM filenames contain "Mass" or "Calc" — parse before calling predict()
finding_type = "MASS" if "Mass" in image_path else "CALCIFICATION"
```

**Output contract**:
```python
{
    "label":            str,    # "BENIGN" or "MALIGNANT"
    "confidence":       float,
    "finding_type":     str,    # "MASS" or "CALCIFICATION"
    "needs_validation": True,
    "explanation":      str
}
```

**Checkpoint path**: `agents/image_analysis_agent/mammography_agent/models/mammography_efficientnet.pth`  
**label_map.json**: `{"0": "BENIGN", "1": "MALIGNANT"}`

### 8.4 Agent 3: HAM10000_AGENT

**Folder**: `agents/image_analysis_agent/ham10000_agent/`

**Class**: `HAM10000Classification` in `ham10000_inference.py`

**Dataset**: HAM10000 (Kaggle)  
`kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection`  
~10,015 dermoscopy images, 7 classes. Severe class imbalance.

**Classes (7)**:
```python
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
FULL_NAMES = {
    'akiec': 'Actinic Keratosis / Intraepithelial Carcinoma',
    'bcc':   'Basal Cell Carcinoma',
    'bkl':   'Benign Keratosis-like Lesions',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Melanocytic Nevi',
    'vasc':  'Vascular Lesions'
}
```

**Model**: EfficientNet-B3:
```python
model = models.efficientnet_b3(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
```

**IMAGE_SIZE = 300**

**Class imbalance**: `nv` (67%) dominates. Use both `WeightedRandomSampler` AND `CrossEntropyLoss(weight=class_weights)`.  
Compute class_weights as: `n_samples / (n_classes * class_counts)`.

**predict() signature**:
```python
def predict(self, image_path: str) -> dict:
```

**Output contract**:
```python
{
    "label":            str,    # abbreviation, e.g. "mel"
    "confidence":       float,
    "needs_validation": True,
    "explanation":      str     # include full name and clinical note
}
```

**Checkpoint path**: `agents/image_analysis_agent/ham10000_agent/models/ham10000_efficientnet.pth`

---

## 9. Integration Protocol — How All Pieces Connect

### 9.1 Integration Flow

```
Phase 1 — Independent development (Sanika, Rushil, Jeet, Srujan work in parallel)
    Each person delivers their agent folder(s) with:
        - inference .py file with the exact class name specified
        - trained .pth checkpoint at the exact path specified
        - label_map.json at the correct location
        - metrics.json with test set performance

Phase 2 — Amritesh retrofits existing system
    Step 2.1: Update covid_chest_xray_inference.py to return dict
    Step 2.2: Update skin_lesion_inference.py to return dict
    Step 2.3: Implement brain_tumor_inference.py and train
    Step 2.4: Update config.py with all new paths
    Step 2.5: Update image_classifier.py to 12 modalities

Phase 3 — Amritesh wires each new agent
    For each incoming agent folder, do these steps in order:
        3a. Copy agent folder into agents/image_analysis_agent/
        3b. Verify checkpoint exists at the path specified in config.py
        3c. Add import to __init__.py
        3d. Add instantiation to ImageAnalysisAgent.__init__()
        3e. Add method to ImageAnalysisAgent
        3f. Add run_*_agent function to agent_decision.py
        3g. Add workflow.add_node() call
        3h. Add routing entry in routing_map
        3i. Add conditional edge for the agent node
        3j. Run standalone test on the new agent

Phase 4 — End-to-end test
    Launch app.py with uvicorn
    Test each new agent via the /upload endpoint
    Verify artifact_path is served correctly for segmentation agents
```

### 9.2 Handoff Deliverable Checklist

Before any contributor hands off to Amritesh, verify all of the following:

```
□ Folder exists at: agents/image_analysis_agent/<agent_name>_agent/
□ Inference file:   agents/image_analysis_agent/<agent_name>_agent/<agent_name>_inference.py
□ Checkpoint:       agents/image_analysis_agent/<agent_name>_agent/models/<checkpoint>.pth
□ Label map:        agents/image_analysis_agent/<agent_name>_agent/models/label_map.json
□ Metrics:          agents/image_analysis_agent/<agent_name>_agent/models/metrics.json
□ Class name matches spec (exact string match)
□ predict() returns dict with all required keys
□ needs_validation is boolean True (not string, not 1)
□ confidence is float between 0.0 and 1.0
□ No unhandled exceptions — try/except wraps the inference body
□ Standalone test passes (assert block from Section 3.11)
□ No hardcoded model paths inside the class
□ Model is in eval() mode after loading
```

### 9.3 Integration — config.py Read Pattern

When Amritesh wires a new agent in `__init__.py`, the instantiation always uses config:
```python
from config import config
self.<agent>_classifier = <ClassName>(config.medical_cv.<model_path_field>)
```

The config field names are defined in Section 4.4 of this document.

### 9.4 Integration — __init__.py Pattern

The method in `ImageAnalysisAgent` that wraps a classification agent:
```python
def classify_<agent>(self, image_path: str) -> dict:
    return self.<agent>_classifier.predict(image_path)
```

The method that wraps a segmentation agent:
```python
def segment_<agent>(self, image_path: str, output_path: str) -> dict:
    return self.<agent>_segmenter.predict(image_path, output_path)
```

### 9.5 Integration — agent_decision.py Pattern

The standard run_*_agent function for a classification agent:
```python
def run_<agent_name>_agent(state: AgentState) -> AgentState:
    image_path = state.get("image_path")
    if not image_path:
        state["messages"].append(AIMessage(content="No image provided for <AGENT_NAME>."))
        return state
    try:
        result = AgentConfig.image_analyzer.classify_<agent>(image_path)
        state["analysis_result"] = result
        state["messages"].append(AIMessage(content=str(result)))
    except Exception as e:
        logger.error(f"<AGENT_NAME> error: {e}")
        state["messages"].append(AIMessage(content=f"<AGENT_NAME> failed: {str(e)}"))
    return state
```

For segmentation agents, also set `state["artifact_path"]`:
```python
def run_<agent_name>_agent(state: AgentState) -> AgentState:
    image_path = state.get("image_path")
    if not image_path:
        state["messages"].append(AIMessage(content="No image provided."))
        return state
    try:
        output_dir = config.medical_cv.<output_path_field>
        output_path = os.path.join(output_dir, f"{uuid.uuid4()}_overlay.png")
        result = AgentConfig.image_analyzer.segment_<agent>(image_path, output_path)
        state["analysis_result"] = result
        state["artifact_path"]   = result.get("overlay_path", "")
        state["messages"].append(AIMessage(content=str(result)))
    except Exception as e:
        logger.error(f"<AGENT_NAME> error: {e}")
        state["messages"].append(AIMessage(content=f"<AGENT_NAME> failed: {str(e)}"))
    return state
```

### 9.6 app.py Artifact Response Pattern

After Amritesh's fix, the generic artifact block in both `/chat` and `/upload` is:
```python
artifact_path = response_data.get("artifact_path")
if artifact_path and os.path.exists(artifact_path):
    url_path = "/" + artifact_path.lstrip("./").lstrip("/")
    result["result_image"] = url_path
```

This works for all segmentation agents (skin lesion, pneumothorax, colon polyp) without any agent-specific branching.

---

## 10. Final Report Requirements

When each contributor finishes their work, they must produce a **Final Integration Report** document. The report is placed at:
```
Work/Reports/<PersonName>_Report.md
```

### 10.1 Report Sections (Mandatory)

Every report must contain exactly these sections:

#### Section 1: Agents Implemented
- List each agent by name (e.g., `TB_AGENT`)
- State the exact class name (e.g., `TBClassification`)
- State the checkpoint file path
- State the dataset used (name + size)

#### Section 2: Model Performance
For each agent, provide a table:
| Metric    | Value  |
|-----------|--------|
| Accuracy  | X.XX%  |
| Precision | X.XX%  |
| Recall    | X.XX%  |
| F1        | X.XX%  |
| AUC       | X.XX   |

For segmentation agents:
| Metric | Value |
|--------|-------|
| Dice   | X.XX  |
| IoU    | X.XX  |

#### Section 3: File Inventory
Complete list of every file created, with path and purpose:
```
agents/image_analysis_agent/tb_agent/tb_inference.py      — inference wrapper
agents/image_analysis_agent/tb_agent/train.py             — training script
agents/image_analysis_agent/tb_agent/models/tb_densenet121.pth  — checkpoint
agents/image_analysis_agent/tb_agent/models/label_map.json      — class index
agents/image_analysis_agent/tb_agent/models/metrics.json        — test metrics
```

#### Section 4: Integration Handoff — What Amritesh Needs
State the exact config key and value for each model path, e.g.:
```
config.medical_cv.tb_model_path = "./agents/image_analysis_agent/tb_agent/models/tb_densenet121.pth"
```
State the `__init__.py` import line:
```python
from agents.image_analysis_agent.tb_agent.tb_inference import TBClassification
```
State the instantiation line:
```python
self.tb_classifier = TBClassification(config.medical_cv.tb_model_path)
```
State the `ImageAnalysisAgent` method signature:
```python
def classify_tb(self, image_path: str) -> dict
```
State the `agent_decision.py` function name:
```python
def run_tb_agent(state: AgentState) -> AgentState
```

#### Section 5: Known Limitations
Any dataset limitations, edge cases, or failure modes observed during testing.

#### Section 6: Environment & Dependencies
Exact `pip install` commands required beyond the base `requirements.txt`.

---

## 11. Agent Routing Table — Complete Reference

This table is the authoritative source for the LangGraph decision logic.

| Agent Name                  | Trigger Conditions                                                        | Image Modality      | Output Type     |
|-----------------------------|---------------------------------------------------------------------------|---------------------|-----------------|
| `BRAIN_TUMOR_AGENT`         | Brain MRI, tumor in brain, glioma, meningioma, pituitary                  | BRAIN_MRI           | Classification  |
| `CHEST_XRAY_AGENT`          | COVID pneumonia, chest X-ray COVID, covid19                               | CHEST_XRAY          | Classification  |
| `TB_AGENT`                  | Tuberculosis, TB, lung TB, Shenzhen, Montgomery                           | CHEST_XRAY          | Classification  |
| `PNEUMOTHORAX_AGENT`        | Pneumothorax, collapsed lung, pleural air                                 | CHEST_XRAY          | Segmentation    |
| `MULTI_LABEL_CXR_AGENT`     | Multiple chest findings, CheXpert, multi-label, general chest X-ray report| CHEST_XRAY          | Multi-label     |
| `DIABETIC_RETINOPATHY_AGENT`| Diabetic retinopathy, fundus, retinal grading, APTOS                     | FUNDUS              | Classification  |
| `OCT_AGENT`                 | OCT, optical coherence tomography, CNV, DME, drusen                      | OCT                 | Classification  |
| `HISTOPATHOLOGY_AGENT`      | Histopathology, tissue biopsy, H&E, colorectal cancer, colon tissue      | HISTOPATHOLOGY      | Classification  |
| `COLON_POLYP_AGENT`         | Colonoscopy, polyp, colon polyp, endoscopy                               | COLONOSCOPY         | Segmentation    |
| `DIABETIC_FOOT_ULCER_AGENT` | Foot ulcer, wound, diabetic foot, gangrene                               | FOOT_WOUND          | Classification  |
| `FRACTURE_AGENT`            | Fracture, bone, musculoskeletal X-ray, MURA, wrist, shoulder, elbow      | MUSCULOSKELETAL     | Classification  |
| `MAMMOGRAPHY_AGENT`         | Mammography, breast finding, BIRADS, mass, calcification                 | MAMMOGRAPHY         | Classification  |
| `HAM10000_AGENT`            | Skin lesion type, melanoma, dermoscopy classification, HAM10000          | SKIN_DERM           | Classification  |
| `SKIN_LESION_AGENT`         | Skin lesion segmentation, lesion boundary, segment lesion                 | SKIN_LESION         | Segmentation    |
| `RAG_AGENT`                 | General medical question, no image, document query                        | None                | Text            |
| `WEB_SEARCH_AGENT`          | Current guidelines, recent research, latest treatment                     | None                | Text            |
| `GENERAL_AGENT`             | No match to any of the above                                              | None                | Text            |

---

## 12. Output Contract Reference — All 17 Agents

Every `predict()` return value must exactly match the following schemas. **Do not add extra keys. Do not rename keys.**

### Classification Agents (standard)

Applies to: `BRAIN_TUMOR_AGENT`, `CHEST_XRAY_AGENT`, `TB_AGENT`, `OCT_AGENT`, `HISTOPATHOLOGY_AGENT`, `DIABETIC_FOOT_ULCER_AGENT`, `HAM10000_AGENT`

```python
{
    "label":            str,    # exact class name as string
    "confidence":       float,  # round(float, 4), range [0.0, 1.0]
    "needs_validation": True,   # boolean True always
    "explanation":      str     # clinical description
}
```

### Fracture Agent (extended classification)

```python
{
    "label":            str,    # "Normal" or "Abnormal"
    "confidence":       float,
    "body_part":        str,    # e.g., "WRIST", "SHOULDER"
    "needs_validation": True,
    "explanation":      str
}
```

### Mammography Agent (extended classification)

```python
{
    "label":            str,    # "BENIGN" or "MALIGNANT"
    "confidence":       float,
    "finding_type":     str,    # "MASS" or "CALCIFICATION"
    "needs_validation": True,
    "explanation":      str
}
```

### Diabetic Retinopathy Agent (grade-based)

```python
{
    "grade":            int,    # 0, 1, 2, 3, or 4
    "severity_text":    str,    # "No DR", "Mild DR", etc.
    "confidence":       float,
    "needs_validation": True,
    "explanation":      str
}
```

### Multi-Label CXR Agent

```python
{
    "findings": [
        {
            "label":      str,   # one of 14 CheXpert labels
            "confidence": float,
            "status":     str    # "positive" or "negative"
        },
        # ... up to 14 entries
    ],
    "needs_validation": True,
    "explanation":      str      # summary sentence of positive findings
}
```

### Segmentation Agents

Applies to: `SKIN_LESION_AGENT`, `PNEUMOTHORAX_AGENT`, `COLON_POLYP_AGENT`

```python
{
    "label":            str,    # e.g., "lesion_segmented", "pneumothorax_detected", "polyp_detected"
    "confidence":       float,  # mean mask activation, rounded to 4 dp
    "mask_path":        str,    # absolute or relative path to binary mask PNG
    "overlay_path":     str,    # path to the color overlay PNG
    "needs_validation": True,
    "explanation":      str
}
```

`overlay_path` is the value Amritesh stores in `state["artifact_path"]` and `app.py` serves to the frontend as `result["result_image"]`.

---

## 13. Completion Test Checklists — Master Reference

Each contributor has a detailed per-agent test checklist in their individual file. This section is the **master integration gate** — it lists the system-level tests that Amritesh must run after all agents are wired in. Individual agent checklists live in:

- `Work/Amritesh.md` — Section "Completion Test Checklist" (T1–T9)
- `Work/Sanika.md` — Section "Completion Test Checklist" (T1–T4)
- `Work/Rushil.md` — Section "Completion Test Checklist" (T1–T3)
- `Work/Jeet.md` — Section "Completion Test Checklist" (T1–T4)
- `Work/Srujan.md` — Section "Completion Test Checklist" (T1–T4)

### 13.1 Per-Person Completion Gates

The table below defines the minimum tests each person must pass before their work is considered done and ready for Amritesh to wire in.

| Person   | Agent(s)                                          | Required Passing Tests                                                                                                   |
|----------|---------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| Amritesh | Brain Tumor + system wiring                       | T1 (brain tumor), T2 (chest xray retrofit), T3 (skin lesion retrofit), T4 (config.py), T5 (classifier), T6 (__init__), T7 (agent_decision), T8 (app.py), T9 (end-to-end) |
| Sanika   | TB, Pneumothorax, Multi-label CXR                 | T1 (TB), T2 (Pneumothorax), T3 (Multi-label CXR), T4 (integration readiness)                                            |
| Rushil   | Diabetic Retinopathy, OCT                         | T1 (DR), T2 (OCT), T3 (integration readiness)                                                                            |
| Jeet     | Histopathology, Colon Polyp, Diabetic Foot Ulcer  | T1 (histopathology), T2 (colon polyp), T3 (DFU), T4 (integration readiness)                                             |
| Srujan   | Fracture, Mammography, HAM10000                   | T1 (fracture), T2 (mammography), T3 (HAM10000), T4 (integration readiness)                                              |

### 13.2 Universal Tests — Apply to Every Agent Written by Every Person

These tests must pass for every `predict()` method in the entire codebase. Run them against every agent before marking any work complete.

**UC1 — Return type**
```python
result = clf.predict(image_path)
assert isinstance(result, dict), f"predict() must return dict, got {type(result)}"
```

**UC2 — needs_validation is boolean True**
```python
assert result["needs_validation"] is True, \
    f"needs_validation must be True (bool), got {result['needs_validation']!r} ({type(result['needs_validation']).__name__})"
```

**UC3 — confidence is float in range**
```python
assert isinstance(result["confidence"], float), \
    f"confidence must be float, got {type(result['confidence']).__name__}"
assert 0.0 <= result["confidence"] <= 1.0, \
    f"confidence out of range: {result['confidence']}"
```

**UC4 — No exception on bad input**
```python
try:
    result = clf.predict("/nonexistent/path/image.jpg")
    assert isinstance(result, dict), "Error path must still return dict"
    assert result["confidence"] == 0.0, "Error path must return confidence=0.0"
except Exception as e:
    raise AssertionError(f"predict() must not raise on bad input — got: {e}")
```

**UC5 — Model is in eval mode**
```python
assert not clf.model.training, "Model must be in eval() mode after __init__"
```

**UC6 — No hardcoded path in class**
Search the inference file source for literal path strings. No string matching `"./agents"`, `"/home/"`, `"/Users/"`, or `"C:\\"` may appear inside the class body.

### 13.3 System Integration Tests — Run by Amritesh After All Wiring

These are end-to-end tests that verify the entire pipeline from HTTP request to JSON response. Run with `uvicorn app:app --port 8000`.

#### SI1 — All 14 Image Agents Reachable

For each agent, POST to `/upload` with an appropriate test image and verify:
- HTTP 200 status
- Response JSON contains `"analysis_result"` key
- `analysis_result` is not null
- `analysis_result["needs_validation"]` is `true`

| Agent                       | Test Image Type          | Expected `analysis_result` key(s) to check            |
|-----------------------------|--------------------------|--------------------------------------------------------|
| BRAIN_TUMOR_AGENT           | Brain MRI JPEG           | `label` in {glioma, meningioma, no_tumor, pituitary}   |
| CHEST_XRAY_AGENT            | Chest X-ray PNG          | `label` in {covid19, normal}                           |
| TB_AGENT                    | Chest X-ray PNG          | `label` in {TB, Normal}                                |
| PNEUMOTHORAX_AGENT          | Chest X-ray PNG          | `label` in {pneumothorax_detected, no_pneumothorax}    |
| MULTI_LABEL_CXR_AGENT       | Chest X-ray PNG          | `findings` is list of length 14                        |
| DIABETIC_RETINOPATHY_AGENT  | Fundus JPEG              | `grade` in {0,1,2,3,4}, `severity_text` non-empty     |
| OCT_AGENT                   | OCT JPEG                 | `label` in {CNV, DME, DRUSEN, NORMAL}                  |
| HISTOPATHOLOGY_AGENT        | H&E patch PNG            | `label` in 9-class set                                 |
| COLON_POLYP_AGENT           | Colonoscopy frame JPEG   | `label` in {polyp_detected, no_polyp}                  |
| DIABETIC_FOOT_ULCER_AGENT   | Wound photo JPEG         | `label` in {Ulcer, Infection, Normal, Gangrene}        |
| FRACTURE_AGENT              | X-ray PNG                | `label` in {Normal, Abnormal}, `body_part` non-empty  |
| MAMMOGRAPHY_AGENT           | Mammogram PNG            | `label` in {BENIGN, MALIGNANT}, `finding_type` non-empty |
| HAM10000_AGENT              | Dermoscopy JPEG          | `label` in 7-class set                                 |
| SKIN_LESION_AGENT           | Skin photo JPEG          | `label` in {lesion_segmented, no_lesion_detected}      |

#### SI2 — Segmentation Agents Serve Overlay Image

For SKIN_LESION_AGENT, PNEUMOTHORAX_AGENT, and COLON_POLYP_AGENT:

- [ ] `/upload` response JSON contains `"result_image"` key
- [ ] `result_image` value is a string starting with `"/uploads/"`
- [ ] GET `http://localhost:8000{result_image}` returns HTTP 200
- [ ] The response Content-Type is `image/png`
- [ ] The PNG file is non-zero in size (greater than 1KB)

#### SI3 — No Hardcoded Agent Name in app.py

- [ ] `grep -n "SKIN_LESION_AGENT" app.py` returns zero results
- [ ] `grep -n "result_image" app.py` returns exactly the 2 lines with the generic `artifact_path` block (one per handler)

#### SI4 — Text-Only Queries Still Work

- [ ] POST `/chat` with body `{"query": "What are the symptoms of pneumonia?"}` returns HTTP 200 with non-empty `response` field
- [ ] POST `/chat` with body `{"query": "Latest treatment guidelines for diabetes 2024"}` returns HTTP 200

#### SI5 — Input Guardrails Still Functional

- [ ] POST `/chat` with a clearly non-medical query (e.g., `"How do I make pasta?"`) is either answered politely or rejected — it must NOT trigger an image agent
- [ ] The guardrails module (`agents/guardrails/local_guardrails.py`) is still imported and called in `agent_decision.py`

#### SI6 — Server Startup

- [ ] `uvicorn app:app` starts without Python exceptions in the terminal
- [ ] All output directories exist after startup: `uploads/skin_lesion_output/`, `uploads/pneumothorax_output/`, `uploads/colon_polyp_output/`
- [ ] Server starts even when running on CPU (no CUDA device)

### 13.4 Minimum Performance Thresholds — Summary

These thresholds are the minimum required to ship. If any agent falls below its threshold, the gap must be documented in the report with an explanation.

| Agent                       | Primary Metric              | Minimum Threshold |
|-----------------------------|-----------------------------|-------------------|
| BRAIN_TUMOR_AGENT           | Test Accuracy               | 85%               |
| CHEST_XRAY_AGENT            | Test Accuracy               | 90%               |
| TB_AGENT                    | Test AUC                    | 0.90              |
| PNEUMOTHORAX_AGENT          | Dice Coefficient            | 0.75              |
| MULTI_LABEL_CXR_AGENT       | Mean AUC (14 labels)        | 0.80              |
| DIABETIC_RETINOPATHY_AGENT  | Quadratic Weighted Kappa    | 0.80              |
| OCT_AGENT                   | Test Accuracy               | 90%               |
| HISTOPATHOLOGY_AGENT        | Macro F1                    | 0.90              |
| COLON_POLYP_AGENT           | Dice Coefficient            | 0.80              |
| DIABETIC_FOOT_ULCER_AGENT   | Macro F1                    | 0.75              |
| FRACTURE_AGENT              | Test AUC                    | 0.88              |
| MAMMOGRAPHY_AGENT           | Test AUC                    | 0.80              |
| HAM10000_AGENT              | Macro AUC                   | 0.85              |
| SKIN_LESION_AGENT           | Visual overlay quality      | Manual inspection |

### 13.5 Definition of Done — Per Role

A role is considered **complete** when:

**Sanika / Rushil / Jeet / Srujan:**
1. All agent folders exist at the specified paths
2. All checkpoints exist at the specified paths
3. All `label_map.json` and `metrics.json` files exist
4. All standalone tests (UC1–UC6 + individual T* tests) pass
5. Performance thresholds in Section 13.4 are met or documented
6. `Work/Reports/<Name>_Report.md` is written with all 6 mandatory sections
7. Integration readiness checklist (T4 in each file) is fully ticked

**Amritesh:**
1. All items T1–T8 in `Work/Amritesh.md` are complete
2. All 14 image agents are registered in `__init__.py`, `agent_decision.py`, and `config.py`
3. System integration tests SI1–SI6 all pass
4. `Work/Reports/Amritesh_Report.md` is written with all 5 mandatory sections

---

*End of WORK_DIVISION.md*

# Zero-Ambiguity Extension Plan

This document is the definitive extension plan for the current repository. It is based on the live codebase in [app.py](app.py), [config.py](config.py), [agents/agent_decision.py](agents/agent_decision.py), [agents/image_analysis_agent/__init__.py](agents/image_analysis_agent/__init__.py), [agents/image_analysis_agent/image_classifier.py](agents/image_analysis_agent/image_classifier.py), [agents/README.md](agents/README.md), [DATASETS.md](DATASETS.md), [DISEASES_AND_MODELS.md](DISEASES_AND_MODELS.md), [EXTEND_DISEASES_PLAN.md](EXTEND_DISEASES_PLAN.md), [PROJECT_CAPABILITIES.md](PROJECT_CAPABILITIES.md), and [assets/extra_details.md](assets/extra_details.md).

## 1. Current State, Without Assumptions

The repository currently supports three image-analysis branches in code, but only two are fully implemented:

- `CHEST_XRAY_AGENT` is implemented and backed by a saved checkpoint in [agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth](agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth).
- `SKIN_LESION_AGENT` is implemented and backed by a saved checkpoint in [agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar](agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar).
- `BRAIN_TUMOR_AGENT` is routed in the system design, but the implementation is still a placeholder in [agents/image_analysis_agent/brain_tumor_agent/brain_tumor_inference.py](agents/image_analysis_agent/brain_tumor_agent/brain_tumor_inference.py).

The actual routing logic lives in [agents/agent_decision.py](agents/agent_decision.py). The image modality detector is [agents/image_analysis_agent/image_classifier.py](agents/image_analysis_agent/image_classifier.py). The runtime entrypoints are [app.py](app.py), which exposes `/chat`, `/upload`, `/validate`, and `/transcribe`, and [config.py](config.py), which centralizes model paths, validation rules, and API settings.

The dataset story is split into two separate layers:

1. Training datasets for vision agents, documented in [DATASETS.md](DATASETS.md) and partially referenced in [assets/extra_details.md](assets/extra_details.md).
2. Retrieval documents for the RAG system, stored in `data/raw/`, parsed into `data/parsed_docs/`, indexed into `data/qdrant_db/`, and chunked into `data/docs_db/` by [ingest_rag_data.py](ingest_rag_data.py).

This means the next extension work is not just "add more disease names". It requires a deliberate expansion of the image-modality classifier, the agent registry, the configuration schema, the validation policy, and the dataset governance rules.

## 2. Non-Negotiable Design Rules

Use these rules for every future extension. They remove ambiguity and keep the codebase consistent with the current architecture.

### 2.1 One disease capability must have one explicit inference contract

Every new disease capability must expose a deterministic `predict(...)` method that returns a structured payload with at least:

- `label`
- `confidence`
- `needs_validation`
- `explanation`
- `artifact_path` when the model produces an image or mask

If a model produces multiple outputs, the output contract must be explicit. For example:

- classification-only: `label`, `confidence`
- segmentation: `label`, `confidence`, `mask_path`, `overlay_path`
- multi-label classification: `findings` as a list of `{label, confidence}` items

### 2.2 One modality classifier must decide the image family before disease inference

The current [agents/image_analysis_agent/image_classifier.py](agents/image_analysis_agent/image_classifier.py) only distinguishes `BRAIN MRI SCAN`, `CHEST X-RAY`, `SKIN LESION`, `OTHER`, and `NON-MEDICAL`. That is sufficient for the current project but not for the planned disease list in [DATASETS.md](DATASETS.md).

For the next extension phase, the modality classifier must be expanded to recognize at least:

- `BRAIN MRI SCAN`
- `CHEST X-RAY`
- `SKIN LESION`
- `FUNDUS RETINA`
- `OCT`
- `MAMMOGRAM`
- `HISTOPATHOLOGY PATCH`
- `MSK X-RAY`
- `COLONOSCOPY FRAME`
- `FOOT ULCER IMAGE`
- `OTHER`
- `NON-MEDICAL`

If the classifier confidence is below the chosen acceptance threshold, the system must return `OTHER` and request user clarification rather than guessing.

### 2.3 Do not overload existing agents with unrelated diseases

Keep the current structure of one disease family per agent folder. Do not turn `chest_xray_agent` into a catch-all pathology folder. That would make preprocessing, validation, and output handling ambiguous.

The only exception is shared base code. Shared base code is allowed in reusable utilities, but each disease capability still needs its own explicit inference wrapper and checkpoint path.

### 2.4 Validation is required by default for high-risk medical outputs

The current validation policy in [config.py](config.py) already marks the existing image agents as requiring validation. That rule must be extended to every new imaging agent that can affect diagnosis or triage.

High-risk output classes that must default to `needs_validation = True` include:

- cancer screening and grading
- lesion segmentation
- fracture detection
- tuberculosis screening
- pneumothorax detection
- multi-label chest findings

### 2.5 The RAG system remains separate from image training datasets

Do not mix the training datasets described in [DATASETS.md](DATASETS.md) with the document corpus used by RAG. The RAG pipeline is driven by medical PDFs under `data/raw/` and their derived Qdrant/doc-store artifacts, not by the image datasets used for CV training.

## 3. What Must Be Added to the Codebase

This section is the concrete change map. If an extension is added, these files are the required control points.

### 3.1 `config.py`

Add one model path per new disease capability under `MedicalCVConfig`. The existing pattern is already established there for brain tumor, chest X-ray, and skin lesion models.

Add or update the following fields as needed:

- `brain_tumor_model_path`
- `chest_xray_model_path`
- `skin_lesion_model_path`
- `tb_model_path`
- `pneumothorax_model_path`
- `multi_label_cxr_model_path`
- `diabetic_retinopathy_model_path`
- `mammography_model_path`
- `histopathology_model_path`
- `fracture_model_path`
- `oct_model_path`
- `diabetic_foot_ulcer_model_path`
- `colon_polyp_model_path`
- `ham10000_model_path`

Also extend `ValidationConfig.require_validation` so every new imaging agent has an explicit rule.

### 3.2 `agents/image_analysis_agent/__init__.py`

Add a dedicated inference wrapper import and class member for every new disease agent. Each wrapper must be initialized with a path from `config.medical_cv`.

### 3.3 `agents/image_analysis_agent/image_classifier.py`

Expand the modality labels and update the prompt so the classifier distinguishes all supported imaging families.

If classification confidence is uncertain, the classifier must return `OTHER` instead of forcing a wrong disease family.

### 3.4 `agents/agent_decision.py`

Extend the decision prompt and the routing logic so each modality maps to one unambiguous agent name.

The routing table must explicitly include the new agent names and must not rely on "closest match" behavior.

### 3.5 `app.py`

The response handling for uploaded images is currently written with a skin-lesion-specific branch. Replace that with a generic artifact-response mechanism so each image agent can return its own output artifact path without requiring special-case code in `app.py`.

### 3.6 `DATASETS.md`

Keep this file as the canonical dataset catalog. When a dataset is adopted for training, update this file with:

- source URL
- class labels
- number of images
- storage size
- task type
- whether the dataset is used for training, validation, or both

### 3.7 `PROJECT_CAPABILITIES.md`

Update the capabilities matrix after the code changes land. This file should describe only implemented or immediately available features, not speculative future work.

### 3.8 `agents/README.md`

Document the final validation flow, the image modality routing rules, and the new agent registry.

## 4. Required New Agents and Models

This section answers the "do we need more models or agents" question directly.

Yes. More agents and more trained models are required if the project is extended to the diseases listed in [DATASETS.md](DATASETS.md) and [EXTEND_DISEASES_PLAN.md](EXTEND_DISEASES_PLAN.md).

The recommended implementation is one disease-capability agent per disease, with shared utility code only where the input modality and output shape are truly the same.

## 5. Exact Agent List to Add

### 5.1 `BRAIN_TUMOR_AGENT`

Status in the codebase: scaffolded, but not finished.

Required artifact:

- one trained MRI classifier or segmenter, depending on the chosen final task

Recommended dataset:

- [Brain Tumor MRI Dataset — Kaggle (Masoud Nickparvar)](DATASETS.md)

Recommended task:

- 4-class classification: `glioma`, `meningioma`, `pituitary`, `no_tumor`

Recommended model:

- EfficientNet or ResNet classifier for the 4-class version
- BraTS-style segmentation only if tumor localization is a hard requirement and the dataset includes masks

Required files:

- [agents/image_analysis_agent/brain_tumor_agent/brain_tumor_inference.py](agents/image_analysis_agent/brain_tumor_agent/brain_tumor_inference.py)
- `agents/image_analysis_agent/brain_tumor_agent/models/<final_checkpoint>.pth`
- `agents/image_analysis_agent/brain_tumor_agent/train.py` if training is done in-repo

Output contract:

- `label`
- `confidence`
- `needs_validation = True`
- `explanation`

### 5.2 `TB_AGENT`

Dataset sources:

- NIH Chest X-ray14
- Shenzhen TB Chest X-ray
- Montgomery TB Chest X-ray

Recommended task:

- binary classification: TB vs normal

Recommended model:

- DenseNet121 or EfficientNet classifier

Preprocessing:

- resize to a fixed resolution
- grayscale normalization if the selected backbone expects it
- optional CLAHE
- optional lung-field crop

Output contract:

- `label`
- `confidence`
- `needs_validation = True`
- `explanation`

Required files:

- `agents/image_analysis_agent/tb_agent/tb_inference.py`
- `agents/image_analysis_agent/tb_agent/models/<final_checkpoint>.pth`
- `agents/image_analysis_agent/tb_agent/train.py`

### 5.3 `PNEUMOTHORAX_AGENT`

Dataset source:

- SIIM-ACR Pneumothorax Segmentation

Recommended task:

- segmentation preferred
- binary presence classification acceptable only if segmentation is explicitly out of scope

Recommended model:

- U-Net or DeepLabV3+ for segmentation

Output contract:

- `label`
- `confidence`
- `mask_path`
- `overlay_path`
- `needs_validation = True`

Required files:

- `agents/image_analysis_agent/pneumothorax_agent/pneumothorax_inference.py`
- `agents/image_analysis_agent/pneumothorax_agent/models/<final_checkpoint>.pth`
- `agents/image_analysis_agent/pneumothorax_agent/train.py`

### 5.4 `MULTI_LABEL_CXR_AGENT`

Dataset source:

- CheXpert

Recommended task:

- multi-label classification with 14 output heads

Recommended model:

- EfficientNet or DenseNet with sigmoid outputs

Output contract:

- `findings`: list of `{label, confidence, status}`
- `needs_validation = True`

Required files:

- `agents/image_analysis_agent/multi_label_cxr_agent/multi_label_cxr_inference.py`
- `agents/image_analysis_agent/multi_label_cxr_agent/models/<final_checkpoint>.pth`
- `agents/image_analysis_agent/multi_label_cxr_agent/train.py`

### 5.5 `DIABETIC_RETINOPATHY_AGENT`

Dataset sources:

- APTOS 2019 Blindness Detection
- EyePACS as a supplementary dataset when more data is needed

Recommended task:

- ordinal classification with 5 grades

Recommended model:

- EfficientNet-B4 or similar pretrained fundus classifier

Required preprocessing:

- circular crop to remove black borders
- contrast normalization
- eye-fundus-specific resize and color normalization

Output contract:

- `grade`
- `severity_text`
- `confidence`
- `needs_validation = True`

Required files:

- `agents/image_analysis_agent/diabetic_retinopathy_agent/diabetic_retinopathy_inference.py`
- `agents/image_analysis_agent/diabetic_retinopathy_agent/models/<final_checkpoint>.pth`
- `agents/image_analysis_agent/diabetic_retinopathy_agent/train.py`

### 5.6 `MAMMOGRAPHY_AGENT`

Dataset source:

- CBIS-DDSM

Recommended task:

- binary classification per finding type: benign vs malignant

Recommended model:

- EfficientNet classifier

Recommended sub-outputs:

- mass classification
- calcification classification

Output contract:

- `finding_type`
- `label`
- `confidence`
- `needs_validation = True`

Required files:

- `agents/image_analysis_agent/mammography_agent/mammography_inference.py`
- `agents/image_analysis_agent/mammography_agent/models/<final_checkpoint>.pth`
- `agents/image_analysis_agent/mammography_agent/train.py`

### 5.7 `HISTOPATHOLOGY_AGENT`

Dataset source:

- NCT-CRC-HE-100K

Optional supplementary dataset:

- PanNuke

Recommended task:

- patch-level multi-class classification

Recommended model:

- EfficientNet or ResNet patch classifier

Required preprocessing:

- stain normalization
- patch filtering
- fixed-size patch resizing

Output contract:

- `tissue_class`
- `confidence`
- `needs_validation = True`

Required files:

- `agents/image_analysis_agent/histopathology_agent/histopathology_inference.py`
- `agents/image_analysis_agent/histopathology_agent/models/<final_checkpoint>.pth`
- `agents/image_analysis_agent/histopathology_agent/train.py`

### 5.8 `FRACTURE_AGENT`

Dataset source:

- MURA

Recommended task:

- binary abnormal vs normal classification with body-part context

Recommended model:

- DenseNet169 or EfficientNet with body-part metadata

Output contract:

- `body_part`
- `label`
- `confidence`
- `needs_validation = True`

Required files:

- `agents/image_analysis_agent/fracture_agent/fracture_inference.py`
- `agents/image_analysis_agent/fracture_agent/models/<final_checkpoint>.pth`
- `agents/image_analysis_agent/fracture_agent/train.py`

### 5.9 `OCT_AGENT` or `AMD_AGENT`

Dataset source:

- Kermany 2018 OCT dataset

Recommended task:

- 4-class classification: `CNV`, `DME`, `DRUSEN`, `NORMAL`

Recommended model:

- EfficientNet-B3 or EfficientNet-B4

Output contract:

- `label`
- `confidence`
- `needs_validation = True`

Required files:

- `agents/image_analysis_agent/oct_agent/oct_inference.py`
- `agents/image_analysis_agent/oct_agent/models/<final_checkpoint>.pth`
- `agents/image_analysis_agent/oct_agent/train.py`

### 5.10 `DIABETIC_FOOT_ULCER_AGENT`

Dataset sources:

- DFU Kaggle dataset
- DFUC 2021 as a supplement

Recommended task:

- 4-class classification: `Ulcer`, `Infection`, `Normal`, `Gangrene`

Recommended model:

- EfficientNet classifier

Output contract:

- `label`
- `confidence`
- `needs_validation = True`

Required files:

- `agents/image_analysis_agent/diabetic_foot_ulcer_agent/diabetic_foot_ulcer_inference.py`
- `agents/image_analysis_agent/diabetic_foot_ulcer_agent/models/<final_checkpoint>.pth`
- `agents/image_analysis_agent/diabetic_foot_ulcer_agent/train.py`

### 5.11 `COLON_POLYP_AGENT`

Dataset sources:

- Kvasir-SEG
- CVC-ClinicDB

Recommended task:

- segmentation is preferred
- binary presence classification is secondary

Recommended model:

- U-Net or DeepLabV3+

Output contract:

- `mask_path`
- `overlay_path`
- `confidence`
- `needs_validation = True`

Required files:

- `agents/image_analysis_agent/colon_polyp_agent/colon_polyp_inference.py`
- `agents/image_analysis_agent/colon_polyp_agent/models/<final_checkpoint>.pth`
- `agents/image_analysis_agent/colon_polyp_agent/train.py`

### 5.12 `HAM10000_AGENT` or `DERMATOLOGY_MULTI_CLASS_AGENT`

Dataset source:

- HAM10000 / Skin Cancer MNIST

Recommended task:

- 7-class lesion classification

Recommended model:

- EfficientNet classifier paired with the existing segmentation branch if segmentation is retained

Output contract:

- `label`
- `confidence`
- `needs_validation = True`

Required files:

- `agents/image_analysis_agent/ham10000_agent/ham10000_inference.py`
- `agents/image_analysis_agent/ham10000_agent/models/<final_checkpoint>.pth`
- `agents/image_analysis_agent/ham10000_agent/train.py`

## 6. Dataset-by-Dataset Specification

This section removes any remaining ambiguity about what to train, from what data, and with what objective.

### 6.1 Brain Tumor MRI

- Source: [DATASETS.md](DATASETS.md)
- Use: 4-class classification first
- Labels: `glioma`, `meningioma`, `pituitary`, `no_tumor`
- Input format: resized MRI slices
- Model: EfficientNet or ResNet classifier
- Validation: mandatory

### 6.2 COVID-19 Chest X-ray

- Current state: already implemented
- Primary source reference: [assets/extra_details.md](assets/extra_details.md)
- Use: maintain existing binary classifier unless you intentionally retrain it
- Labels: `covid19`, `normal`
- Validation: mandatory

### 6.3 Skin Lesion / ISIC segmentation

- Current state: already implemented
- Source reference: [assets/extra_details.md](assets/extra_details.md)
- Use: maintain the segmentation model and add classification only if you also adopt HAM10000
- Validation: mandatory

### 6.4 Tuberculosis

- Source: [DATASETS.md](DATASETS.md)
- Labels: `TB`, `Normal`
- Model: DenseNet121 or EfficientNet
- Optional preprocessing: lung crop + CLAHE
- Validation: mandatory

### 6.5 Pneumothorax

- Source: [DATASETS.md](DATASETS.md)
- Labels: segmentation mask or binary presence
- Model: U-Net or DeepLabV3+
- Validation: mandatory

### 6.6 Multi-label CXR findings

- Source: [DATASETS.md](DATASETS.md)
- Labels: 14 CheXpert pathologies
- Model: DenseNet/EfficientNet with sigmoid heads
- Validation: mandatory

### 6.7 Diabetic retinopathy

- Source: [DATASETS.md](DATASETS.md)
- Labels: 5 grades
- Model: EfficientNet-B4
- Validation: mandatory

### 6.8 Mammography

- Source: [DATASETS.md](DATASETS.md)
- Labels: benign vs malignant per finding
- Model: EfficientNet classifier
- Validation: mandatory

### 6.9 Histopathology

- Source: [DATASETS.md](DATASETS.md)
- Labels: 9 tissue classes
- Model: EfficientNet or ResNet
- Validation: mandatory

### 6.10 Fracture detection

- Source: [DATASETS.md](DATASETS.md)
- Labels: normal vs abnormal per body part
- Model: DenseNet169 or EfficientNet
- Validation: mandatory

### 6.11 AMD / OCT

- Source: [DATASETS.md](DATASETS.md)
- Labels: `CNV`, `DME`, `DRUSEN`, `NORMAL`
- Model: EfficientNet-B3/B4
- Validation: mandatory

### 6.12 Diabetic Foot Ulcer

- Source: [DATASETS.md](DATASETS.md)
- Labels: `Ulcer`, `Infection`, `Normal`, `Gangrene`
- Model: EfficientNet classifier
- Validation: mandatory

### 6.13 Colon Polyp

- Source: [DATASETS.md](DATASETS.md)
- Labels: polyp mask or polyp present/absent
- Model: U-Net / DeepLabV3+
- Validation: mandatory

### 6.14 Expanded Dermatology / HAM10000

- Source: [DATASETS.md](DATASETS.md)
- Labels: `mel`, `nv`, `bcc`, `akiec`, `bkl`, `df`, `vasc`
- Model: EfficientNet classifier plus optional segmentation branch
- Validation: mandatory

## 7. Training Standard for Every New Model

Every new model must follow the same creation standard so that runtime code stays predictable.

### 7.1 Required training directory layout

Use one dataset-specific directory per disease, for example:

- `data/<disease_name>/train`
- `data/<disease_name>/val`
- `data/<disease_name>/test`

If the source dataset already comes with official splits, preserve them.

### 7.2 Required training script behavior

Every training script must:

- fix the random seed
- log dataset version and source URL
- log class mapping
- save the final checkpoint under `agents/image_analysis_agent/<disease>_agent/models/`
- save the label map alongside the model file
- export preprocessing constants used at training time

### 7.3 Required evaluation behavior

Every trained model must report metrics appropriate to the task:

- classification: accuracy, precision, recall, F1, AUC
- multi-label classification: per-label AUC, micro F1, macro F1
- segmentation: Dice, IoU, pixel precision/recall

### 7.4 Required inference behavior

Inference code must recreate the exact preprocessing pipeline used in training. If the training preprocessing is not stored, the model is not considered production-ready.

## 8. Validation and Safety Policy

Add these rules to [config.py](config.py) and keep them consistent across the backend.

### 8.1 Validation defaults

Default `needs_validation = True` for all medical image outputs except explicitly low-risk internal routing helpers.

### 8.2 Confidence thresholds

Use explicit thresholds rather than implicit judgment:

- modality classifier acceptance threshold: choose one fixed threshold and keep it constant in code
- image agent auto-accept threshold: only for non-critical outputs
- medical risk threshold: if the disease can alter treatment or triage, validation stays on regardless of confidence

### 8.3 Human-in-the-loop contract

[app.py](app.py) already exposes `/validate`. The response payloads from every high-risk agent must surface enough information for that endpoint and the frontend to show a review state.

## 9. How the RAG Side Should Be Extended

The RAG pipeline is independent from the CV models, but it should be extended in a controlled way.

### 9.1 RAG corpus location

Use `data/raw/` for medical PDFs and run [ingest_rag_data.py](ingest_rag_data.py) to populate the local document store and vector index.

### 9.2 RAG topic expansion rule

If a disease is supported as an image agent, the textual support for that disease should also be added to the RAG corpus when trustworthy PDFs are available. That keeps diagnostic guidance, red flags, and follow-up recommendations grounded in source documents.

### 9.3 RAG does not replace disease-specific image models

Do not use RAG as a substitute for image inference. RAG is for context, explanation, and up-to-date medical guidance. The vision agents remain the source of image-level predictions.

## 10. File-Level Work Plan, Organized by Dependency

This is not a timeline. It is the dependency order required for a deterministic build.

### 10.1 Foundation layer

1. Update [config.py](config.py) with model paths and validation flags.
2. Extend [agents/image_analysis_agent/image_classifier.py](agents/image_analysis_agent/image_classifier.py) with all supported modalities.
3. Update [agents/agent_decision.py](agents/agent_decision.py) with explicit routing rules for every supported agent.

### 10.2 Model integration layer

1. Add the new agent folder for each disease.
2. Add the inference wrapper.
3. Add the trained checkpoint file under that agent’s `models/` directory.
4. Add the label map and preprocessing metadata.

### 10.3 Runtime layer

1. Update [agents/image_analysis_agent/__init__.py](agents/image_analysis_agent/__init__.py) so the orchestrator can call the new wrapper.
2. Update [app.py](app.py) so artifact responses are generic and not hard-coded to skin lesion segmentation.
3. Extend the frontend only if the new agent returns masks, overlays, tables, or per-label findings that the current UI cannot already display.

### 10.4 Documentation layer

1. Update [DATASETS.md](DATASETS.md) after any dataset is adopted.
2. Update [DISEASES_AND_MODELS.md](DISEASES_AND_MODELS.md) when the implementation status changes.
3. Update [PROJECT_CAPABILITIES.md](PROJECT_CAPABILITIES.md) after implementation lands.
4. Keep [EXTEND_DISEASES_PLAN.md](EXTEND_DISEASES_PLAN.md) as a high-level roadmap, but let this file be the operational plan.

## 11. Recommended Output Contract Per Output Type

Use one of these payload shapes and do not invent a new one unless the task requires it.

### 11.1 Classification

```json
{
  "label": "string",
  "confidence": 0.0,
  "needs_validation": true,
  "explanation": "string"
}
```

### 11.2 Segmentation

```json
{
  "label": "string",
  "confidence": 0.0,
  "mask_path": "string",
  "overlay_path": "string",
  "needs_validation": true,
  "explanation": "string"
}
```

### 11.3 Multi-label classification

```json
{
  "findings": [
    { "label": "string", "confidence": 0.0, "status": "positive|negative|uncertain" }
  ],
  "needs_validation": true,
  "explanation": "string"
}
```

## 12. What Not to Do

- Do not add a new disease by editing only the prompt in [agents/agent_decision.py](agents/agent_decision.py).
- Do not add a new model path in [config.py](config.py) without adding the corresponding inference wrapper.
- Do not add training data references only in prose without updating [DATASETS.md](DATASETS.md).
- Do not mix medical training datasets with RAG PDFs.
- Do not leave validation behavior implicit for high-risk outputs.

## 13. Final Recommendation

If the project is extended exactly as the attached documentation implies, the correct implementation strategy is:

1. keep the current chest X-ray and skin lesion agents intact,
2. finish the brain tumor agent instead of leaving it as a stub,
3. add one explicit agent per new disease listed in [DATASETS.md](DATASETS.md),
4. expand the modality classifier to cover the new imaging families,
5. keep validation enabled for all high-risk findings,
6. store every checkpoint under the relevant `agents/image_analysis_agent/<disease>_agent/models/` directory,
7. update the dataset catalog and capability docs every time a new dataset or checkpoint is adopted.

That is the only low-ambiguity path that matches the repository’s current architecture.
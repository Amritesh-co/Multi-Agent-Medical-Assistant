# Datasets Reference

This document lists all datasets used or planned for the Multi-Agent Medical Assistant project — existing, in-progress, and future extensions. Organized by agent/disease.

---

## Table of Contents

- [Already Implemented](#already-implemented)
  - [COVID-19 Chest X-ray](#1-covid-19-chest-x-ray)
  - [Skin Lesion (ISIC)](#2-skin-lesion-isic)
- [Placeholder — Needs Dataset & Model](#placeholder--needs-dataset--model)
  - [Brain Tumor MRI](#3-brain-tumor-mri)
- [RAG Knowledge Base (PDF Documents)](#rag-knowledge-base-pdf-documents)
- [Planned Extensions — Image Agents](#planned-extensions--image-agents)
  - [Tuberculosis — Chest X-ray](#4-tuberculosis--chest-x-ray)
  - [Pneumothorax — Chest X-ray](#5-pneumothorax--chest-x-ray)
  - [Multi-label Chest X-ray Findings](#6-multi-label-chest-x-ray-findings)
  - [Diabetic Retinopathy](#7-diabetic-retinopathy)
  - [Mammography — Breast Cancer](#8-mammography--breast-cancer)
  - [Histopathology — Colorectal Cancer Tissue](#9-histopathology--colorectal-cancer-tissue)
  - [Musculoskeletal Fracture Detection](#10-musculoskeletal-fracture-detection)
  - [Age-related Macular Degeneration — OCT](#11-age-related-macular-degeneration--oct)
  - [Diabetic Foot Ulcer](#12-diabetic-foot-ulcer)
  - [Colon Polyp Detection](#13-colon-polyp-detection)
  - [Expanded Dermatology — ISIC Multi-class](#14-expanded-dermatology--isic-multi-class)

---

## Already Implemented

### 1. COVID-19 Chest X-ray

| Field | Details |
|---|---|
| **Agent** | `CHEST_XRAY_AGENT` |
| **Model** | `agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth` |
| **Status** | Trained and integrated |

**What it contains:**
Chest X-ray images labeled as COVID-19 positive or Normal. Binary classification dataset widely used for COVID-19 radiological screening research.

**Classes:** `covid19` / `normal`

**Task:** Binary image classification

**How it's used:**
The trained `.pth` model (PyTorch) is loaded at inference time. The `CHEST_XRAY_AGENT` runs the uploaded image through the model and returns a positive/negative COVID-19 result, followed by human validation.

---

### 2. Skin Lesion (ISIC)

| Field | Details |
|---|---|
| **Agent** | `SKIN_LESION_AGENT` |
| **Model** | `agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar` |
| **Status** | Trained and integrated |

**What it contains:**
Dermoscopic images of skin lesions from the ISIC (International Skin Imaging Collaboration) archive. The model performs semantic segmentation — producing a binary mask that highlights the lesion region.

**Task:** Semantic segmentation

**How it's used:**
The model segments the lesion area from the background and outputs a segmentation plot saved to `uploads/skin_lesion_output/segmentation_plot.png`, which is returned to the frontend alongside the text response. Human validation is required.

---

## Placeholder — Needs Dataset & Model

### 3. Brain Tumor MRI

| Field | Details |
|---|---|
| **Agent** | `BRAIN_TUMOR_AGENT` |
| **Model** | Not yet implemented (stub in `agents/agent_decision.py:455`) |
| **Status** | Agent routing exists; model returns a placeholder string |

#### Recommended Dataset

**[Brain Tumor MRI Dataset — Kaggle (Masoud Nickparvar)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)**

**What it contains:**
7,023 MRI brain scan images across 4 classes collected from multiple sources (figshare, SARTAJ, Br35H). Standard JPEG format, pre-organized into train/test folders.

| Class | Description |
|---|---|
| `glioma` | Malignant tumor in glial cells |
| `meningioma` | Tumor in meninges (usually benign) |
| `pituitary` | Tumor in the pituitary gland |
| `no_tumor` | Healthy brain MRI |

**Total images:** ~7,023  
**Format:** JPEG, variable resolution (resized to 224×224 for training)  
**Access:** Free, no registration required  
**License:** Public domain / research use  

**Task:** Multi-class classification (4 classes)

**How it will be used:**
Train an EfficientNet/ResNet classifier on the 4 classes. Replace the current placeholder response in `run_brain_tumor_agent()` with actual model inference. Human validation appended as with other CV agents.

**Alternative — Segmentation approach:**
For tumor localization use **[BraTS Challenge](https://www.synapse.org/Synapse:syn51156910/wiki/)** (Brain Tumor Segmentation) — provides ground truth segmentation masks but requires registration.

---

## RAG Knowledge Base (PDF Documents)

These are ingested into the Qdrant vector store via `ingest_rag_data.py` and used by the `RAG_AGENT`.

| Document | Location | Covers |
|---|---|---|
| `brain_tumor_2024.pdf` | `data/raw/` | Deep learning techniques for brain tumor detection |
| `brain_tumors_ucni.pdf` | `data/raw/` | Introduction to brain tumors |
| `covid_chest_xray_2024.pdf` | `data/raw/` | Deep learning for COVID-19 detection from CXR |
| `skin_lesion_2023.pdf` | `data/raw/` | Skin lesion classification techniques |

**To extend the RAG knowledge base:** Drop any medical PDF into `data/raw/` and run:
```bash
python ingest_rag_data.py --file ./data/raw/<filename>.pdf
```

---

## Planned Extensions — Image Agents

---

### 4. Tuberculosis — Chest X-ray

#### Primary Dataset

**[NIH Chest X-ray Dataset — Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)**

**What it contains:**
112,120 frontal-view chest X-rays from 30,805 unique patients. Each image has up to 14 disease labels extracted from radiology reports using NLP. TB is included as one of the labels.

| Detail | Value |
|---|---|
| Total images | 112,120 |
| Format | PNG, 1024×1024 |
| Size | ~42 GB |
| Access | Free on Kaggle |
| License | Public domain (NIH) |

**Classes used for TB agent:** `Infiltration` / `No Finding` (filter TB-relevant labels)

#### Supplementary Datasets

**[Shenzhen & Montgomery TB Chest X-ray — Kaggle](https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities)**

Smaller dedicated TB datasets (662 + 138 images) with binary labels (TB / Normal). Useful for fine-tuning.

**Task:** Binary classification (TB positive / negative)

**How it will be used:**
Train DenseNet121/EfficientNet on TB vs. Normal labels. New agent `tb_agent/` following the same pattern as `chest_xray_agent/`. Reuses existing CXR preprocessing pipeline.

---

### 5. Pneumothorax — Chest X-ray

**[SIIM-ACR Pneumothorax Segmentation — Kaggle](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)**

**What it contains:**
12,954 chest X-ray images with pixel-level segmentation masks for pneumothorax (collapsed lung) regions. Part of the SIIM-ACR 2019 Kaggle competition.

| Detail | Value |
|---|---|
| Total images | 12,954 |
| Format | DICOM → PNG (after conversion) |
| Masks | RLE-encoded binary masks |
| Access | Free on Kaggle (account required) |
| License | Research use |

**Task:** Segmentation (pneumothorax region) or binary presence classification

**How it will be used:**
For classification: train EfficientNet on presence/absence. For segmentation: train UNet on pixel masks. New agent `pneumothorax_agent/` — can share an agent folder with TB as `chest_xray_extended_agent/`.

---

### 6. Multi-label Chest X-ray Findings

**[CheXpert Dataset — Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/)**

**What it contains:**
224,316 chest X-rays from 65,240 patients, labeled for 14 pathologies using an automated labeler. One of the largest CXR datasets available.

| Detail | Value |
|---|---|
| Total images | 224,316 |
| Format | JPEG |
| Size | ~439 GB (full) / ~11 GB (small version) |
| Access | Free — requires registration (instant) |
| License | Research use only |

**Classes (14 labels):** Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion, and 9 others

**Uncertainty labels:** `positive` / `negative` / `uncertain` per finding

**Task:** Multi-label classification (multiple conditions per image)

**How it will be used:**
Train a multi-label EfficientNet (sigmoid output, one node per class). The agent returns a structured list of detected findings with confidence scores per finding. Use the **small version** (~11GB) for faster iteration.

---

### 7. Diabetic Retinopathy

**[APTOS 2019 Blindness Detection — Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection)**

**What it contains:**
3,662 retinal fundus photographs taken with fundus cameras under various imaging conditions in India. Graded by clinicians on a 0–4 severity scale.

| Detail | Value |
|---|---|
| Total images | 3,662 |
| Format | PNG, variable resolution |
| Size | ~9 GB |
| Access | Free on Kaggle (account required) |
| License | Competition terms (research use) |

**Classes (DR grading scale):**

| Grade | Severity |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

**Task:** Ordinal classification (5-class grading)

**Supplementary:**
**[EyePACS — Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection)** — 88,702 images, same grading scale, much larger but noisier labels.

**How it will be used:**
Train EfficientNet-B4 with preprocessing: CLAHE contrast enhancement + circular crop to remove black borders. Output: DR grade + severity description. New agent `diabetic_retinopathy_agent/`. Uses fundus image type routing (update `image_classifier.py` prompt to recognize fundus images).

---

### 8. Mammography — Breast Cancer

**[CBIS-DDSM Breast Cancer Image Dataset — Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)**

**What it contains:**
Preprocessed version of the Curated Breast Imaging Subset of DDSM (Digital Database for Screening Mammography). Converted from DICOM to PNG, organized into train/test splits. Two sub-tasks: mass detection and calcification detection.

| Detail | Value |
|---|---|
| Total images | ~10,239 (ROI crops + full mammograms) |
| Format | PNG (pre-converted from DICOM) |
| Size | ~6.8 GB |
| Access | Free on Kaggle — no DUA required |
| License | Public research use |

**Sub-datasets & Classes:**

| Sub-task | Classes |
|---|---|
| Mass classification | `BENIGN` / `MALIGNANT` |
| Calcification classification | `BENIGN` / `MALIGNANT` |
| Mass detection | Bounding box coordinates |

**ROI crops are pre-extracted** — no tiling pipeline needed.

**Task:** Binary classification (benign vs. malignant) per finding type

**How it will be used:**
Train two separate EfficientNet classifiers — one for masses, one for calcifications. New agent `mammography_agent/` with a two-stage output: finding type (mass/calcification) + malignancy prediction. Human validation mandatory (clinical sensitivity). Update `image_classifier.py` to recognize mammography images.

---

### 9. Histopathology — Colorectal Cancer Tissue

**[NCT-CRC-HE-100K — Zenodo (Kather et al.)](https://zenodo.org/record/1214456)**

**What it contains:**
100,000 non-overlapping image patches from hematoxylin & eosin (H&E) stained histological images of human colorectal cancer and normal tissue. All patches are 224×224 pixels at 0.5 microns/pixel.

| Detail | Value |
|---|---|
| Total images | 100,000 |
| Format | PNG, 224×224 pixels |
| Size | ~7.4 GB |
| Access | Free on Zenodo (no registration) |
| License | CC BY 4.0 |

**Classes (9 tissue types):**

| Label | Tissue |
|---|---|
| ADI | Adipose |
| BACK | Background |
| DEB | Debris |
| LYM | Lymphocytes |
| MUC | Mucus |
| MUS | Smooth muscle |
| NORM | Normal colon mucosa |
| STR | Cancer-associated stroma |
| TUM | Colorectal adenocarcinoma epithelium |

**Task:** Multi-class patch classification (9 classes)

**Supplementary — Nuclei Segmentation:**
**[PanNuke — from the Histopathology Datasets Index](https://github.com/maduc7/Histopathology-Datasets)**
189,744 nuclei patches across 19 tissue types. For nuclei-level analysis if needed.

**How it will be used:**
Train EfficientNet on 9-class tissue classification (patch-level). Input: cropped histopathology image patch. Output: tissue type distribution and tumor presence confidence. New agent `histopathology_agent/`. Standard JPEG/PNG input — no OpenSlide or WSI pipeline needed (patch-based only). Update `image_classifier.py` to recognize histopathology slide images.

---

### 10. Musculoskeletal Fracture Detection

**[MURA — Stanford ML Group](https://stanfordmlgroup.github.io/competitions/mura/)**

**What it contains:**
40,561 musculoskeletal X-ray images from 14,863 studies across 7 body parts. Each study is labeled as normal or abnormal (fracture/lesion) by board-certified radiologists.

| Detail | Value |
|---|---|
| Total images | 40,561 |
| Format | PNG |
| Size | ~6 GB |
| Access | Free — requires registration (instant) |
| License | Research use only |

**Body parts covered:** Elbow, Finger, Forearm, Hand, Humerus, Shoulder, Wrist

**Classes:** `normal` / `abnormal` (per body part)

**Task:** Binary classification per body part study

**How it will be used:**
Train DenseNet169 (architecture from original MURA paper) per body part, or a single model with body part as input context. New agent `fracture_agent/`. Output: abnormality detected / not detected + body part identified. Update `image_classifier.py` to recognize orthopedic X-ray images.

---

### 11. Age-related Macular Degeneration — OCT

**[Retinal OCT Images — Kermany 2018 — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)**

**What it contains:**
84,495 validated retinal OCT images organized into train/test folders. Collected from multiple hospitals, labeled by medical experts with a 2-stage quality check.

| Detail | Value |
|---|---|
| Total images | 84,495 |
| Format | JPEG |
| Size | ~6.3 GB |
| Access | Free on Kaggle |
| License | CC BY 4.0 |

**Classes (4 retinal conditions):**

| Class | Description |
|---|---|
| `CNV` | Choroidal neovascularization (wet AMD) |
| `DME` | Diabetic macular edema |
| `DRUSEN` | Early/intermediate dry AMD (drusen deposits) |
| `NORMAL` | Healthy retina |

**Task:** Multi-class classification (4 classes)

> **Note:** The [Farsiu 2014 dataset](https://www.kaggle.com/datasets/paultimothymooney/farsiu-2014) (also by the same uploader) was evaluated but rejected — it stores data in MATLAB `.mat` format and focuses on retinal layer boundary segmentation rather than disease classification, making it a poor fit for this project.

**How it will be used:**
Train EfficientNet-B3 on 4-class OCT classification. New agent `oct_agent/` (or `amd_agent/`). Output: condition name + clinical description. Update `image_classifier.py` to recognize retinal OCT scan images.

---

### 12. Diabetic Foot Ulcer

#### Primary Dataset

**[Diabetic Foot Ulcer (DFU) — Kaggle (laithjj)](https://www.kaggle.com/datasets/laithjj/diabetic-foot-ulcer-dfu)**

**What it contains:**
~1,426 high-definition images of diabetic foot ulcers collected from DFUC 2020 and Xiangya Hospital. Labeled across 4 clinical categories.

| Detail | Value |
|---|---|
| Total images | ~1,426 |
| Format | JPEG/PNG |
| Size | ~500 MB |
| Access | Free on Kaggle |
| License | CC BY-NC |

**Classes:** `Ulcer` / `Infection` / `Normal` / `Gangrene`

#### Supplementary Dataset (Recommended for Better Training)

**[DFUC 2021 Challenge Dataset](https://dfu-challenge.github.io/dfuc2021.html)**

15,683 DFU patches with the same 4-class structure. Requires application (usually approved within days for academic use).

| Detail | Value |
|---|---|
| Total images | 15,683 (5,955 train / 5,734 test / 3,994 unlabeled) |
| Format | JPEG patches |
| Access | Application required at dfu-challenge.github.io |

**Task:** Multi-class classification (4 classes)

**How it will be used:**
Train EfficientNet on 4-class DFU classification using the Kaggle dataset, optionally augmented with DFUC 2021 for better generalization. New agent `dfu_agent/`. Output: ulcer type + severity description. Update `image_classifier.py` to recognize foot wound images.

---

### 13. Colon Polyp Detection

**[Kvasir-SEG — Simula Research Lab](https://datasets.simula.no/kvasir-seg/)**

**What it contains:**
1,000 gastrointestinal polyp images with corresponding pixel-level segmentation masks from colonoscopy procedures. Part of the larger Kvasir dataset.

| Detail | Value |
|---|---|
| Total images | 1,000 |
| Masks | 1,000 binary segmentation masks |
| Format | JPEG images + JPEG masks |
| Resolution | 332×487 to 1920×1072 |
| Size | ~44 MB |
| Access | Free, direct download |
| License | CC BY 4.0 |

**Task:** Binary segmentation (polyp region vs. background)

**Supplementary:**
**[CVC-ClinicDB](https://www.cvc.uab.es/CVC-Clinic/)** — 612 colonoscopy frames with masks. Combine with Kvasir-SEG for a total ~1,612 samples.

**How it will be used:**
Train UNet or DeepLabV3+ for polyp segmentation. New agent `polyp_agent/`. Output: segmentation overlay image showing detected polyp region. Update `image_classifier.py` to recognize colonoscopy frame images.

---

### 14. Expanded Dermatology — ISIC Multi-class

**[HAM10000 / Skin Cancer MNIST — Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)**

**What it contains:**
10,015 dermoscopic images spanning 7 types of skin lesions, collected from two major dermatoscopy sources (HAM10000 — Human Against Machine with 10,000 training images). A significant expansion over the current binary skin lesion agent.

| Detail | Value |
|---|---|
| Total images | 10,015 |
| Format | JPEG, 600×450 |
| Size | ~2.4 GB |
| Access | Free on Kaggle |
| License | CC BY-NC 4.0 |

**Classes (7 lesion types):**

| Code | Full Name | Malignancy |
|---|---|---|
| `mel` | Melanoma | Malignant |
| `nv` | Melanocytic nevi | Benign |
| `bcc` | Basal cell carcinoma | Malignant |
| `akiec` | Actinic keratoses / Intraepithelial carcinoma | Pre-malignant |
| `bkl` | Benign keratosis-like lesions | Benign |
| `df` | Dermatofibroma | Benign |
| `vasc` | Vascular lesions | Benign |

**Class imbalance:** `nv` dominates (~67%). Use weighted sampling or focal loss.

**Task:** Multi-class classification (7 classes) — upgrades the current binary skin lesion agent

**How it will be used:**
The current `SKIN_LESION_AGENT` performs segmentation only. This dataset enables a classification upgrade — identify the lesion type in addition to segmenting it. Add a classification head alongside the existing segmentation model, or train a separate classifier and run both. Human validation mandatory.

---

## Summary Table

| # | Disease | Dataset | Images | Size | Task | Diseases / Conditions Detected | Access | Agent Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | COVID-19 CXR | Internal | — | — | Classification | COVID-19 Positive, Normal | — | Done |
| 2 | Skin Lesion | ISIC | — | — | Segmentation | Lesion region (binary mask) | — | Done |
| 3 | Brain Tumor | [Kaggle (Nickparvar)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | 7,023 | ~700 MB | Classification | Glioma, Meningioma, Pituitary Tumor, No Tumor | Free | Placeholder |
| 4 | Tuberculosis | [NIH CXR14 (Kaggle)](https://www.kaggle.com/datasets/nih-chest-xrays/data) + [Shenzhen/Montgomery (Kaggle)](https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities) | 112k+ | ~42 GB + ~1 GB | Classification | Tuberculosis, Normal | Free | Planned |
| 5 | Pneumothorax | [SIIM-ACR (Kaggle)](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation) | 12,954 | ~10 GB | Segmentation | Pneumothorax region, No Finding | Free | Planned |
| 6 | Multi-label CXR | [CheXpert (Stanford)](https://stanfordmlgroup.github.io/competitions/chexpert/) | 224,316 | ~11 GB (small) / ~439 GB (full) | Multi-label Classification | Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion + 9 more | Registration | Planned |
| 7 | Diabetic Retinopathy | [APTOS 2019 (Kaggle)](https://www.kaggle.com/c/aptos2019-blindness-detection) | 3,662 | ~9 GB | Ordinal Classification | No DR, Mild DR, Moderate DR, Severe DR, Proliferative DR | Free | Planned |
| 8 | Mammography | [CBIS-DDSM (Kaggle)](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) | ~10,239 | ~6.8 GB | Classification | Benign Mass, Malignant Mass, Benign Calcification, Malignant Calcification | Free | Planned |
| 9 | Histopathology | [Kather NCT-CRC (Zenodo)](https://zenodo.org/record/1214456) | 100,000 | ~7.4 GB | Multi-class Classification | Adipose, Background, Debris, Lymphocytes, Mucus, Smooth Muscle, Normal Mucosa, Cancer Stroma, Tumor Epithelium | Free | Planned |
| 10 | Fracture | [MURA (Stanford)](https://stanfordmlgroup.github.io/competitions/mura/) | 40,561 | ~6 GB | Classification | Normal, Abnormal — across Elbow, Finger, Forearm, Hand, Humerus, Shoulder, Wrist | Registration | Planned |
| 11 | AMD / OCT | [Kermany 2018 (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) | 84,495 | ~6.3 GB | Multi-class Classification | CNV (Wet AMD), DME (Diabetic Macular Edema), DRUSEN (Dry AMD), Normal | Free | Planned |
| 12 | Diabetic Foot Ulcer | [laithjj (Kaggle)](https://www.kaggle.com/datasets/laithjj/diabetic-foot-ulcer-dfu) + [DFUC 2021](https://dfu-challenge.github.io/dfuc2021.html) | 1,426 + 15,683 | ~500 MB + ~2 GB | Classification | Ulcer, Infection, Normal, Gangrene | Free + Application | Planned |
| 13 | Colon Polyp | [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) + [CVC-ClinicDB](https://www.cvc.uab.es/CVC-Clinic/) | 1,000 + 612 | ~44 MB + ~100 MB | Segmentation | Polyp region (binary mask), No Polyp | Free | Planned |
| 14 | Expanded Dermatology | [HAM10000 (Kaggle)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) | 10,015 | ~2.4 GB | Multi-class Classification | Melanoma, Melanocytic Nevi, Basal Cell Carcinoma, Actinic Keratoses, Benign Keratosis, Dermatofibroma, Vascular Lesions | Free | Planned |

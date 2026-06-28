# VizWiz-VQA: Vision-Language Models for Visual Question Answering & Answerability Prediction

This repository contains a comprehensive research codebase that applies **pretrained Vision-Language Models (VLMs)** to the **VizWiz dataset** for two related tasks: **Visual Question Answering (VQA)** and **Answerability Prediction (AP)**. The work evaluates and compares six transformer-based VLMs (CLIP, Open CLIP, SigLIP, BLIP, ViLT, and BERT+ViT), explores dual-encoder backbone ablations, and combines the strongest models into an **ensemble of classifiers** that improves over any single model. It also includes a full exploratory analysis of the VizWiz dataset and a dedicated notebook for visualizing all experimental results.

The published research can be found here: [IEEE Xplore Conference Paper](https://doi.org/10.1109/ICCIT68739.2025.11490380)

---

## Table of Contents

1. [Background & Motivation](#background--motivation)
2. [Tasks](#tasks)
3. [Repository Structure](#repository-structure)
4. [Notebook-by-Notebook Guide](#notebook-by-notebook-guide)
   - [Dataset Analysis](#1-vizwiz-dataset-analysis)
   - [Code Dissection](#2-vizwiz-vqa-code-dissection)
   - [Single-Model Pipelines (Six Models)](#3-single-model-pipelines)
   - [Ensemble of Classifiers](#4-ensemble-of-classifiers)
   - [Experimental Results Visualization](#5-experimental-results-visualization)
5. [Common Pipeline Architecture](#common-pipeline-architecture)
6. [Models & Encoders Used](#models--encoders-used)
7. [Results Summary](#results-summary)
8. [How to Run](#how-to-run)
9. [Environment & Dependencies](#environment--dependencies)
10. [License & Acknowledgements](#license--acknowledgements)

---

## Background & Motivation

**VizWiz** is a real-world VQA dataset consisting of images captured by blind users, paired with spoken questions and 10 crowdsourced answers per question. The dataset is particularly challenging because:

- Many images are **blurry, poorly framed, or under-exposed** (taken without sight).
- A large fraction of questions are **unanswerable** from the image alone.
- Answers span four canonical answer types: `yes/no`, `number`, `other`, and `unanswerable`.

This research work investigates whether modern pretrained **vision-language models** can be reused as **frozen feature extractors** to build accurate, efficient VQA and Answerability classifiers on VizWiz, without re-training the underlying backbone.

---

## Tasks

Each notebook addresses both tasks jointly:

| Task    | Description                                                                           | Output                            |
| ------- | ------------------------------------------------------------------------------------- | --------------------------------- |
| **VQA** | Given an image and a question, predict the correct answer from the answer vocabulary. | Top-1 predicted answer (`answer`) |
| **AP**  | Given an image and a question, predict whether the question is answerable.            | Scalar probability (`answerable`) |

Both predictions are submitted to the VizWiz evaluation server in the expected JSON format.

---

## Repository Structure

```
.
├── README.md
├── .gitignore
│
├── vizwiz-dataset-analysis.ipynb
├── vizwiz-vqa-code-dissection.ipynb
│
├── vizwiz-vqa-using-clip-rn50.ipynb
├── vizwiz-vqa-using-clip-feature-extractors.ipynb
├── vizwiz-vqa-using-open-clip-convnet.ipynb
├── vizwiz-vqa-using-open-clip-feature-extractors.ipynb
├── vizwiz-vqa-using-siglip-feature-extractors.ipynb
├── vizwiz-vqa-using-blip.ipynb
├── vizwiz-vqa-using-vilt.ipynb
├── vizwiz-vqa-using-bert-vit-patch-16.ipynb
├── vizwiz-vqa-using-ensemble-of-classifiers.ipynb
│
└── experimental-results-visualization.ipynb
```

---

## Notebook-by-Notebook Guide

### 1. VizWiz Dataset Analysis

**File:** `vizwiz-dataset-analysis.ipynb`

An exploratory data analysis (EDA) of the VizWiz 2023 Edition dataset. Each section is self-contained.

- **Exploratory Data Analysis**
  - _Analysis of Questions_ — top-10 question bar charts per split + per-split word clouds.
  - _Analysis of Answers_ — pie charts of `answer_type` and `answerable`; per-split answer word clouds.
  - _Analysis of Images_ — sample image per answer type for the train and val splits.

This notebook produces the foundation understanding used by every VQA/AP pipeline in the repository.

---

### 2. VizWiz-VQA Code Dissection

**File:** `vizwiz-vqa-code-dissection.ipynb`

A pedagogical notebook that walks step-by-step through the building blocks of a VQA pipeline before any full training run. It is intended as a reference / sanity-check notebook for the other VLM pipelines.

- **Basic structure of the data**
- **Image and Text Encoder output analysis**
- **Answer vocabulary creation analysis** — Replicates the canonical VizWiz 3-stage tie-breaking policy on the training set, then on the validation set, and finally on a **joint train+val** vocabulary:
  1. Take the most-frequent answer among the 10.
  2. Break ties using the global answer frequency.
  3. Break further ties using pairwise Levenshtein distance.

  This analysis motivates the decision (used by every downstream notebook) to **build a single joint answer vocabulary from train + val**.

---

### 3. Single-Model Pipelines

Each notebook follows the same canonical pipeline so that the only variable is the **pretrained encoder** used to embed (image, question) pairs.

Common structure of every model notebook:

1. **Processing Images & Questions** — Loading the encoder (and tokenizer/processor) from 🤗 `transformers`, freezing its parameters.
2. **Feature Extraction of Image-Question pairs & Save To File** — Forward passes through the frozen encoder for every (image, question); embeddings are **pickled** to disk so subsequent epochs do not re-encode the data.
3. **Loading Processed Embeddings** — Reloads the cached embeddings.
4. **Creating the Answer Vocabulary** — Same 3-stage tie-breaking logic from the dissection notebook.
5. **Creating Dataset Class** — `VizWizDataset` wraps (embedding, answer, answerable) triples; `VizWizTestDataset` wraps (embedding, image) for inference.
6. **Preparing dataloaders** — Train/val/test `DataLoader`s.
7. **Building The VQA Model's Architecture** — `VQAModel`: two-layer MLP (`LayerNorm → Dropout → Linear → LayerNorm → Linear`) over the joint image-question embedding; cross-entropy loss over the answer vocabulary.
8. **Training the VQA Model** — Standard supervised training loop with metric tracking (loss + VizWiz accuracy on train and val).
9. **Building The AP Model's Architecture** — `APModel`: same MLP trunk, single logit, `BCELoss`.
10. **Training the AP Model** — Train/val tracking of binary cross-entropy and answerability score.
11. **Inference (Testing split)**
    - _Inference for the VQA Model_ — Produces `submission.json` with `[{image, answer}]`.
    - _Inference for the AP Model_ — Produces `ap_submission.json` with `[{image, answerable}]`.

#### Individual notebooks

| Notebook                                              | Encoder                                          | Notes                                                                                  |
| ----------------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------- |
| `vizwiz-vqa-using-clip-rn50.ipynb`                    | **CLIP (RN50x64)**                               | Dual-encoder; convolution-based image tower.                                           |
| `vizwiz-vqa-using-clip-feature-extractors.ipynb`      | **CLIP (ViT-L/14 @ 336px)**                      | Default CLIP feature extractors used for embedding.                                    |
| `vizwiz-vqa-using-open-clip-convnet.ipynb`            | **Open CLIP ConvNeXt-L**                         | OpenCLIP backbone; convolutional image tower.                                          |
| `vizwiz-vqa-using-open-clip-feature-extractors.ipynb` | **Open CLIP (ViT-L/14)**                         | OpenCLIP dual-encoder.                                                                 |
| `vizwiz-vqa-using-siglip-feature-extractors.ipynb`    | **SigLIP**                                       | Sigmoid-loss variant of CLIP; strongest single VQA model in this work.                 |
| `vizwiz-vqa-using-blip.ipynb`                         | **BLIP**                                         | Bootstrapped Language-Image Pretraining.                                               |
| `vizwiz-vqa-using-vilt.ipynb`                         | **ViLT**                                         | Vision-and-Language Transformer (single-stream).                                       |
| `vizwiz-vqa-using-bert-vit-patch-16.ipynb`            | **BERT (`bert-base-uncased`) + ViT (`patch16`)** | Modular combination of two independent unimodal encoders, fused by the downstream MLP. |

---

### 4. Ensemble of Classifiers

**File:** `vizwiz-vqa-using-ensemble-of-classifiers.ipynb`

Rather than retrain, this notebook **reuses the pretrained image-question embeddings from the three strongest single models (BLIP, SigLIP, CLIP)** and trains new VQA and AP classifiers on their **concatenated embeddings**.

- **Loading BLIP, SigLIP and CLIP model states and embeddings** — Loads cached pickle files from the three single-model runs.
- **Creating the Answer Vocabulary** — Joint train+val vocabulary (canonical 3-stage tie-breaking).
- **Creating Dataset Class** — `VizWizDataset` now returns a tuple `(blip_emb, siglip_emb, clip_emb, answer, answerable)` so the dataloader can be iterated independently for each sub-model.
- **Preparing dataloaders** — Three parallel dataloaders (one per source model).
- **Building The VQA Model's Architecture** — `EnsembleVQAModel` consumes BLIP, SigLIP, and CLIP embeddings through parallel MLP towers, then fuses them via concatenation followed by a final classification MLP that predicts over the answer vocabulary.
- **Building The AP Model's Architecture** — `EnsembleAPModel` mirrors the VQA architecture but with a single sigmoid output.
- **Loading the models** — Loads the previously trained VQA and AP ensemble checkpoints.
- **Inference for the VQA Model** — Generates `submission.json`.
- **Inference for the AP Model** — Generates `ap_submission.json`.
- **Qualitative Analysis** — Side-by-side inspection of predictions vs. ground-truth answers on sample test images.

This is the **best-performing configuration** in the repository on both VQA accuracy and answerability precision.

---

### 5. Experimental Results Visualization

**File:** `experimental-results-visualization.ipynb`

A standalone visualization notebook that consumes the **tabular results** produced by the training notebooks and turns them into publication-ready plots.

It includes:

- **Overall VQA accuracy comparison** — Bar chart of VQA accuracy across all six single models (CLIP, Open CLIP, SigLIP, BLIP, ViLT, BERT+ViT).
- **Overall Answerability comparison** — Grouped bar chart of AP Average Precision and AP F1-Score across the six models.
- **Per-answer-type VQA accuracy** — Per-answer-type (`other`, `unans.`, `yes/no`, `number`) accuracy grouped by model.
- **test-dev vs. test-std (top-3 models)** — Side-by-side comparison of VQA accuracy on the two official VizWiz test splits for SigLIP, BLIP, and CLIP.
- **Combined AP performance (test-dev / test-std)** — Average Precision and F1-Score bar chart for the top-3 models on both splits.
- **Per-answer-type, per-test-set faceted plot** — One subplot per candidate model showing per-answer-type accuracy on `test-dev` vs. `test-std`.
- **Ensemble overall performance** — VQA accuracy and AP metrics (Avg. Precision, F1) on `test-dev` and `test-std`.
- **Ensemble per-answer-type performance** — Per-answer-type accuracy for the ensemble on both test splits.
- **Dual-encoder backbone ablation** — VQA accuracy and AP metrics comparing ViT-based vs. Convolution-based backbones for CLIP and OpenCLIP.
- **Dual-encoder per-answer-type ablation** — Per-answer-type line plots contrasting backbone choices.
- **Ensemble composition ablation** — VQA accuracy and AP metrics for the four possible 2- and 3-way combinations of SigLIP, BLIP, and CLIP.
- **Ensemble per-answer-type ablation** — Per-answer-type accuracy across the four ensemble configurations.

All plots are exported as high-resolution PNGs (300 dpi) for inclusion in reports/papers.

---

## Common Pipeline Architecture

The single-model and ensemble notebooks all conform to the same architecture:

```
Image + Question
        │
        ▼
┌──────────────────────────────┐
│   Frozen Pretrained Encoder  │   ◄── CLIP / OpenCLIP / SigLIP /
│   (image tower + text tower) │       BLIP / ViLT / BERT+ViT
└──────────────────────────────┘
        │
        ▼
  Image-Question Embedding
        │
        ├──► VQA Head: Linear(768→hidden) → Linear(hidden→|V|) → argmax
        │
        └──► AP Head:  Linear(768→hidden) → Linear(hidden→1) → sigmoid
```

In the **ensemble** notebook the three encoder outputs are concatenated before the heads.

---

## Models & Encoders Used

| Model                   | Image Tower                         | Text Tower            |
| ----------------------- | ----------------------------------- | --------------------- |
| CLIP (ViT-L/14 @ 336px) | ViT-L/14                            | Transformer           |
| CLIP (RN50x64)          | ResNet-50x64                        | Transformer           |
| Open CLIP (ViT-L/14)    | ViT-L/14                            | Transformer           |
| Open CLIP (ConvNeXt-L)  | ConvNeXt-L                          | Transformer           |
| SigLIP                  | ViT                                 | Transformer           |
| BLIP                    | ViT                                 | Transformer           |
| ViLT                    | ViT (single-stream)                 | Embedded into patches |
| BERT + ViT (patch16)    | `google/vit-base-patch16-224-in21k` | `bert-base-uncased`   |

All backbones are used **frozen** — only the downstream MLP heads are trained.

---

## Results Summary

### Single-model VQA & AP on test-dev

| Model      | VQA Acc. (%) | AP Avg. Precision | AP F1-Score |
| ---------- | -----------: | ----------------: | ----------: |
| CLIP       |        61.36 |             80.53 |       68.37 |
| Open CLIP  |        61.14 |             77.34 |       64.61 |
| **SigLIP** |    **62.21** |             79.58 |   **69.14** |
| BLIP       |        61.50 |             75.64 |       66.53 |
| ViLT       |        54.17 |             68.82 |       50.54 |
| BERT + ViT |        54.22 |             64.75 |       58.38 |

### Ensemble (SigLIP + BLIP + CLIP) on test-dev / test-std

| Split    | VQA Acc. (%) | AP Avg. Precision | AP F1-Score |
| -------- | -----------: | ----------------: | ----------: |
| test-dev |    **64.02** |         **82.54** |   **69.87** |
| test-std |        63.22 |             82.02 |       69.33 |

The **ensemble of classifiers** outperforms every single model on VQA accuracy (≈+1.8 over the best single model) and on average precision for answerability.

---

## How to Run

The notebooks were originally developed on **Kaggle** with the VizWiz 2023 Edition dataset. Each notebook assumes this exact path layout.

### Option A — Kaggle (recommended)

1. Create a new Kaggle Notebook.
2. Add the **VizWiz 2023 Edition** dataset.
3. Upload one of the notebooks from this repo.
4. Run all cells. Trained checkpoints are pickled to `/kaggle/working/`.
5. Submission files (`submission.json`, `ap_submission.json`) are written to the working directory and can be downloaded.

### Option B — Local

1. Clone the repository.
2. Download the VizWiz 2023 Edition dataset and arrange it as:
   ```
   /kaggle/input/vizwiz-2023-edition/
       Annotations/{train,val,test}.json
       train/train/*.jpg
       val/val/*.jpg
       test/test/*.jpg
   ```
   (Either symlink or edit the `INPUT_PATH` constants at the top of each notebook.)
3. Create a Python environment (Python 3.10+) and install dependencies (see below).
4. Launch Jupyter and run notebooks end-to-end.

### Recommended execution order

1. `vizwiz-dataset-analysis.ipynb` — explore the dataset first.
2. `vizwiz-vqa-code-dissection.ipynb` — understand the building blocks.
3. Each `vizwiz-vqa-using-*.ipynb` notebook — train each single model.
4. `vizwiz-vqa-using-ensemble-of-classifiers.ipynb` — reuses embeddings from BLIP, SigLIP, CLIP.
5. `experimental-results-visualization.ipynb` — visualize the results.

---

## Environment & Dependencies

The notebooks use the following stack (versions are the ones used during development; recent equivalents should work):

- **Python** 3.10+
- **PyTorch** (CUDA build recommended)
- **transformers** (Hugging Face) — for AutoTokenizer, AutoFeatureExtractor, AutoModel
- **pandas**, **numpy**
- **scikit-learn** — `OneHotEncoder`, `train_test_split`, `average_precision_score`
- **Pillow (PIL)** — image loading
- **matplotlib**, **wordcloud**, **seaborn** — visualization
- **Levenshtein** — answer tie-breaking (used in the code-dissection notebook and downstream vocab construction)
- **tqdm** — progress bars
- **tabulate** — formatted tables
- **pickle** — caching encoder embeddings
- **ftfy** — text cleanup

Install with:

```bash
pip install torch transformers pandas numpy scikit-learn pillow matplotlib wordcloud seaborn python-Levenshtein tqdm tabulate ftfy
```

A `.gitignore` is included to exclude Python build artifacts, virtual environments, and Jupyter checkpoints.

---

## License & Acknowledgements

This repository is intended as a research / educational resource.

- **Dataset:** VizWiz 2023 Edition — please refer to the original VizWiz dataset license and terms.
- **Pretrained models:** All encoders are publicly released on the Hugging Face Hub under their respective licenses (CLIP: OpenAI, OpenCLIP: LAION, SigLIP: Google, BLIP: Salesforce, ViLT: dandelin, BERT/ViT: Google).

If you use this work, please cite the original VizWiz dataset paper, the respective model papers, and the associated research paper published in IEEE ICCIT 2025 for this work.

# 3DReconstruction_SpatialTranscriptomics

## Overview

This repository banchmarks methods for **3D reconstruction of spatial transcriptomics data** on the small MERFISH (Moffit et al. 2018 - "Molecular, spatial, and functional single-cell profiling of the
             hypothalamic preoptic region") and STARmap datasets (Wang et al. 2018 - "Three-dimensional intact-tissue sequencing of single-cell transcriptional states").

It focuses on reconstructing continuous spatial structures from aligned slices and comparing different computational frameworks for this task (SpatialZ, UOT-based, SpatialZ + PCA/scVI, etc.).

---

## Structure

### Folders

* **`data/`** – input datasets
* **`exploration/`** – exploratory notebooks and experiments
* **`inference/`** – inference outputs and figures
* **`output/`** – saved results and reconstructions
* **`uot/`** – unbalanced optimal transport (UOT) implementation
* **`utils/`** – helper functions

---

### Core Scripts

#### Reconstruction & methods

* **`SpatialZ.py`** – main SpatialZ method

---

#### Custom methods

* **`my_method_pca.py`** – reconstruction using SpatialZ (for cell locations + cell type) and PCA (for gene expression)
* **`my_method_scvi.py`** – reconstruction using SpatialZ (for cell locations + cell type) and scVI (for gene expression)
* **`my_method_pretrained_scvi.py`** – reconstruction using SpatialZ (for cell locations + cell type) and pre-trained scVI (for gene expression)

---

#### Comparison scripts

* **`compare_merfish_small.py`** – comparison on MERFISH dataset
* **`compare_starmap.py`** – comparison on STARmap dataset
* **`compare_location_models.ipynb`** – notebook comparing other spatial models (KDE vs. OT)

---

#### Evaluation

* **`evaluation_merfish_small.py`** – evaluation on MERFISH
* **`evaluation_starmap.py`** – evaluation on STARmap

---

#### Data processing & experiments

* **`cut_dataset.ipynb`** – dataset preprocessing and slice splitting
* **`split_train_test*.ipynb`** – train/test splitting
* **`optimization_vs_kde.ipynb`** – experiments comparing other spatial models (KDE vs. OT)

---

### Requirements

* **`requirements_compare.txt`** – dependencies for comparisons
* **`requirements_scvi.txt`** – dependencies for scVI models
* **`requirements_spatialz.txt`** – dependencies for SpatialZ

---
license: cc-by-4.0
---

<a name="readme-top"></a>

# What Benchmarks Don’t Test: An Existential Gap in Concept Drift Detection


---

## Abstract
Business processes evolve over time, and event logs recorded across these changes often mix traces from multiple process versions.
Concept drift detection identifies if and when such changes occur. Respective detection algorithms are evaluated almost exclusively on two synthetic benchmarks, the Concept Drift Log Generator (CDLG) dataset and CDRIFT, on which state-of-the-art methods report strong detection accuracy. 
We show that this accuracy is conditional on a structural limitation neither benchmark discloses: both cover only process changes that alter directly-follows relations, leaving the entire class of pure existential changes entirely untested. Through a systematic coverage analysis of all change operations present in CDLG and CDRIFT, we demonstrate that no existing evaluation can reveal whether a detector handles this class of drift.
To support future evaluation, we contribute a dataset of 25 synthetic event logs with existential-only drifts across four structural categories, realizing 11 of the 19 admissible existential transitions. Based on our findings, we propose the temporal/existential decomposition as a coverage criterion for concept drift benchmark design.

---

## Experiments

To demonstrate the impact of including existential-only drift scenarios when evaluating detection algorithms, we
conduct experiments on the CDLG and CDRIFT datasets. For each dataset, we run the CV4CDD-D4 framework by Kraus
and van der Aa based on a DFG representation, and an adapted version of it based on the ARM representation. The results of this can be found in the VerificationResults folder.

## Drift Classification of the Benchmarks

[`drift_classification/`](drift_classification) classifies every drift of the existential dataset, CDRIFT and CDLG as temporal-only, existential-only, mixed or existence drift, and maps it to the transition IDs of the temporal and existential transition tables.
For each change point, the ARM is derived from the traces directly before and after the change (a margin around the change point and the full window of gradual changes are excluded) with threshold τ = 0.95 and no artificial start/end activities. Only pairs of distinct activities are considered; for repeated activities, temporal precedence is taken from their first occurrence in a trace.

Result tables, one row per drift (CDLG: per change event):

| File | Logs | Drifts |
|---|---|---|
| [`own_drift_classes.csv`](drift_classification/results/own_drift_classes.csv) | 25 (existential dataset) | 25 |
| [`cdrift_drift_classes.csv`](drift_classification/results/cdrift_drift_classes.csv) | 115 (all in-scope CDRIFT logs, incl. noisy ones) | 157 |
| [`cdlg_drift_classes.csv`](drift_classification/results/cdlg_drift_classes.csv) | first 1,000 of the 2,858 noise-free CDLG test logs with drifts | 3,892 |

Main columns:
* `drift_class`: classification as described above.
* `class_incl_self_pairs`: the same, but also counting changes of an activity's relation to itself (an activity becoming repeatable or not).
* `class_dfg_based`: the same, with the temporal relations derived from the directly-follows graph (loop-aware) instead of first occurrences.
* `temporal_transitions`, `existential_transitions`: transition IDs with the number of activity pairs realizing them, e.g. `E2x3` = transition E2 in three pairs.
* `activities_added`, `activities_removed`: changes of the activity set (existence drift).
* CDLG only: `tree_before`, `tree_after` and `documented_*` as documented by the generator. Own dataset only: `expected_from_name`, the transition encoded in the log name.

To reproduce (Python ≥ 3.9; about 15 minutes, dominated by CDLG):
```sh
cd drift_classification
pip install -r requirements.txt
./fetch_data.sh   # CDRIFT and CDLG logs from HuggingFace into data/
./run_all.sh      # raw outputs in out/, tables in results/
```
`class_dfg_based` is only filled if [armature](https://github.com/INSM-TUM/armature) is installed (Python ≥ 3.10) or `ARMATURE_REPO` points to a checkout.

## Repository & Branch Structure

Please note the branch organization when reproducing different phases of this project:

* **`main` (Verification Branch - Current Branch):**  
  This branch was used to run the Verification for CDLG and CDRIFT on the COMA-Cluster.
* **`training` (Training Branch):**  
  This branch was used to run the preprocessing and model training on the COMA-Cluster.
* **`custom` (Existence only Branch):**  
  This branch was used to generate the results for the existence only results and some locally run computations.

---

## Data & Fine-Tuned Models

### Benchmark (Original) Data & DFG Models
The original datasets (benchmarks) and pre-trained CV4CDD-4D models provided by Kraus & van der Aa are available at:
* **HuggingFace Repository:** [pm-science/cv4cdd_4d](https://huggingface.co/datasets/pm-science/cv4cdd_4d/tree/main)

### Existence Only Dataset
The dataset is included in [`existence_only/`](existence_only): 25 event logs (gzipped XES) in four categories, `isolated`, `surrounding_activities`, `gradual` (5 logs each) and `related conditions` (10 logs). Each folder has a `drift_info.csv` with the change points in the CDLG format. Every log contains one existential-only drift; the realized transitions are listed in [`drift_classification/results/own_drift_classes.csv`](drift_classification/results/own_drift_classes.csv).

### ARM Fine-Tuned Model
The fine-tuned model for ARM cdd can be found here:
* **Zenodo Model:** [ARM Model](https://doi.org/10.5281/zenodo.22775563)
---

## Setup & Installation

### Prerequisites
* **Python:** 3.9 (Local execution) / 3.11 (COMA Cluster environment)
* **Poetry:** Dependency management tool ([installation instructions](https://python-poetry.org/))
* **Git** & **Git LFS**

### Local Setup
1. Clone the repository and navigate to the project root:
   ```sh
   git clone <repository-url>
   cd cv4cdd
   ```
2. Install dependencies using Poetry (creates a virtual environment):
   ```sh
   poetry install
   ```
3. Activate the Poetry virtual environment:
   ```sh
   poetry shell
   ```
4. *(Optional - for training/model garden compatibility)* Clone the TensorFlow Model Garden repository into `./models`:
   ```sh
   git clone https://github.com/tensorflow/models.git ./models
   cd models && git checkout 3256e1018a402bf30179ffa9b82e01024fa61fc2 && cd ..
   ```

---

## How to Run

### 1. Local Evaluation & Prediction

To run concept drift prediction on an event log directory using a pre-trained computer vision model:

```sh
poetry shell
cd approaches/object_detection
python predict.py \
  --model-path <path_to_unzipped_pretrained_model> \
  --log-dir <path_to_event_log_directory> \
  --encoding arm \
  --n-windows 200 \
  --output-dir <path_to_output_directory>
```

Supported `--encoding` configurations:
* `dfg`: Standard baseline Directly-Follows Graph.
* `dfg_io`: DFG with artificial Start ($S$) and End ($E$) trace boundary activities.
* `arm`: Continuous 8D Activity Relationship Matrix representation.

---

### 2. Running on the COMA Cluster

For model retraining and large-scale synthetic benchmark generation on the **COMA Cluster**, follow these steps:

#### Step 1: Environment Setup
Load Python via Spack and set up the Python virtual environment:
```sh
module load python/3.11.7-gcc-11.4.1-6677fey
python -m venv .venv
source .venv/bin/activate
pip install poetry
poetry install
```

#### Step 2: Data Acquisition
Download the required dataset partitions and TF Records from HuggingFace:
```sh
pip install -U "huggingface_hub[cli]"
huggingface-cli download pm-science/cv4cdd_4d --include "input_cdlg/*" "tf_records/*" --local-dir ./data
```
Clone the TensorFlow Models repository into `./models`:
```sh
git clone https://github.com/tensorflow/models.git ./models
cd models && git checkout 3256e1018a402bf30179ffa9b82e01024fa61fc2 && cd ..
```

#### Step 3: Submitting SLURM Jobs
To reproduce model training, switch to the `training` branch and submit the SLURM job:
```sh
git checkout training
sbatch slurm/scripts/sbatch_train.sh
```

## References & Citation

This project extends the **CV4CDD-4D** framework:

1. **A. Kraus and H. van der Aa.** *"Machine learning-based detection of concept drift in business processes."* Process Science 2.5 (2025).

---

## License

This repository is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). See `LICENSE.txt` for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

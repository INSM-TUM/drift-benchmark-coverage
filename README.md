---
license: cc-by-4.0
---

<a name="readme-top"></a>

# Rethinking Behavioral Representations in Concept Drift Detection

> **Bachelor's Thesis in Information Engineering**  
> **Author:** Fabian Deigner

---

## Abstract
Organizations operate in constantly changing environments where business processes
have to adapt due to evolving regulations, shifting strategic goals, and technological
advances. Accurately detecting concept drift is critical for effective process mining
and operational monitoring. However, historical research never investigated the
impact of the expressiveness of the underlying representation used for log abstraction
when detection concept drifts. The state-of-the-art computer-vision-based CV4CDD-
4D, relies on the Directly-Follows Graph (DFG) as the underlying log abstraction .
Although DFGs capture immediate, adjacent activity transitions well, they operate
strictly on local sequences and do not explicitly represent global, non-adjacent co-
occurrence relationships across a full trace.
To address this limitation, this thesis evaluates whether replacing the baseline DFG
with the more expressive Activity Relationship Matrix (ARM) enhances automated
drift detection. To resolve the sensitivity of discrete activity relationships to trace-
level noise, we formulate a continuous, 8-dimensional feature vector that breaks down
relational rules into temporal and existential frequency weights. By aligning these
matrices across chronological trace windows, we generate a similarity visualization
that is fed to the downstream machine learning pipeline.
Through evaluation across large-scale synthetic benchmarks (CDLG, CDRIFT) and
custom edge-case datasets, this work demonstrates that the choice of behavioral repre-
sentation establishes a strict upper bound on downstream drift detection performance.
While standard DFGs perform adequately on basic process trees where structural shifts
alter close-range directly follow relations, they remain completely blind to concept drifts
that modify long-range conditional dependencies without altering local directly-follows
probabilities. Under these non-directly-follows drift scenarios, the explicit inclusion of
global existential dependencies in the ARM representation exposes clear visual drift
boundaries, enabling accurate detection where traditional adjacency baselines fail.
Additionally, by analyzing boundary limitations on isolated activity transitions, we
show that standard DFGs can be enhanced by introducing start and end events (DFG-
IO) to capture secondary boundary effects. Ultimately, this thesis demonstrates that
moving beyond simple directly-follows abstractions to more expressive, relation-aware
representations is essential to overcome theoretical blind spots and unlock fine-grained,
interpretable process analytics

---

## Repository & Branch Structure

Please note the branch organization when reproducing different phases of this project:

* **`main` (Evaluation Branch - Current Branch):**  
  This branch was used to run all evaluation experiments presented in the thesis.
* **`training` (Training Branch):**  
  This branch was used to run the preprocessing and model training.
* **`custom` (Evaluation Branch - Current Branch):**  
  This branch was used to run custom evaluation and some locally run computations.

---

## Data & Fine-Tuned Models

### Benchmark (Original) Data & DFG Models
The original datasets (benchmarks) and pre-trained CV4CDD-4D models provided by Kraus & van der Aa are available at:
* **HuggingFace Repository:** [pm-science/cv4cdd_4d](https://huggingface.co/datasets/pm-science/cv4cdd_4d/tree/main)

### Custom Data & ARM Model
The custom datasets and retrained model for ARM detection can be found here:
* **Zenodo Dataset:** [Custom-Data](https://zenodo.org/records/21813069)

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
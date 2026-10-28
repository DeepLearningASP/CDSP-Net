# CDSP-Net

PyTorch implementation of **CDSP-Net: Covariance-Driven Dynamic Sparse Pursuit for Robust DOA Estimation Under Non-Ideal Conditions**.

**Authors:** Zeyu Gong, Jian Zhang, Jianhui Wang, Weijia Cui, and Bin Ba.

## Overview

CDSP-Net is a covariance-driven neural network for direction-of-arrival (DOA) estimation. It integrates an observation-dependent covariance dictionary with differentiable sparse pursuit and continuous-angle regression. The framework considers narrowband signals received by a uniform linear array, including coherent sources, mutual coupling, and sensor-position errors.

The network consists of three components:

1. **Cross-attention dynamic dictionary learning (CA-DDLN):** combines upper-triangular vectorized sample covariance features with a nominal array-manifold covariance dictionary to generate observation-dependent dictionary corrections.
2. **Differentiable sparse pursuit:** progressively constructs selected-atom representations through soft atom aggregation, least-squares fitting, and residual updates.
3. **Atom-to-DOA mapping (AD-MapNet):** maps the selected-atom representations to continuous DOA estimates.

Throughout the code, `M` denotes the number of sources, `N` the number of sensors, and `T` the number of snapshots. The number of sources is assumed known.

## Repository Structure

```text
CDSP-Net/
├── data_handler/          # Synthetic data generation and preprocessing
├── models/                # Network architectures and baseline algorithms
│   ├── CDSP_Net.py
│   ├── AblationAlgorithm.py
│   ├── BaselineAlgorithm.py
│   ├── SubspaceNet.py
│   ├── DeepCNN.py
│   ├── SubNet/            # CA-DDLN and AD-MapNet
│   └── Layers/            # Layers used by the subnetworks
├── main_train/            # Training configuration and execution
├── main_test/             # Experiment-specific evaluation
├── tools/                 # Shared utilities
└── requirements.txt
```

### Data Generation

[data_handler/CreateDataset.py](data_handler/CreateDataset.py) generates synthetic array observations, DOA labels, and covariance features. It supports configurable SNRs, snapshot counts, source coherence, mutual coupling, and sensor-position errors. Generated datasets can be saved and reused during training and evaluation.

### Models

| File or directory | Description |
| --- | --- |
| [models/CDSP_Net.py](models/CDSP_Net.py) | Complete CDSP-Net architecture |
| [models/SubNet/](models/SubNet/) | CA-DDLN and AD-MapNet implementations |
| [models/Layers/](models/Layers/) | Attention and feature-processing layers used by the subnetworks |
| [models/AblationAlgorithm.py](models/AblationAlgorithm.py) | CDSP-Net variants for input and architecture ablations |
| [models/BaselineAlgorithm.py](models/BaselineAlgorithm.py) | Classical sparse-pursuit and subspace-based DOA estimators |
| [models/SubspaceNet.py](models/SubspaceNet.py) | SubspaceNet baseline |
| [models/DeepCNN.py](models/DeepCNN.py) | DeepCNN baseline |

### Training

The [main_train/](main_train/) directory contains three files:

- `config_train_parameters.py`: defines shared parameters and a `TASK_LIST` of training experiments.
- `run_train.py`: prepares datasets, constructs the model, and initializes the training run.
- `train_solver.py`: performs training and validation, records losses, and saves checkpoints.

### Evaluation

Each experiment in [main_test/](main_test/) follows the same organization: `config_*.py` defines the settings, `run_*.py` prepares the experiment, and `*_solver.py` performs evaluation and saves numerical results.

| Directory | Evaluation purpose | Entry point |
| --- | --- | --- |
| `SNR_Snapshot` | Performance versus SNR or snapshot count | [config_ss.py](main_test/SNR_Snapshot/config_ss.py) |
| `Ablation` | Input representations and architecture variants | [config_ablation.py](main_test/Ablation/config_ablation.py) |
| `Grid` | Dictionary grid spacing | [config_grid.py](main_test/Grid/config_grid.py) |
| `SrcDist` | Angular separation between sources | [config_sd.py](main_test/SrcDist/config_sd.py) |
| `SrcNum` | Number of sources | [config_sn.py](main_test/SrcNum/config_sn.py) |
| `Coup` | Mutual-coupling strength | [config_coup.py](main_test/Coup/config_coup.py) |
| `Pos-Err` | Sensor-position errors | [config_pe.py](main_test/Pos-Err/config_pe.py) |
| `CDF` | Empirical distribution of estimation errors | [config_cdf.py](main_test/CDF/config_cdf.py) |
| `Temp` | Soft-aggregation temperature | [config_temp.py](main_test/Temp/config_temp.py) |
| `Weight` | Export of atom-aggregation weights | [config_weight.py](main_test/Weight/config_weight.py) |
| `Generalization` | Generalization across signal and array conditions | [config_generalization.py](main_test/Generalization/config_generalization.py) |
| `Complexity` | Inference runtime and scaling with source/sensor counts | [config_complexity.py](main_test/Complexity/config_complexity.py) |

### Utilities

The [tools/](tools/) directory provides loss functions (`criterions.py`), model construction (`model_config.py`), checkpoint loading (`load_checkpoint.py`), checkpoint cleanup (`cleanup_checkpoints.py`), random-seed control (`seed.py`), logging (`logger.py`), and filename formatting (`filename_handler.py`).

## Installation

The reference environment uses Python 3.9.13 and PyTorch 2.5.1. Install the dependencies in your Python environment:

```bash
git clone https://github.com/DeepLearningASP/CDSP-Net.git
cd CDSP-Net
python -m pip install -r requirements.txt
```

For the CUDA 12.1 build, install PyTorch before installing the remaining dependencies:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The default training and most test configurations use CUDA and require a compatible NVIDIA GPU and driver.

## Usage

### 1. Configure Paths

Update the local paths before running:

- `datasets_save_dir` in the relevant configuration: directory for generated or cached datasets.
- `RESULTS_ROOT` in `main_train/run_train.py`: directory for training outputs.
- `save_dir1` in test configurations: directory for evaluation outputs.
- `checkpoint_path` or the corresponding checkpoint mapping: location of trained model files.

Where available, set `is_creat = "True"` to generate data or `is_creat = "False"` to reuse a saved dataset. Use `"True"` for the first run if no dataset has been generated.

Run the commands below from the repository root. Set the project import path in the same terminal:

**PowerShell:**

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

**Bash:**

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

### 2. Train a Model

Open `main_train/config_train_parameters.py`, set the shared training parameters, and select the experiments in `TASK_LIST`. Each entry overrides the shared settings for one run; all uncommented entries execute sequentially.

For example, use the following task list to train CDSP-Net with 200 snapshots and an SNR of 10 dB:

```python
TASK_LIST = [
    {
        "model_type": "CDSP-Net",
        "T": [200],
        "snr": [10, 10],
        "coh_type": "NonCoherent",
        "is_creat": "True",
    },
]
```

Start training:

```bash
python main_train/config_train_parameters.py
```

Training outputs are saved under `RESULTS_ROOT/<model_name>/<experiment_timestamp>/`. They include a training log, a loss CSV file, and model checkpoints named `Model_epoch<number>.pth`. After training completes successfully, checkpoint cleanup retains the model with the lowest validation loss.

To resume a run, set `resume = True` and specify the saved checkpoint in `checkpoint_path`.

### 3. Evaluate a Model

1. Place the trained `.pth` checkpoint files in a local directory.
2. Open the configuration for the required experiment and update `checkpoint_path` or its checkpoint mapping. Follow the model ordering or labels specified in that file.
3. Select `test_algorithm`, configure the test conditions and output paths, and choose whether to generate or reuse test data. Use checkpoints matching the selected model, source/sensor counts, dictionary grid, and temperature.
4. Run the experiment's configuration file.

For an SNR or snapshot sweep:

```bash
python main_test/SNR_Snapshot/config_ss.py
```

Set `test_type = "SNR"` or `test_type = "Snapshot"` in that configuration to select the experiment.

For generalization evaluation:

```bash
python main_test/Generalization/config_generalization.py
```

Configure the `UnifiedMixed`, `NoCoup`, and `Unified` checkpoint paths. A single run evaluates the SNR/coherence/coupling conditions, the coupling sweep, and the position-error sweep.

For runtime evaluation:

```bash
python main_test/Complexity/config_complexity.py
```

Set the learned-method checkpoints in `checkpoint_path`. For CDSP-Net size-scaling tests, select `test_algorithm = "CDSP-Net"`, set `M` and `N`, and provide the corresponding entry in `cdsp_checkpoint_paths`.

Other experiments are launched in the same way using the entry points listed above. Evaluation outputs include numerical tables and logs saved beneath `save_dir1`; the generalization and complexity modules also export configuration or runtime metadata.

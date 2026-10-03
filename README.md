# CircuitSynth Data Pipeline

This repository contains the implementation and experiments developed for the CircuitSynth final project. The main pipeline follows the workflow of [CircuitSynth: Reliable Synthetic Data Generation](https://aclanthology.org/2026.findings-acl.1770/) and is evaluated on WebNLG, DART, and a deterministic ZebraLogic-compatible benchmark. The same eight-stage workflow is extended to Vietnamese with ViSFD.

The repository is organized so that the English experiments and the Vietnamese experiment can be run independently, while the submitted Jupyter Notebook presents the complete project in one place.

## Project at a glance

CircuitSynth is a synthetic data generation pipeline designed to keep generated text consistent with structured information and explicit constraints.

Instead of generating text in one step, the pipeline first turns the required information into a structured semantic plan. That plan is checked before generation, its distribution is adjusted under the configured constraints, and a fine-tuned Student model then produces the final text. The generated output is evaluated again for structure, factual consistency, coverage, and distributional behavior.

A simplified view of the workflow is:

```text
Dataset / structured information
        ↓
Semantic plan
        ↓
Verification
        ↓
Probabilistic prior and distribution adjustment
        ↓
Plan sampling
        ↓
Student model fine-tuning
        ↓
FSA-constrained generation
        ↓
Evaluation
```

The main experiment follows the CircuitSynth paper on English benchmarks. The same workflow is also adapted to Vietnamese using ViSFD, with separate preprocessing, schema rules, verification logic, prompting, constrained generation rules, and evaluation.

## Implementation scope

The project covers the full pipeline rather than only model training or a single notebook experiment. The implementation includes:

- implemented the eight-stage pipeline in Python;
- prepared and normalized data for WebNLG, DART, and a deterministic ZebraLogic-compatible benchmark;
- built semantic-plan generation and symbolic verification;
- implemented the internal probabilistic circuit workflow and the distribution-adjustment step;
- trained the Qwen2.5-0.5B Student model with QLoRA;
- added FSA-constrained generation so required plan information is preserved in the output;
- implemented evaluation for schema validity, constraint violations, factuality, coverage, distributional drift, bits per token, and perplexity;
- added checkpoints, resume support, tests, saved configurations, and reusable experiment artifacts;
- adapted the pipeline to Vietnamese with ViSFD and kept that version runnable as a separate experiment.

The repository includes source code, tests, configuration, the project notebook, the report, saved metrics, training reports, and representative generated samples so the work can be inspected without rerunning the full training process.

## Quick results

The saved evaluation artifacts in this repository contain the following full-pipeline results:

| Experiment | Evaluated outputs | Schema validity | Fact coverage | Factuality precision |
| --- | ---: | ---: | ---: | ---: |
| WebNLG | 318 | 100.0% | 99.8% | 99.0% |
| DART | 894 | 99.2% | 87.9% | 90.2% |
| ZebraLogic-compatible | 1,788 | 100.0% | 100.0% | 100.0% |
| ViSFD | 200 | 100.0% | 100.0% | 72.0% |

Detailed results are available in [`artifacts/metrics/metrics.json`](artifacts/metrics/metrics.json) and [`CircuitSynth_ViSFD/artifacts/metrics/metrics.json`](CircuitSynth_ViSFD/artifacts/metrics/metrics.json).

For the Vietnamese experiment, the full pipeline produced 200 evaluated outputs with a 0% constraint-violation rate. The corresponding direct and less-constrained baselines are also saved in the ViSFD metrics so the effect of the complete pipeline can be inspected rather than inferred from a single final score.

## Public repository note

This public repository keeps the source code, configuration, tests, reports, metrics, and representative result samples. Large training checkpoints, model weights, adapters, tokenizer files, and full generated datasets are intentionally left out of Git history to keep the repository practical to clone and review.

The detailed sections below describe the complete experiment workflow. Where an artifact is produced by a full local run but excluded from the public repository because of its size, that distinction is stated explicitly.


## 1. Repository structure

```text
.
├── README.md
├── run.py
├── config.yaml
├── pyproject.toml
├── requirements.lock
├── src/
│   ├── data.py
│   ├── schema.py
│   ├── teacher.py
│   ├── verifier.py
│   ├── circuits.py
│   ├── student.py
│   ├── fsa.py
│   ├── evaluate.py
│   └── checkpoint.py
├── tests/
│   ├── test_core.py
│   └── test_pipeline.py
├── artifacts/
├── notebooks/
│   └── CircuitSynth_Project.ipynb
└── CircuitSynth_ViSFD/
    ├── README.md
    ├── run.py
    ├── config.yaml
    ├── pyproject.toml
    ├── requirements.lock
    ├── scripts/
    ├── src/
    ├── tests/
    └── artifacts/
```

The root directory contains the main English pipeline. The Vietnamese implementation is kept under [`CircuitSynth_ViSFD/`](CircuitSynth_ViSFD/) because it has its own data processing, schema, verifier rules, prompting, FSA realization rules, configuration, and evaluation while preserving the same overall pipeline.

The main notebook for the complete project is [`notebooks/CircuitSynth_Project.ipynb`](notebooks/CircuitSynth_Project.ipynb).

## 2. Reference paper

The project is based on:

- Zehua Cheng, Wei Dai, Jiahao Sun, and Thomas Lukasiewicz. **CircuitSynth: Reliable Synthetic Data Generation.** Findings of ACL 2026.
- ACL Anthology: [https://aclanthology.org/2026.findings-acl.1770/](https://aclanthology.org/2026.findings-acl.1770/)

The implementation preserves the main workflow of separating semantic-plan construction from surface realization, applying symbolic verification, fitting a probabilistic semantic prior, adjusting the distribution under soft constraints, sampling plans, training a conditional Student model, and applying constrained decoding.

## 3. Datasets

### 3.1 WebNLG

The configured source is [`GEM/web_nlg`](https://huggingface.co/datasets/GEM/web_nlg), configuration `en`.

The project uses 50% of the configured population:

```text
train        6,606
validation     834
test           890
```

### 3.2 DART

The configured source is [`GEM/dart`](https://huggingface.co/datasets/GEM/dart).

The project uses 50% of the configured population:

```text
train       25,000
test         2,500
```

### 3.3 ZebraLogic-compatible benchmark

The submitted pipeline does **not** download ZebraLogic from an external dataset repository. It generates a deterministic ZebraLogic-compatible pool in [`src/data.py`](src/data.py), using the `generated/ZebraLogic` source label and the `paper_compatible_grid` configuration defined in [`config.yaml`](config.yaml).

The configured full pool is:

```text
easy        3,000
medium      4,000
hard        3,000
```

With `sample_fraction: 0.5`, the selected pool contains 5,000 puzzles:

```text
easy        1,500
medium      2,000
hard        1,500
```

These records are then divided into the research split used by the experiment:

```text
research train        4,000
research validation     500
research test           500
```

For background on the benchmark design, see [ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning](https://arxiv.org/abs/2502.01100).

### 3.4 Vietnamese dataset: ViSFD

The Vietnamese experiment uses [`visolex/ViSFD`](https://huggingface.co/datasets/visolex/ViSFD).

The public dataset contains 11,122 records. Its `type` field contains:

```text
train       7,786
dev         1,112
test        2,224
```

The ViSFD pipeline normalizes `dev` to `validation` and selects:

```text
train         700
validation    100
test          200
```

ViSFD aspect-sentiment annotations are converted into structured semantic plans. For this experiment, the accepted semantic plans originate from the human annotations provided by ViSFD and are checked by the ViSFD verifier before being used by the later stages.

The ViSFD configuration is available at [`CircuitSynth_ViSFD/config.yaml`](CircuitSynth_ViSFD/config.yaml).

## 4. Models

The configured models are:

- Teacher: [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- Student: [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

Both model configurations use 4-bit NF4 loading. The Student is fine-tuned with QLoRA.

### Main Student configuration

```text
epochs                  1
micro batch size        1
gradient accumulation  16
maximum sequence      1024
learning rate          2e-5
QLoRA rank                8
QLoRA alpha              16
QLoRA dropout          0.05
optimizer             AdamW
scheduler             linear
```

### ViSFD Student configuration

```text
epochs                  1
micro batch size        1
gradient accumulation   8
maximum sequence       512
learning rate          2e-5
QLoRA rank                8
QLoRA alpha              16
QLoRA dropout          0.05
optimizer             AdamW
scheduler             linear
```

A full local training run writes the final Student adapter and tokenizer artifacts to:

- English Student: `artifacts/student/final/`
- ViSFD Student: `CircuitSynth_ViSFD/artifacts/student/final/`

These generated model artifacts are intentionally excluded from the public GitHub repository. The training configuration, training reports, evaluation metrics, and representative samples remain available in version control.

The base Qwen models are not duplicated inside the repository; they are downloaded from the verified Hugging Face model pages above.

## 5. Pipeline workflow

The two experiments use the same eight-stage orchestration.

### Stage 1: data and schema

Dataset records are loaded or generated, sampled according to the configuration, normalized into a common record format, and paired with domain-specific schemas.

Main outputs are stored under [`artifacts/data/`](artifacts/data/).

### Stage 2: semantic plans and verification

For WebNLG, DART, and the ZebraLogic-compatible benchmark, the Teacher proposes semantic plans and the symbolic verifier checks the required schema and grounding constraints.

For ViSFD, semantic plans are constructed from the human aspect-sentiment annotations and checked by the ViSFD verifier.

Only accepted plan-text pairs continue as silver data.

Completed-run counts:

```text
English silver accepted     4,000
ViSFD silver accepted         700
```

### Stage 3: probabilistic semantic prior

The submitted implementation uses the repository's internal circuit backend:

```text
backend    internal_psdd
compiler   deterministic_plan_circuit
```

The prior is fitted over verifier-approved semantic-plan support. Circuit artifacts are stored under [`artifacts/circuits/`](artifacts/circuits/).

### Stage 4: distributional optimization

The semantic distribution is adjusted under the configured soft constraints. The implementation tracks attribute marginals, relation marginals, and rare pair combinations.

The completed runs converge with zero invalid-support mass in the saved projection reports.

### Stage 5: semantic-plan sampling

Plans are sampled from the optimized distribution and verified before they are written to [`artifacts/plans/`](artifacts/plans/).

Configured sample counts:

```text
English sampled plans     4,000
ViSFD sampled plans         200
```

### Stage 6: Student training

The Student is trained as a conditional realizer from serialized semantic plan to text using QLoRA.

Completed English run:

```text
silver accepted        4,000
Student train          3,800
Student validation       200
optimizer steps          238
```

Completed ViSFD run:

```text
silver accepted          700
Student train            665
Student validation        35
optimizer steps           84
```

Training reports are stored under the corresponding `artifacts/student/` directories. Final adapters are produced locally under `artifacts/student/final/` and `CircuitSynth_ViSFD/artifacts/student/final/`, but those generated model files are excluded from the public repository.

### Stage 7: FSA-constrained generation

The trained Student generates text from semantic plans. A finite-state automaton constrains legal realizations so that required plan elements remain represented in the generated output.

During a full run, generated records are written under [`artifacts/outputs/`](artifacts/outputs/) and [`CircuitSynth_ViSFD/artifacts/outputs/`](CircuitSynth_ViSFD/artifacts/outputs/). The public repository keeps the generation manifests and representative samples while excluding the full generated-output files.

### Stage 8: evaluation

The evaluation stage computes:

- Schema Validity
- Constraint Violation Rate
- Factuality Precision
- Fact Coverage
- Coverage Error
- Jensen-Shannon divergence
- Rare-Combination Coverage
- Bits per Token
- Perplexity

The evaluation also contains the following variants:

```text
Teacher
Distill
Prior
Prior + Projection
Prior + FSA
Full
```

The completed full-pipeline evaluation uses 3,000 English outputs and 200 ViSFD outputs. Results are stored in the corresponding `artifacts/metrics/` directories and are presented in the submitted notebook.

## 6. Environment

The packages and supported Python version are defined in [`pyproject.toml`](pyproject.toml) and [`requirements.lock`](requirements.lock).

Minimum project requirement:

```text
Python >= 3.10
```

A CUDA-capable NVIDIA GPU is required for the full Teacher, Student-training, and neural-generation stages in the submitted configuration. CPU execution is sufficient for source checks, tests, reading saved artifacts, data exploration, result tables, and the lightweight FSA demonstration.

For a CUDA-enabled PyTorch installation, use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) and choose the command that matches the operating system and CUDA environment.

Check the current Python/CUDA environment with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 7. Installation

### 7.1 Windows

Open a terminal in the repository root.

Create the environment:

```powershell
python -m venv .venv
```

Install the project using the Python executable inside the environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[full,test]"
```

Using the environment's Python executable directly avoids depending on PowerShell script activation policy.

If the full GPU stages will be rerun, install a CUDA-compatible PyTorch build using the official [PyTorch installation instructions](https://pytorch.org/get-started/locally/) before running the model stages.

### 7.2 Linux / Google Colab / Kaggle

Upload or extract the complete repository and keep the directory structure unchanged.

From the repository root:

```bash
python -m pip install --upgrade pip
pip install -e ".[full,test]"
```

Use a GPU runtime when rerunning Teacher inference, Student training, or neural constrained generation.

## 8. Running the main pipeline

All commands in this section are run from the repository root.

### Check current status

```bash
python run.py status
```

### Run all eight stages

```bash
python run.py run
```

### Resume an interrupted run

```bash
python run.py resume
```

### Run one stage

```bash
python run.py stage 1
```

For example, run evaluation only:

```bash
python run.py stage 8
```

A stage run separately still requires the artifacts produced by its preceding stages.

The main runtime configuration is defined in [`config.yaml`](config.yaml).

## 9. Running the ViSFD pipeline

Move into the Vietnamese project:

```bash
cd CircuitSynth_ViSFD
```

Check status:

```bash
python run.py status
```

Run the complete ViSFD pipeline:

```bash
python run.py run
```

Resume from the latest compatible checkpoint:

```bash
python run.py resume
```

Run one stage:

```bash
python run.py stage 1
```

The ViSFD runtime configuration is defined in [`CircuitSynth_ViSFD/config.yaml`](CircuitSynth_ViSFD/config.yaml). Additional ViSFD-specific notes are kept in [`CircuitSynth_ViSFD/README.md`](CircuitSynth_ViSFD/README.md).

## 10. Configuration

### 10.1 Main experiment

Important values from the completed run are defined in [`config.yaml`](config.yaml):

```yaml
data:
  sample_fraction: 0.5

silver:
  accepted_total_override: 4000

sampling:
  total_plans: 4000

student:
  epochs: 1
  micro_batch_size: 1
  gradient_accumulation_steps: 16
  max_sequence_length: 1024
  learning_rate: 2.0e-5
  lora_rank: 8
  lora_alpha: 16
  lora_dropout: 0.05
```

The CLI supports configuration overrides. Examples:

```bash
python run.py run --set data.sample_fraction=0.25
python run.py run --set silver.accepted_total_override=2000
```

### 10.2 ViSFD experiment

Important values are defined in [`CircuitSynth_ViSFD/config.yaml`](CircuitSynth_ViSFD/config.yaml):

```text
train                     700
validation                100
test                      200
silver accepted           700
sampled plans             200
Student epochs              1
micro batch size            1
gradient accumulation       8
maximum sequence          512
learning rate            2e-5
QLoRA rank                  8
QLoRA alpha                16
QLoRA dropout            0.05
seed                       42
```

The exact resolved configuration used by a completed run is saved in:

- [`artifacts/resolved_config.yaml`](artifacts/resolved_config.yaml)
- [`CircuitSynth_ViSFD/artifacts/resolved_config.yaml`](CircuitSynth_ViSFD/artifacts/resolved_config.yaml)

## 11. Artifacts and results

The main pipeline uses:

```text
artifacts/
├── data/
├── silver/
├── circuits/
├── plans/
├── student/
├── outputs/
├── metrics/
└── checkpoints/
```

ViSFD uses the same structure under [`CircuitSynth_ViSFD/artifacts/`](CircuitSynth_ViSFD/artifacts/).

Useful main-run files:

- [`artifacts/data/manifests.json`](artifacts/data/manifests.json)
- [`artifacts/silver/quota_report.json`](artifacts/silver/quota_report.json)
- [`artifacts/circuits/projection_report.json`](artifacts/circuits/projection_report.json)
- [`artifacts/plans/sampling_report.json`](artifacts/plans/sampling_report.json)
- [`artifacts/student/training_report.json`](artifacts/student/training_report.json)
- [`artifacts/metrics/metrics.json`](artifacts/metrics/metrics.json)

The corresponding ViSFD files are located under [`CircuitSynth_ViSFD/artifacts/`](CircuitSynth_ViSFD/artifacts/).

## 12. Jupyter Notebook

The notebook submitted for the complete project is:

[`notebooks/CircuitSynth_Project.ipynb`](notebooks/CircuitSynth_Project.ipynb)

It contains the required end-to-end experimental presentation:

1. library installation and environment setup;
2. repository and pipeline checks;
3. dataset loading and reading;
4. data exploration and preprocessing;
5. model and training configuration;
6. English test evaluation;
7. baseline and ablation comparison;
8. English error analysis;
9. ViSFD loading and exploration;
10. ViSFD preprocessing and semantic plans;
11. ViSFD training and optimization artifacts;
12. ViSFD test evaluation;
13. ViSFD baseline and ablation comparison;
14. ViSFD error analysis;
15. demonstration on a new Vietnamese semantic plan.

The notebook does not hard-code a personal machine path. It searches for the repository root and supports the directory layouts used by local execution, Google Colab, and Kaggle.

By default, **Run All** validates the repository, runs the tests, reads the submitted artifacts, performs the data exploration/evaluation sections, and runs the lightweight Vietnamese demonstration. It does not retrain the two complete pipelines.

The final notebook cell contains:

```python
RERUN_MAIN_PIPELINE = False
RERUN_VISFD_PIPELINE = False
```

Set the corresponding flag to `True` only when a complete experiment rerun is required.

For Colab or Kaggle, upload or extract the complete repository before opening the notebook. The relative paths among `src/`, `artifacts/`, `CircuitSynth_ViSFD/`, and `notebooks/` must remain unchanged.

## 13. Tests

Run the main test suite from the repository root:

```bash
python -m pytest -q
```

Run the ViSFD tests:

```bash
cd CircuitSynth_ViSFD
python -m pytest -q
```

The submitted runs pass both test suites. Tests cover the core pipeline behavior; experimental test-set evaluation is handled separately in Stage 8 and in the notebook.

## 14. Final submission package

The project submission is organized to include the required deliverables:

```text
CircuitSynth_report.pdf
README.md
run.py
config.yaml
pyproject.toml
requirements.lock
src/
tests/
artifacts/
notebooks/
│   └── CircuitSynth_Project.ipynb
└── CircuitSynth_ViSFD/
    ├── README.md
    ├── run.py
    ├── config.yaml
    ├── pyproject.toml
    ├── requirements.lock
    ├── scripts/
    ├── src/
    ├── tests/
    └── artifacts/
```

The original submission package included the ACL-format report, source code, Jupyter Notebook, run instructions, dataset sources, experiment artifacts, and trained Student adapters. The public GitHub version keeps the reproducible source, reports, metrics, and representative samples, while intentionally excluding the large trained adapter and model artifacts.

Do not include local or generated cache directories in the final ZIP:

```text
.venv/
.pytest_cache/
__pycache__/
*.pyc
*.egg-info/
```

Temporary nested ZIP files are also not required in the final submission.

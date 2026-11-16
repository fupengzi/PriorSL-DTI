# PriorSL-DTI

![Overview of the PriorSL-DTI architecture](assets/architecture.png)

PriorSL-DTI is a sequence-based framework for drug-target interaction prediction. It combines pretrained MoLFormer and ESM-2 representations with multi-scale convolution, then uses a protein pseudo-binding prior to guide feature modulation, local aggregation and long-range candidate selection. Alternating short-range dense attention (SDA) and long-range sparse attention (LSA) blocks integrate drug and protein information through bidirectional cross-attention. An adaptive calibration layer refines the updated representations, followed by attention pooling and complementary feature fusion for prediction.

This repository provides code for data preparation, feature extraction, model training and deletion-faithfulness analysis.

## Code

```text
model.py       Model architecture, including SDA and LSA
data.py        Conversion of fixed warm/cold splits
features.py    MoLFormer and ESM-2 feature extraction
main.py        Training, evaluation and ten repeated runs
deletion.py    Deletion-faithfulness analysis and plotting
```

## Installation

Python 3.10 and a CUDA GPU are recommended.
The experiments used an NVIDIA GeForce RTX 3080 Ti (12 GB). We recommend the same GPU to reproduce the paper results.

```bash
conda create -n priorsl_dti python=3.10 -y
conda activate priorsl_dti
pip install -r requirements.txt
```

For GPU training, install a CUDA-compatible PyTorch build for your environment.

## Data and Features

Each input CSV must contain `SMILES`, `Protein` and `Y`. The expected split layout is:

```text
data/<dataset>/{random,e2,e3,e4}/{0..9}/{train,validation,test}.csv
```

Convert the splits and extract pretrained features:

```bash
python data.py convert --input_root data/BioSNAP --output_root data/converted/biosnap --scenarios random
python features.py drug --csv data/converted/biosnap/random --out_dir embeddings/biosnap/drug
python features.py protein --csv data/converted/biosnap/random --out_dir embeddings/biosnap/protein
```

The default encoders are MoLFormer-XL and ESM-2-150M. A local Hugging Face model directory may be supplied with `--model_name`.

## Training

Run one dataset and one setting per command. `random` is the warm-start setting; `e2`, `e3` and `e4` are the cold-start settings. Each command runs ten folds using `config.yaml` (batch size 64, 150 epochs, FP16).

BioSNAP:

```bash
python -u main.py run --input_root data/BioSNAP --converted_root data/converted/biosnap --convert --extract --train --scenarios random --output_dir outputs/biosnap_random
python -u main.py run --input_root data/BioSNAP --converted_root data/converted/biosnap --convert --extract --train --scenarios e2 --output_dir outputs/biosnap_e2
python -u main.py run --input_root data/BioSNAP --converted_root data/converted/biosnap --convert --extract --train --scenarios e3 --output_dir outputs/biosnap_e3
python -u main.py run --input_root data/BioSNAP --converted_root data/converted/biosnap --convert --extract --train --scenarios e4 --output_dir outputs/biosnap_e4
```

BindingDB:

```bash
python -u main.py run --input_root data/BindingDB --converted_root data/converted/bindingdb --convert --extract --train --scenarios random --output_dir outputs/bindingdb_random
python -u main.py run --input_root data/BindingDB --converted_root data/converted/bindingdb --convert --extract --train --scenarios e2 --output_dir outputs/bindingdb_e2
python -u main.py run --input_root data/BindingDB --converted_root data/converted/bindingdb --convert --extract --train --scenarios e3 --output_dir outputs/bindingdb_e3
python -u main.py run --input_root data/BindingDB --converted_root data/converted/bindingdb --convert --extract --train --scenarios e4 --output_dir outputs/bindingdb_e4
```

DrugBank:

```bash
python -u main.py run --input_root data/DrugBank --converted_root data/converted/drugbank --convert --extract --train --scenarios random --output_dir outputs/drugbank_random
python -u main.py run --input_root data/DrugBank --converted_root data/converted/drugbank --convert --extract --train --scenarios e2 --output_dir outputs/drugbank_e2
python -u main.py run --input_root data/DrugBank --converted_root data/converted/drugbank --convert --extract --train --scenarios e3 --output_dir outputs/drugbank_e3
python -u main.py run --input_root data/DrugBank --converted_root data/converted/drugbank --convert --extract --train --scenarios e4 --output_dir outputs/drugbank_e4
```

Converted data and embedding caches are reused across settings. Separate output directories preserve each setting's `summary.csv` and `fold_results.csv`; fold logs, checkpoints and test metrics are under `outputs/<dataset>_<setting>/<setting>/<fold>/`.

## Deletion Faithfulness

```bash
python deletion.py run --config outputs/bindingdb_random/random/0/config.yaml --checkpoint outputs/bindingdb_random/random/0/best_model.pt --modality protein --random_repeats 20 --output_dir outputs/deletion/protein
```

Use `--modality drug` for drug-token deletion. Combine protein and drug results with:

```bash
python deletion.py plot --protein_summary outputs/deletion/protein/deletion_summary.csv --drug_summary outputs/deletion/drug/deletion_summary.csv --output outputs/deletion_figure
```

Training uses all configured epochs and saves the checkpoint with the best validation AUROC. 

To reproduce the experiments, use the provided splits. Conversion preserves every row, its order and split membership without deduplication or re-splitting. All warm/cold runs use the shared 512-dimensional configuration. Thresholds are selected on validation MCC. Historical `lda` checkpoint keys are mapped to `lsa` when loading. Ablation experiment scripts are not included.

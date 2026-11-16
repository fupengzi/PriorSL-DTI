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
python data.py convert --input_root data/BioSNAP --output_root data/converted/biosnap
python features.py drug --csv data/converted/biosnap --out_dir embeddings/biosnap/drug
python features.py protein --csv data/converted/biosnap --out_dir embeddings/biosnap/protein
```

The default encoders are MoLFormer-XL and ESM-2-150M. A local Hugging Face model directory may be supplied with `--model_name`.

## Training

Run the ten predefined warm- and cold-start splits:

```bash
python -u main.py run --train --converted_root data/converted/biosnap
```

The dataset name, embedding directories and output directory are inferred from `converted_root`. Use the optional command arguments to override them. Replace `biosnap` with `bindingdb` or `drugbank` for the other datasets.

### Reproduction

Start with one BioSNAP fold to check the setup. Training uses batch size 64, 150 epochs and FP16 as specified in `config.yaml`.

```bash
python -u main.py run --input_root data/BioSNAP --converted_root data/converted/biosnap --convert --extract --train --scenarios random --folds 0 --output_dir outputs/smoke_biosnap
```

After the single fold succeeds, run all four predefined settings and ten folds for each dataset. Existing embedding files are reused.

```bash
python -u main.py run --input_root data/BioSNAP --converted_root data/converted/biosnap --convert --extract --train
python -u main.py run --input_root data/BindingDB --converted_root data/converted/bindingdb --convert --extract --train
python -u main.py run --input_root data/DrugBank --converted_root data/converted/drugbank --convert --extract --train
```

Check `outputs/<dataset>/summary.csv`, `fold_results.csv` and each `<setting>/<fold>/train.log`, `best_model.pt` and `test_metrics.json`.

## Deletion Faithfulness

```bash
python deletion.py run --config outputs/bindingdb/random/0/config.yaml --checkpoint outputs/bindingdb/random/0/best_model.pt --modality protein --random_repeats 20 --output_dir outputs/deletion/protein
```

Use `--modality drug` for drug-token deletion. Combine protein and drug results with:

```bash
python deletion.py plot --protein_summary outputs/deletion/protein/deletion_summary.csv --drug_summary outputs/deletion/drug/deletion_summary.csv --output outputs/deletion_figure
```

Training uses all configured epochs and saves the checkpoint with the best validation AUROC. 

To reproduce the experiments, use the provided splits. Conversion preserves every row, its order and split membership without deduplication or re-splitting. All warm/cold runs use the shared 512-dimensional configuration. Thresholds are selected on validation MCC. Historical `lda` checkpoint keys are mapped to `lsa` when loading. Ablation experiment scripts are not included.

# Calibration-First Monte Carlo Dropout for Cardiac MRI Segmentation

A reproducible benchmark that treats calibration as the primary outcome and compares Monte Carlo (MC) Dropout across four encoder architectures on multi-structure cardiac cine-MRI (LV cavity, RV cavity, LV myocardium). The codebase covers data partitioning, model construction, training, MC inference, calibration analysis, ablations, per-pathology stratification, and uncertainty-ranked quality control.

## What is implemented

- Four interchangeable encoders behind a shared LinkNet decoder: EfficientNet-B3, EfficientNet-B4, SwinUNet, MobileViT-S.
- MC Dropout with configurable placement (encoder-only, decoder-only, encoder+decoder) and a dropout layer that stays active at inference.
- Composite Dice + Focal + Cross-Entropy loss.
- Calibration metrics: ECE, Adaptive ECE (equal-mass bins), Brier score, NLL, reliability diagrams, and a bin-count sensitivity sweep.
- Temperature Scaling as a post-hoc calibration baseline.
- Segmentation metrics: DSC, HD95, precision, recall.
- Volumetric indices (EDV, ESV, EF) with Pearson r, MAE, and Bland-Altman style agreement.
- Structured ablations: backbone trade-off, dropout rate, MC pass count, dropout placement.
- Per-pathology stratification with patient-level bootstrap confidence intervals on delta-ECE.
- Multi-seed robustness over five training seeds.
- Quality-control triage: uncertainty-ranked review with NPV/PPV (Wilson intervals), enrichment, and voxel-level error-detection AUROC.

## Layout

```
cardiac_mc_dropout/
  config.py                 hyperparameters and constants
  data/acdc.py              ACDC loading, stratified partitioning, preprocessing
  models/                   encoders, LinkNet decoder, MC dropout, model factory
  training/                 loss, augmentation, data sequence, training loop
  evaluation/               metrics, calibration, MC inference, volumetric, QC, plots
  experiments/              cross-validation, ablation, per-pathology, robustness
  main.py                   pipeline entry point
```

## Requirements

```
pip install -r requirements.txt
```

The EfficientNet encoders load ImageNet weights through `tf.keras.applications`. The SwinUNet and MobileViT-S encoders are defined from scratch in `models/swin.py` and `models/mobilevit.py` and train from random initialisation unless external pretrained weights are loaded into the corresponding layers.

## Data

This code expects the ACDC dataset on disk in its original layout, with each patient folder containing `Info.cfg`, the cine frames as `patientXXX_frameYY.nii.gz`, and the expert masks as `patientXXX_frameYY_gt.nii.gz`. Point the loader at the database root:

```
export ACDC_ROOT=/path/to/ACDC/database
export OUTPUT_ROOT=/path/to/outputs
```

Label convention used throughout: 0 = background, 1 = RV cavity, 2 = LV myocardium, 3 = LV cavity. Slices are centre-cropped to 100x110 and z-score normalised per slice. Partitioning is patient-level and stratified by pathology (NOR, MINF, DCM, HCM, ARV): 100 training patients with 5-fold cross-validation and 50 held-out test patients.

No data is bundled or synthesised; every number is produced by running the pipeline against the real dataset.

## Running

```
python main.py --stage cv          # 5-fold cross-validation for all variants
python main.py --stage test        # held-out test evaluation, calibration, QC, per-pathology
python main.py --stage ablation    # dropout rate, MC passes, placement
python main.py --stage robustness  # five-seed delta-ECE stability
python main.py --stage all         # everything
```

Results are written as JSON under `OUTPUT_ROOT/results`, checkpoints under `OUTPUT_ROOT/checkpoints`.

## Configuration

Key settings live in `config.py`: crop size, fold count, training schedule (100 epochs, Adam, lr 8e-5, ReduceLROnPlateau), loss weights, dropout rate (0.3), MC pass count (10), placement, ECE bin counts, QC thresholds, the failure threshold (DSC < 0.85), and the seed list (42, 123, 256, 512, 777).

## Reproducibility

The primary results use seed 42, which fixes weight initialisation and data ordering. MC inference remains stochastic because each of the T forward passes draws an independent dropout mask, so the reported per-voxel variance reflects genuine sampling spread. The five-seed robustness stage repeats the full pipeline to confirm calibration stability across training runs.

## Notes and limitations

- Everything is 2D, slice by slice.
- Only epistemic uncertainty is estimated; aleatoric uncertainty from image noise and inter-annotator disagreement is not recovered.
- The SwinUNet encoder runs at a 100x110 crop below its native input size; its window partitioning is sensitive to input size.
- The fixed crop assumes the heart lies near the image centre, as in ACDC; a localiser would be needed for less controlled acquisitions.

# Biplanar Bone Reconstruction

Reconstructing 3D bone shapes from two X-ray images: one front (AP) view and one side (lateral) view. This repository has
the code, data splits and per-case results of a benchmark of three deep learning models on six datasets: proximal femur,
hip, rib cage, two vertebra datasets (VerSe19, VerSe20) and pelvis (CTPelvic). All three models were trained and tested
with one pipeline and the same training settings.

| Model | Idea | Parameters |
|---|---|---|
| **DualViewCNN** | small two-view CNN: each view is encoded, expanded into 3D along its X-ray direction, joined and decoded | 0.56M |
| **MultiScale** | two-view CNN that places the features of both views into the 3D grid at four resolutions | 3.73M |
| **SwinX2S** | Swin-X2S transformer (Tiny), adapted from the [official code](https://github.com/liukuan5625/Swin-X2S) | 32.5M |

Trained models: [Hugging Face, `jeevan1234/biplanar-bone-reconstruction-models`](https://huggingface.co/jeevan1234/biplanar-bone-reconstruction-models).

## Results

Test Dice (%) on held-out test splits (297 cases in total), scored once. Higher is better.

| Dataset | Test cases | DualViewCNN | MultiScale | SwinX2S |
|---|---|---|---|---|
| Femur | 75 | 92.5 | **94.9** | 93.6 |
| Hip | 35 | 80.8 | **87.0** | 84.7 |
| Rib | 39 | 47.5 | **56.4** | 45.4 |
| VerSe19 | 94 | 66.5 | **88.3** | 85.5 |
| VerSe20 | 10 | 69.4 | **89.2** | 88.3 |
| CTPelvic | 44 | 79.0 | **86.6** | 82.4 |

MultiScale is best on all six datasets.
- Against DualViewCNN, the difference is significant on all six (paired Wilcoxon test, Holm correction).
- Against SwinX2S, it is significant on five; the exception is VerSe20, which has only 10 test cases.

`results/` has the per-case scores (Dice, IoU, precision, recall, HD95, ASSD, normalized surface Dice) of every validation
and test case, and `results/results_fair.json` has the paired statistics.

## Placing the two views in 3D

The two X-rays must be placed in the 3D grid along the directions their rays travel:
- the AP image along front-to-back;
- the lateral image along left-to-right.

The stored AP images follow the usual display convention, mirrored relative to the volume, so they are mirrored back when
they are loaded. Two checks make sure this is right:

```bash
python scripts/check_geometry.py --part data --flip-ap   # do the loaded images match projections of the 3D masks?
python scripts/check_geometry.py --part models           # does each model put each view on the correct axis?
python tests/test_swin_windows.py                        # Swin window attention against a simple reference
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # pick the build for your driver
pip install -r requirements.txt
```

## Data

The datasets are not included. They come from TotalSegmentator, VerSe and CTPelvic1K, and were prepared with NAAMII's
open-source [XrayTo3DPreprocess](https://github.com/naamiinepal/XrayTo3DPreprocess) pipeline: digitally reconstructed
radiographs (DRRs) of a fixed-size box around the bone, plus a mask of the target bone. Expected layout:

```text
datasets/Total Segmentator/{Femur,Hip,Rib}/train/p_XXXX/{x-ray/*ap*.png, x-ray/*lat*.png, ct-scan/*.nii.gz}
datasets/Verse/{verse19,verse20}/train/p_XXXX/...
datasets/ctpelvik/train/p_XXXX/...
```

The splits in `splits/` select training, validation and test samples from the `train/` folders: seed 42, 80/10/10. For
CTPelvic, `splits/ctpelvic_seed42_dedup.json` keeps one copy of each scan.

## Training

Each model and dataset has a config named `fair_<model>_<dataset>`, for example `fair_multiscale_hip`. All configs use at
most 200 epochs, batch size 2, early stopping after 40 epochs without a better validation Dice, and the checkpoint with the
best validation Dice.

```bash
python train.py --list_configs
python train.py --config fair_multiscale_hip --no_wandb
bash scripts/run_fair_queue.sh none 2 fair_swinx2s_verse19 fair_multiscale_femur   # several runs, 2 at a time
```

`scripts/modal_fair.py` runs the same training on cloud GPUs (Modal), if you have an account.

## Evaluation

```bash
# validation first: rescores the validation split and checks it against the value stored with each checkpoint
python scripts/evaluate_fair.py --split val --hf-dir /path/to/biplanar-bone-reconstruction-models --out my_results
# then the test split, once
python scripts/evaluate_fair.py --split test --hf-dir /path/to/biplanar-bone-reconstruction-models --out my_results
python scripts/analyze_fair.py   # paired statistics of results/
```

`--hf-dir` points to a downloaded copy of the Hugging Face model repository. Without it, the script reads the original
training run folders listed in `scripts/evaluate_fair.py`.

## Limitations

- Trained and tested only on simulated X-ray images (DRRs), not on real radiographs, and not clinically validated.
- Splits are by sample, so vertebrae of one spine, or the two femurs of one scan, can fall into different splits.
- Each model was trained once (one seed), and the learning rates were not tuned.

## Licences

- **Data:** TotalSegmentator (CC BY 4.0), VerSe (CC BY-SA 4.0) and CTPelvic1K (CC BY-NC-SA). Models trained on
  CTPelvic1K must not be used commercially.
- **Swin-X2S:** the adapted Swin-X2S code is under the MIT licence; see `THIRD_PARTY_NOTICES.md`.

## Acknowledgements

We thank the Nepal Applied Mathematics and Informatics Institute for Research (NAAMII) for the processed datasets and the
XrayTo3DPreprocess pipeline, the Department of Electronics and Computer Engineering, Pulchowk Campus, Institute of
Engineering, Tribhuvan University, and the creators of TotalSegmentator, VerSe and CTPelvic1K.

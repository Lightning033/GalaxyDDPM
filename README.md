# Morphology-Conditioned Galaxy Image Synthesis using DDPMs

A conditional Denoising Diffusion Probabilistic Model (DDPM) trained to generate synthetic galaxy images conditioned on morphological class labels from the Galaxy Zoo 2 (GZ2) dataset. A conditional DCGAN baseline is included for comparison.

---

## Research Question

Can a conditional DDPM generate realistic, morphologically-diverse galaxy images across all 8 GZ2 classes, and does it outperform a conditional DCGAN baseline on FID?

---

## Results

| Metric | DDPM | DCGAN |
|---|---|---|
| FID Score ↓ | **84.16** | — |
| Parameters | 52.6M | ~4M |
| Epochs | 100 | 100 |

FID computed on 200 generated vs 200 real GZ2 validation images via Inception v3. Lower is better — random noise scores above 400.

---

## Project Structure

```
GalaxyDDPM/
├── src/
│   ├── dataset.py      # GZ2 data loading, per-class thresholding, augmentation
│   ├── unet.py         # Conditional U-Net — 52.6M parameters
│   ├── ddpm.py         # DDPM forward/reverse process and training loop
│   ├── dcgan.py        # Conditional DCGAN baseline
│   └── evaluate.py     # FID, MCA, nearest-neighbour evaluation
├── train_ddpm.py       # DDPM training entry point
├── train_dcgan.py      # DCGAN training entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
git clone https://github.com/Lightning033/GalaxyDDPM.git
cd GalaxyDDPM
pip install -r requirements.txt
```

Place dataset in `data/`:
```
data/
├── images_training_rev1/
└── training_solutions_rev1.csv
```

Download from: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge

---

## Training

```bash
python train_ddpm.py
python train_dcgan.py
```

Checkpoints saved every 10 epochs to `checkpoints/`.

---

## Architecture

### Conditional U-Net (DDPM)
- Parameters: 52.6M
- Channel schedule: [128, 256, 256, 512]
- Self-attention at 8×8 resolution only
- Sinusoidal timestep + learned class embedding injected at every ResBlock
- T=1000 timesteps, linear β: 1e-4 → 0.02
- AdamW lr=2e-4, cosine LR, gradient clipping=1.0

### Conditional DCGAN (Baseline)
- Generator: 5-layer transposed convolution, z_dim=128
- Discriminator: 5-layer strided convolution with class map conditioning
- Adam β₁=0.5, BCEWithLogitsLoss

---

## Key Design Decisions

- **Per-class thresholding** — GZ2 uses a hierarchical decision tree. Global τ=0.70 missed 5 classes. Per-class thresholds with priority ordering (Merger ≥ 0.05, Spirals ≥ 0.50, Ellipticals ≥ 0.469) fixed this.
- **Weighted random sampler** — Merger class has only 652 samples (1.7%). Oversampling ensures all 8 classes appear equally per batch.
- **64×64 resolution** — U-Net self-attention is O(H²W²). 64² fits T4/A100 VRAM while preserving key morphological features.
- **Iterative refinement** — global τ failed → per-class thresholds → OOM → batch=32 → weighted sampler → cosine LR + gradient clipping.

---

## Class Distribution

| Class | Samples | % |
|---|---|---|
| Round Elliptical | 9,285 | 24.9% |
| In-Between Elliptical | 7,708 | 20.7% |
| Cigar Elliptical | 4,719 | 12.7% |
| Edge-On Disc | 2,631 | 7.1% |
| Barred Spiral | 5,377 | 14.4% |
| Unbarred Spiral | 1,932 | 5.2% |
| Irregular | 4,982 | 13.4% |
| Merger | 652 | 1.7% |

---

## References

1. Ho et al. (2020). Denoising diffusion probabilistic models. NeurIPS.
2. Willett et al. (2013). Galaxy Zoo 2. MNRAS, 435(3).
3. Hart et al. (2016). GZ2 debiased classifications. MNRAS, 461(4).
4. Dhariwal & Nichol (2021). Diffusion models beat GANs. NeurIPS.
5. Ronneberger et al. (2015). U-Net. MICCAI.

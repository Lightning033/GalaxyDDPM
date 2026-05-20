import sys
import torch

sys.path.append("src")

from dataset import get_dataloaders
from unet import UNet
from ddpm import DDPM

CSV_PATH   = "data/training_solutions_rev1.csv"
IMG_DIR    = "data/images_training_rev1/"
SAVE_DIR   = "checkpoints/"
SAMPLE_DIR = "samples/"

BATCH_SIZE = 64
N_EPOCHS   = 100
LR         = 2e-4
SAVE_EVERY = 10
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"Device: {DEVICE}")

    train_loader, val_loader, _ = get_dataloaders(
        csv_path    = CSV_PATH,
        img_dir     = IMG_DIR,
        batch_size  = BATCH_SIZE,
        num_workers = 2,
    )

    model = UNet(
        in_ch       = 3,
        base_ch     = 128,
        ch_mult     = (1, 2, 2, 4),
        num_classes = 8,
        emb_dim     = 256,
    ).to(DEVICE)

    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    ddpm = DDPM(model=model, n_steps=1000, device=DEVICE)

    ddpm.train_loop(
        train_loader = train_loader,
        val_loader   = val_loader,
        n_epochs     = N_EPOCHS,
        lr           = LR,
        save_dir     = SAVE_DIR,
        sample_dir   = SAMPLE_DIR,
        save_every   = SAVE_EVERY,
    )


if __name__ == "__main__":
    main()

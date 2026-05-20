import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class DDPM:
    def __init__(
        self,
        model,
        n_steps    = 1000,
        beta_start = 1e-4,
        beta_end   = 0.02,
        device     = "cuda",
    ):
        self.model   = model
        self.n_steps = n_steps
        self.device  = device

        betas               = torch.linspace(beta_start, beta_end, n_steps).to(device)
        alphas              = 1.0 - betas
        alphas_cumprod      = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]])

        self.betas                         = betas
        self.alphas                        = alphas
        self.alphas_cumprod                = alphas_cumprod
        self.alphas_cumprod_prev           = alphas_cumprod_prev
        self.sqrt_alphas_cumprod           = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.posterior_variance            = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha     = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    def p_losses(self, x0, t, c):
        xt, noise  = self.q_sample(x0, t)
        noise_pred = self.model(xt, t, c)
        return nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(self, xt, t, c):
        betas_t        = self.betas[t][:, None, None, None]
        sqrt_recip     = (1.0 / self.alphas[t]).sqrt()[:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        noise_pred     = self.model(xt, t, c)
        mean           = sqrt_recip * (xt - betas_t / sqrt_one_minus * noise_pred)
        var            = self.posterior_variance[t][:, None, None, None]
        noise          = torch.randn_like(xt) if t[0].item() > 0 else torch.zeros_like(xt)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, n_samples, class_id, image_size=64, channels=3):
        self.model.eval()
        x = torch.randn(n_samples, channels, image_size, image_size).to(self.device)
        c = torch.full((n_samples,), class_id, dtype=torch.long).to(self.device)
        for t_val in reversed(range(self.n_steps)):
            t = torch.full((n_samples,), t_val, dtype=torch.long).to(self.device)
            x = self.p_sample(x, t, c)
        return x.clamp(-1, 1)

    def train_loop(
        self,
        train_loader,
        val_loader,
        n_epochs   = 100,
        lr         = 2e-4,
        save_dir   = "checkpoints",
        sample_dir = "samples",
        save_every = 10,
    ):
        os.makedirs(save_dir,   exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_val_loss = float("inf")
        train_losses  = []
        val_losses    = []

        for epoch in range(1, n_epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for imgs, labels in train_loader:
                imgs   = imgs.to(self.device)
                labels = labels.to(self.device)
                t      = torch.randint(0, self.n_steps, (imgs.size(0),), device=self.device)
                loss   = self.p_losses(imgs, t, labels)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_train = epoch_loss / len(train_loader)
            train_losses.append(avg_train)
            scheduler.step()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs   = imgs.to(self.device)
                    labels = labels.to(self.device)
                    t      = torch.randint(0, self.n_steps, (imgs.size(0),), device=self.device)
                    val_loss += self.p_losses(imgs, t, labels).item()

            avg_val = val_loss / len(val_loader)
            val_losses.append(avg_val)

            print(f"Epoch {epoch:03d}/{n_epochs}  train={avg_train:.4f}  val={avg_val:.4f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save({
                    "epoch":     epoch,
                    "model":     self.model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss":  best_val_loss,
                }, os.path.join(save_dir, "ddpm_best.pt"))

            if epoch % save_every == 0:
                torch.save({
                    "epoch":        epoch,
                    "model":        self.model.state_dict(),
                    "train_losses": train_losses,
                    "val_losses":   val_losses,
                }, os.path.join(save_dir, f"ddpm_epoch_{epoch:03d}.pt"))

        torch.save({
            "epoch":        n_epochs,
            "model":        self.model.state_dict(),
            "train_losses": train_losses,
            "val_losses":   val_losses,
        }, os.path.join(save_dir, "ddpm_final.pt"))

        print(f"Training complete. Best val loss: {best_val_loss:.4f}")
        return train_losses, val_losses

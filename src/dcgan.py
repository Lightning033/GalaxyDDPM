import os
import torch
import torch.nn as nn
from torch.optim import Adam


class Generator(nn.Module):
    def __init__(self, z_dim=128, num_classes=8, emb_dim=128, base_ch=64):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, emb_dim)
        in_dim = z_dim + emb_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, base_ch * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_ch, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, c):
        c_emb = self.class_emb(c).unsqueeze(-1).unsqueeze(-1)
        x     = torch.cat([z, c_emb], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, num_classes=8, base_ch=64):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, 64 * 64)
        self.net = nn.Sequential(
            nn.Conv2d(3 + 1, base_ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x, c):
        B     = x.size(0)
        c_map = self.class_emb(c).view(B, 1, 64, 64)
        x     = torch.cat([x, c_map], dim=1)
        return self.net(x).view(B)


class DCGAN:
    def __init__(
        self,
        z_dim       = 128,
        num_classes = 8,
        device      = "cuda",
        lr          = 2e-4,
        beta1       = 0.5,
        beta2       = 0.999,
    ):
        self.z_dim  = z_dim
        self.device = device

        self.G = Generator(z_dim=z_dim, num_classes=num_classes).to(device)
        self.D = Discriminator(num_classes=num_classes).to(device)

        self.G.apply(self._weights_init)
        self.D.apply(self._weights_init)

        self.opt_G = Adam(self.G.parameters(), lr=lr, betas=(beta1, beta2))
        self.opt_D = Adam(self.D.parameters(), lr=lr, betas=(beta1, beta2))

        self.criterion = nn.BCEWithLogitsLoss()

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)

    def train_loop(
        self,
        train_loader,
        n_epochs   = 100,
        save_dir   = "checkpoints",
        save_every = 10,
    ):
        os.makedirs(save_dir, exist_ok=True)

        g_losses = []
        d_losses = []

        for epoch in range(1, n_epochs + 1):
            epoch_g = 0.0
            epoch_d = 0.0

            for imgs, labels in train_loader:
                B      = imgs.size(0)
                real   = imgs.to(self.device)
                labels = labels.to(self.device)
                real_t = torch.ones(B,  device=self.device)
                fake_t = torch.zeros(B, device=self.device)

                self.opt_D.zero_grad()
                d_real = self.D(real, labels)
                z      = torch.randn(B, self.z_dim, 1, 1, device=self.device)
                fake   = self.G(z, labels).detach()
                d_fake = self.D(fake, labels)
                loss_D = (self.criterion(d_real, real_t) +
                          self.criterion(d_fake, fake_t)) / 2
                loss_D.backward()
                self.opt_D.step()

                self.opt_G.zero_grad()
                z      = torch.randn(B, self.z_dim, 1, 1, device=self.device)
                fake   = self.G(z, labels)
                d_fake = self.D(fake, labels)
                loss_G = self.criterion(d_fake, real_t)
                loss_G.backward()
                self.opt_G.step()

                epoch_g += loss_G.item()
                epoch_d += loss_D.item()

            avg_g = epoch_g / len(train_loader)
            avg_d = epoch_d / len(train_loader)
            g_losses.append(avg_g)
            d_losses.append(avg_d)

            print(f"Epoch {epoch:03d}/{n_epochs}  G={avg_g:.4f}  D={avg_d:.4f}")

            if epoch % save_every == 0:
                torch.save({
                    "epoch":    epoch,
                    "G":        self.G.state_dict(),
                    "D":        self.D.state_dict(),
                    "g_losses": g_losses,
                    "d_losses": d_losses,
                }, os.path.join(save_dir, f"dcgan_epoch_{epoch:03d}.pt"))

        torch.save({
            "epoch":    n_epochs,
            "G":        self.G.state_dict(),
            "D":        self.D.state_dict(),
            "g_losses": g_losses,
            "d_losses": d_losses,
        }, os.path.join(save_dir, "dcgan_final.pt"))

        print("DCGAN training complete.")
        return g_losses, d_losses

    @torch.no_grad()
    def sample(self, n_samples, class_id):
        self.G.eval()
        z = torch.randn(n_samples, self.z_dim, 1, 1, device=self.device)
        c = torch.full((n_samples,), class_id, dtype=torch.long, device=self.device)
        return self.G(z, c).clamp(-1, 1)

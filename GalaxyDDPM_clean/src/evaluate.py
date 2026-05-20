import torch
import torch.nn as nn
import numpy as np
from torchvision.models import inception_v3, resnet18
from torchvision import transforms
from scipy.linalg import sqrtm


def get_inception_features(imgs, model, device):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    with torch.no_grad():
        imgs  = torch.stack([transform(img) for img in imgs]).to(device)
        feats = model(imgs)
    return feats.cpu().numpy()


def compute_fid(real_imgs, gen_imgs, device="cuda", batch_size=64):
    inception = inception_v3(pretrained=True, transform_input=False)
    inception.fc = nn.Identity()
    inception = inception.to(device).eval()

    def get_feats(imgs):
        all_feats = []
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i:i+batch_size]
            feats = get_inception_features(batch, inception, device)
            all_feats.append(feats)
        return np.concatenate(all_feats, axis=0)

    real_feats = get_feats(real_imgs)
    gen_feats  = get_feats(gen_imgs)

    mu_r, sig_r = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu_g, sig_g = gen_feats.mean(0),  np.cov(gen_feats,  rowvar=False)

    diff    = mu_r - mu_g
    covmean = sqrtm(sig_r @ sig_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sig_r + sig_g - 2 * covmean)
    return float(fid)


def train_morphology_classifier(train_loader, val_loader, device="cuda", n_epochs=20):
    model    = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 8)
    model    = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc   = 0.0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds    = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        acc = correct / total
        print(f"Classifier epoch {epoch:02d}/{n_epochs}  val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"Best classifier accuracy: {best_acc:.4f}")
    return model


def compute_mca(generated_imgs, class_labels, classifier, device="cuda", batch_size=64):
    classifier.eval()
    correct = total = 0

    for i in range(0, len(generated_imgs), batch_size):
        imgs   = generated_imgs[i:i+batch_size]
        labels = class_labels[i:i+batch_size]
        imgs_t = torch.stack(imgs).to(device)
        labs_t = torch.tensor(labels, dtype=torch.long, device=device)

        with torch.no_grad():
            preds = classifier(imgs_t).argmax(dim=1)
        correct += (preds == labs_t).sum().item()
        total   += labs_t.size(0)

    return correct / total


def compute_per_class_mca(generated_imgs, class_labels, classifier, device="cuda"):
    classifier.eval()
    per_class = {i: {"correct": 0, "total": 0} for i in range(8)}

    for img, label in zip(generated_imgs, class_labels):
        img_t = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = classifier(img_t).argmax(dim=1).item()
        per_class[label]["total"]   += 1
        per_class[label]["correct"] += int(pred == label)

    return {
        cid: (vals["correct"] / vals["total"] if vals["total"] > 0 else 0.0)
        for cid, vals in per_class.items()
    }


def nearest_neighbour_check(generated_imgs, train_imgs):
    gen   = torch.stack(generated_imgs).view(len(generated_imgs), -1).numpy()
    train = torch.stack(train_imgs).view(len(train_imgs), -1).numpy()

    distances = []
    for g in gen:
        dists = np.linalg.norm(train - g, axis=1)
        distances.append(dists.min())

    return {
        "mean_nn_dist": float(np.mean(distances)),
        "std_nn_dist":  float(np.std(distances)),
        "min_nn_dist":  float(np.min(distances)),
    }

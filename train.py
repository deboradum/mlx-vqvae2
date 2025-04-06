import os
import time

import numpy as np
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim


from PIL import Image


from models.vqvae import VQVAE2
from dataset import get_dataloaders


def save_snapshot(net, loader, path="results/0/"):
    net.train(False)
    os.makedirs(path, exist_ok=True)

    batch, _ = next(iter(test_loader))
    batch = mx.array(batch.numpy()).transpose(0, 2, 3, 1)
    x_hat, _, _, _ = net(batch)

    x_hat = np.array(x_hat)
    batch = np.array(batch)

    for i in range(batch.shape[0]):
        orig = batch[i]
        recon = x_hat[i]

        orig = np.clip(orig * 255, 0, 255).astype(np.uint8)
        recon = np.clip(recon * 255, 0, 255).astype(np.uint8)

        h, w, c = orig.shape

        separator_img = np.full((h, 1, c), (0, 255, 0), dtype=np.uint8)

        combined = np.concatenate([orig, separator_img, recon], axis=1).astype(np.uint8)
        img = Image.fromarray(combined)

        img.save(os.path.join(path, f"{i}.png"))


def loss_fn(net, X):
    x_hat, top_loss, btm_loss, perplexity = net(X)
    recon_loss = ((x_hat - X) ** 2).mean() / x_train_var
    loss = recon_loss + top_loss + btm_loss

    metrics["total_loss"].append(loss.item())
    metrics["recon_loss"].append(recon_loss.item())
    metrics["top_loss"].append(top_loss.item())
    metrics["btm_loss"].append(btm_loss.item())
    metrics["perplexity"].append(perplexity.item())

    return loss


def get_avg(values):
    return sum(values) / len(values)


def get_avg_metrics(window):
    total_loss = get_avg(metrics["total_loss"][-window:-1])
    recon_loss = get_avg(metrics["recon_loss"][-window:-1])
    perplexity = get_avg(metrics["perplexity"][-window:-1])

    return total_loss, recon_loss, perplexity


def train(epochs, net, optimizer, train_loader, test_loader, x_train_var, log_every=10):
    loss_and_grad_fn = nn.value_and_grad(net, loss_fn)
    s = time.perf_counter()
    for epoch in range(epochs):
        save_snapshot(net, test_loader, path=f"results/{epoch}")
        net.train(True)
        for i, (X, _) in enumerate(train_loader):
            X = mx.array(X.numpy()).transpose(0, 2, 3, 1)

            loss, grads = loss_and_grad_fn(net, X)
            optimizer.update(net, grads)
            mx.eval(net.parameters(), optimizer.state)

            if i % log_every == 0 and i > 0:
                took = time.perf_counter() - s
                time_per_it = took / log_every
                avg_total_loss, avg_recon_loss, avg_perplexity = get_avg_metrics(log_every)
                s = time.perf_counter()
                print(
                    f"Epoch {epoch}, step {i}/{len(train_loader)} - loss: {avg_total_loss:.5f}, recon_loss: {avg_recon_loss:.5f}, perplexity: {avg_perplexity:.5f}. Took {took:.3f}s ({time_per_it:.3f} s/it)"
                )

if __name__ == "__main__":
    dataset = "geoguessr"
    metrics = {
        "total_loss": [],
        "recon_loss": [],
        "top_loss": [],
        "btm_loss": [],
        "perplexity": [],
    }
    net = VQVAE2(128, 64, 2, 512, 64, 0.25)

    optimizer = optim.Adam(learning_rate=1e-4)

    train_loader, test_loader, x_train_var = get_dataloaders(dataset, batch_size=4)
    train(10, net, optimizer, train_loader, test_loader, x_train_var)

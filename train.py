# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import ViTMoE
from dataset import CIFAR10Small
import matplotlib.pyplot as plt
import time
from tqdm import tqdm  # progress bar

if __name__ == "__main__":  # necessary for Windows

    # -----------------------------
    # Configuration
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "img_size": 32,
        "patch_size": 4,
        "emb_size": 128,
        "depth": 6,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "dropout": 0.3,
        "num_classes": 10,
        "batch_size": 128,
        "train_size": 20000,
        "test_size": 5000,
        "epochs": 100,
        "lr": 3e-4,
        "seed": 42
    }

    torch.manual_seed(config["seed"])

    print("Loading CIFAR-10 small dataset...")
    dataset = CIFAR10Small(train_size=config["train_size"], test_size=config["test_size"], batch_size=config["batch_size"], seed=config["seed"])
    trainloader, testloader = dataset.loaders()
    print(f"Train batches: {len(trainloader)}, Test batches: {len(testloader)}")

    # -----------------------------
    # Model, Loss, Optimizer
    # -----------------------------
    print("Initializing ViT-MoE model...")
    model = ViTMoE(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        num_classes=config["num_classes"],
        emb_size=config["emb_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        dropout=config["dropout"]
    ).to(device)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    # Optional: parameters per module type
    param_summary = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_type = name.split('.')[0]
            param_summary[layer_type] = param_summary.get(layer_type, 0) + param.numel()

    print("Trainable parameters per module:")
    for k, v in param_summary.items():
        print(f"  {k}: {v:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # -----------------------------
    # Training Loop
    # -----------------------------
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("Starting training...\n")
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        start_time = time.time()

        # Progress bar for batches
        loop = tqdm(trainloader, desc=f"Epoch {epoch}/{config['epochs']} [Train]", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Compute training accuracy
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            loop.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        train_loss = running_loss / len(trainloader.dataset)
        train_acc = running_correct / running_total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            val_loop = tqdm(testloader, desc=f"Epoch {epoch}/{config['epochs']} [Val]  ", leave=False)
            for images, labels in val_loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                val_loop.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        val_loss /= len(testloader.dataset)
        val_losses.append(val_loss)
        val_acc = correct / total
        val_accuracies.append(val_acc)

        scheduler.step()
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch}/{config['epochs']}] "
              f"Train Loss: {train_loss:.4f} "
              f"Train Acc: {train_acc*100:.2f}% "
              f"Val Loss: {val_loss:.4f} "
              f"Val Acc: {val_acc*100:.2f}% "
              f"Time: {epoch_time:.1f}s")

    # -----------------------------
    # Plot training curves
    # -----------------------------
    plt.figure(figsize=(12,4))

    plt.subplot(1,2,1)
    plt.plot(range(1, config["epochs"]+1), train_losses, label="Train Loss")
    plt.plot(range(1, config["epochs"]+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1, config["epochs"]+1), train_accuracies, label="Train Acc")
    plt.plot(range(1, config["epochs"]+1), val_accuracies, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()

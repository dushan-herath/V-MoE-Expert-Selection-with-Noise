import torch
import torch.nn as nn
import torch.optim as optim
from model import ViTMoE
from dataset import CIFAR10Small
import matplotlib.pyplot as plt
import time
from tqdm import tqdm  


# this method is to check expertbias during the training
def check_expert_bias(model, dataloader, device, max_batches=10):

    model.eval()
    num_experts = model.moe.num_experts
    expert_counts = torch.zeros(num_experts, device=device)

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= max_batches:
                break

            images = images.to(device)
            _, routing = model(images, return_routing=True)  # [B, N, k]

            for e in range(num_experts):
                expert_counts[e] += (routing == e).sum()

    probs = expert_counts / expert_counts.sum()
    return probs.cpu()

def save_training_curves(
    train_losses, val_losses,
    train_accuracies, val_accuracies,
    epoch, save_path
):
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epoch + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 1), train_accuracies, label="Train Acc")
    plt.plot(range(1, epoch + 1), val_accuracies, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  


if __name__ == "__main__":  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "img_size": 32,
        "patch_size": 4,
        "emb_size": 128,
        "num_heads": 8,
        "dropout": 0.2,
        "num_classes": 10,
        "batch_size": 32,
        "train_size": 50000,
        "test_size": 10000,
        "epochs": 100,
        "lr": 3e-4,
        "seed": 42
    }

    torch.manual_seed(config["seed"])

    print("Loading dataset...")
    dataset = CIFAR10Small(train_size=config["train_size"],
                            test_size=config["test_size"],
                            batch_size=config["batch_size"],
                            seed=config["seed"])
    trainloader, testloader = dataset.loaders()
    print(f"Train batches: {len(trainloader)}, Test batches: {len(testloader)}")


    print("Initializing ViT-MoE model...")
    model = ViTMoE(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        num_classes=config["num_classes"],
        emb_size=config["emb_size"],
        num_heads=config["num_heads"],
        dropout=config["dropout"]
    ).to(device)

    # print trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    param_summary = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_type = name.split('.')[0]
            param_summary[layer_type] = param_summary.get(layer_type, 0) + param.numel()

    print("trainable parameters per module:")
    for k, v in param_summary.items():
        print(f"  {k}: {v:,}")


    # loss function, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # Training Loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_acc = 0.0
    best_model_path = "best_vit_moe.pth"
    
    print("starting training...\n")
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        start_time = time.time()

        loop = tqdm(trainloader, desc=f"Epoch {epoch}/{config['epochs']} [Train]", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            #outputs = model(images)
            #loss = criterion(outputs, labels)
            outputs, _, load_loss = model(images, return_routing=True, return_load_loss=True)
            loss = criterion(outputs, labels) + 0.01 * load_loss  # small weight for balancing
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

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

        # Expert usage check
        expert_probs = check_expert_bias(model, trainloader, device)
        expert_str = " | ".join([f"E{i}: {p:.2f}" for i, p in enumerate(expert_probs)])

        print(f"Epoch [{epoch}/{config['epochs']}] "
              f"Train Loss: {train_loss:.4f} "
              f"Train Acc: {train_acc*100:.2f}% "
              f"Val Loss: {val_loss:.4f} "
              f"Val Acc: {val_acc*100:.2f}% "
              f"Experts → {expert_str} "
              f"Time: {epoch_time:.1f}s")

        # Save the best model

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, best_model_path)

            curve_path = f"training_curves_best_epoch_{epoch}.png"

            save_training_curves(
                train_losses,
                val_losses,
                train_accuracies,
                val_accuracies,
                epoch,
                curve_path
            )

            print(
                f"--> Best model saved at epoch {epoch} "
                f"with Val Acc: {val_acc*100:.2f}% "
                f"| Curves saved to {curve_path}"
            )


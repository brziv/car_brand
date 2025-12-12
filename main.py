import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from model import get_model
from dataset import train_dataloader, val_dataloader, test_dataloader
from train import run_epoch
import matplotlib.pyplot as plt
import json
import seaborn as sns
import os

# config
num_classes_brand = 22
num_classes_color = 9
num_epochs = 50
lr = 1e-4
model_names = [
    "efficientnet_b0",
    "resnet50"
]

# early stopping
patience = 5

# Create results directory if not exists
os.makedirs("results/", exist_ok=True)

for model_name in model_names:
    print(f"\n--- Training {model_name} ---")
    
    # model
    model = get_model(model_name, num_classes_brand, num_classes_color).to("cuda" if torch.cuda.is_available() else "cpu")
    criterion_brand = nn.CrossEntropyLoss()
    criterion_color = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    best_val_loss = float("inf")
    no_improve = 0
    
    # create dicts
    train_metrics = {"loss": [], "acc_brand": [], "acc_color": [], "precision_brand": [], "recall_brand": [], "f1_brand": [], "precision_color": [], "recall_color": [], "f1_color": []}
    val_metrics = {"loss": [], "acc_brand": [], "acc_color": [], "precision_brand": [], "recall_brand": [], "f1_brand": [], "precision_color": [], "recall_color": [], "f1_color": []}
    
    # main loop
    for epoch in range(num_epochs):
        train_loss, train_acc_brand, train_acc_color, train_prec_brand, train_rec_brand, train_f1_brand, train_prec_color, train_rec_color, train_f1_color, _, _ = run_epoch(
            model, train_dataloader, criterion_brand, criterion_color, optimizer, mode="Train"
        )
        val_loss, val_acc_brand, val_acc_color, val_prec_brand, val_rec_brand, val_f1_brand, val_prec_color, val_rec_color, val_f1_color, _, _ = run_epoch(
            model, val_dataloader, criterion_brand, criterion_color, optimizer=None, mode="Validation"
        )

        # save metrics to dict
        train_metrics["loss"].append(train_loss)
        train_metrics["acc_brand"].append(train_acc_brand)
        train_metrics["acc_color"].append(train_acc_color)
        train_metrics["precision_brand"].append(train_prec_brand)
        train_metrics["recall_brand"].append(train_rec_brand)
        train_metrics["f1_brand"].append(train_f1_brand)
        train_metrics["precision_color"].append(train_prec_color)
        train_metrics["recall_color"].append(train_rec_color)
        train_metrics["f1_color"].append(train_f1_color)

        val_metrics["loss"].append(val_loss)
        val_metrics["acc_brand"].append(val_acc_brand)
        val_metrics["acc_color"].append(val_acc_color)
        val_metrics["precision_brand"].append(val_prec_brand)
        val_metrics["recall_brand"].append(val_rec_brand)
        val_metrics["f1_brand"].append(val_f1_brand)
        val_metrics["precision_color"].append(val_prec_color)
        val_metrics["recall_color"].append(val_rec_color)
        val_metrics["f1_color"].append(val_f1_color)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Brand Acc: {train_acc_brand:.2f}%, Color Acc: {train_acc_color:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Brand Acc: {val_acc_brand:.2f}%, Color Acc: {val_acc_color:.2f}%\n")

        scheduler.step(val_loss)

        # early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            # save best model
            torch.save(model.state_dict(), f"results/{model_name}_best.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs for {model_name}.")
                break

    # test
    test_loss, test_acc_brand, test_acc_color, test_prec_brand, test_rec_brand, test_f1_brand, test_prec_color, test_rec_color, test_f1_color, test_cm_brand, test_cm_color = run_epoch(
        model, test_dataloader, criterion_brand, criterion_color, optimizer=None, mode="Test"
    )
    print(f"Test Loss: {test_loss:.4f}, Brand Acc: {test_acc_brand:.2f}%, Color Acc: {test_acc_color:.2f}%")

    # save test metrics
    test_metrics = {
        "loss": test_loss,
        "acc_brand": test_acc_brand,
        "acc_color": test_acc_color,
        "precision_brand": test_prec_brand,
        "recall_brand": test_rec_brand,
        "f1_brand": test_f1_brand,
        "precision_color": test_prec_color,
        "recall_color": test_rec_color,
        "f1_color": test_f1_color,
        "confusion_matrix_brand": test_cm_brand.tolist(),
        "confusion_matrix_color": test_cm_color.tolist()
    }
    with open(f"results/{model_name}_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    # plot metrics
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    metrics_to_plot = ["loss", "acc_brand", "acc_color", "precision_brand", "recall_brand", "f1_brand", "precision_color", "recall_color", "f1_color"]

    for i, metric in enumerate(metrics_to_plot):
        axes[i].plot(train_metrics[metric], label="Train")
        axes[i].plot(val_metrics[metric], label="Validation")
        axes[i].set_title(f"{metric.replace('_', ' ').capitalize()} per Epoch")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(metric.replace('_', ' ').capitalize())
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.savefig(f"results/{model_name}_metrics.png")
    plt.close()

    # plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(test_cm_brand, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix Brand")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"results/{model_name}_cmatrix_brand.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(test_cm_color, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix Color")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"results/{model_name}_cmatrix_color.png")
    plt.close()
    
    # Clear memory
    del model
    torch.cuda.empty_cache()

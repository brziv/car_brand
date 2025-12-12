import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from dataset import train_dataloader

# config
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes_brand = 22
num_classes_color = 9
max_steps = 50
lr = 1e-4
model_names = [
    "efficientnet_b0",
    "efficientnet_v2_s",
    "resnet50"
]

# train one batch
images, labels_brand, labels_color = next(iter(train_dataloader))
images, labels_brand, labels_color = images.to(device), labels_brand.to(device), labels_color.to(device)

for model_name in model_names:
    print(f"\n--- Overfitting {model_name} ---")
    
    model = get_model(model_name, num_classes_brand, num_classes_color).to(device)
    criterion_brand = nn.CrossEntropyLoss()
    criterion_color = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    converged_brand = False
    converged_color = False
    for step in range(max_steps):
        model.train()
        optimizer.zero_grad()
        outputs_brand, outputs_color = model(images)
        loss_brand = criterion_brand(outputs_brand, labels_brand)
        loss_color = criterion_color(outputs_color, labels_color)
        loss = loss_brand + loss_color
        loss.backward()
        optimizer.step()

        _, preds_brand = outputs_brand.max(1)
        _, preds_color = outputs_color.max(1)
        acc_brand = (preds_brand == labels_brand).float().mean().item() * 100
        acc_color = (preds_color == labels_color).float().mean().item() * 100

        print(f"Step {step+1}/{max_steps}, Loss: {loss.item():.4f}, Brand Acc: {acc_brand:.2f}%, Color Acc: {acc_color:.2f}%")
        
        if acc_brand == 100.:
            print(f"{model_name} Brand Converged at step {step+1}!")
            converged_brand = True
        if acc_color == 100.:
            print(f"{model_name} Color Converged at step {step+1}!")
            converged_color = True
        if converged_brand and converged_color:
            break
    
    if not converged_brand:
        print(f"{model_name} Brand did not converge within {max_steps} steps.")
    if not converged_color:
        print(f"{model_name} Color did not converge within {max_steps} steps.")
    
    # Clear memory
    del model
    torch.cuda.empty_cache()

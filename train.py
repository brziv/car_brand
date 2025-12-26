from tqdm import tqdm
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def run_epoch(model, dataloader, criterion_brand, criterion_color, optimizer=None, mode="Train", num_classes_brand=None, num_classes_color=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    scaler = torch.amp.GradScaler(device, enabled=is_train)
    
    running_loss, running_corrects_brand, running_corrects_color, total_samples = 0.0, 0, 0, 0
    all_labels_brand, all_preds_brand = [], []
    all_labels_color, all_preds_color = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels_brand, labels_color in tqdm(dataloader, desc=mode, leave=False):
            images, labels_brand, labels_color = images.to(device), labels_brand.to(device), labels_color.to(device)
            
            if is_train:
                optimizer.zero_grad()
            
            with torch.amp.autocast(device, enabled=True):
                outputs_brand, outputs_color = model(images)
                loss_brand = criterion_brand(outputs_brand, labels_brand)
                loss_color = criterion_color(outputs_color, labels_color)
                loss = loss_brand + loss_color
            
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            _, preds_brand = outputs_brand.max(1)
            _, preds_color = outputs_color.max(1)
            running_loss += loss.item() * images.size(0)
            running_corrects_brand += (preds_brand == labels_brand).sum().item()
            running_corrects_color += (preds_color == labels_color).sum().item()
            total_samples += images.size(0)
            
            all_labels_brand.extend(labels_brand.cpu().numpy())
            all_preds_brand.extend(preds_brand.cpu().numpy())
            all_labels_color.extend(labels_color.cpu().numpy())
            all_preds_color.extend(preds_color.cpu().numpy())
    
    epoch_loss = running_loss / total_samples
    epoch_acc_brand = running_corrects_brand / total_samples * 100
    epoch_acc_color = running_corrects_color / total_samples * 100

    precision_brand = precision_score(all_labels_brand, all_preds_brand, average='macro', zero_division=0) * 100
    recall_brand = recall_score(all_labels_brand, all_preds_brand, average='macro', zero_division=0) * 100
    f1_brand = f1_score(all_labels_brand, all_preds_brand, average='macro', zero_division=0) * 100

    precision_color = precision_score(all_labels_color, all_preds_color, average='macro', zero_division=0) * 100
    recall_color = recall_score(all_labels_color, all_preds_color, average='macro', zero_division=0) * 100
    f1_color = f1_score(all_labels_color, all_preds_color, average='macro', zero_division=0) * 100
    
    # ensure confusion matrices include all classes
    if num_classes_brand is not None:
        labels_brand = list(range(num_classes_brand))
        conf_matrix_brand = confusion_matrix(all_labels_brand, all_preds_brand, labels=labels_brand)
    else:
        conf_matrix_brand = confusion_matrix(all_labels_brand, all_preds_brand)

    if num_classes_color is not None:
        labels_color = list(range(num_classes_color))
        conf_matrix_color = confusion_matrix(all_labels_color, all_preds_color, labels=labels_color)
    else:
        conf_matrix_color = confusion_matrix(all_labels_color, all_preds_color)
    
    return epoch_loss, epoch_acc_brand, epoch_acc_color, precision_brand, recall_brand, f1_brand, precision_color, recall_color, f1_color, conf_matrix_brand, conf_matrix_color

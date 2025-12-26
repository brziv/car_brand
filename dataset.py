import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

batch_size = 32
img_dir = "images"
train_path = "annot/train.txt"
val_path = "annot/val.txt"

class CarDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.samples = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue  # skip invalid
                img_name = " ".join(parts[:-2])
                brand_label = int(parts[-2])
                color_label = int(parts[-1])
                self.samples.append((img_name, brand_label, color_label))

        self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_name, brand_label, color_label = self.samples[index]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, brand_label, color_label
    
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),

    # transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(5),

    # transforms.ColorJitter(
    #     brightness=0.1,
    #     contrast=0.1,
    #     saturation=0.05,
    #     hue=0.0
    # ),
    # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# input
train_dataset = CarDataset(img_dir, train_path, transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CarDataset(img_dir, val_path, transform=val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
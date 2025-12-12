import torch.nn as nn
from torchvision import models

def get_model(name, num_classes_brand, num_classes_color):
    name = name.lower()

    if name.startswith("efficientnet"):
        if name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif name == "efficientnet_v2_s":
            model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unknown EfficientNet model: {name}")
        
        original_forward = model.forward
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Identity()
        model.dropout = nn.Dropout(0.2)
        model.brand_head = nn.Linear(in_features, num_classes_brand)
        model.color_head = nn.Linear(in_features, num_classes_color)
        
        def forward(self, x):
            x = original_forward(x)
            x = self.dropout(x)
            return self.brand_head(x), self.color_head(x)
        model.forward = forward.__get__(model, type(model))
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        original_forward = model.forward
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        model.dropout = nn.Dropout(0.2)
        model.brand_head = nn.Linear(in_features, num_classes_brand)
        model.color_head = nn.Linear(in_features, num_classes_color)
        
        def forward(self, x):
            x = original_forward(x)
            x = self.dropout(x)
            return self.brand_head(x), self.color_head(x)
        model.forward = forward.__get__(model, type(model))
    else:
        raise ValueError(f"Unknown model name: {name}")

    return model

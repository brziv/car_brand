import torch.nn as nn
from torchvision import models

def get_model(name, num_classes_brand, num_classes_color):
    name = name.lower()

    # --- ResNet / ResNeXt / RegNet ---
    if name.startswith("resnet") or name.startswith("resnext") or name.startswith("regnet"):
        if name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif name == "resnext50":
            model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        elif name == "regnet_x_1_6gf":
            model = models.regnet_x_1_6gf(weights=models.RegNet_X_1_6GF_Weights.IMAGENET1K_V2)
        elif name == "regnet_y_1_6gf":
            model = models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V2)
        elif name == "regnet_x_3_2gf":
            model = models.regnet_x_3_2gf(weights=models.RegNet_X_3_2GF_Weights.IMAGENET1K_V2)
        elif name == "regnet_y_3_2gf":
            model = models.regnet_y_3_2gf(weights=models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V2)
        elif name == "regnet_x_800mf":
            model = models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V2)
        elif name == "regnet_y_800mf":
            model = models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unknown model name: {name}")
        
        original_forward = model.forward
        in_features = model.fc.in_features
        model.fc = nn.Identity()  # remove original fc
        model.brand_head = nn.Linear(in_features, num_classes_brand)
        model.color_head = nn.Linear(in_features, num_classes_color)
        
        def forward(self, x):
            x = original_forward(x)
            return self.brand_head(x), self.color_head(x)
        model.forward = forward.__get__(model, type(model))

    # --- EfficientNet ---
    elif name.startswith("efficientnet"):
        if name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif name == "efficientnet_b1":
            model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V2)
        elif name == "efficientnet_b2":
            model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        elif name == "efficientnet_v2_s":
            model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unknown model name: {name}")
        
        original_forward = model.forward
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Identity()
        model.brand_head = nn.Linear(in_features, num_classes_brand)
        model.color_head = nn.Linear(in_features, num_classes_color)
        
        def forward(self, x):
            x = original_forward(x)
            return self.brand_head(x), self.color_head(x)
        model.forward = forward.__get__(model, type(model))

    # --- ConvNeXt ---
    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        original_forward = model.forward
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Identity()
        model.brand_head = nn.Linear(in_features, num_classes_brand)
        model.color_head = nn.Linear(in_features, num_classes_color)
        
        def forward(self, x):
            x = original_forward(x)
            return self.brand_head(x), self.color_head(x)
        model.forward = forward.__get__(model, type(model))

    # --- Swin Transformer ---
    elif name == "swin_t":
        model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        original_forward = model.forward
        in_features = model.head.in_features
        model.head = nn.Identity()
        model.brand_head = nn.Linear(in_features, num_classes_brand)
        model.color_head = nn.Linear(in_features, num_classes_color)
        
        def forward(self, x):
            x = original_forward(x)
            return self.brand_head(x), self.color_head(x)
        model.forward = forward.__get__(model, type(model))
        
    # --- DenseNet ---
    elif name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        original_forward = model.forward
        in_features = model.classifier.in_features
        model.classifier = nn.Identity()
        model.brand_head = nn.Linear(in_features, num_classes_brand)
        model.color_head = nn.Linear(in_features, num_classes_color)
        
        def forward(self, x):
            x = original_forward(x)
            return self.brand_head(x), self.color_head(x)
        model.forward = forward.__get__(model, type(model))

    else:
        raise ValueError(f"Unknown model name: {name}")

    return model

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import segmentation_models_pytorch as smp
import joblib
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -----------------------
# Device Helper
# -----------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Model Loading Functions
# -----------------------
def load_segmentation_model(device, model_path: str):
    try:
        # Make sure the model path exists
        if not os.path.exists(model_path):
            print(f"Warning: Model path {model_path} does not exist.")
            # Try to find the model in the models directory
            alt_path = os.path.join("models", os.path.basename(model_path))
            if os.path.exists(alt_path):
                model_path = alt_path
                print(f"Using alternative path: {model_path}")
            else:
                raise FileNotFoundError(f"Could not find model at {model_path} or {alt_path}")

        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,  # Disable download of pretrained weights
            in_channels=3,
            classes=1,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Loaded DeepLabV3+ weights from:", model_path)
        return model
    except Exception as e:
        print(f"Error loading segmentation model: {str(e)}")
        raise

def load_efficientnet_model(device, model_path: str):
    try:
        # Make sure the model path exists
        if not os.path.exists(model_path):
            print(f"Warning: Model path {model_path} does not exist.")
            # Try to find the model in the models directory
            alt_path = os.path.join("models", os.path.basename(model_path))
            if os.path.exists(alt_path):
                model_path = alt_path
                print(f"Using alternative path: {model_path}")
            else:
                raise FileNotFoundError(f"Could not find model at {model_path} or {alt_path}")

        backbone = timm.create_model('tf_efficientnetv2_m', pretrained=False, num_classes=0)
        # Remove original classifier head
        backbone.classifier = nn.Identity()

        # Define the final model using a custom module
        class FinalBoneAgeModel(nn.Module):
            def __init__(self, backbone, gender_embed_dim, dropout_rate, num_out=1):
                super(FinalBoneAgeModel, self).__init__()
                self.backbone = backbone
                self.gender_embedding = nn.Linear(1, gender_embed_dim)
                in_features = self.backbone.num_features
                self.fc = nn.Sequential(
                    nn.Linear(in_features + gender_embed_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(256, num_out)
                )
            def forward(self, image, gender):
                features = self.backbone(image)
                gender_emb = self.gender_embedding(gender)
                x = torch.cat([features, gender_emb], dim=1)
                output = self.fc(x)
                return output.squeeze(1)

        # Hyperparameters (from tuning)
        best_dropout = 0.44341782415973996
        best_gender_embed_dim = 8
        model = FinalBoneAgeModel(backbone, best_gender_embed_dim, best_dropout, num_out=1)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Loaded final EfficientNetV2M model from:", model_path)
        return model
    except Exception as e:
        print(f"Error loading EfficientNet model: {str(e)}")
        raise

def load_lin_reg_model(model_path: str):
    try:
        # Make sure the model path exists
        if not os.path.exists(model_path):
            print(f"Warning: Model path {model_path} does not exist.")
            # Try to find the model in the models directory
            alt_path = os.path.join("models", os.path.basename(model_path))
            if os.path.exists(alt_path):
                model_path = alt_path
                print(f"Using alternative path: {model_path}")
            else:
                raise FileNotFoundError(f"Could not find model at {model_path} or {alt_path}")

        model = joblib.load(model_path)
        print("Loaded linear regression calibration model from:", model_path)
        return model
    except Exception as e:
        print(f"Error loading linear regression model: {str(e)}")
        raise

# -----------------------
# Helper Functions for Image Processing
# -----------------------
def get_bounding_box(mask_np):
    """Compute tight bounding box from a binary mask (numpy array)."""
    coords = np.column_stack(np.where(mask_np > 0))
    if coords.size == 0:
        # If mask is empty, return full dimensions (assumed 512x512 here)
        return 0, 0, mask_np.shape[1], mask_np.shape[0]
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return x_min, y_min, x_max, y_max

def crop_and_resize(image_pil, mask_np, target_size=(480,480)):
    """
    Given a PIL image and a corresponding binary mask (numpy array),
    compute the bounding box, crop the image, pad to square if needed,
    and finally resize to target_size.
    """
    x_min, y_min, x_max, y_max = get_bounding_box(mask_np)
    width, height = image_pil.size
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(width, x_max), min(height, y_max)
    cropped = image_pil.crop((x_min, y_min, x_max, y_max))
    cropped_width, cropped_height = cropped.size
    max_side = max(cropped_width, cropped_height)
    padded = Image.new("RGB", (max_side, max_side), (0, 0, 0))
    paste_x = (max_side - cropped_width) // 2
    paste_y = (max_side - cropped_height) // 2
    padded.paste(cropped, (paste_x, paste_y))
    resized = padded.resize(target_size, Image.BILINEAR)
    return resized

# -----------------------
# Preprocessing Transforms
# -----------------------
inference_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# Define a list of Test-Time Augmentation (TTA) transforms
tta_transforms = []
for _ in range(10):
    tta_transforms.append(
        A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
    )

# -----------------------
# Inference Pipeline
# -----------------------
def process_image_for_inference(image: Image.Image, gender: float, seg_model, final_model, lin_reg, device):
    # 1. Preprocessing for Segmentation: Resize to 512x512
    seg_image = image.resize((512, 512), Image.BILINEAR)
    seg_tensor = inference_transform(seg_image).unsqueeze(0).to(device)

    # 2. Run segmentation
    with torch.no_grad():
        seg_output = seg_model(seg_tensor)
        seg_prob = torch.sigmoid(seg_output).cpu().numpy()[0, 0]
    binary_mask = (seg_prob > 0.5).astype(np.uint8) * 255
    mask_pil = Image.fromarray(binary_mask).resize((512, 512), Image.NEAREST)

    # 3. Crop and Resize Image Using the Mask
    cropped_img = crop_and_resize(seg_image, np.array(mask_pil), target_size=(480,480))

    # 4. TTA and Prediction
    tta_preds = []
    cropped_np = np.array(cropped_img)
    for tta in tta_transforms:
        augmented = tta(image=cropped_np)
        aug_image = augmented['image']  # shape [C, H, W]
        aug_image = aug_image.unsqueeze(0).to(device)
        # Convert gender to a tensor of shape [1, 1]
        gender_tensor = torch.tensor([gender], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = final_model(aug_image, gender_tensor)
        tta_preds.append(output.item())
    raw_pred = np.mean(tta_preds)
    # Calibrate the prediction using the linear regression model
    calibrated_pred = lin_reg.predict(np.array(raw_pred).reshape(-1, 1))[0]
    return calibrated_pred

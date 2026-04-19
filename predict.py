import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# =========================
# 1. class names
# =========================
class_names = [
    '三杯雞', '什錦炒麵', '咖哩雞', '塔香海茸', '大陸妹', '客家小炒', '小番茄', '有機小松菜',
    '有機青松菜', '木瓜', '柳丁', '棗子', '橘子', '沙茶肉片', '油菜', '洋蔥炒蛋',
    '滷蛋', '滷雞腿', '玉米炒蛋', '瓜仔肉', '番茄炒蛋', '白米飯', '白菜滷', '福山萵苣',
    '空心菜', '糖醋雞丁', '紅蘿蔔炒蛋', '義大利麵', '芥藍菜', '菠菜', '葡萄', '蒜泥白肉',
    '蒸蛋', '蓮霧', '螞蟻上樹', '西瓜', '豆芽菜', '關東煮', '青江菜', '香蕉',
    '香酥魚排', '馬鈴薯燉肉', '高麗菜', '鳳梨', '鵝白菜', '鹽酥雞', '麥克雞塊',
    '麻婆豆腐', '麻油雞', '黑胡椒豬柳'
]

# =========================
# 2. device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 3. transform
# =========================
img_size = 224

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# 4. build model
# =========================
def build_model(num_classes):
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features

    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

# =========================
# 5. load model
# =========================
def load_model(weight_path):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weights not found: {weight_path}")

    model = build_model(len(class_names))
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# =========================
# 6. predict one image
# =========================
def predict_image(model, image_path, topk=5):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        topk = min(topk, len(class_names))
        top_probs, top_indices = torch.topk(probs, topk, dim=1)

    top_probs = top_probs[0].cpu().numpy()
    top_indices = top_indices[0].cpu().numpy()

    print(f"\nImage: {image_path}")
    print("\nTop predictions:")
    for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), start=1):
        print(f"{rank}. {class_names[idx]}: {prob * 100:.2f}%")

    print(f"\nPredicted class: {class_names[top_indices[0]]}")
    print(f"Confidence: {top_probs[0] * 100:.2f}%")

# =========================
# 7. main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a food image using ConvNeXt-Tiny")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="best_model.pth", help="Path to model weights")
    parser.add_argument("--topk", type=int, default=5, help="Number of top predictions to show")
    args = parser.parse_args()

    model = load_model(args.weights)
    predict_image(model, args.image, topk=args.topk)
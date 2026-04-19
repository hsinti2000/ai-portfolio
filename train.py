# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# %%
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

base_dirtrain = 'D:/ai/train'#50 class ,1 class about 500 pic  
base_dirtest = 'D:/ai/test' #50 class ,1  class about 100 pic
img_size = 256
batchsize = 32

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# train dataset
train_dataset = datasets.ImageFolder(
    root=base_dirtrain,
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root=base_dirtest,
    transform=transform
)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)

# class names
class_names = train_dataset.classes

print(class_names)
print("train:", len(train_dataset))
print("val:", len(val_dataset))

# %%
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Microsoft JhengHei'  # 微軟正黑體
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(20, 20))
images, labels = next(iter(train_loader))
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
for i in range(32):
    ax = plt.subplot(8, 8, i + 1)
    img = images[i].permute(1,2,0).numpy() # [H, W, C]
    plt.imshow(img)
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()

# %%
img_size = 256
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(12),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.03
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# %%
import numpy as np
from torch.utils.data import WeightedRandomSampler
# 取得每張訓練圖片的類別標籤
train_labels = [label for _, label in train_dataset.samples]

# 統計每個類別的樣本數
class_counts = np.bincount(train_labels)

print("Class counts:", class_counts)

# 每個類別的權重 = 1 / 該類別數量
class_weights = 1.0 / class_counts

# 每張圖片的抽樣權重
sample_weights = [class_weights[label] for label in train_labels]

# 轉成 tensor
sample_weights = torch.DoubleTensor(sample_weights)

# 建立 sampler
train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# %%
train_loader = DataLoader(
    train_dataset,
    batch_size=batchsize,
    sampler=train_sampler,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batchsize,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
train_dataset = datasets.ImageFolder(
    root=base_dirtrain,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=base_dirtest,
    transform=val_transform
)

# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = len(class_names)

# =========================
# Training stages
# =========================
stage1_epochs = 5
total_epochs = 30

# =========================
# 1. Model
# =========================
model = models.convnext_tiny(weights="IMAGENET1K_V1")
in_features = model.classifier[2].in_features

model.classifier = nn.Sequential(
    model.classifier[0],
    model.classifier[1],
    nn.Dropout(p=0.2),
    nn.Linear(in_features, num_classes)
)

for param in model.features.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(device)

# =========================
# 2. Loss / Optimizer / Scheduler
# =========================
criterion = nn.CrossEntropyLoss()

# Stage 1 optimizer / scheduler
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=stage1_epochs,
    eta_min=1e-6
)

# =========================
# 3. Training settings
# =========================
patience = 5
checkpoint_filepath = "tmp/best_model.pth"
os.makedirs("tmp", exist_ok=True)

best_val_loss = float("inf")
early_stop_counter = 0

history = {
    "train_loss": [],
    "train_acc": [],
    "train_top5": [],
    "val_loss": [],
    "val_acc": [],
    "val_top5": []
}

# =========================
# 4. Metrics
# =========================
def correct_count(outputs, labels):
    preds = outputs.argmax(dim=1)
    return (preds == labels).sum().item()

def top5_correct_count(outputs, labels):
    k = min(5, outputs.size(1))
    _, topk = outputs.topk(k, dim=1)
    correct = topk.eq(labels.view(-1, 1))
    return correct.any(dim=1).sum().item()

# =========================
# 5. Train loop
# =========================
for epoch in range(total_epochs):

    
    if epoch == stage1_epochs:
        print("=== Stage 2: unfreeze last two feature blocks ===")

        for param in model.features[-2].parameters():
            param.requires_grad = True

        for param in model.features[-1].parameters():
            param.requires_grad = True

        
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-5,
            weight_decay=1e-4
        )

        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - stage1_epochs,
            eta_min=1e-6
        )

    model.train()
    running_loss = 0.0
    train_correct = 0
    train_top5_correct = 0
    train_total = 0
    train_batches = 0
    skipped_train_batches = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        # 檢查輸入
        if torch.isnan(images).any() or torch.isinf(images).any():
            print(f"[WARNING] Bad images at epoch {epoch+1}, batch {batch_idx}, skipped")
            skipped_train_batches += 1
            continue

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print(f"[WARNING] Bad outputs at epoch {epoch+1}, batch {batch_idx}, skipped")
            skipped_train_batches += 1
            continue

        loss = criterion(outputs, labels)

        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARNING] Bad loss at epoch {epoch+1}, batch {batch_idx}, skipped")
            print("labels min/max:", labels.min().item(), labels.max().item())
            skipped_train_batches += 1
            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        train_correct += correct_count(outputs, labels)
        train_top5_correct += top5_correct_count(outputs, labels)
        train_total += labels.size(0)
        train_batches += 1

    train_loss = running_loss / train_batches if train_batches > 0 else float("nan")
    train_acc = train_correct / train_total if train_total > 0 else 0.0
    train_top5 = train_top5_correct / train_total if train_total > 0 else 0.0

    # =========================
    # 6. Validation loop
    # =========================
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_top5_correct = 0
    val_total = 0
    val_batches = 0
    skipped_val_batches = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            if torch.isnan(images).any() or torch.isinf(images).any():
                print(f"[WARNING] Bad val images at epoch {epoch+1}, batch {batch_idx}, skipped")
                skipped_val_batches += 1
                continue

            outputs = model(images)

            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"[WARNING] Bad val outputs at epoch {epoch+1}, batch {batch_idx}, skipped")
                skipped_val_batches += 1
                continue

            loss = criterion(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARNING] Bad val loss at epoch {epoch+1}, batch {batch_idx}, skipped")
                skipped_val_batches += 1
                continue

            val_running_loss += loss.item()
            val_correct += correct_count(outputs, labels)
            val_top5_correct += top5_correct_count(outputs, labels)
            val_total += labels.size(0)
            val_batches += 1

    val_loss = val_running_loss / val_batches if val_batches > 0 else float("nan")
    val_acc = val_correct / val_total if val_total > 0 else 0.0
    val_top5 = val_top5_correct / val_total if val_total > 0 else 0.0

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["train_top5"].append(train_top5)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_top5"].append(val_top5)

    print(
        f"Epoch [{epoch+1}/{total_epochs}] | "
        f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | train_top5: {train_top5:.4f} | "
        f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | val_top5: {val_top5:.4f}"
    )
    print(f"Skipped train batches: {skipped_train_batches} | Skipped val batches: {skipped_val_batches}")

    if not torch.isnan(torch.tensor(val_loss)) and val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), checkpoint_filepath)
        print(f"Saved best model to {checkpoint_filepath}")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"EarlyStopping counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    scheduler.step()

# =========================
# 7. Load best model
# =========================
state_dict = torch.load(checkpoint_filepath, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print("Loaded best model weights.")

# %%
epochs_range = range(1, len(history["train_loss"]) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history["train_loss"], label="train_loss")
plt.plot(epochs_range, history["val_loss"], label="val_loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# %%
plt.figure()

plt.plot(history["train_acc"], label="Train")
plt.plot(history["val_acc"], label="Validation")

plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(loc="upper left")

plt.show()

# %%
#confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 1. 重新建立和訓練時完全一樣的模型
# =========================
model = models.convnext_tiny(weights="IMAGENET1K_V1")
in_features = model.classifier[2].in_features

model.classifier = nn.Sequential(
    model.classifier[0],
    model.classifier[1],
    nn.Dropout(p=0.2),   # 要跟你訓練時一致
    nn.Linear(in_features, num_classes)
)

# =========================
# 2. 載入 checkpoint
# =========================
state_dict = torch.load("tmp/best_model.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

# =========================
# 3. 跑完整個 val_loader / test_loader
# =========================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:   # 如果你有 test_loader，也可以改成 test_loader
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# =========================
# 4. Confusion Matrix
# =========================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# =========================
# 5. Normalized Confusion Matrix
# =========================
cm_norm = confusion_matrix(all_labels, all_preds, normalize="true")

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.show()

# =========================
# 6. Classification Report
# =========================
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# %%
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 重新建立和訓練時完全一樣的模型
model = models.convnext_tiny(weights="IMAGENET1K_V1")
in_features = model.classifier[2].in_features

model.classifier = nn.Sequential(
    model.classifier[0],
    model.classifier[1],
    nn.Dropout(p=0.5),
    nn.Linear(in_features, num_classes)
)

# 載入正確的 checkpoint
state_dict = torch.load("tmp/best_model.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

lab_rand = class_names[np.random.randint(0, len(class_names))]
testdir = os.path.join(base_dirtest, lab_rand)

image_files = [
    f for f in os.listdir(testdir)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
]

testimg = os.path.join(
    testdir,
    image_files[np.random.randint(0, len(image_files))]
)

img = Image.open(testimg).convert("RGB")
x = val_transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(x)
    probs = F.softmax(outputs, dim=1)[0]

top_probs, top_inds = torch.topk(probs, 5)
pred_class = class_names[top_inds[0].item()]

print(f"True class: {lab_rand}")
print(f"Predicted class: {pred_class}")
print("Top-5 predictions:")
for p, i in zip(top_probs, top_inds):
    print(f"{p.item():.3f}  {class_names[i.item()]}")

plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.title(f"True: {lab_rand}\nPred: {pred_class}")
plt.axis("off")
plt.show()

# %%
torch.save(model.state_dict(), "best_model.pth")





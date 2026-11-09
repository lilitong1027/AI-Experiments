import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
# 1. 超参数设置 (先用较少 epoch 跑通)
# ==========================================
BATCH_SIZE = 128
EPOCHS = 80
LEARNING_RATE = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"当前使用的设备: {DEVICE}")

# ==========================================
# 2. 数据准备与划分
# ==========================================
print("开始准备数据...")
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 下载并加载完整训练集
full_trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=val_test_transform
)

# 分层抽样划分 80% 训练，20% 验证
targets = np.array(full_trainset.targets)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, val_idx in sss.split(np.zeros(len(targets)), targets):
    train_subset = Subset(full_trainset, train_idx)
    val_subset = Subset(full_trainset, val_idx)

# 训练集应用数据增强的 transform
train_subset.dataset.transform = train_transform

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=val_test_transform
)


train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(testset,      batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("数据加载完毕！")

# ==========================================
# 3. 定义 CNN 模型
# ==========================================
class CIFARNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn4   = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn5   = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn6   = nn.BatchNorm2d(256)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, 2)

        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

model = CIFARNet().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1) # 注意这里 step_size 改为了 15，配合较小的 EPOCHS

# ==========================================
# 4. 训练与验证
# ==========================================
def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total

best_val_acc = 0.0
patience, patience_counter = 15, 0
train_losses, val_losses, train_accs, val_accs = [], [], [], []

print("开始训练...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, running_correct, seen = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        running_correct += pred.eq(labels).sum().item()
        seen += images.size(0)

    train_loss = running_loss / seen
    train_acc = running_correct / seen
    val_loss, val_acc = evaluate(model, val_loader)

    train_losses.append(train_loss); val_losses.append(val_loss)
    train_accs.append(train_acc);   val_accs.append(val_acc)

    print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"早停触发！停止于 Epoch {epoch}")
            break

    scheduler.step()

# ==========================================
# 5. 测试与可视化
# ==========================================
print("开始测试最佳模型...")
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
overall_acc = (all_preds == all_labels).mean()
print(f"=== 最终测试集准确率: {overall_acc*100:.2f}% ===")

# 绘制并保存 Loss/Acc 曲线
epochs_range = range(1, len(train_losses) + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Train")
plt.plot(epochs_range, val_losses, label="Validation")
plt.title("Loss Curve"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accs, label="Train")
plt.plot(epochs_range, val_accs, label="Validation")
plt.title("Accuracy Curve"); plt.legend()
plt.savefig("training_curves.png", dpi=150)
print("训练曲线已保存为 training_curves.png")

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=testset.classes, yticklabels=testset.classes)
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Normalized Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=150)
print("混淆矩阵已保存为 confusion_matrix.png")

# 分类报告
print("\n=== 各类别分类报告 ===")
print(classification_report(all_labels, all_preds, target_names=testset.classes, digits=4))

# 误分类样本可视化
wrong_idx = np.where(all_preds != all_labels)[0]
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for ax, idx in zip(axes.flat, wrong_idx[:12]):
    img = testset.data[idx]
    ax.imshow(img)
    ax.set_title(f"True: {testset.classes[all_labels[idx]]}\nPred: {testset.classes[all_preds[idx]]}", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("misclassified.png", dpi=150)
print("误分类样本图已保存为 misclassified.png")
print("全部流程执行完毕！")
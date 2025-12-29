import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# =========================
# 基本配置
# =========================
DATA_DIR = r"E:\traffic_light_recognition\data" #数据路径
BATCH_SIZE = 32 #一次训练数量
NUM_EPOCHS = 8 #训练轮数
LEARNING_RATE = 1e-4 #学习率

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 数据预处理
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),    #ResNet18 要求输入尺寸固定
    transforms.RandomHorizontalFlip(),   #数据增强，随机水平翻转
    transforms.ToTensor(),   #把图片转换成 PyTorch 张量
    transforms.Normalize(     #归一化到模型预训练的分布
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(    #根据文件夹名称自动分配标签，go-0,stop-1
    root=os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_DIR, "val"),
    transform=val_transform
)

train_loader = DataLoader(      #提供批量数据迭代
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True      #训练时打乱顺序
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("类别映射：", train_dataset.class_to_idx)

# =========================
# 加载 ResNet18（预训练）
# =========================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  #使用 预训练的 ResNet18
model.fc = nn.Linear(model.fc.in_features, 2)   #修改 fc 全连接层输出为 2 类（go / stop）表示模型对每个类别的预测得分
model = model.to(DEVICE)   #移动模型到 GPU

# =========================
# 损失函数 & 优化器
# =========================
criterion = nn.CrossEntropyLoss()   #分类任务常用损失函数
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)   #自适应学习率优化器

train_losses = []   #列表
val_losses = []
train_accs = []
val_accs = []

# =========================
# 训练 & 验证
# =========================
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")

    # ---- 训练 ----
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()    #清空梯度
        outputs = model(images)
        loss = criterion(outputs, labels)   #criterion(...)：计算 预测 vs 真实标签 的差距，返回一个 标量 loss
        loss.backward()     #反向传播
        optimizer.step()   #更新参数

        train_loss += loss.item() * images.size(0)  #将每个 batch 的损失乘以样本数，累加得到总损失，最后计算整个 epoch 的平均损失。
        _, predicted = torch.max(outputs, 1)  #在 类别维度 上找最大值，0是go,1是stop
        total += labels.size(0)   #total = 当前 epoch 一共看了多少张图
        correct += (predicted == labels).sum().item()  #统计预测正确的数量

    train_loss /= total #每张图像的平均 loss
    train_acc = correct / total   #训练准确率
    print(f"训练集 Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

    # ---- 验证 ----
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():   #不计算梯度，节省显存
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= total
    val_acc = correct / total
    print(f"验证集 Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # ---- 现在再记录 ----
    train_losses.append(train_loss)  #用列表记录每个 epoch 的训练/验证损失和准确率
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

# =========================
# 保存模型
# =========================
torch.save(model.state_dict(), "resnet18_go_stop.pth")  #保存模型参数到文件，后续可以直接加载进行预测
print("\n模型已保存为 resnet18_go_stop.pth")

# =========================
# 画曲线
# =========================
epochs = range(1, NUM_EPOCHS + 1)

# Loss 曲线
plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()

# Accuracy 曲线
plt.figure()
plt.plot(epochs, train_accs, label="Train Accuracy")
plt.plot(epochs, val_accs, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_curve.png")
plt.show()

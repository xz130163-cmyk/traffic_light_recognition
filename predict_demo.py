import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import os
import random

# ====== 路径配置 ======
MODEL_PATH = "resnet18_go_stop.pth"
DATA_DIR = r"E:\traffic_light_recognition\data\val"

# ====== 类别映射（和训练时一致） ======
class_names = ["go", "stop"]

# ====== 设备 ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 模型加载 ======
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ====== 图像预处理 ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ====== 随机选一张图片 ======
label = random.choice(class_names)
img_name = random.choice(os.listdir(os.path.join(DATA_DIR, label)))
img_path = os.path.join(DATA_DIR, label, img_name)

img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

# ====== 推理 ======
with torch.no_grad():
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1).item()

pred_label = class_names[pred]

# ====== 显示结果 ======
plt.imshow(img)
plt.title(f"Prediction: {pred_label}")
plt.axis("off")
plt.show()

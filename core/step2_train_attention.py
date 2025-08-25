import os
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import umap  # 新增：聚类可视化
from sklearn.decomposition import PCA  # 新增：聚类可视化

# -------- 超参数 --------
IMG_SIZE = 460
BS = 32
VALID_PCT = 0.2
SEED = 33
EPOCHS = 10
LR = 1e-4
N_WORKERS = 4
ALPHA = 0.2  # Mixup参数

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------- 路径 --------
npy_path = Path('./npy_specs_center')
MODEL_SAVE_PATH = Path('./model/best_attention_model.pth')
report_dir = Path('./report/report_attention_model')
report_dir.mkdir(parents=True, exist_ok=True)

# -------- 设置随机种子 --------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# -------- 标签函数 --------
def label_func(p: Path):
    return p.parent.name

# -------- 数据增强函数 --------
def freq_mask(x, F=20):
    _, H, W = x.shape
    f = random.randint(0, F)
    f0 = random.randint(0, H - f)
    x_masked = x.clone()
    x_masked[:, f0:f0+f, :] = 0
    return x_masked

def time_mask(x, T=20):
    _, H, W = x.shape
    t = random.randint(0, T)
    t0 = random.randint(0, W - t)
    x_masked = x.clone()
    x_masked[:, :, t0:t0+t] = 0
    return x_masked

# -------- 读取npy文件函数 --------
def load_npy(path):
    arr = np.load(path)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)  # (1,H,W)
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 5, axis=0)  # 复制单通道到5通道
    elif arr.shape[0] == 3:
        arr = np.concatenate([arr, arr[:2, :, :]], axis=0)  # 3+2=5
    if arr.shape[0] != 5:
        raise ValueError(f"通道数错误: {arr.shape[0]}")
    return torch.tensor(arr, dtype=torch.float32)

# -------- 划分数据集函数 --------
def balanced_train_valid_split(items, valid_pct=VALID_PCT, seed=SEED, get_label=label_func):
    random.seed(seed)
    class_items = defaultdict(list)
    for item in items:
        class_items[get_label(item)].append(item)

    train_items = []
    valid_items = []
    for cls in classes:
        cls_files = class_items[cls]
        random.shuffle(cls_files)
        cls_valid_size = max(1, int(len(cls_files) * valid_pct))
        valid_items.extend(cls_files[:cls_valid_size])
        train_items.extend(cls_files[cls_valid_size:])
    return train_items, valid_items

# -------- Dataset定义 --------
class NPYSpectrumDatasetWithMemory(Dataset):
    def __init__(self, files, classes, label_func, transform=None):
        self.files = files
        self.classes = classes
        self.label_func = label_func
        self.class2idx = {c:i for i,c in enumerate(classes)}
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        x = load_npy(f)
        if self.transform:
            x = self.transform(x)
        y_label = self.label_func(f)
        y = self.class2idx[y_label]
        return x, y

# -------- 获取数据和类别 --------
all_files = list(npy_path.rglob('*.npy'))
if not all_files:
    raise FileNotFoundError("未找到.npy文件")

all_labels = [label_func(f) for f in all_files]
classes = sorted(list(set(all_labels)))
print(f"检测到 {len(classes)} 个类别: {classes}")

# -------- 划分数据集 --------
train_files, valid_files = balanced_train_valid_split(all_files)
print(f"训练样本: {len(train_files)}, 验证样本: {len(valid_files)}")

# -------- 数据增强 --------
def train_transform(x):
    return time_mask(freq_mask(x, F=30), T=30)

# -------- 创建DataLoader --------
train_ds = NPYSpectrumDatasetWithMemory(train_files, classes, label_func, transform=train_transform)
valid_ds = NPYSpectrumDatasetWithMemory(valid_files, classes, label_func, transform=None)

train_dl = DataLoader(train_ds, BS, shuffle=True, num_workers=N_WORKERS, pin_memory=True)
valid_dl = DataLoader(valid_ds, BS, shuffle=False, num_workers=N_WORKERS, pin_memory=True)

# -------- 模型架构 --------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction),
            nn.ReLU(),
            nn.Linear(channel//reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AttentionFusion(nn.Module):
    def __init__(self, in_ch1, in_ch2, fused_ch, reduction=16):
        super().__init__()
        self.se1 = SEBlock(in_ch1, reduction)
        self.se2 = SEBlock(in_ch2, reduction)
        self.conv1 = ConvBlock(in_ch1, fused_ch, 1, padding=0)
        self.conv2 = ConvBlock(in_ch2, fused_ch, 1, padding=0)
        self.se_fusion = SEBlock(fused_ch*2, reduction)
        self.final_conv = ConvBlock(fused_ch*2, fused_ch, 1, padding=0)

    def forward(self, x1, x2):
        x1 = self.se1(x1)
        x2 = self.se2(x2)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        combined = torch.cat([x1, x2], dim=1)
        combined = self.se_fusion(combined)
        return self.final_conv(combined)

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 分支1: 3通道输入
        self.branch1 = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2)
        )
        # 分支2: 2通道输入
        self.branch2 = nn.Sequential(
            ConvBlock(2, 32),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2)
        )
        # 融合模块
        self.fusion = AttentionFusion(128, 128, 256)
        # 公共主干
        self.common = nn.Sequential(
            ConvBlock(256, 512),
            nn.MaxPool2d(2),
            ConvBlock(512, 512),
            nn.MaxPool2d(2)
        )
        # 分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x1 = x[:, :3]
        x2 = x[:, 3:5]
        f1 = self.branch1(x1)
        f2 = self.branch2(x2)
        fused = self.fusion(f1, f2)
        x = self.common(fused)
        x = self.global_pool(x).flatten(1)
        return self.classifier(x)

model = CustomCNN(len(classes)).to(device)
print(f"模型参数总量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# -------- 训练配置 --------
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------- Mixup函数 --------
def mixup_data(x, y, alpha=1.0, device=device):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1-lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, output, y_a, y_b, lam):
    return lam * criterion(output, y_a).mean() + (1-lam) * criterion(output, y_b).mean()

# -------- 训练/验证函数 --------
def train_epoch(model, dataloader, criterion, optimizer, alpha=ALPHA):
    model.train()
    running_loss = 0
    preds, targets = [], []
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        mixed_x, y_a, y_b, lam = mixup_data(xb, yb, alpha)

        optimizer.zero_grad()
        output = model(mixed_x)
        loss = mixup_criterion(criterion, output, y_a, y_b, lam)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            orig_output = model(xb)
            preds.extend(torch.argmax(orig_output, 1).cpu().numpy())
            targets.extend(yb.cpu().numpy())
            running_loss += loss.item() * xb.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(targets, preds)
    report = classification_report(targets, preds, target_names=classes, zero_division=0)
    return epoch_loss, report, acc

def valid_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            output = model(xb)
            loss = criterion(output, yb).mean()
            running_loss += loss.item() * xb.size(0)

            preds.extend(torch.argmax(output, 1).cpu().numpy())
            targets.extend(yb.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(targets, preds)
    report = classification_report(targets, preds, target_names=classes, zero_division=0)
    return epoch_loss, report, acc

# -------- 训练主循环 --------
best_valid_acc = 0.0
train_losses, valid_losses = [], []
train_accs, valid_accs = [], []  # 新增：记录准确率

for epoch in range(1, EPOCHS+1):
    train_loss, train_report, train_acc = train_epoch(model, train_dl, criterion, optimizer)
    valid_loss, valid_report, valid_acc = valid_epoch(model, valid_dl, criterion)

    # 记录指标
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)

    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
    print(f"Valid: Loss={valid_loss:.4f}, Acc={valid_acc:.4f}")
    print(f"分类报告:\n{valid_report}\n{'-'*60}")

    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"✅ 保存最佳模型，验证准确率: {best_valid_acc:.4f}")

# -------- 绘制准确率曲线 --------
def plot_accuracy_curve(train_accs, valid_accs, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS+1), train_accs, 'o-', label='Training Accuracy')
    plt.plot(range(1, EPOCHS+1), valid_accs, 's-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

plot_accuracy_curve(train_accs, valid_accs, report_dir / 'accuracy_curve.png')

# -------- 评估指标可视化 --------
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

all_targets = []
all_probs = []
all_preds = []
all_features = []  # 新增：特征提取

# 注册特征提取Hook
feature_hook = None
def hook_fn(module, input, output):
    global feature_hook
    feature_hook = output.detach().cpu().numpy()

hook = model.common[-2].register_forward_hook(hook_fn)  # 在最后一个卷积层后提取特征

with torch.no_grad():
    for xb, yb in valid_dl:
        xb = xb.to(device)
        output = model(xb)

        # 保存特征
        if feature_hook is not None:
            features = feature_hook.reshape(xb.size(0), -1)  # 展平特征
            all_features.append(features)

        probs = torch.softmax(output, 1).cpu().numpy()
        preds = np.argmax(probs, 1)
        all_probs.append(probs)
        all_preds.extend(preds)
        all_targets.extend(yb.numpy())

hook.remove()
all_features = np.vstack(all_features) if all_features else None
all_probs = np.vstack(all_probs)
all_targets = np.array(all_targets)
all_preds = np.array(all_preds)

# -------- 绘制聚类图 --------
# -------- 绘制聚类图 --------
def plot_clustering(features, labels, classes, save_path, method='umap'):
    if features is None:
        print("警告: 未提取到特征，无法绘制聚类图")
        return

    plt.figure(figsize=(12, 10))
    if method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=SEED)
    else:
        reducer = PCA(n_components=2, random_state=SEED)

    reduced = reducer.fit_transform(features)
    cmap = plt.cm.get_cmap('tab10', len(classes))

    for cls_idx, cls_name in enumerate(classes):
        mask = (labels == cls_idx)
        plt.scatter(reduced[mask, 0], reduced[mask, 1], c=cmap(cls_idx), label=cls_name, alpha=0.8)

    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'{method.upper()} Clustering of Features')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# 绘制4分类聚类图（使用不同形状和颜色）
def plot_4class_clustering(features, labels, classes, save_path, method='umap'):
    if features is None:
        print("警告: 未提取到特征，无法绘制聚类图")
        return

    plt.figure(figsize=(12, 10))

    if method.lower() == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=SEED)
        method_name = 'UMAP'
    else:
        reducer = PCA(n_components=2, random_state=SEED)
        method_name = 'PCA'

    reduced = reducer.fit_transform(features)

    # 确保类别数量为4
    if len(classes) != 4:
        print(f"警告: 预期4个类别，但实际有{len(classes)}个类别")

    # 自定义颜色映射（适合4分类）
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33F6']  # 更鲜明的颜色
    markers = ['o', 's', '^', 'D']  # 不同形状增强区分度

    # 绘制每个类别
    for cls_idx, cls_name in enumerate(classes):
        mask = (labels == cls_idx)
        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            c=colors[cls_idx],
            label=cls_name,
            alpha=0.8,
            s=60,  # 增大点大小
            marker=markers[cls_idx]  # 不同类别使用不同形状
        )

    # 添加中心点标记
    for cls_idx in range(len(classes)):
        mask = (labels == cls_idx)
        center_x = np.mean(reduced[mask, 0])
        center_y = np.mean(reduced[mask, 1])
        plt.scatter(
            center_x,
            center_y,
            c='black',
            s=150,
            alpha=0.7,
            marker='*',  # 星形标记中心点
            edgecolors='white',
            linewidth=1.5
        )

    # 美化图表
    plt.title(f'{method_name} Clustering for {len(classes)}-Class Problem', fontsize=14)
    plt.xlabel(f'{method_name} Component 1', fontsize=12)
    plt.ylabel(f'{method_name} Component 2', fontsize=12)
    plt.legend(loc='best', fontsize=12, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

# -------- 多分类ROC曲线 --------
def plot_multiclass_roc(y_true, y_prob, classes, save_path):
    plt.figure(figsize=(10, 8))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(classes)

    # One-hot编码
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{classes[i]} (AUC = {roc_auc[i]:.3f})')

    # 计算macro avg ROC曲线
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc["macro"] = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color='navy', linestyle=':', linewidth=4,
             label=f'Macro Avg (AUC = {roc_auc["macro"]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(save_path)
    plt.close()

# -------- 多分类PR曲线 --------
def plot_multiclass_pr(y_true, y_prob, classes, save_path):
    plt.figure(figsize=(10, 8))
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = len(classes)

    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        average_precision[i] = auc(recall[i], precision[i])
        plt.plot(recall[i], precision[i], lw=2,
                 label=f'{classes[i]} (AUC = {average_precision[i]:.3f})')

    # 计算macro avg PR曲线
    all_recall = np.unique(np.concatenate([recall[i] for i in range(n_classes)]))
    mean_precision = np.zeros_like(all_recall)
    for i in range(n_classes):
        mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])
    mean_precision /= n_classes
    average_precision["macro"] = auc(all_recall, mean_precision)
    plt.plot(all_recall, mean_precision, color='navy', linestyle=':', linewidth=4,
             label=f'Macro Avg (AUC = {average_precision["macro"]:.3f})')

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-class Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid()
    plt.savefig(save_path)
    plt.close()

# -------- 画混淆矩阵 --------
def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

# -------- 保存所有报告图片 --------
plot_4class_clustering(all_features, all_targets, classes, report_dir / '4class_clustering.png')
plot_multiclass_roc(all_targets, all_probs, classes, report_dir / 'roc_curve.png')
plot_multiclass_pr(all_targets, all_probs, classes, report_dir / 'pr_curve.png')
plot_confusion_matrix(all_targets, all_preds, classes, report_dir / 'confusion_matrix.png')

# 打印最终报告
print(f"所有报告图片已保存至 {report_dir.resolve()}")
print(f"最佳验证准确率: {best_valid_acc:.4f}")
print(f"最终分类报告:\n{classification_report(all_targets, all_preds, target_names=classes, zero_division=0)}")
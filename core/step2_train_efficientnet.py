import os
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from collections import defaultdict
import umap  # 新增：用于聚类可视化
from sklearn.decomposition import PCA  # 新增：用于聚类可视化

# -------- 超参数 --------
IMG_SIZE = 460
BS = 32
VALID_PCT = 0.2
SEED = 33
EPOCHS = 2
LR = 1e-4
N_WORKERS = 4
# Mixup 参数
ALPHA = 0.2  # Mixup的alpha参数，通常在0.1到0.4之间

EFFICIENTNET_MODEL_NAME = 'tf_efficientnet_b0.ns_jft_in1k'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------- 路径 --------
npy_path = Path('./npy_specs_center')
MODEL_SAVE_PATH = Path('./model/best_efficientnet_model.pth')
report_dir = Path('./report/report_efficientnet_model')  # 新增：报告目录
report_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在

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
# 频率遮蔽
def freq_mask(x, F=20):
    _, H, W = x.shape
    f = random.randint(0, F)
    f0 = random.randint(0, H - f)
    x_masked = x.clone()
    x_masked[:, f0:f0+f, :] = 0
    return x_masked

# 时间遮蔽
def time_mask(x, T=20):
    _, H, W = x.shape
    t = random.randint(0, T)
    t0 = random.randint(0, W - t)
    x_masked = x.clone()
    x_masked[:, :, t0:t0+t] = 0
    return x_masked

# -------- 读取npy --------
def load_npy(path):
    arr = np.load(path)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)  # (1,H,W)
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 5, axis=0)  # 复制单通道到5通道
    elif arr.shape[0] == 3:
        arr = np.concatenate([arr, arr[:2, :, :]], axis=0)  # 3+2=5
    # 确保通道数是5
    if arr.shape[0] != 5:
        raise ValueError(f"Expected 5 channels, but got {arr.shape[0]} for file {path}")
    return torch.tensor(arr, dtype=torch.float32)

# -------- 划分函数 --------
def balanced_train_valid_split(items, valid_pct=VALID_PCT, seed=SEED, get_label=label_func):
    random.seed(seed)
    # 使用 defaultdict 来按类别分组文件
    class_items = defaultdict(list)
    for item in items:
        class_items[get_label(item)].append(item)

    train_items = []
    valid_items = []

    for cls in classes:  # 确保每个类别都被处理
        cls_files = class_items[cls]
        random.shuffle(cls_files)  # 打乱每个类别的文件
        cls_valid_size = max(1, int(len(cls_files) * valid_pct))  # 确保每个类别至少有一个验证样本

        valid_items.extend(cls_files[:cls_valid_size])
        train_items.extend(cls_files[cls_valid_size:])

    return train_items, valid_items

# -------- Dataset定义 --------
class NPYSpectrumDatasetWithMemory(Dataset):
    def __init__(self, files, classes, label_func, transform=None):
        self.files = files
        self.classes = classes
        self.label_func = label_func
        self.class2idx = {c: i for i, c in enumerate(classes)}
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

# -------- 获取所有文件和类别 --------
all_files = list(npy_path.rglob('*.npy'))
if not all_files:
    raise FileNotFoundError(f"No .npy files found in {npy_path}. Please check the path and ensure data is preprocessed.")

all_labels = [label_func(f) for f in all_files]
classes = sorted(list(set(all_labels)))  # 使用 list(set(...)) 确保唯一性并排序
print(f"Found {len(classes)} classes: {classes}")

# -------- 划分训练验证 --------
train_files, valid_files = balanced_train_valid_split(all_files, valid_pct=VALID_PCT, seed=SEED, get_label=label_func)
print(f"Training samples: {len(train_files)}, Validation samples: {len(valid_files)}")

# -------- 定义训练集数据增强组合 --------
def train_transform(x):
    x = freq_mask(x, F=30)
    x = time_mask(x, T=30)
    return x

# -------- 创建Dataset和DataLoader --------
train_ds = NPYSpectrumDatasetWithMemory(train_files, classes, label_func, transform=train_transform)
valid_ds = NPYSpectrumDatasetWithMemory(valid_files, classes, label_func, transform=None)

train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=N_WORKERS, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size=BS, shuffle=False, num_workers=N_WORKERS, pin_memory=True)

# -------- 构建模型 --------
class EfficientNetClassifier(nn.Module):
    def __init__(self, model_name, num_classes, in_channels=5):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=in_channels)

        # 替换分类头
        if hasattr(self.backbone, 'classifier'):
            num_ftrs = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(num_ftrs, num_classes)
        elif hasattr(self.backbone, 'fc'):
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        elif hasattr(self.backbone, 'head') and hasattr(self.backbone.head, 'fc'):
            num_ftrs = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Linear(num_ftrs, num_classes)
        else:
            raise AttributeError("Could not find a standard classifier/fc layer in the EfficientNet backbone.")

    def forward(self, x):
        return self.backbone(x)

model = EfficientNetClassifier(EFFICIENTNET_MODEL_NAME, num_classes=len(classes), in_channels=5).to(device)

# -------- 损失和优化器 --------
criterion = nn.CrossEntropyLoss(reduction='none')  # 用于Mixup
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------- Mixup 辅助函数 --------
def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, outputs, y_a, y_b, lam):
    return lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

# -------- 训练函数 --------
def train_epoch(model, dataloader, criterion, optimizer, device, alpha=ALPHA):
    model.train()
    running_loss = 0
    all_preds = []
    all_targets = []

    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)

        # 应用 Mixup
        mixed_xb, y_a, y_b, lam = mixup_data(xb, yb, alpha, device)

        optimizer.zero_grad()
        outputs = model(mixed_xb)

        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # 使用原始数据获取预测，以便计算准确率
            original_outputs = model(xb)
            preds = torch.argmax(original_outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(yb.cpu().numpy())

        running_loss += loss.item() * xb.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    report = classification_report(all_targets, all_preds, target_names=classes, zero_division=0)
    acc = accuracy_score(all_targets, all_preds)

    return epoch_loss, report, acc

# -------- 验证函数 --------
def valid_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb).mean()
            running_loss += loss.item() * xb.size(0)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(yb.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    report = classification_report(all_targets, all_preds, target_names=classes, zero_division=0)
    acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, report, acc

# -------- 解析macro F1分数 --------
def parse_macro_f1(report_str):
    lines = report_str.split('\n')
    for line in lines:
        if 'macro avg' in line:
            parts = line.split()
            f1_index = 4 if len(parts) > 4 else 3
            try:
                return float(parts[f1_index])
            except:
                return 0.0
    return 0.0

# -------- 画损失曲线 --------
def plot_loss_curve(train_losses, valid_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, EPOCHS+1), valid_losses, label='Valid Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# -------- 画准确率曲线 --------
def plot_accuracy_curve(train_accs, valid_accs, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS+1), train_accs, label='Train Accuracy', marker='o')
    plt.plot(range(1, EPOCHS+1), valid_accs, label='Valid Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# -------- 画F1曲线 --------
def plot_f1_curve(train_f1s, valid_f1s, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS+1), train_f1s, label='Train Macro F1', marker='o')
    plt.plot(range(1, EPOCHS+1), valid_f1s, label='Valid Macro F1', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1 Score')
    plt.title('Training and Validation Macro F1 Curve')
    plt.legend()
    plt.grid(True)
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
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')

    # 计算macro avg ROC曲线
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc["macro"] = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color='navy', linestyle=':', linewidth=4,
             label=f'macro-average ROC curve (area = {roc_auc["macro"]:.2f})')

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

    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        average_precision[i] = auc(recall[i], precision[i])
        plt.plot(recall[i], precision[i], lw=2,
                 label=f'PR curve of class {classes[i]} (area = {average_precision[i]:.2f})')

    # 计算macro avg PR曲线
    all_recall = np.unique(np.concatenate([recall[i] for i in range(n_classes)]))
    mean_precision = np.zeros_like(all_recall)
    for i in range(n_classes):
        mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])
    mean_precision /= n_classes
    average_precision["macro"] = auc(all_recall, mean_precision)
    plt.plot(all_recall, mean_precision, color='navy', linestyle=':', linewidth=4,
             label=f'macro-average PR curve (area = {average_precision["macro"]:.2f})')

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-class Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid()
    plt.savefig(save_path)
    plt.close()

# -------- 混淆矩阵 --------
def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -------- 提取验证集特征并绘制聚类图 --------
def plot_clustering(features, labels, classes, save_path, method='umap', n_components=2):
    """
    features: 特征矩阵 (n_samples, n_features)
    labels: 真实标签列表
    method: 'umap' 或 'pca'
    """
    plt.figure(figsize=(12, 10))

    if method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=SEED)
    else:
        reducer = PCA(n_components=n_components, random_state=SEED)

    reduced_features = reducer.fit_transform(features)

    # 颜色映射
    cmap = plt.cm.get_cmap('tab10', len(classes))

    for cls_idx, cls_name in enumerate(classes):
        mask = (labels == cls_idx)
        plt.scatter(
            reduced_features[mask, 0],
            reduced_features[mask, 1],
            c=cmap(cls_idx),
            label=cls_name,
            alpha=0.8,
            s=50
        )

    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'{method.upper()} Clustering of Validation Set Features')
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -------- 训练主循环 --------
train_losses, valid_losses = [], []
train_accs, valid_accs = [], []
train_f1s, valid_f1s = [], []
best_valid_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_report, train_acc = train_epoch(model, train_dl, criterion, optimizer, device, alpha=ALPHA)
    valid_loss, valid_report, valid_acc = valid_epoch(model, valid_dl, criterion, device)

    # 解析F1分数
    train_f1 = parse_macro_f1(train_report)
    valid_f1 = parse_macro_f1(valid_report)

    # 记录指标
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    train_f1s.append(train_f1)
    valid_f1s.append(valid_f1)

    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Macro F1: {train_f1:.4f}")
    print(f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f} | Valid Macro F1: {valid_f1:.4f}")
    print(f"Valid Classification Report:\n{valid_report}")

    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"**模型已保存! 新的最佳验证准确率: {best_valid_acc:.4f}**")
    else:
        print(f"当前最佳验证准确率: {best_valid_acc:.4f}")

    print('-' * 60)

# -------- 绘制训练指标曲线 --------
plot_loss_curve(train_losses, valid_losses, report_dir / 'loss_curve.png')
plot_accuracy_curve(train_accs, valid_accs, report_dir / 'accuracy_curve.png')
plot_f1_curve(train_f1s, valid_f1s, report_dir / 'f1_curve.png')

# -------- 用最佳模型预测验证集 --------
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

all_targets = []
all_features = []  # 存储特征向量
all_probs = []  # 预测概率，用于ROC/PR曲线
all_preds = []

# 提取特征的hook函数
feature_extractor = None

def get_feature_hook(module, input, output):
    global feature_extractor
    feature_extractor = output.detach()

# 注册hook到最后一个特征层
if hasattr(model.backbone, 'global_pool'):
    hook_handle = model.backbone.global_pool.register_forward_hook(get_feature_hook)
else:
    hook_handle = model.backbone.register_forward_hook(get_feature_hook)

with torch.no_grad():
    for xb, yb in valid_dl:
        xb = xb.to(device)
        outputs = model(xb)

        # 获取特征
        if feature_extractor is not None:
            batch_features = feature_extractor.cpu().numpy()
            # 展平特征
            if len(batch_features.shape) > 2:
                batch_features = batch_features.reshape(batch_features.shape[0], -1)
            all_features.append(batch_features)

        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_probs.append(probs)
        all_preds.extend(preds)
        all_targets.extend(yb.numpy())

# 释放hook
hook_handle.remove()

all_probs = np.vstack(all_probs)
all_targets = np.array(all_targets)
all_preds = np.array(all_preds)
all_features = np.vstack(all_features) if all_features else None

# -------- 绘制评估图表 --------
plot_multiclass_roc(all_targets, all_probs, classes, report_dir / 'roc_curve.png')
plot_multiclass_pr(all_targets, all_probs, classes, report_dir / 'pr_curve.png')
plot_confusion_matrix(all_targets, all_preds, classes, report_dir / 'confusion_matrix.png')

# -------- 绘制聚类图 --------
if all_features is not None:
    plot_clustering(all_features, all_targets, classes, report_dir / 'umap_clustering.png', method='umap')
    plot_clustering(all_features, all_targets, classes, report_dir / 'pca_clustering.png', method='pca')
    print("特征聚类图已生成 (UMAP和PCA)")
else:
    print("警告: 未能提取特征，聚类图未生成")

print(f"所有报告图片已保存至 {report_dir.resolve()}")
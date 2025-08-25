import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from sklearn.metrics import classification_report
from tqdm import tqdm

class GatedChannelGroupAttention2d(nn.Module):
    def __init__(self, channels, groups=16, reduction_ratio=4):
        super().__init__()
        self.groups = groups
        self.channels_per_group = channels // groups
        self.gate_conv = nn.Sequential(
            nn.Conv1d(self.channels_per_group, self.channels_per_group // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv1d(self.channels_per_group // reduction_ratio, self.channels_per_group, 1),
            nn.Sigmoid()
        )
        self.cross_gate = nn.Sequential(
            nn.Linear(groups, groups // 2),
            nn.ReLU(),
            nn.Linear(groups // 2, groups),
            nn.Sigmoid()
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch, C, H, W = x.shape
        x = self._apply_gating(x, first_pass=True)
        x = self._full_channel_shuffle(x)
        x = self._apply_gating(x, first_pass=False)
        x = self._cross_group_interaction(x)
        return x

    def _apply_gating(self, x, first_pass=True):
        batch, C, H, W = x.shape
        x_grouped = x.view(batch, self.groups, self.channels_per_group, H, W)
        attended = []
        for i in range(self.groups):
            group_feat = x_grouped[:, i]
            pooled = self.gap(group_feat).view(batch, -1)
            gate = self.gate_conv(pooled.unsqueeze(-1)).squeeze(-1)
            attended.append(group_feat * gate.view(batch, -1, 1, 1))
        return torch.cat(attended, dim=1)

    def _full_channel_shuffle(self, x):
        batch, C, H, W = x.shape
        x = x.view(batch, self.groups, self.channels_per_group, H, W)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, C, H, W)

    def _cross_group_interaction(self, x):
        batch, C, H, W = x.shape
        assert C == self.groups * self.channels_per_group, \
            f"Channels {C} != groups {self.groups} * channels_per_group {self.channels_per_group}"
        x_grouped = x.view(batch, self.groups, self.channels_per_group, H, W)
        group_features_list = []
        for g in range(self.groups):
            feat = x_grouped[:, g]
            pooled = self.gap(feat)
            flattened = pooled.squeeze(-1).squeeze(-1)
            group_features_list.append(flattened)
        group_features = torch.stack(group_features_list, dim=1)
        global_features = group_features.mean(dim=2)
        group_weights = self.cross_gate(global_features)
        weighted = x_grouped * group_weights.view(batch, self.groups, 1, 1, 1)
        return weighted.permute(0, 2, 1, 3, 4).contiguous().view(batch, C, H, W)

class MarineSpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = []

        for label_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = label_idx
                self.idx_to_class.append(class_name)
                for npy_file in os.listdir(class_path):
                    if npy_file.endswith('.npy'):
                        self.data_paths.append(os.path.join(class_path, npy_file))
                        self.labels.append(label_idx)

        print(f"Found {len(self.data_paths)} samples belonging to {len(self.class_to_idx)} classes.")
        print(f"Class mapping: {self.class_to_idx}")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        npy_path = self.data_paths[idx]
        label = self.labels[idx]
        data = np.load(npy_path).astype(np.float32)

        if data.ndim == 2:
            data = np.expand_dims(data, axis=0)
        elif data.ndim == 1:
            data = np.expand_dims(data, axis=0)
            data = np.expand_dims(data, axis=0)

        data = torch.from_numpy(data)

        if self.transform:
            data = self.transform(data)
        return data, label

class SpectrogramClassifier(nn.Module):
    def __init__(self, num_classes, input_channels=1, initial_features=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, initial_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_features),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(initial_features, initial_features * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_features * 2),
            nn.ReLU(),
            GatedChannelGroupAttention2d(channels=initial_features * 2, groups=16),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(initial_features * 2, initial_features * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(initial_features * 4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self._set_classifier(num_classes, input_channels, initial_features)

    def _set_classifier(self, num_classes, input_channels, initial_features):
        with torch.no_grad():
            root_dir = './npy_specs_center'
            dummy_input_height = 32
            dummy_input_width = 32
            dummy_input_channels = input_channels

            if os.path.exists(root_dir) and os.listdir(root_dir):
                first_class_dir = os.path.join(root_dir, os.listdir(root_dir)[0])
                if os.path.isdir(first_class_dir) and os.listdir(first_class_dir):
                    first_npy = np.load(os.path.join(first_class_dir, os.listdir(first_class_dir)[0]))
                    if first_npy.ndim == 2:
                        dummy_input_channels = 1
                        dummy_input_height, dummy_input_width = first_npy.shape
                    elif first_npy.ndim == 3:
                        dummy_input_channels, dummy_input_height, dummy_input_width = first_npy.shape
                    else:
                        dummy_input_channels = 1
                        dummy_input_height = 1
                        dummy_input_width = first_npy.shape[0]
            else:
                print("Warning: Dataset directory not found or empty. Using default input dimensions (1, 32, 32).")

            dummy_input = torch.randn(1, dummy_input_channels, dummy_input_height, dummy_input_width)
            output_features = self.features(dummy_input)
            flattened_size = output_features.view(output_features.size(0), -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        num_epochs,
        device,
        class_names
):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            train_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")

        model.eval()
        val_predictions = []
        val_true_labels = []
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_true_labels.extend(labels.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())
                val_total_samples += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()
                val_bar.set_postfix(loss=loss.item())

        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_accuracy = val_correct_predictions / val_total_samples
        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}")
        print("\nClassification Report (Validation Set):")
        print(classification_report(val_true_labels, val_predictions, target_names=class_names))

if __name__ == "__main__":
    # --- Configuration ---
    DATA_ROOT_DIR = './npy_specs_center'
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    VALIDATION_SPLIT = 0.2
    RANDOM_SEED = 42 # For reproducible split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Splitting ---
    dataset = MarineSpectrogramDataset(root_dir=DATA_ROOT_DIR)
    num_classes = len(dataset.class_to_idx)
    class_names = dataset.idx_to_class

    # Calculate split sizes
    total_size = len(dataset)
    val_size = int(VALIDATION_SPLIT * total_size)
    train_size = total_size - val_size

    # Split dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Determine input channels dynamically
    input_channels = 1
    if hasattr(dataset, 'data_paths') and len(dataset.data_paths) > 0:
        sample_data = np.load(dataset.data_paths[0]).astype(np.float32)
        if sample_data.ndim == 3:
            input_channels = sample_data.shape[0]
        # else: default to 1 as handled in dataset if ndim is 2 or 1
    print(f"Detected input channels: {input_channels}")

    # --- Model, Loss, Optimizer ---
    model = SpectrogramClassifier(num_classes=num_classes, input_channels=input_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.6f}M")

    # --- Training ---
    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        NUM_EPOCHS,
        device,
        class_names
    )
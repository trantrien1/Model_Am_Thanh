import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, features, rows, statistics, augment=False):
        self.features = features
        self.rows = rows
        self.mean = torch.tensor(statistics["mean"], dtype=torch.float32).view(-1, 1)
        self.std = torch.tensor(statistics["std"], dtype=torch.float32).view(-1, 1)
        self.augment = augment
        if augment and any(row["split"] != "train" for row in rows):
            raise ValueError("Augmentation is train-only.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        feature = torch.from_numpy(np.array(self.features[index], dtype=np.float32, copy=True))
        feature = (feature - self.mean) / self.std
        if self.augment:
            # Light spectrogram masking; no waveform crop and no inserted-word deletion.
            for axis, maximum in ((0, 4), (1, 8)):
                width = int(torch.randint(0, maximum + 1, (1,)).item())
                start = int(torch.randint(0, feature.shape[axis] - width + 1, (1,)).item())
                if axis == 0:
                    feature[start:start + width, :] = 0
                else:
                    feature[:, start:start + width] = 0
        return feature.unsqueeze(0), torch.tensor(float(self.rows[index]["label"]), dtype=torch.float32)

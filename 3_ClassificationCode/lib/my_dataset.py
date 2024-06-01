
import torch
from torch.utils.data import Dataset

class CustomTableDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        df: Pandas DataFrame，with columns named "label"
        transform: data preprocessing
        """
        super().__init__()
        self.df = df.drop(columns=['label'])  # data
        self.labels = df['label'].values  # labels
        self.transform = transform

    def __len__(self):
        # get the number of samples
        return len(self.df)

    def __getitem__(self, idx):
        # get samples by index = idx
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get data and labels by index
        features = self.df.iloc[idx].values
        label = self.labels[idx]

        # numpy array to tensor
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        # apply transform
        if self.transform:
            features = self.transform(features)

        return features, label


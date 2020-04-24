from torch.utils.data import Dataset, DataLoader
import numpy as np


class DummyDataset(Dataset):
    def __init__(self):
        self.len = 10000
        # channel first!!
        self.contents = \
            [(np.random.rand(160*128).reshape(1, -1).astype(np.float32), -1)] * self.len
            # [(np.arange(160*128).reshape(1, -1).astype(np.float32), -1),
            #  (np.arange(160*128).reshape(1, -1).astype(np.float32), -1)]  # channels first

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.contents[idx]


def get_dummy_data_loader(batch_size):
    return DataLoader(DummyDataset(), batch_size=batch_size, shuffle=True)


class OverfitWaikatoSnippet(Dataset):
    def __init__(self):
        data = np.load("/shared_volume/preprocessed_4m_waikato.npy")
        seq_len = 160*128
        offset = np.random.randint(0, seq_len//2)
        pruned_data = data[offset:-((len(data) - offset) % seq_len)]
        assert len(pruned_data) % seq_len == 0, len(pruned_data)
        self.len = len(pruned_data) // seq_len
        self.contents = np.split(pruned_data.reshape(1, -1).astype(np.float32), self.len, axis=1)
        self.contents = [(c, -1) for c in self.contents]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.contents[idx]


def get_waikato_snippet(batch_size):
    return DataLoader(OverfitWaikatoSnippet(), batch_size=batch_size)

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler
import numpy as np


class OffsetSampler(Sampler):
    def __init__(self, num_datapoints, seq_len):
        offset = np.random.randint(0, seq_len//2)
        indices = np.arange(offset, num_datapoints, seq_len)
        if (num_datapoints - offset) % seq_len != 0:
            indices = indices[:-1]
        np.random.shuffle(indices)
        self.shuffled_indices = indices

    def __iter__(self):
        for split in self.shuffled_indices:
            yield split

    def __len__(self):
        return len(self.shuffled_indices)


class DummyDataset(Dataset):
    def __init__(self):
        self.len = 10000
        # channel first!!
        self.contents = \
            [(np.random.rand(160*128).reshape(1, -1).astype(np.float32), -1)] * self.len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.contents[idx]


def get_dummy_data_loader(batch_size):
    return DataLoader(DummyDataset(), batch_size=batch_size, shuffle=True)


class OverfitWaikatoSnippet(Dataset):
    def __init__(self):
        self.contents = np.load("/shared_volume/preprocessed_4m_waikato.npy")
        self.contents = self.contents.astype(np.float32)
        self.seq_len = 160*128

    def __len__(self):
        return len(self.contents) - self.seq_len

    def __getitem__(self, idx):
        return (self.contents[idx:idx+self.seq_len].reshape(1, -1), -1)

    def metadata(self):
        return "Waikato peak hours, 4 million IATs"

    def getDataLoader(self, batchSize):
        sampler = OffsetSampler(len(self.contents), self.seq_len)
        return DataLoader(self, sampler=sampler, batch_size=batchSize, drop_last=True)

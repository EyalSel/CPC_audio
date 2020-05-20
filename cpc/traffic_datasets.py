from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import Sampler
import numpy as np
from pathlib import Path


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


def get_dummy_data_loader(batch_size):
    return DataLoader(DummyDataset(), batch_size=batch_size, shuffle=True)


class OverfitWaikatoSnippet(Dataset):
    def __init__(self, contents=None):
        self.contents = contents
        if self.contents is None:
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


class MultitraceOffsetSampler(Sampler):
    def __init__(self, len_per_dataset, seq_len):

        def get_indices(num_datapoints, base_start):
            offset = np.random.randint(0, seq_len//2)
            indices = np.arange(offset, num_datapoints, seq_len)+base_start
            if (num_datapoints - offset) % seq_len != 0:
                indices = indices[:-1]
            return indices

        base_indices = np.cumsum(
            np.concatenate([[0], np.array(len_per_dataset)[1:] - seq_len+1]))
        self.all_indices = np.concatenate([get_indices(*pair) for pair in
                                           zip(len_per_dataset, base_indices)])
        np.random.shuffle(self.all_indices)

    def __iter__(self):
        for split in self.all_indices:
            yield split

    def __len__(self):
        return len(self.all_indices)


class TraceSnippet(Dataset):
    seq_len = 160*128

    def __init__(self, contents):
        self.contents = contents.astype(np.float32)

    def __len__(self):
        return len(self.contents) - self.seq_len + 1

    def __getitem__(self, idx):
        return (self.contents[idx:idx+self.seq_len].reshape(1, -1), -1)

    def metadata(self):
        return "Trace Snippet"

    def getDataLoader(self, batchSize):
        sampler = OffsetSampler(len(self.contents), self.seq_len)
        return DataLoader(self, sampler=sampler, batch_size=batchSize, drop_last=True)


def combined_trace_dataloader(deltas_contents_list, batchSize):
    datasets = [TraceSnippet(deltas) for deltas in deltas_contents_list]
    combined = ConcatDataset(datasets)
    sampler = MultitraceOffsetSampler([len(d) for d in deltas_contents_list], TraceSnippet.seq_len)
    return DataLoader(combined, sampler=sampler, batch_size=batchSize, drop_last=True)


class CombinedDatasetObject(Dataset):

    def __init__(self, dataset_path, mode):
        assert mode in ["train", "validation"], mode
        self.dataset_path = Path(dataset_path)
        dataset_path = Path(dataset_path)
        if mode == "train":
            trace_files = dataset_path.glob("train/*.npy")
        else:
            trace_files = dataset_path.glob("val/*.npy")
        deltas_contents_list = [np.load(f) for f in trace_files]
        self.deltas_contents_lengths = [len(contents) for contents in deltas_contents_list]
        datasets = [TraceSnippet(contents) for contents in deltas_contents_list]
        self.combined_dataset = ConcatDataset(datasets)

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, idx):
        self.combined_dataset[idx]

    def metadata(self):
        return "Dataset from {}".format(self.dataset_path)

    def getDataLoader(self, batchSize):
        sampler = MultitraceOffsetSampler(self.deltas_contents_lengths, TraceSnippet.seq_len)
        return DataLoader(self.combined_dataset, sampler=sampler, batch_size=batchSize, drop_last=True)

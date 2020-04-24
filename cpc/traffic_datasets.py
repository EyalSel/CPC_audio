from torch.utils.data import Dataset, DataLoader


class DummyDataset(Dataset):
    def __init__(self):
        self.contents = [(None, None), (None, None)]

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        self.contents[idx]


def get_dummy_data_loader():
    return DataLoader(DummyDataset(), batch_size=1, shuffle=True)

from torch.utils.data import Dataset
import torch
import os
from PIL import Image


class TrashDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, pandas_df, root_dir, transform=None):
        """
        Args:
            pandas_df: a Pandas dataframe containing files info with their annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.trash_frame = pandas_df
        self.root_dir = root_dir
        self.transform = transform
        self.classes = self.trash_frame['label'].unique()

    def __len__(self):
        return len(self.trash_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.trash_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image = image.convert('RGB')
        target = self.trash_frame.loc[idx, 'cat_index']

        if self.transform:
            image = self.transform(image)

        return image, target

    def get_column_obs(self, col_name):
        return self.trash_frame[col_name]
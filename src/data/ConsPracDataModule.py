import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from src.data.ConsPrac import ConsPrac
from src.data.utils import get_split_idxs


class ConsPracDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # self.dims = (1, 28, 28)
        # self.num_classes = 10

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            x_fit_df = pd.read_csv(
                self.data_dir + "/train_features.csv", index_col="id"
            )
            y_fit_df = pd.read_csv(self.data_dir + "/train_labels.csv", index_col="id")

            train_idxs, validation_idxs = get_split_idxs(
                df=x_fit_df,
                train_frac=0.8,
                random_state=23,
            )

            consprac_train = ConsPrac(x_fit_df.filepath.to_frame(), y_fit_df)

            self.consprac_train = Subset(consprac_train, train_idxs)
            self.consprac_val = Subset(consprac_train, validation_idxs)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            x_test_df = pd.read_csv(
                self.data_dir + "/test_features.csv", index_col="id"
            )
            self.consprac_test = ConsPrac(x_test_df.filepath.to_frame())

    def train_dataloader(self):
        return DataLoader(self.consprac_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.consprac_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.consprac_test, batch_size=self.batch_size)

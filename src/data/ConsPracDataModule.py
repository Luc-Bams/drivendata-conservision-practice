import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.config import Config
from src.data.ConsPrac import ConsPrac
from src.data.utils import get_split_idxs


class ConsPracDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = Config.DATA_DIR, batch_size: int = Config.BATCH_SIZE
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            fit = pd.read_csv(
                f"{self.data_dir}/{Config.DATA_FIT}", index_col=Config.INDEX_COLUMN
            )
            x_fit = fit.filepath.to_frame()

            y_fit = fit[Config.FEATURE_COLUMNS]

            train_idxs, validation_idxs = get_split_idxs(
                df=fit,
                train_frac=0.8,
                random_state=23,
            )

            x_train = x_fit.loc[train_idxs]
            y_train = y_fit.loc[train_idxs]
            x_val = x_fit.loc[validation_idxs]
            y_val = y_fit.loc[validation_idxs]

            self.consprac_train = ConsPrac(x_train, y_train)
            self.consprac_val = ConsPrac(x_val, y_val)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            test = pd.read_csv(
                f"{self.data_dir}/{Config.DATA_TEST}", index_col=Config.INDEX_COLUMN
            )
            x_test = test.filepath.to_frame()

            self.consprac_test = ConsPrac(x_test, None)

    def train_dataloader(self):
        return DataLoader(
            self.consprac_train,
            batch_size=self.batch_size,
            num_workers=8,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.consprac_val,
            batch_size=self.batch_size,
            num_workers=8,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(self.consprac_test, batch_size=self.batch_size, num_workers=8)

import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from src.config import Config
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
            fit = pd.read_csv(f"{self.data_dir}/fit.csv", index_col="id")
            x_fit = fit.filepath.to_frame()

            feature_columns = [
                "antelope_duiker",
                "bird",
                "blank",
                "civet_genet",
                "hog",
                "leopard",
                "monkey_prosimian",
                "rodent",
            ]
            y_fit = fit[feature_columns]

            sgkf = StratifiedGroupKFold(
                n_splits=2, shuffle=True, random_state=Config.RANDOM_STATE
            )
            for i, (idxs_1, idxs_2) in enumerate(
                sgkf.split(fit, y_fit.idxmax(axis=1), fit["site"])
            ):
                train_idxs = idxs_1 if len(idxs_1) > len(idxs_2) else idxs_2
                validation_idxs = idxs_2 if len(idxs_1) > len(idxs_2) else idxs_1
                print(
                    f"Train: {len(train_idxs)} ({len(train_idxs)/len(fit.index):.2f}), Validation: {len(validation_idxs)} ({len(validation_idxs)/len(fit.index):.2f})"
                )
                print(
                    f"# Overlapping sites: {len(set(fit.iloc[train_idxs].site) & set(fit.iloc[validation_idxs].site))}"
                )
                break

            x_train = x_fit.iloc[train_idxs]
            y_train = y_fit.iloc[train_idxs]
            x_val = x_fit.iloc[validation_idxs]
            y_val = y_fit.iloc[validation_idxs]

            self.consprac_train = ConsPrac(x_train, y_train)
            self.consprac_val = ConsPrac(x_val, y_val)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            test = pd.read_csv(f"{self.data_dir}/test_features.csv", index_col="id")
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

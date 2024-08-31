import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class ConsPrac(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(self, x_df: pd.DataFrame, y_df: pd.DataFrame = None, augment=False):

        self.data = x_df
        self.label = y_df

        if augment:
            self.transform = v2.Compose(
                [
                    v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
                    v2.ToDtype(
                        torch.uint8, scale=True
                    ),  # optional, most input are already uint8 at this point
                    v2.Resize(size=(224, 224), antialias=True),
                    v2.RandomHorizontalFlip(p=0.25),
                    v2.ToDtype(
                        torch.float32, scale=True
                    ),  # Normalize expects float input
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.Resize((224, 224)),
                    v2.ToTensor(),
                    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

    def __getitem__(self, index):
        image = Image.open(self.data.iloc[index]["filepath"]).convert("RGB")
        image = self.transform(image)
        image_id = self.data.index[index]
        # if we don't have labels (e.g. for test set) just return the image and image id
        if self.label is None:
            sample = {"image_id": image_id, "image": image}
        else:
            label = torch.tensor(self.label.iloc[index].values, dtype=torch.float)
            sample = {
                "image_id": image_id,
                "image": image,
                "label": label,
            }
        return sample

    def __len__(self):
        return len(self.data)

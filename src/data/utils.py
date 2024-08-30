import pandas as pd


def get_split_idxs(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    # val_frac: float = 0.2,
    random_state: int = 23,
):
    # Calculate the number of images per site
    site_counts = df["site"].value_counts()
    site_counts.columns = ["site", "count"]

    # Total number of images
    total_images = site_counts.sum()

    # Desired number of images for training and validation splits
    train_target = total_images * train_frac
    # val_target = total_images * val_frac

    # # Shuffle the sites to randomize assignment
    site_counts = site_counts.sample(frac=1, random_state=random_state)

    # # Initialize variables to hold training and validation data
    train_sites = []
    val_sites = []
    # test_sites = []
    train_count = 0
    val_count = 0
    # test_count = 0

    # # Iterate over shuffled sites and assign them to either train or val set
    for index, value in site_counts.items():
        if train_count + value <= train_target:
            train_sites.append(index)
            train_count += value
        else:
            val_sites.append(index)
            val_count += value
        # else:
        #     test_sites.append(index)
        #     test_count += value

    train_idxs = df[df["site"].isin(train_sites)].index
    validation_idxs = df[df["site"].isin(val_sites)].index
    # test_idxs = df[df["site"].isin(test_sites)].index

    return train_idxs, validation_idxs
    # return train_idxs, validation_idxs, test_idxs

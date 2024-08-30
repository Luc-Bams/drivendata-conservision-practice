import pandas as pd


def get_split_idxs(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    random_state: int = 23,
):
    # Calculate the number of images per site
    site_counts = df["site"].value_counts().reset_index()
    site_counts.columns = ["site", "count"]

    # Total number of images
    total_images = site_counts["count"].sum()

    # Desired number of images for training and validation splits
    train_target = total_images * train_frac

    # # Shuffle the sites to randomize assignment
    site_counts = site_counts.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )

    # # Initialize variables to hold training and validation data
    train_sites = []
    val_sites = []
    train_count = 0
    val_count = 0

    # # Iterate over shuffled sites and assign them to either train or val set
    for index, row in site_counts.iterrows():
        site = row["site"]
        count = row["count"]

        if train_count + count <= train_target:
            train_sites.append(site)
            train_count += count
        else:
            val_sites.append(site)
            val_count += count

    df.reset_index(inplace=True)
    train_idxs = df[df["site"].isin(train_sites)].index
    validation_idxs = df[df["site"].isin(val_sites)].index

    return train_idxs, validation_idxs

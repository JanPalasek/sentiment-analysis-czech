import csv
import os
import random

from .utils import write_into_file, train_val_test_split


def preprocess(in_csv_paths, out_dir_path, val_split, test_split):
    dataset = []
    for path in in_csv_paths:
        with open(path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                text, label = row["text"], row["label"]
                dataset.append((text, label))

    random.shuffle(dataset)

    train_dataset, val_dataset, test_dataset = train_val_test_split(dataset, val_split, test_split)

    write_into_file(dataset, out_path=os.path.join(out_dir_path, "data_full.csv"))
    write_into_file(train_dataset, out_path=os.path.join(out_dir_path, "data_train.csv"))
    write_into_file(val_dataset, out_path=os.path.join(out_dir_path, "data_val.csv"))
    write_into_file(test_dataset, out_path=os.path.join(out_dir_path, "data_test.csv"))

